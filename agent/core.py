"""Agent core: routes a query through the tool fallback chain, then answers.

Fallback order: rag_retriever -> web_search -> wikipedia. The first tool that
returns >= 100 chars within the timeout wins and its output becomes the context.
"""

import asyncio
from typing import AsyncGenerator

from tools.registry import ToolRegistry

FALLBACK_CHAIN = ["rag_retriever", "web_search", "wikipedia", "arxiv_search"]
TOOL_TIMEOUT = 5.0
MIN_USABLE_CHARS = 100


async def gather_context(
    query: str,
    registry: ToolRegistry,
    sources_tried: list | None = None,
) -> str:
    """Run the fallback chain and return the first usable context string.

    Appends per-tool outcomes ("rag_retriever", "web_search:failed", ...) to
    *sources_tried* if a list is provided. Returns "" if no tool succeeds.
    """
    if sources_tried is None:
        sources_tried = []

    context = ""
    for tool_name in FALLBACK_CHAIN:
        try:
            tool_fn = registry.get(tool_name)
            result = await asyncio.wait_for(tool_fn(query), timeout=TOOL_TIMEOUT)
        except asyncio.TimeoutError:
            sources_tried.append(f"{tool_name}:timeout")
            continue
        except Exception:
            sources_tried.append(f"{tool_name}:failed")
            continue

        if result and len(result) >= MIN_USABLE_CHARS:
            context = result
            sources_tried.append(tool_name)
            break

        # Ran but returned nothing usable (too short / empty).
        sources_tried.append(f"{tool_name}:insufficient")

    return context


async def run_agent(
    query: str,
    registry: ToolRegistry,
    llm=None,
    sources_tried: list | None = None,
) -> AsyncGenerator[str | dict, None]:
    """Route *query* through the fallback chain and yield the answer.

    The **first** yielded value is always a metadata dict:
        {"tool_used": "rag_retriever"|...|None, "context_chars": int,
         "sources_tried": [...]}
    All subsequent yields are str chunks (the answer text).

    This lets callers (main.py, test clients) distinguish tool-grounded
    answers from LLM-parametric fallbacks.
    """
    if sources_tried is None:
        sources_tried = []

    context = await gather_context(query, registry, sources_tried)

    # Determine which tool actually provided the context.
    # The winning tool is the last entry in sources_tried that doesn't
    # end with :failed/:timeout/:insufficient — i.e., the one that
    # returned usable context.
    tool_used = None
    for entry in sources_tried:
        if ":" not in entry:  # no suffix = success
            tool_used = entry

    metadata = {
        "tool_used": tool_used,
        "context_chars": len(context),
        "sources_tried": list(sources_tried),
    }
    yield metadata

    if llm is None:
        # No LLM yet — surface the raw context for verification.
        if context:
            yield context
        else:
            yield "No usable context found from any tool."
        return

    # Stream the LLM's answer.
    async for chunk in llm.stream(query=query, context=context):
        yield chunk

