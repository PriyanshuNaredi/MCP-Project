"""Phase 3 manual test harness: verify the fallback chain in run_agent().

Run with:  python -m tests.test_agent

Note on realism: the real rag_retriever (ChromaDB similarity_search) always
returns the top-k nearest chunks for ANY query, so it never naturally falls
below the 100-char threshold and would always win the chain. To verify that
the web_search and wikipedia fallback stages are actually reachable, tests 2
and 3 substitute empty-returning stubs for the earlier tools -- simulating the
real condition "that tool found nothing usable". The fallback logic under test
(run_agent / gather_context) is exercised unmodified.
"""

import asyncio

from agent.core import run_agent
from tools.registry import ToolRegistry
from tools.rag_tool import rag_tool
from tools.web_search_tool import web_search_tool
from tools.wikipedia_tool import wikipedia_tool


async def _empty_tool(query: str) -> str:
    """Stub tool that always finds nothing (returns empty string)."""
    return ""


def make_registry(rag=rag_tool, web=web_search_tool, wiki=wikipedia_tool) -> ToolRegistry:
    """Build a registry, optionally swapping in stubs for specific tools."""
    registry = ToolRegistry()
    registry.register("rag_retriever", rag, "Search local document knowledge base")
    registry.register("web_search", web, "Search the web via Tavily")
    registry.register("wikipedia", wiki, "Retrieve Wikipedia article summaries")
    return registry


def _banner(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


async def run_case(label: str, query: str, registry: ToolRegistry, expected: str):
    _banner(f"{label}\nQUERY: {query!r}\nEXPECTED WINNER: {expected}")

    sources_tried: list = []
    chunks = []
    async for chunk in run_agent(query, registry, llm=None, sources_tried=sources_tried):
        chunks.append(chunk)
    context = "".join(chunks)

    winner = sources_tried[-1] if sources_tried else "(none)"
    print(f"Tools tried (in order): {sources_tried}")
    print(f"Winning tool:           {winner}")
    print(f"Context length:         {len(context)} chars")
    preview = context[:500] + ("..." if len(context) > 500 else "")
    print(f"\nContext returned:\n{preview}")


async def main():
    # --- Test 1: RAG should answer (query about a topic in the local docs) ---
    await run_case(
        "TEST 1 — RAG should win",
        "What is retrieval-augmented generation and what are the stages of a RAG pipeline?",
        make_registry(),
        expected="rag_retriever",
    )

    # --- Test 2: RAG finds nothing -> web_search should answer ---
    await run_case(
        "TEST 2 — RAG empty, web_search should win",
        "What are the latest large language model releases and news in 2026?",
        make_registry(rag=_empty_tool),
        expected="web_search",
    )

    # --- Test 3: RAG + web empty -> wikipedia should answer ---
    await run_case(
        "TEST 3 — RAG + web empty, wikipedia should win",
        "Albert Einstein",
        make_registry(rag=_empty_tool, web=_empty_tool),
        expected="wikipedia",
    )


if __name__ == "__main__":
    asyncio.run(main())
