"""FastAPI entry point: wires up tools + LLM provider and streams answers over SSE."""

import asyncio
import json
import logging
import os
import time

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import StreamingResponse

from agent.core import run_agent
from agent.llm_provider import LLMProvider
from tools.registry import ToolRegistry
from tools.rag_tool import rag_tool
from tools.web_search_tool import web_search_tool
from tools.wikipedia_tool import wikipedia_tool
from tools.arxiv_tool import arxiv_tool

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("research-agent")

# --- Wire up the registry (Phase 2) and LLM provider (Phase 4) at startup. ---
registry = ToolRegistry()
registry.register("rag_retriever", rag_tool, "Search local document knowledge base")
registry.register("web_search", web_search_tool, "Search the web via Tavily")
registry.register("wikipedia", wikipedia_tool, "Retrieve Wikipedia article summaries")
registry.register("arxiv_search", arxiv_tool, "Search arXiv for academic papers")

llm_provider = LLMProvider(
    gemini_api_key=os.getenv("GEMINI_API_KEY"),
    openai_api_key=os.getenv("OPENAI_API_KEY"),
)

app = FastAPI(title="MCP Research Assistant")


def sse_event(data: str) -> str:
    """Format *data* as a valid SSE event, prefixing every line with 'data: '.

    Splitting on newlines keeps multi-line LLM chunks from breaking the SSE
    frame (a lone '\\n\\n' inside the data would otherwise end the event early).
    """
    lines = data.split("\n")
    return "".join(f"data: {line}\n" for line in lines) + "\n"

def sse_named_event(event_type: str, data: str) -> str:
    """Format a named SSE event (e.g. 'event: meta\\ndata: ...\\n\\n')."""
    lines = data.split("\n")
    return f"event: {event_type}\n" + "".join(f"data: {line}\n" for line in lines) + "\n"


@app.get("/query")
async def query_endpoint(q: str):
    async def event_generator():
        start = time.perf_counter()
        sources_tried: list = []
        first_chunk_logged = False
        got_output = False
        metadata_sent = False

        logger.info(f"Query received: {q!r}")

        try:
            async for item in run_agent(q, registry, llm=llm_provider, sources_tried=sources_tried):
                # First yield is always the metadata dict.
                if isinstance(item, dict) and not metadata_sent:
                    metadata_sent = True
                    yield sse_named_event("meta", json.dumps(item))
                    continue

                chunk = item
                if not first_chunk_logged:
                    ttfc_ms = (time.perf_counter() - start) * 1000
                    logger.info(f"Time-to-first-chunk: {ttfc_ms:.1f} ms (tools tried so far: {sources_tried})")
                    first_chunk_logged = True
                if chunk:
                    got_output = True
                    yield sse_event(chunk)
        except Exception as e:
            # Never let the stream crash the connection — report and close cleanly.
            logger.exception(f"Error while streaming answer for {q!r}")
            yield sse_event(f"Error: {e}")
        finally:
            if not got_output:
                # Fallback chain + LLM produced nothing usable — respond anyway.
                yield sse_event("No answer could be generated for this query.")
            yield sse_event("[DONE]")
            duration_ms = (time.perf_counter() - start) * 1000
            logger.info(
                f"Request complete: {q!r} | tools tried: {sources_tried} | "
                f"total duration: {duration_ms:.1f} ms"
            )

    return StreamingResponse(event_generator(), media_type="text/event-stream")


@app.get("/health")
async def health():
    return {"status": "ok", "tools": list(registry.all_tools().keys())}
