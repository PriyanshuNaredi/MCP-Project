"""RAG retriever tool: queries the local ChromaDB store and formats the hits."""

import asyncio
import sys

from rag.retriever import retrieve_with_metadata


async def rag_tool(query: str, k: int = 3) -> str:
    """Retrieve the top-*k* document chunks for *query* as a formatted string.

    Returns an empty string if nothing is retrieved or the store errors out.
    """
    try:
        # retrieve_with_metadata is synchronous (embedding + Chroma search) and
        # can take seconds on a cold model load; run it off the event loop so it
        # never blocks the async caller (critical for the MCP stdio transport).
        hits = await asyncio.to_thread(retrieve_with_metadata, query, k)
    except Exception as e:
        print(f"[rag_tool] retrieval failed: {e}", file=sys.stderr)
        return ""

    if not hits:
        return ""

    parts = []
    for i, hit in enumerate(hits, 1):
        source = hit.get("metadata", {}).get("source", "unknown")
        text = hit.get("text", "").strip()
        parts.append(f"[Chunk {i} | source: {source}]\n{text}")

    return "\n\n".join(parts)
