"""MCP server: exposes the 3 research tools over the MCP protocol via stdio.

Reuses the exact same tool callables the FastAPI agent uses (through a
ToolRegistry), so there is a single source of truth for tool logic.
"""

import asyncio
import sys

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

from tools.registry import ToolRegistry
from tools.rag_tool import rag_tool
from tools.web_search_tool import web_search_tool
from tools.wikipedia_tool import wikipedia_tool
from tools.arxiv_tool import arxiv_tool

# --- Single source of truth: the same registry/tools used by the agent. ---
registry = ToolRegistry()
registry.register("rag_retriever", rag_tool, "Search the local document knowledge base (RAG)")
registry.register("web_search", web_search_tool, "Search the web for current information via Tavily")
registry.register("wikipedia", wikipedia_tool, "Retrieve a Wikipedia article summary")
registry.register("arxiv_search", arxiv_tool, "Search arXiv for academic papers")

server = Server("research-assistant")

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "description": "The search query or question to answer.",
        }
    },
    "required": ["query"],
}


@server.list_tools()
async def list_tools() -> list[types.Tool]:
    """Advertise the registered tools (name + description + input schema)."""
    return [
        types.Tool(name=name, description=description, inputSchema=_QUERY_SCHEMA)
        for name, description in registry.all_tools().items()
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    """Execute a registered tool by name and return its output as text."""
    try:
        tool_fn = registry.get(name)
    except KeyError:
        return [types.TextContent(type="text", text=f"Error: unknown tool {name!r}")]

    query = (arguments or {}).get("query", "")
    result = await tool_fn(query)
    if not result:
        result = f"(no result from {name})"
    return [types.TextContent(type="text", text=result)]


async def _prewarm():
    """Load the embedding model + vector store once, before serving requests.

    The sentence-transformers/torch model load is heavy and, if it happens
    lazily during a live tool call, it can block the event loop long enough to
    starve the MCP stdio transport (manifesting as a hang or dropped connection).
    Doing it up front — off the event loop — makes every later rag call fast.
    """
    from rag.retriever import retrieve

    try:
        await asyncio.to_thread(retrieve, "warmup", 1)
        print("[mcp_server] embedding model pre-warmed", file=sys.stderr)
    except Exception as e:
        print(f"[mcp_server] pre-warm skipped: {e}", file=sys.stderr)


async def main():
    await _prewarm()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
