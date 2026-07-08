"""Phase 6 manual test: exercise the MCP server through the MCP protocol.

Launches mcp_server/server.py as a stdio subprocess and talks to it as a real
MCP client — list_tools() then call_tool() for each tool — proving the MCP
layer works end to end, not just the underlying tool functions.

Run with:  python -m tests.test_mcp_server
"""

import asyncio
import sys

from mcp import types
from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

# Launch the server with the SAME interpreter running this test (venv-aware).
SERVER_PARAMS = StdioServerParameters(
    command=sys.executable,
    args=["-m", "mcp_server.server"],
)

CALLS = [
    ("rag_retriever", "What is retrieval-augmented generation?"),
    ("web_search", "latest large language model releases 2026"),
    ("wikipedia", "Alan Turing"),
]


def _banner(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


async def main():
    async with stdio_client(SERVER_PARAMS) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            _banner("list_tools() over MCP")
            tools = await session.list_tools()
            for t in tools.tools:
                print(f"  - {t.name}: {t.description}")

            for name, query in CALLS:
                _banner(f"call_tool({name!r}, query={query!r}) over MCP")
                result = await session.call_tool(name, {"query": query})
                for content in result.content:
                    if isinstance(content, types.TextContent):
                        text = content.text
                        preview = text[:400] + ("..." if len(text) > 400 else "")
                        print(f"[chars: {len(text)}]\n{preview}")


if __name__ == "__main__":
    asyncio.run(main())
