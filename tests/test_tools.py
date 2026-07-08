"""Phase 2 manual test harness: register all 3 tools and call each one.

Run with:  python -m tests.test_tools
Prints each tool's output so you can visually verify they work independently.
"""

import asyncio

from tools.registry import ToolRegistry
from tools.rag_tool import rag_tool
from tools.web_search_tool import web_search_tool
from tools.wikipedia_tool import wikipedia_tool


def build_registry() -> ToolRegistry:
    """Register all three Phase 2 tools into a fresh registry."""
    registry = ToolRegistry()
    registry.register("rag_retriever", rag_tool, "Search local document knowledge base")
    registry.register("web_search", web_search_tool, "Search the web via Tavily")
    registry.register("wikipedia", wikipedia_tool, "Retrieve Wikipedia article summaries")
    return registry


def _banner(title: str):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


async def main():
    registry = build_registry()

    print("Registered tools:")
    for name, desc in registry.all_tools().items():
        print(f"  - {name}: {desc}")

    # One representative query per tool.
    cases = [
        ("rag_retriever", "What is retrieval-augmented generation?"),
        ("web_search", "latest advances in large language models 2025"),
        ("wikipedia", "Transformer (deep learning architecture)"),
    ]

    for tool_name, query in cases:
        _banner(f"TOOL: {tool_name}  |  QUERY: {query!r}")
        tool_fn = registry.get(tool_name)
        result = await tool_fn(query)

        if result:
            print(f"[chars: {len(result)}]\n")
            print(result)
        else:
            print("(empty / no result — tool handled failure gracefully)")


if __name__ == "__main__":
    asyncio.run(main())
