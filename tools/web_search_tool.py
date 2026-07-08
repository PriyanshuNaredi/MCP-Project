"""Web search tool: queries the Tavily API and returns top result snippets."""

import asyncio
import os

from dotenv import load_dotenv

load_dotenv()

_TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

_client = None


def _get_client():
    """Lazily construct (and cache) the Tavily client; None if no valid key."""
    global _client
    if _client is not None:
        return _client
    if not _TAVILY_API_KEY or _TAVILY_API_KEY.startswith("tvly-...") or _TAVILY_API_KEY == "":
        return None
    try:
        from tavily import TavilyClient

        _client = TavilyClient(api_key=_TAVILY_API_KEY)
    except Exception as e:
        print(f"[web_search_tool] client init failed: {e}")
        return None
    return _client


async def web_search_tool(query: str, max_results: int = 3) -> str:
    """Search the web via Tavily and return up to *max_results* snippets.

    Returns an empty string on missing key, network error, or no results.
    """
    client = _get_client()
    if client is None:
        print("[web_search_tool] no valid TAVILY_API_KEY configured")
        return ""

    try:
        # Tavily's client is synchronous; run it off the event loop.
        response = await asyncio.to_thread(
            client.search,
            query=query,
            max_results=max_results,
        )
    except Exception as e:
        print(f"[web_search_tool] search failed: {e}")
        return ""

    results = response.get("results", []) if isinstance(response, dict) else []
    if not results:
        return ""

    parts = []
    for i, r in enumerate(results, 1):
        title = r.get("title", "").strip()
        content = r.get("content", "").strip()
        url = r.get("url", "").strip()
        parts.append(f"[Result {i}] {title}\n{content}\nSource: {url}")

    return "\n\n".join(parts)
