"""Wikipedia fallback tool: resolve a title, then fetch its REST summary."""

import asyncio

import requests

_USER_AGENT = "MCP-Research-Assistant/0.1 (portfolio project)"
_OPENSEARCH_URL = "https://en.wikipedia.org/w/api.php"
_REST_SUMMARY_URL = "https://en.wikipedia.org/api/rest_v1/page/summary/"
_HTTP_TIMEOUT = 5


def _resolve_title(query: str) -> str:
    """Return the best-matching article title for *query* via OpenSearch, or ""."""
    resp = requests.get(
        _OPENSEARCH_URL,
        params={
            "action": "opensearch",
            "search": query,
            "limit": 1,
            "namespace": 0,
            "format": "json",
        },
        headers={"User-Agent": _USER_AGENT},
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    titles = resp.json()[1]
    return titles[0] if titles else ""


def _lookup(query: str, max_chars: int) -> str:
    """Resolve the best title, then fetch its intro summary from the REST API."""
    title = _resolve_title(query)
    if not title:
        return ""

    resp = requests.get(
        _REST_SUMMARY_URL + requests.utils.quote(title, safe=""),
        headers={"User-Agent": _USER_AGENT},
        timeout=_HTTP_TIMEOUT,
    )
    resp.raise_for_status()
    extract = resp.json().get("extract", "")
    return extract[:max_chars].strip()


async def wikipedia_tool(query: str, max_chars: int = 1500) -> str:
    """Return the summary of the best-matching Wikipedia article for *query*.

    Returns an empty string if no article is found or the API errors out.
    """
    try:
        return await asyncio.to_thread(_lookup, query, max_chars)
    except Exception as e:
        print(f"[wikipedia_tool] lookup failed: {e}")
        return ""
