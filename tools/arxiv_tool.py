"""arXiv search tool: queries the arXiv API for academic papers and returns titles + abstracts.

Onboarding proof for the '<30 min to add a new tool' claim.  Follows the same
async-wrapper-around-sync-client pattern used by every other tool in the project.
"""

import asyncio


def _search_arxiv(query: str, max_results: int = 3) -> str:
    """Search arXiv for *query* and return formatted paper summaries (sync)."""
    import arxiv  # lazy import so the rest of the app doesn't break if not installed

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance,
    )

    parts = []
    for i, result in enumerate(client.results(search), 1):
        authors = ", ".join(a.name for a in result.authors[:3])
        if len(result.authors) > 3:
            authors += f" et al. ({len(result.authors)} authors)"
        parts.append(
            f"[Paper {i}] {result.title}\n"
            f"Authors: {authors}\n"
            f"Published: {result.published.strftime('%Y-%m-%d')}\n"
            f"Abstract: {result.summary[:400].strip()}\n"
            f"URL: {result.entry_id}"
        )

    return "\n\n".join(parts)


async def arxiv_tool(query: str, max_results: int = 3) -> str:
    """Search arXiv for papers matching *query* and return formatted summaries.

    Returns an empty string if the API errors out or finds nothing.
    """
    try:
        return await asyncio.to_thread(_search_arxiv, query, max_results)
    except Exception as e:
        print(f"[arxiv_tool] search failed: {e}")
        return ""
