"""Phase 7 -- 20-query metrics suite against the live FastAPI server.

Hits http://localhost:8000/query over SSE for each query, recording:
  - whether the query resolved (non-empty, substantive answer)
  - grounding classification: tool-grounded vs LLM-parametric
  - time-to-first-chunk (TTFC) and total wall-clock time

The server emits an "event: meta" SSE event with JSON payload containing
tool_used, context_chars, and sources_tried before the answer chunks.
This lets us honestly classify whether the answer came from a tool or the
LLM's own knowledge.

Requires the server to be running:
    python -m uvicorn main:app --reload

Run with:
    python -m tests.test_metrics
"""

import asyncio
import json
import statistics
import time

import httpx

BASE_URL = "http://localhost:8000/query"

# -- 20 test queries, categorised by expected primary tool --
# Categories are for REPORTING only; the real fallback chain decides.

QUERIES = [
    # --- RAG-targeted (topics covered in data/docs/) ---
    ("rag", "What is retrieval-augmented generation?"),
    ("rag", "What are the three stages of a RAG pipeline?"),
    ("rag", "How does chunking work in a RAG system and what is chunk overlap?"),
    ("rag", "What is the Model Context Protocol and who created it?"),
    ("rag", "How does MCP use JSON-RPC 2.0 for communication?"),
    ("rag", "What is the stdio transport in MCP?"),
    ("rag", "What is LLM provider failover and why is it important?"),
    ("rag", "How does a circuit breaker pattern work for LLM providers?"),
    ("rag", "What is the difference between RAG and fine-tuning?"),
    ("rag", "What are MCP tools, resources, and prompts?"),

    # --- Web search-targeted (current events not in local docs) ---
    ("web", "What are the latest large language model releases in 2026?"),
    ("web", "Recent developments in AI regulation in the European Union 2026"),
    ("web", "Latest breakthroughs in quantum computing 2026"),
    ("web", "SpaceX Starship latest launch updates 2026"),
    ("web", "New features announced for Python 3.14"),
    ("web", "What companies are leading in AI chip development in 2026?"),

    # --- General knowledge (LLM can answer with parametric + RAG context) ---
    ("general", "Explain the transformer architecture and self-attention mechanism"),
    ("general", "What is cosine similarity and how is it used in vector search?"),
    ("general", "What is the difference between keyword search and semantic search?"),
    ("general", "How does an embedding model convert text to vectors?"),
]


# Phrases that indicate the LLM could not answer from the provided context.
_REFUSAL_PHRASES = [
    "does not contain information",
    "do not contain information",
    "no information about",
    "cannot answer",
    "i don't have",
    "i don't know",
    "not mentioned in the context",
    "not provided in the context",
    "outside the scope",
    "No answer could be generated",
]

# RAG doc topics — if the query is about one of these, RAG context is relevant.
# Used to classify grounded vs parametric when rag_retriever wins the chain.
_RAG_TOPIC_KEYWORDS = [
    "retrieval-augmented generation", "retrieval augmented generation", "rag",
    "model context protocol", "mcp", "json-rpc", "stdio",
    "failover", "fallback", "circuit breaker", "llm provider",
    "chunking", "chunk overlap", "embedding", "vector", "chromadb",
    "fine-tuning", "fine tuning",
    "tools, resources, and prompts",
]


def _is_rag_relevant(query: str) -> bool:
    """Heuristic: does the query match a topic in the local RAG docs?"""
    q_lower = query.lower()
    return any(kw in q_lower for kw in _RAG_TOPIC_KEYWORDS)


async def run_single_query(client: httpx.AsyncClient, query: str, timeout: float = 60.0):
    """Hit /query and collect the SSE stream.

    Returns dict with keys: answer, ttfc_ms, total_ms, resolved, grounding,
    tool_used, context_chars, sources_tried, error.
    """
    start = time.perf_counter()
    ttfc_ms = None
    chunks: list[str] = []
    error = None
    metadata = None

    try:
        async with client.stream("GET", BASE_URL, params={"q": query}, timeout=timeout) as resp:
            resp.raise_for_status()
            current_event_type = None
            async for raw_line in resp.aiter_lines():
                # SSE spec: "event: <type>" sets the event type for the next data.
                if raw_line.startswith("event: "):
                    current_event_type = raw_line[len("event: "):].strip()
                    continue

                if raw_line.startswith("data: "):
                    payload = raw_line[len("data: "):]

                    # Handle the metadata event
                    if current_event_type == "meta":
                        try:
                            metadata = json.loads(payload)
                        except json.JSONDecodeError:
                            pass
                        current_event_type = None
                        continue

                    # Regular data event
                    current_event_type = None
                    if payload == "[DONE]":
                        break
                    if ttfc_ms is None:
                        ttfc_ms = (time.perf_counter() - start) * 1000
                    chunks.append(payload)
    except Exception as e:
        error = str(e)

    total_ms = (time.perf_counter() - start) * 1000
    answer = " ".join(chunks).strip()
    answer_len = len(answer)

    # --- Resolution check ---
    resolved = bool(answer) and answer_len > 20
    if resolved:
        if answer.startswith("Error:"):
            resolved = False
        elif answer_len < 200:
            answer_lower = answer.lower()
            for phrase in _REFUSAL_PHRASES:
                if phrase in answer_lower:
                    resolved = False
                    break

    # --- Grounding classification ---
    # "grounded"   = a tool found relevant context and the LLM likely used it
    # "parametric" = the LLM answered from its own knowledge (tool context was
    #                irrelevant or absent)
    # "none"       = query was not resolved
    tool_used = metadata.get("tool_used") if metadata else None
    context_chars = metadata.get("context_chars", 0) if metadata else 0

    if not resolved:
        grounding = "none"
    elif tool_used and _is_rag_relevant(query):
        # RAG found something AND the query matches a topic in the docs
        grounding = "grounded"
    elif tool_used and tool_used != "rag_retriever":
        # A non-RAG tool won (web_search, wikipedia, arxiv) — always grounded
        grounding = "grounded"
    elif tool_used == "rag_retriever" and not _is_rag_relevant(query):
        # RAG won but query is off-topic — LLM is using parametric knowledge
        grounding = "parametric"
    elif context_chars == 0:
        # No tool returned context — pure parametric
        grounding = "parametric"
    else:
        grounding = "grounded"

    return {
        "answer": answer,
        "ttfc_ms": ttfc_ms,
        "total_ms": total_ms,
        "resolved": resolved,
        "grounding": grounding,
        "tool_used": tool_used,
        "context_chars": context_chars,
        "sources_tried": metadata.get("sources_tried", []) if metadata else [],
        "answer_len": answer_len,
        "error": error,
    }


def compute_stats(values: list[float]) -> dict:
    """Return min / median / mean / max for a list of floats."""
    if not values:
        return {"min": 0, "median": 0, "mean": 0, "max": 0, "count": 0}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "mean": statistics.mean(values),
        "max": max(values),
        "count": len(values),
    }


def print_report(results: list[dict]):
    """Print the final metrics summary with honest grounding breakdown."""
    total = len(results)
    resolved_count = sum(1 for r in results if r["resolved"])
    resolution_pct = (resolved_count / total) * 100 if total else 0

    grounded_count = sum(1 for r in results if r["grounding"] == "grounded")
    parametric_count = sum(1 for r in results if r["grounding"] == "parametric")
    grounded_pct = (grounded_count / total) * 100 if total else 0
    parametric_pct = (parametric_count / total) * 100 if total else 0

    # Collect TTFC values
    all_ttfc = [r["ttfc_ms"] for r in results if r["ttfc_ms"] is not None]
    ttfc_stats = compute_stats(all_ttfc)

    # Per-category breakdown
    cat_names = sorted(set(cat for cat, _ in QUERIES))
    categories = {c: [] for c in cat_names}
    for (cat, _query), result in zip(QUERIES, results):
        categories[cat].append(result)

    cat_labels = {"rag": "RAG queries", "web": "Web Search queries",
                  "general": "General knowledge"}

    print()
    print("=" * 74)
    print("  PHASE 7 METRICS -- 20-QUERY TEST SUITE")
    print("=" * 74)

    # -- Resolution Rate (honest breakdown) --
    print(f"\n{'-' * 74}")
    print(f"  QUERY RESOLUTION RATE (with grounding breakdown)")
    print(f"{'-' * 74}")
    print(f"  Total queries:      {total}")
    print(f"  Resolved (any):     {resolved_count}/{total} ({resolution_pct:.0f}%)")
    print(f"    Tool-grounded:    {grounded_count}/{total} ({grounded_pct:.0f}%)")
    print(f"    LLM-parametric:   {parametric_count}/{total} ({parametric_pct:.0f}%)")
    print(f"  Unresolved:         {total - resolved_count}/{total}")
    print()
    print(f"  * Tool-grounded  = a tool found relevant context, LLM used it")
    print(f"  * LLM-parametric = tool context was irrelevant, LLM used own knowledge")

    # Per-category
    print()
    for cat_name in cat_names:
        cat_results = categories[cat_name]
        cat_resolved = sum(1 for r in cat_results if r["resolved"])
        cat_grounded = sum(1 for r in cat_results if r["grounding"] == "grounded")
        cat_parametric = sum(1 for r in cat_results if r["grounding"] == "parametric")
        cat_total = len(cat_results)
        label = cat_labels.get(cat_name, cat_name)
        print(f"    {label}:")
        print(f"      resolved {cat_resolved}/{cat_total}  "
              f"(grounded={cat_grounded}, parametric={cat_parametric})")

    # -- TTFC Distribution --
    print(f"\n{'-' * 74}")
    print(f"  ENDPOINT-LEVEL TTFC (time-to-first-chunk) -- ALL {total} QUERIES")
    print(f"{'-' * 74}")
    if all_ttfc:
        print(f"  Min:    {ttfc_stats['min']:>8.1f} ms")
        print(f"  Median: {ttfc_stats['median']:>8.1f} ms")
        print(f"  Mean:   {ttfc_stats['mean']:>8.1f} ms")
        print(f"  Max:    {ttfc_stats['max']:>8.1f} ms")
        print(f"  N:      {ttfc_stats['count']:>8d}")
    else:
        print("  (no TTFC data -- all queries failed)")

    # -- Per-category TTFC --
    print(f"\n{'-' * 74}")
    print(f"  TTFC BY QUERY CATEGORY")
    print(f"{'-' * 74}")
    for cat_name in cat_names:
        cat_ttfc = [r["ttfc_ms"] for r in categories[cat_name] if r["ttfc_ms"] is not None]
        stats = compute_stats(cat_ttfc)
        label = cat_labels.get(cat_name, cat_name)
        if cat_ttfc:
            print(f"  {label:20s}  min={stats['min']:>7.1f}  median={stats['median']:>7.1f}  max={stats['max']:>7.1f} ms  (n={stats['count']})")

    # -- Per-query detail table --
    print(f"\n{'-' * 74}")
    print(f"  PER-QUERY DETAIL")
    print(f"{'-' * 74}")
    print(f"  {'#':>2}  {'Cat':7s}  {'TTFC':>7s}  {'Total':>7s}  {'Len':>5s}  {'Ground':8s}  {'Tool':14s}  Query")
    print(f"  {'--':2}  {'-------':7}  {'-------':7}  {'-------':7}  {'-----':5}  {'--------':8}  {'-'*14}  {'-'*24}")
    for i, ((cat, query), result) in enumerate(zip(QUERIES, results), 1):
        ttfc_str = f"{result['ttfc_ms']:.0f}" if result['ttfc_ms'] is not None else "--"
        grnd = result["grounding"][:8]
        tool = (result["tool_used"] or "--")[:14]
        q_short = query[:24] + ("..." if len(query) > 24 else "")
        print(f"  {i:>2}  {cat:7s}  {ttfc_str:>7s}  {result['total_ms']:>7.0f}  {result['answer_len']:>5d}  {grnd:8s}  {tool:14s}  {q_short}")

    print(f"\n{'=' * 74}\n")


async def main():
    print("Phase 7 -- Running 20-query metrics suite against live server")
    print(f"Target: {BASE_URL}")
    print(f"Queries: {len(QUERIES)}")
    print()

    # Verify server is up
    async with httpx.AsyncClient() as client:
        try:
            health = await client.get("http://localhost:8000/health", timeout=5.0)
            health.raise_for_status()
            info = health.json()
            print(f"Server healthy: tools={info.get('tools', '?')}")
        except Exception as e:
            print(f"ERROR: Cannot reach server at http://localhost:8000 -- {e}")
            print("Start the server first:  python -m uvicorn main:app")
            return

    results = []
    async with httpx.AsyncClient() as client:
        for i, (cat, query) in enumerate(QUERIES, 1):
            print(f"  [{i:>2}/{len(QUERIES)}] ({cat:7s}) {query[:50]}...", end="", flush=True)
            result = await run_single_query(client, query)
            grnd_tag = f"[{result['grounding'][:5]}]"
            ttfc_str = f"{result['ttfc_ms']:.0f}ms" if result["ttfc_ms"] is not None else "--"
            print(f"  {grnd_tag:7s}  TTFC={ttfc_str}  total={result['total_ms']:.0f}ms")
            if result["answer"].startswith("Error: 429"):
                print(f"         ^^ Gemini rate limit hit (no OpenAI failover key configured)")
            results.append(result)
            # Gemini free-tier rate limit is 15 RPM. Space queries to stay
            # under the limit without triggering 429 errors.
            if i < len(QUERIES):
                await asyncio.sleep(4.0)

    print_report(results)
    return results


if __name__ == "__main__":
    asyncio.run(main())
