"""Phase 4 manual test harness: verify LLM provider streaming + failover.

Run with:  python -m tests.test_llm_provider

Test 1 — valid Gemini key: confirm a streamed answer comes back.
Test 2 — invalid Gemini key: confirm failover to OpenAI, or a graceful error
         string if no OPENAI_API_KEY is configured.
Both print time-to-first-chunk (feeds the <200ms perceived-latency claim).
"""

import asyncio
import os
import time

from dotenv import load_dotenv

from agent.llm_provider import LLMProvider

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SAMPLE_QUERY = "What is retrieval-augmented generation?"
SAMPLE_CONTEXT = (
    "Retrieval-Augmented Generation (RAG) combines a retrieval system with a "
    "language model: relevant passages are fetched from a knowledge base and "
    "passed as context to the model so its answer is grounded in real documents."
)


def _banner(title: str):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


async def stream_and_time(provider: LLMProvider) -> tuple[str, float]:
    """Consume the stream, returning (full_text, time_to_first_chunk_seconds)."""
    start = time.perf_counter()
    ttfc = None
    chunks = []
    async for chunk in provider.stream(query=SAMPLE_QUERY, context=SAMPLE_CONTEXT):
        if ttfc is None:
            ttfc = time.perf_counter() - start
        chunks.append(chunk)
    if ttfc is None:
        ttfc = time.perf_counter() - start  # stream produced nothing
    return "".join(chunks), ttfc


async def test_normal():
    _banner("TEST 1 — valid Gemini key (expect streamed Gemini answer)")
    if not GEMINI_API_KEY:
        print("SKIPPED: GEMINI_API_KEY not set in .env")
        return

    provider = LLMProvider(gemini_api_key=GEMINI_API_KEY, openai_api_key=OPENAI_API_KEY)
    answer, ttfc = await stream_and_time(provider)
    print(f"Time-to-first-chunk: {ttfc * 1000:.1f} ms")
    print(f"Answer length:       {len(answer)} chars\n")
    print(f"Answer:\n{answer}")


async def test_failover():
    _banner("TEST 2 — invalid Gemini key (expect OpenAI fallback or graceful error)")
    provider = LLMProvider(gemini_api_key="INVALID_KEY_FORCED_FAILURE", openai_api_key=OPENAI_API_KEY)
    answer, ttfc = await stream_and_time(provider)
    print(f"Time-to-first-chunk: {ttfc * 1000:.1f} ms")
    print(f"OpenAI fallback configured: {bool(OPENAI_API_KEY)}")
    print(f"Answer length:              {len(answer)} chars\n")
    print(f"Answer:\n{answer}")


async def main():
    await test_normal()
    await test_failover()


if __name__ == "__main__":
    asyncio.run(main())
