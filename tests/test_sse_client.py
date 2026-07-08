"""Phase 5 manual test: hit the running server and watch the SSE stream arrive.

Prereq — start the server in another terminal first:
    ./.venv/Scripts/python.exe -m uvicorn main:app --reload

Then run:
    ./.venv/Scripts/python.exe -m tests.test_sse_client
    ./.venv/Scripts/python.exe -m tests.test_sse_client "your own question here"

Each line prints the elapsed time when it arrived, so you can visually confirm
chunks stream incrementally (rising timestamps) rather than landing all at once.
Also equivalent to:
    curl -N "http://localhost:8000/query?q=What+is+RAG?"
(the -N / --no-buffer flag disables curl buffering so you see chunks live).
"""

import asyncio
import sys
import time

import httpx

BASE_URL = "http://localhost:8000/query"


async def main():
    query = sys.argv[1] if len(sys.argv) > 1 else "What is retrieval-augmented generation?"
    print(f"Querying: {query!r}\n")

    start = time.perf_counter()
    ttfc = None
    n_events = 0

    async with httpx.AsyncClient(timeout=60.0) as client:
        async with client.stream("GET", BASE_URL, params={"q": query}) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                payload = line[len("data: "):]
                elapsed_ms = (time.perf_counter() - start) * 1000
                if ttfc is None:
                    ttfc = elapsed_ms
                n_events += 1
                marker = "[DONE]" if payload == "[DONE]" else repr(payload)
                print(f"  +{elapsed_ms:7.1f} ms | {marker}")

    total_ms = (time.perf_counter() - start) * 1000
    print("\n--- summary ---")
    print(f"time-to-first-chunk (client-side): {ttfc:.1f} ms" if ttfc else "no data received")
    print(f"total events received:             {n_events}")
    print(f"total wall time:                   {total_ms:.1f} ms")


if __name__ == "__main__":
    asyncio.run(main())
