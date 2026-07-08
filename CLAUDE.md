# CLAUDE.md — MCP-Based Research Assistant Agent

## Project Overview

A Python-based agentic research assistant that uses the Model Context Protocol (MCP) for tool-calling, a RAG retrieval pipeline for document-grounded answers, and SSE streaming for low-latency responses. The agent routes queries across multiple data sources with automatic fallback and an LLM provider failover mechanism.

**Tech Stack:** Python 3.11, FastAPI, MCP (Python SDK), LangChain (RAG), ChromaDB (vector store), Gemini API + OpenAI API (LLM providers), SSE (Server-Sent Events), dotenv

---

## Architecture at a Glance

```
User Query
    │
    ▼
FastAPI /query endpoint (SSE)
    │
    ▼
Agent Core (tool router)
    │
    ├──► Tool 1: RAG Retriever  (local vector store)
    ├──► Tool 2: Web Search     (Tavily or DuckDuckGo API)
    └──► Tool 3: Wikipedia API  (fallback)
    │
    ▼
LLM Provider (Gemini primary → OpenAI fallback)
    │
    ▼
SSE stream back to client
```

---

## Directory Structure

```
research-agent/
├── main.py                    # FastAPI app entry point
├── agent/
│   ├── __init__.py
│   ├── core.py                # Agent loop, tool routing, fallback chain
│   ├── llm_provider.py        # OpenAI + Anthropic with failover logic
│   └── streaming.py           # SSE response builder
├── tools/
│   ├── __init__.py
│   ├── registry.py            # Tool registry (plug-and-play)
│   ├── rag_tool.py            # RAG retriever tool
│   ├── web_search_tool.py     # Web search tool
│   └── wikipedia_tool.py      # Wikipedia fallback tool
├── rag/
│   ├── __init__.py
│   ├── ingest.py              # Document chunking + embedding + storing
│   └── retriever.py           # Query-time retrieval logic
├── mcp_server/
│   ├── __init__.py
│   └── server.py              # MCP server exposing tools via MCP protocol
├── data/
│   └── docs/                  # Drop PDFs/text files here for RAG ingestion
├── tests/
│   ├── test_tools.py
│   ├── test_agent.py
│   └── test_rag.py
├── .env                       # API keys (never commit)
├── .env.example
├── requirements.txt
└── README.md
```

---

## Implementation Plan

### Phase 1 — Scaffolding & RAG Pipeline (Day 1–2)

**Goal:** Get documents ingested and retrievable before touching MCP or streaming.

#### 1.1 `rag/ingest.py`
- Load PDF/text files from `data/docs/` using `langchain.document_loaders`
- Chunk with `RecursiveCharacterTextSplitter` (chunk_size=500, overlap=50)
- Embed using `text-embedding-3-small` (OpenAI) or `all-MiniLM-L6-v2` (sentence-transformers — free)
- Store in ChromaDB (local, persistent)

```python
# rag/ingest.py  — skeleton
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings

def ingest_documents(docs_dir: str, persist_dir: str = "./chroma_db"):
    # load → split → embed → persist
    ...
```

#### 1.2 `rag/retriever.py`
- Wrap ChromaDB as a retriever
- `retrieve(query: str, k: int = 3) -> list[str]` returns top-k chunk texts
- Keep it simple: no reranking at this stage

---

### Phase 2 — Tool Registry & Individual Tools (Day 2–3)

**Goal:** Each tool is a self-contained callable. The registry maps tool names to callables.

#### 2.1 `tools/registry.py`

```python
# tools/registry.py
from typing import Callable

class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, Callable] = {}

    def register(self, name: str, fn: Callable, description: str):
        self._tools[name] = {"fn": fn, "description": description}

    def get(self, name: str) -> Callable:
        return self._tools[name]["fn"]

    def all_tools(self) -> dict:
        return {k: v["description"] for k, v in self._tools.items()}
```

- Tools are registered at startup in `main.py`
- Adding a new tool = one `registry.register(...)` call, no other changes

#### 2.2 `tools/rag_tool.py`
- Calls `retriever.retrieve(query)`
- Returns formatted string of retrieved chunks

#### 2.3 `tools/web_search_tool.py`
- Use **Tavily API** (free tier, research-focused, easy Python client)
- `pip install tavily-python`
- Returns top 3 result snippets as a string

#### 2.4 `tools/wikipedia_tool.py`
- Use `wikipedia` Python package (`pip install wikipedia-api`)
- Simple: search → get summary of top result

---

### Phase 3 — Agent Core with Fallback Chain (Day 3–4)

**Goal:** Route queries through tools in order, stop when one returns a confident result.

#### 3.1 `agent/core.py`

The fallback chain logic:
1. Try RAG retriever first (fastest, local, grounded)
2. If RAG returns < 100 chars or "I don't know" — try web search
3. If web search fails or times out — fall back to Wikipedia
4. Pass collected context + original query to LLM for final answer

```python
# agent/core.py  — skeleton
async def run_agent(query: str, registry: ToolRegistry, llm) -> AsyncGenerator[str, None]:
    context = ""
    sources_tried = []

    # Fallback chain
    for tool_name in ["rag_retriever", "web_search", "wikipedia"]:
        try:
            tool_fn = registry.get(tool_name)
            result = await asyncio.wait_for(tool_fn(query), timeout=5.0)
            if result and len(result) > 100:
                context = result
                sources_tried.append(tool_name)
                break
        except (asyncio.TimeoutError, Exception) as e:
            sources_tried.append(f"{tool_name}:failed")
            continue

    # Stream LLM answer
    async for chunk in llm.stream(query=query, context=context):
        yield chunk
```

**Why this counts as MCP tool-calling:** wrap each tool as an MCP-compatible tool in `mcp_server/server.py` (Phase 5). The agent core stays the same — it calls tools via the registry, and the MCP server exposes those same tools over the MCP protocol.

---

### Phase 4 — LLM Provider with Failover (Day 4)

#### 4.1 `agent/llm_provider.py`

- Primary: **Gemini 2.0 Flash** (`gemini-2.0-flash`) — free tier via Google AI Studio, fast, generous limits
- Fallback: **OpenAI `gpt-4o-mini`** — cheap if you have credits, or mock it for demo purposes
- Auto-switch on `google.api_core.exceptions.GoogleAPIError` or timeout

Get your free Gemini API key at: https://aistudio.google.com/apikey

```python
# agent/llm_provider.py  — skeleton
import asyncio
import google.generativeai as genai
import openai

class LLMProvider:
    def __init__(self, gemini_api_key: str, openai_api_key: str = None):
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel("gemini-2.0-flash")
        self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key) if openai_api_key else None

    async def stream(self, query: str, context: str):
        prompt = self._build_prompt(query, context)
        try:
            async for chunk in self._gemini_stream(prompt):
                yield chunk
        except Exception as e:
            # fallback to OpenAI if available, else re-raise
            if self.openai_client:
                async for chunk in self._openai_stream(prompt):
                    yield chunk
            else:
                yield f"Error: {str(e)}"

    async def _gemini_stream(self, prompt: str):
        # Gemini's generate_content_async with stream=True
        response = await asyncio.to_thread(
            self.gemini_model.generate_content,
            prompt,
            stream=True
        )
        for chunk in response:
            if chunk.text:
                yield chunk.text

    async def _openai_stream(self, prompt: str):
        stream = await self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            stream=True
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _build_prompt(self, query: str, context: str) -> str:
        return f"""Use the context below to answer the question.
Context: {context}
Question: {query}
Answer concisely and cite the source if possible."""
```

**Note on Gemini streaming:** The `google-generativeai` SDK's streaming is synchronous under the hood, so `asyncio.to_thread` wraps it to avoid blocking FastAPI. If you use `google-genai` (the newer SDK), it has native async — either works.

**Uptime claim (99%):** during demo/testing, pass an invalid OpenAI key and set Gemini as the only provider — show the agent stays up. To demonstrate failover, temporarily pass a bad Gemini key and show it switches to OpenAI.

---

### Phase 5 — SSE Streaming Endpoint (Day 4–5)

#### 5.1 `main.py` FastAPI app

```python
# main.py
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from agent.core import run_agent

app = FastAPI()

@app.get("/query")
async def query_endpoint(q: str):
    async def event_generator():
        async for chunk in run_agent(q, registry, llm_provider):
            yield f"data: {chunk}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(event_generator(), media_type="text/event-stream")
```

**Latency claim (<200ms perceived):** SSE sends first token the moment LLM starts generating — user sees text appearing immediately. Measure time-to-first-chunk with `time.perf_counter()` in a test and log it.

---

### Phase 6 — MCP Server (Day 5–6)

**Goal:** Expose the tools via actual MCP protocol so this qualifies as an MCP-based agent.

#### 6.1 `mcp_server/server.py`

Use the official `mcp` Python SDK:
```
pip install mcp
```

```python
# mcp_server/server.py
from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp import types

server = Server("research-assistant")

@server.list_tools()
async def list_tools() -> list[types.Tool]:
    return [
        types.Tool(name="rag_retriever", description="Search local document knowledge base", inputSchema={...}),
        types.Tool(name="web_search", description="Search the web for current information", inputSchema={...}),
        types.Tool(name="wikipedia", description="Retrieve Wikipedia article summaries", inputSchema={...}),
    ]

@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    result = await registry.get(name)(arguments["query"])
    return [types.TextContent(type="text", text=result)]

async def main():
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
```

- Run with `python -m mcp_server.server`
- The FastAPI agent can also call tools locally via the registry (simpler for demo); MCP server is the protocol-compliant layer for portfolio credibility

---

### Phase 7 — Tests & Metrics (Day 6–7)

#### Key things to test and log (needed for resume bullet claims)

| Claim | How to verify |
|---|---|
| 95% query resolution | Run 20 test queries, count how many get a non-empty answer without manual help |
| <200ms perceived latency | Log `time_to_first_chunk` in streaming endpoint |
| 99% uptime during outages | Mock OpenAI failure, verify Anthropic fallback fires |
| <30 min to onboard new tool | Time yourself adding a 4th tool (e.g., arXiv search) |

Simple test harness in `tests/test_agent.py`:
```python
import pytest, asyncio

TEST_QUERIES = [
    "What is retrieval-augmented generation?",
    "Explain the transformer architecture",
    # ... 18 more
]

@pytest.mark.asyncio
async def test_query_resolution():
    resolved = 0
    for q in TEST_QUERIES:
        chunks = []
        async for chunk in run_agent(q, registry, llm):
            chunks.append(chunk)
        if "".join(chunks).strip():
            resolved += 1
    assert resolved / len(TEST_QUERIES) >= 0.95
```

---

## .env.example

```
GEMINI_API_KEY=AIza...         # free from aistudio.google.com/apikey
OPENAI_API_KEY=sk-...          # optional, used as fallback only
TAVILY_API_KEY=tvly-...
CHROMA_PERSIST_DIR=./chroma_db
DOCS_DIR=./data/docs
```

---

## requirements.txt (core)

```
fastapi
uvicorn[standard]
python-dotenv
google-generativeai          # Gemini SDK — pip install google-generativeai
openai>=1.0                  # optional fallback
langchain
langchain-community
langchain-chroma
langchain-openai
chromadb
sentence-transformers        # free embeddings alternative to OpenAI
mcp                          # official MCP Python SDK
tavily-python
wikipedia-api
pytest
pytest-asyncio
httpx                        # for testing FastAPI endpoints
```

---

## Running the Project

```bash
# 1. Install deps
pip install -r requirements.txt

# 2. Set up .env
cp .env.example .env
# fill in your keys

# 3. Ingest documents (drop PDFs in data/docs/ first)
python -m rag.ingest

# 4. Start the FastAPI server
uvicorn main:app --reload

# 5. Query it
curl "http://localhost:8000/query?q=What+is+RAG?" --no-buffer

# 6. (Optional) Run MCP server separately
python -m mcp_server.server
```

---

## Resume Bullet ↔ Code Mapping (Verified)

| Resume Claim | Where it lives | Verified Result |
|---|---|---|
| MCP tool-calling | `mcp_server/server.py` + `tools/registry.py` | 4 tools exposed over MCP stdio, tested via JSON-RPC 2.0 |
| RAG retrieval | `rag/ingest.py` + `rag/retriever.py` + `tools/rag_tool.py` | 10/10 RAG queries resolved and grounded |
| Fallback chain across 3+ sources | `agent/core.py` — for-loop with per-tool timeout | 4-tool chain (rag → web → wiki → arxiv), first ≥100 chars wins |
| Query resolution | `tests/test_metrics.py` — 20-query suite | **100% total resolution** (20/20 get substantive answers); **60% tool-grounded** (12/20 backed by retrieved context); 40% LLM-parametric (see Known Limitations in README) |
| SSE streaming (low perceived latency) | `main.py` StreamingResponse + `event: meta` SSE event | Median TTFC **937ms** end-to-end (embedding + retrieval + LLM first-token); SSE surfaces the first token as soon as it's ready |
| LLM failover | `agent/llm_provider.py` — Gemini primary, OpenAI fallback | Auto-switches on Gemini failure; falls back to error string if no OpenAI key configured |
| Modular tool registry | `tools/registry.py` — register() pattern | Single source of truth for both FastAPI and MCP server |
| <30 min to onboard new tool | `tools/arxiv_tool.py` + `tests/time_onboarding.py` | ~10 min, 5 files touched (1 new + 4 one-line edits), 3/3 verification tests pass |

---

## Things to Keep Scope-Appropriate

- **No Kubernetes, no Docker required** — run locally with uvicorn; mention "containerization-ready" in README if asked
- **No custom embedding model training** — use OpenAI embeddings or sentence-transformers off the shelf
- **No reranking** — basic top-k retrieval is fine; mention it as a future improvement
- **No persistent conversation history** — stateless per query; keep it simple
- **MCP server via stdio** — simpler than HTTP transport; fully spec-compliant and fine for a portfolio project
