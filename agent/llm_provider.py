"""LLM provider with failover: Gemini 2.0 Flash primary, OpenAI gpt-4o-mini fallback.

If Gemini fails before producing any output, the request is retried against
OpenAI (when an OPENAI_API_KEY is configured). If no fallback is available, a
readable error string is yielded instead of raising, so callers never crash.
"""

import asyncio

import google.generativeai as genai

GEMINI_MODEL = "gemini-3.1-flash-lite"
OPENAI_MODEL = "gpt-4o-mini"


class LLMProvider:
    def __init__(self, gemini_api_key: str, openai_api_key: str = None):
        genai.configure(api_key=gemini_api_key)
        self.gemini_model = genai.GenerativeModel(GEMINI_MODEL)

        # OpenAI is optional — only build the client if a key was provided.
        if openai_api_key:
            import openai

            self.openai_client = openai.AsyncOpenAI(api_key=openai_api_key)
        else:
            self.openai_client = None

    async def stream(self, query: str, context: str):
        """Yield answer chunks, failing over Gemini -> OpenAI when possible."""
        prompt = self._build_prompt(query, context)

        yielded = False
        gemini_error = None  # persists past the except block (the `as` name is auto-deleted)
        try:
            async for chunk in self._gemini_stream(prompt):
                yielded = True
                yield chunk
            return
        except Exception as e:
            gemini_error = e  # copy into a normal local so it survives the except block
            if yielded:
                # Partial Gemini output already streamed — can't cleanly restart.
                yield f"\n[Gemini stream interrupted: {gemini_error}]"
                return
            print(f"[llm_provider] Gemini failed, attempting fallback: {gemini_error}")

        # Gemini failed before producing output — try OpenAI if available.
        if self.openai_client:
            try:
                async for chunk in self._openai_stream(prompt):
                    yield chunk
                return
            except Exception as e:
                # openai_error is used only inside this block, so it can't leak
                # the UnboundLocalError; still, keep gemini_error in the message.
                yield f"Error: both providers failed (gemini: {gemini_error}; openai: {e})"
                return

        yield f"Error: {gemini_error}"

    async def _gemini_stream(self, prompt: str):
        """Stream from Gemini, wrapping the sync SDK calls in asyncio.to_thread."""
        response = await asyncio.to_thread(
            self.gemini_model.generate_content,
            prompt,
            stream=True,
        )
        # Pull each chunk off the event loop too, so nothing blocks FastAPI later.
        iterator = iter(response)
        sentinel = object()
        while True:
            chunk = await asyncio.to_thread(next, iterator, sentinel)
            if chunk is sentinel:
                break
            text = getattr(chunk, "text", None)
            if text:
                yield text

    async def _openai_stream(self, prompt: str):
        """Stream from OpenAI's native async client."""
        stream = await self.openai_client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[{"role": "user", "content": prompt}],
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def _build_prompt(self, query: str, context: str) -> str:
        if context:
            return f"""You are a research assistant. Answer the question using the retrieved context below when it is relevant. If the context does not contain relevant information for the question, use your own knowledge to provide a helpful answer instead. Always answer the question — never refuse.

Retrieved context:
{context}

Question: {query}

Answer concisely. Cite the source when using retrieved context."""
        else:
            return f"""You are a research assistant. Answer the following question concisely using your knowledge.

Question: {query}"""
