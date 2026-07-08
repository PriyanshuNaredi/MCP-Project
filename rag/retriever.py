"""Query-time retrieval from the persisted ChromaDB vector store."""

import os

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

_vectorstore: Chroma | None = None


def _get_vectorstore(persist_dir: str = None) -> Chroma:
    """Lazily initialise (and cache) the Chroma vector store."""
    global _vectorstore
    if _vectorstore is not None:
        return _vectorstore

    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )
    _vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embeddings,
    )
    return _vectorstore


def retrieve(query: str, k: int = 3) -> list[str]:
    """Return the top-*k* most similar chunk texts for *query*."""
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return [doc.page_content for doc in results]


def retrieve_with_metadata(query: str, k: int = 3) -> list[dict]:
    """Return the top-*k* chunks as dicts with text and source metadata."""
    vectorstore = _get_vectorstore()
    results = vectorstore.similarity_search(query, k=k)
    return [
        {"text": doc.page_content, "metadata": doc.metadata}
        for doc in results
    ]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python -m rag.retriever 'your search query'")
        sys.exit(1)

    hits = retrieve(sys.argv[1])
    for i, text in enumerate(hits, 1):
        print(f"\n--- Result {i} ---\n{text}")
