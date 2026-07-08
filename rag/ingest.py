"""Document ingestion pipeline: load, chunk, embed, and persist to ChromaDB."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

load_dotenv()

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_embeddings() -> HuggingFaceEmbeddings:
    """Return a sentence-transformers embedding function (free, no API key)."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True},
    )


def load_documents(docs_dir: str) -> list:
    """Load all .pdf and .txt files from *docs_dir* into LangChain Documents."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        print(f"Error: documents directory not found: {docs_dir}")
        sys.exit(1)

    documents = []
    for file_path in sorted(docs_path.iterdir()):
        suffix = file_path.suffix.lower()
        if suffix == ".pdf":
            loader = PyPDFLoader(str(file_path))
            documents.extend(loader.load())
            print(f"  Loaded PDF: {file_path.name}")
        elif suffix == ".txt":
            loader = TextLoader(str(file_path), encoding="utf-8")
            documents.extend(loader.load())
            print(f"  Loaded TXT: {file_path.name}")

    return documents


def split_documents(documents: list, chunk_size: int = CHUNK_SIZE, chunk_overlap: int = CHUNK_OVERLAP) -> list:
    """Split documents into overlapping text chunks."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    return chunks


def ingest_documents(docs_dir: str = None, persist_dir: str = None) -> int:
    """Load, chunk, embed, and persist documents into ChromaDB.

    Returns the number of chunks ingested.
    """
    docs_dir = docs_dir or os.getenv("DOCS_DIR", "./data/docs")
    persist_dir = persist_dir or os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

    print(f"Ingesting documents from: {docs_dir}")
    documents = load_documents(docs_dir)

    if not documents:
        print("No .pdf or .txt documents found. Drop files in data/docs/ and retry.")
        return 0

    print(f"Loaded {len(documents)} document(s). Chunking...")
    chunks = split_documents(documents)
    print(f"Split into {len(chunks)} chunks.")

    print(f"Embedding with {EMBEDDING_MODEL} and persisting to {persist_dir}...")
    embeddings = get_embeddings()
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=persist_dir,
    )

    print(f"Done. {len(chunks)} chunks stored in ChromaDB at {persist_dir}")
    return len(chunks)


if __name__ == "__main__":
    ingest_documents()
