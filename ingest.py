"""ingest.py - load a book, chunk it, embed it, store it in Chroma (and search it later).

Pipeline:  file -> text -> ~500-token chunks (15% overlap) -> embeddings -> Chroma
"""
import re
from pathlib import Path

import chromadb
import tiktoken
from pypdf import PdfReader

from llm import EMBED_MODEL, MIN_SCORE, client

CHUNK_TOKENS = 500
OVERLAP_TOKENS = 75  # 15% of 500
EMBED_BATCH = 100  # chunks per embeddings API call

_encoder = tiktoken.get_encoding("cl100k_base")  # tokenizer used by OpenAI embedding models
_chroma = chromadb.PersistentClient(path="chroma_db")  # local folder, no server


def load_text(path: Path) -> str:
    """Read a .pdf or .txt file into one big string."""
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8", errors="ignore")
    raise ValueError("Only .pdf and .txt files are supported")


def chunk_text(text: str) -> list[str]:
    """Slide a 500-token window over the text, stepping 425 tokens so neighbours share 75."""
    tokens = _encoder.encode(text)
    step = CHUNK_TOKENS - OVERLAP_TOKENS
    chunks = []
    for start in range(0, len(tokens), step):
        window = tokens[start : start + CHUNK_TOKENS]
        chunks.append(_encoder.decode(window))
        if start + CHUNK_TOKENS >= len(tokens):  # last window reached the end
            break
    return chunks


def embed(texts: list[str]) -> list[list[float]]:
    """Turn texts into vectors, in batches to stay under API limits."""
    vectors = []
    for i in range(0, len(texts), EMBED_BATCH):
        resp = client.embeddings.create(model=EMBED_MODEL, input=texts[i : i + EMBED_BATCH])
        vectors.extend(item.embedding for item in resp.data)
    return vectors


def _collection_name(title: str) -> str:
    """Chroma names must be 3-63 chars of [a-zA-Z0-9._-]; derive one from the title."""
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", title).strip("-").lower()[:50]
    return f"book-{slug}"


def ingest_book(path: Path, title: str) -> int:
    """Chunk + embed a book and store it. Re-ingesting replaces the old copy. Returns chunk count."""
    chunks = [c for c in chunk_text(load_text(path)) if c.strip()]
    if not chunks:
        raise ValueError("No text could be extracted from this file")

    name = _collection_name(title)
    try:
        _chroma.delete_collection(name)  # start fresh so re-runs don't duplicate chunks
    except Exception:
        pass
    collection = _chroma.create_collection(name, metadata={"hnsw:space": "cosine"})
    collection.add(
        ids=[f"{name}-{i}" for i in range(len(chunks))],
        documents=chunks,
        embeddings=embed(chunks),
        metadatas=[{"book_title": title, "chunk_index": i} for i in range(len(chunks))],
    )
    return len(chunks)


def retrieve(title: str, question: str, k: int = 4) -> list[str]:
    """Return up to k chunks most similar to the question, dropping any scoring below MIN_SCORE.

    Chroma returns cosine *distance*; similarity = 1 - distance. An empty list means
    "nothing in the book is relevant", which lets the agent say so instead of guessing.
    """
    collection = _chroma.get_collection(_collection_name(title))
    result = collection.query(query_embeddings=embed([question]), n_results=k)
    return [
        doc
        for doc, dist in zip(result["documents"][0], result["distances"][0])
        if 1 - dist >= MIN_SCORE
    ]
