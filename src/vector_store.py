"""
Vector Store Module
Wraps ChromaDB for storing, querying, and managing chunk embeddings.

ChromaDB stores vectors + metadata on disk. It uses HNSW (Hierarchical
Navigable Small World) indexing for fast approximate nearest-neighbor search.
"""

from __future__ import annotations

import chromadb
from pathlib import Path
from src.chunker import Chunk

CHROMA_DIR = "./chroma_db"

_client: chromadb.PersistentClient | None = None


def get_client() -> chromadb.PersistentClient:
    """Lazy-init a persistent ChromaDB client. Data survives restarts."""
    global _client
    if _client is None:
        Path(CHROMA_DIR).mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=CHROMA_DIR)
    return _client


def get_or_create_collection(name: str) -> chromadb.Collection:
    """
    Get an existing collection or create a new one.
    One collection per book keeps things organized.
    """
    client = get_client()
    return client.get_or_create_collection(
        name=name,
        metadata={"hnsw:space": "cosine"},  # use cosine distance
    )


def add_chunks(
    collection: chromadb.Collection,
    chunks: list[Chunk],
    embeddings: list[list[float]],
) -> None:
    """
    Store chunks and their embeddings in ChromaDB.
    Each chunk gets its text, embedding, and metadata stored together.
    """
    collection.add(
        ids=[c.chunk_id for c in chunks],
        documents=[c.text for c in chunks],
        embeddings=embeddings,
        metadatas=[
            {
                "page_number": c.page_number,
                "source_file": c.source_file,
                "strategy": c.strategy,
            }
            for c in chunks
        ],
    )
    print(f"Added {len(chunks)} chunks to collection '{collection.name}'")


def query_collection(
    collection: chromadb.Collection,
    query_embedding: list[float],
    top_k: int = 20,
) -> dict:
    """
    Find the top-K most similar chunks to the query embedding.
    Returns a dict with keys: ids, documents, metadatas, distances.
    Distances are cosine distances (lower = more similar).
    """
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )
    return results


def list_collections() -> list[str]:
    """List all collection names in the DB."""
    client = get_client()
    return [c.name for c in client.list_collections()]


def delete_collection(name: str) -> None:
    """Delete a collection and all its data."""
    client = get_client()
    client.delete_collection(name)
    print(f"Deleted collection '{name}'")


def collection_count(collection: chromadb.Collection) -> int:
    """How many chunks are stored in this collection."""
    return collection.count()
