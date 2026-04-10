"""
Reranker Module
Uses a cross-encoder to re-score retrieved chunks for higher precision.

WHY RERANK?
-----------
The bi-encoder (embedding model) encodes the query and each document INDEPENDENTLY,
then compares their vectors. This is fast but approximate.

A cross-encoder takes the (query, document) PAIR as input and scores them together,
allowing it to capture fine-grained interactions between the query and document.
It's much more accurate but too slow to run on the entire corpus — so we use it
only on the top-K candidates from the bi-encoder stage.

Pipeline:  top-20 from vector search  -->  cross-encoder scores  -->  top-5 best
"""

from __future__ import annotations

from sentence_transformers import CrossEncoder
from src.retriever import RetrievedChunk

RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_reranker: CrossEncoder | None = None


def get_reranker() -> CrossEncoder:
    """Lazy-load the cross-encoder reranker model."""
    global _reranker
    if _reranker is None:
        print(f"Loading reranker model '{RERANKER_MODEL}'...")
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank(
    query: str,
    chunks: list[RetrievedChunk],
    top_n: int = 5,
) -> list[tuple[RetrievedChunk, float]]:
    """
    Re-score each chunk against the query using the cross-encoder,
    then return the top_n chunks sorted by relevance.

    Returns list of (chunk, cross_encoder_score) tuples, highest score first.
    """
    if not chunks:
        return []

    reranker = get_reranker()

    # Cross-encoder expects list of [query, document] pairs
    pairs = [[query, chunk.text] for chunk in chunks]
    scores = reranker.predict(pairs)

    # Pair each chunk with its cross-encoder score
    scored = list(zip(chunks, scores))
    scored.sort(key=lambda x: x[1], reverse=True)  # highest score first

    return scored[:top_n]


def print_reranked(results: list[tuple[RetrievedChunk, float]], show_text: bool = False) -> None:
    """Pretty-print reranked results for debugging."""
    print(f"\nReranked top-{len(results)} chunks:")
    print("-" * 60)
    for i, (chunk, score) in enumerate(results, 1):
        print(f"  {i}. [page {chunk.page_number}] cross-encoder score={score:.4f}  (vector sim={chunk.similarity_score:.4f})")
        if show_text:
            preview = chunk.text[:150].replace("\n", " ")
            print(f"     {preview}...")
    print()
