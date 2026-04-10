"""
Embedding Module
Converts text into dense vectors using sentence-transformers.

The model `all-MiniLM-L6-v2` produces 384-dimensional vectors.
Semantically similar texts will have high cosine similarity (close to 1.0).
"""

from __future__ import annotations

import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_NAME = "all-MiniLM-L6-v2"

_model: SentenceTransformer | None = None


def get_model() -> SentenceTransformer:
    """Lazy-load the embedding model (avoids re-loading on every call)."""
    global _model
    if _model is None:
        print(f"Loading embedding model '{MODEL_NAME}'...")
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed_texts(texts: list[str], batch_size: int = 64) -> np.ndarray:
    """
    Embed a list of texts into vectors.
    Returns an (N, 384) numpy array where N = len(texts).
    """
    model = get_model()
    embeddings = model.encode(texts, batch_size=batch_size, show_progress_bar=True)
    return np.array(embeddings)


def embed_query(query: str) -> np.ndarray:
    """Embed a single query string. Returns a (384,) vector."""
    model = get_model()
    return np.array(model.encode(query))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    Compute cosine similarity between two vectors.
    cos_sim = (a · b) / (||a|| * ||b||)
    Range: -1 (opposite) to 1 (identical meaning).
    """
    dot = np.dot(a, b)
    norm = np.linalg.norm(a) * np.linalg.norm(b)
    if norm == 0:
        return 0.0
    return float(dot / norm)


if __name__ == "__main__":
    # Quick experiment: see how cosine similarity captures meaning
    pairs = [
        ("The cat sat on the mat", "A cat is sitting on a rug"),       # similar
        ("The cat sat on the mat", "Stock prices rose sharply today"),  # different
        ("I love programming", "Coding is my passion"),                 # similar
        ("I love programming", "The weather is sunny"),                 # different
    ]

    print("Cosine Similarity Experiment")
    print("=" * 60)
    for text_a, text_b in pairs:
        vec_a = embed_query(text_a)
        vec_b = embed_query(text_b)
        sim = cosine_similarity(vec_a, vec_b)
        print(f"\n  A: '{text_a}'")
        print(f"  B: '{text_b}'")
        print(f"  Similarity: {sim:.4f}")
