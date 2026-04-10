"""
Retriever Module
Takes a user query, embeds it, and finds the most relevant chunks via vector search.

This is the "R" in RAG — Retrieval. The quality of retrieval directly determines
the quality of the final LLM output. If the right chunks aren't found here,
no amount of prompt engineering will fix the answer.
"""

from dataclasses import dataclass
import chromadb
from src.embedder import embed_query
from src.vector_store import query_collection


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    distance: float  # cosine distance — lower is more similar
    page_number: int
    source_file: str

    @property
    def similarity_score(self) -> float:
        """Convert cosine distance to similarity (1 - distance)."""
        return 1.0 - self.distance


def retrieve(
    collection: chromadb.Collection,
    query: str,
    top_k: int = 20,
) -> list[RetrievedChunk]:
    """
    Embed the query and search the vector store for the top-K nearest chunks.

    Pipeline: query string -> embedding vector -> ChromaDB ANN search -> ranked chunks
    """
    query_vector = embed_query(query).tolist()
    results = query_collection(collection, query_vector, top_k=top_k)

    chunks: list[RetrievedChunk] = []
    ids = results["ids"][0]
    docs = results["documents"][0]
    dists = results["distances"][0]
    metas = results["metadatas"][0]

    for i in range(len(ids)):
        chunks.append(RetrievedChunk(
            chunk_id=ids[i],
            text=docs[i],
            distance=dists[i],
            page_number=metas[i].get("page_number", 0),
            source_file=metas[i].get("source_file", ""),
        ))

    return chunks


def print_results(chunks: list[RetrievedChunk], show_text: bool = False) -> None:
    """Pretty-print retrieval results for debugging."""
    print(f"\nRetrieved {len(chunks)} chunks:")
    print("-" * 60)
    for i, c in enumerate(chunks, 1):
        print(f"  {i}. [page {c.page_number}] similarity={c.similarity_score:.4f}")
        if show_text:
            preview = c.text[:150].replace("\n", " ")
            print(f"     {preview}...")
    print()
