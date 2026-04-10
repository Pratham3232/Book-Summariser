"""
RAG Agent Module
Orchestrates retrieval + reranking + LLM generation.

This is where everything comes together:
  1. User asks a question
  2. Retriever finds top-20 chunks from the vector store
  3. Reranker narrows it to top-5 most relevant chunks
  4. LLM generates an answer grounded in those chunks
"""

import os
from openai import OpenAI
from src.retriever import retrieve, RetrievedChunk
from src.reranker import rerank
from src.vector_store import get_or_create_collection


SYSTEM_PROMPT = """You are a helpful book summarizer assistant. You answer questions 
based ONLY on the provided book excerpts. Follow these rules strictly:

1. Only use information from the provided excerpts to answer.
2. Cite the page number when referencing specific information (e.g., "According to page 42...").
3. If the excerpts don't contain enough information to answer the question, say:
   "I don't have enough information from the book to answer this question."
4. Be concise but thorough. Synthesize information across multiple excerpts when relevant.
5. Do not make up or hallucinate information not present in the excerpts."""


SUMMARY_SYSTEM_PROMPT = """You are a book summarizer. Given excerpts from a book, 
produce a comprehensive summary. Follow these rules:

1. Organize the summary by themes or chapters as they appear in the excerpts.
2. Include key arguments, insights, and examples from the book.
3. Cite page numbers for major points.
4. Be thorough but avoid redundancy.
5. Only summarize what is present in the excerpts — do not add outside knowledge."""


def _get_openai_client() -> OpenAI:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY not set. Run: export OPENAI_API_KEY='sk-...'"
        )
    return OpenAI(api_key=api_key)


def _format_context(chunks: list[tuple[RetrievedChunk, float]]) -> str:
    """Format reranked chunks into a context block for the LLM prompt."""
    parts = []
    for i, (chunk, score) in enumerate(chunks, 1):
        parts.append(
            f"[Excerpt {i} | Page {chunk.page_number} | Relevance: {score:.2f}]\n"
            f"{chunk.text}"
        )
    return "\n\n---\n\n".join(parts)


def query_book(
    collection_name: str,
    question: str,
    top_k_retrieve: int = 20,
    top_n_rerank: int = 5,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Full RAG pipeline: retrieve -> rerank -> generate.
    Returns the LLM's answer grounded in book content.
    """
    collection = get_or_create_collection(collection_name)

    # Step 1: Retrieve top-K chunks via vector search
    print(f"Retrieving top-{top_k_retrieve} chunks...")
    retrieved = retrieve(collection, question, top_k=top_k_retrieve)

    if not retrieved:
        return "No content found in the book for this query."

    # Step 2: Rerank to top-N using cross-encoder
    print(f"Reranking to top-{top_n_rerank}...")
    reranked = rerank(question, retrieved, top_n=top_n_rerank)

    # Step 3: Build prompt and call LLM
    context = _format_context(reranked)
    user_message = (
        f"Based on the following book excerpts, answer this question:\n\n"
        f"**Question:** {question}\n\n"
        f"**Book Excerpts:**\n\n{context}"
    )

    print(f"Generating answer with {model}...")
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,  # low temp for factual, grounded responses
        max_tokens=1024,
    )

    return response.choices[0].message.content


def summarize_book(
    collection_name: str,
    top_k_retrieve: int = 50,
    top_n_rerank: int = 15,
    model: str = "gpt-4o-mini",
) -> str:
    """
    Generate a full book summary by retrieving chunks across broad topics.
    Uses a generic "summarize" query to pull diverse content.
    """
    collection = get_or_create_collection(collection_name)

    broad_queries = [
        "What are the main themes and arguments of this book?",
        "What are the key insights and conclusions?",
        "What examples and evidence does the author present?",
    ]

    all_chunks: dict[str, tuple[RetrievedChunk, float]] = {}

    for q in broad_queries:
        retrieved = retrieve(collection, q, top_k=top_k_retrieve)
        reranked = rerank(q, retrieved, top_n=top_n_rerank)
        for chunk, score in reranked:
            # Keep the highest score if a chunk appears in multiple queries
            if chunk.chunk_id not in all_chunks or all_chunks[chunk.chunk_id][1] < score:
                all_chunks[chunk.chunk_id] = (chunk, score)

    # Sort by page number for chronological order
    sorted_chunks = sorted(all_chunks.values(), key=lambda x: x[0].page_number)
    context = _format_context(sorted_chunks)

    user_message = (
        f"Based on the following excerpts from the book, provide a comprehensive summary:\n\n"
        f"**Book Excerpts:**\n\n{context}"
    )

    print(f"Generating summary with {model}...")
    client = _get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
            {"role": "user", "content": user_message},
        ],
        temperature=0.3,
        max_tokens=2048,
    )

    return response.choices[0].message.content
