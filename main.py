"""
Book Summarizer RAG Agent — CLI Entry Point

Usage:
  python main.py ingest   <pdf_path>  [--strategy recursive]  [--chunk-size 512]
  python main.py query    <collection> <question>
  python main.py summarize <collection>
  python main.py list
  python main.py delete   <collection>
"""

import argparse
import sys

from src.ingestion import extract_text_from_pdf
from src.chunker import fixed_size_chunking, sentence_chunking, recursive_chunking, Chunk
from src.embedder import embed_texts
from src.vector_store import (
    get_or_create_collection, add_chunks, list_collections,
    delete_collection, collection_count,
)
from src.retriever import retrieve, print_results
from src.reranker import rerank, print_reranked
from src.rag_agent import query_book, summarize_book


CHUNKING_STRATEGIES = {
    "fixed": fixed_size_chunking,
    "sentence": sentence_chunking,
    "recursive": recursive_chunking,
}


def cmd_ingest(args: argparse.Namespace) -> None:
    """Ingest a PDF: extract text -> chunk -> embed -> store in ChromaDB."""
    # 1. Extract text from PDF
    print(f"\n[1/4] Extracting text from '{args.pdf_path}'...")
    pages = extract_text_from_pdf(args.pdf_path)
    print(f"       Got {len(pages)} pages")

    # 2. Chunk the text
    strategy_fn = CHUNKING_STRATEGIES[args.strategy]
    print(f"\n[2/4] Chunking with '{args.strategy}' strategy (chunk_size={args.chunk_size})...")
    if args.strategy == "sentence":
        chunks = strategy_fn(pages, max_chars=args.chunk_size)
    else:
        chunks = strategy_fn(pages, chunk_size=args.chunk_size, overlap=args.overlap)
    print(f"       Created {len(chunks)} chunks")

    # 3. Generate embeddings
    print(f"\n[3/4] Generating embeddings for {len(chunks)} chunks...")
    texts = [c.text for c in chunks]
    embeddings = embed_texts(texts).tolist()

    # 4. Store in ChromaDB
    collection_name = args.collection or _make_collection_name(args.pdf_path)
    print(f"\n[4/4] Storing in collection '{collection_name}'...")
    collection = get_or_create_collection(collection_name)
    add_chunks(collection, chunks, embeddings)

    print(f"\nDone! Collection '{collection_name}' now has {collection_count(collection)} chunks.")
    print(f"You can now query it with:\n  python main.py query {collection_name} \"your question here\"")


def cmd_query(args: argparse.Namespace) -> None:
    """Ask a question about an ingested book."""
    print(f"\nQuerying '{args.collection}': {args.question}\n")

    if args.debug:
        # Show retrieval + reranking steps for learning
        collection = get_or_create_collection(args.collection)
        retrieved = retrieve(collection, args.question, top_k=20)
        print_results(retrieved, show_text=True)
        reranked = rerank(args.question, retrieved, top_n=5)
        print_reranked(reranked, show_text=True)

    answer = query_book(
        args.collection,
        args.question,
        top_k_retrieve=args.top_k,
        top_n_rerank=args.top_n,
    )
    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(answer)


def cmd_summarize(args: argparse.Namespace) -> None:
    """Generate a full book summary."""
    print(f"\nGenerating summary for '{args.collection}'...\n")
    summary = summarize_book(args.collection)
    print("\n" + "=" * 60)
    print("BOOK SUMMARY:")
    print("=" * 60)
    print(summary)


def cmd_list(args: argparse.Namespace) -> None:
    """List all ingested book collections."""
    collections = list_collections()
    if not collections:
        print("No books ingested yet. Run: python main.py ingest <pdf_path>")
        return
    print("Ingested books:")
    for name in collections:
        col = get_or_create_collection(name)
        print(f"  - {name} ({collection_count(col)} chunks)")


def cmd_delete(args: argparse.Namespace) -> None:
    """Delete a collection."""
    delete_collection(args.collection)
    print(f"Deleted '{args.collection}'")


def _make_collection_name(pdf_path: str) -> str:
    """Derive a collection name from the PDF filename."""
    from pathlib import Path
    name = Path(pdf_path).stem.lower()
    # ChromaDB collection names: 3-63 chars, alphanumeric + underscores/hyphens
    name = "".join(c if c.isalnum() or c in "-_" else "_" for c in name)
    return name[:63] if len(name) >= 3 else name + "_book"


def main():
    parser = argparse.ArgumentParser(description="RAG Book Summarizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- ingest ---
    p_ingest = subparsers.add_parser("ingest", help="Ingest a PDF book")
    p_ingest.add_argument("pdf_path", help="Path to the PDF file")
    p_ingest.add_argument("--strategy", choices=CHUNKING_STRATEGIES.keys(), default="recursive",
                          help="Chunking strategy (default: recursive)")
    p_ingest.add_argument("--chunk-size", type=int, default=512, help="Chunk size in characters")
    p_ingest.add_argument("--overlap", type=int, default=50, help="Overlap between chunks")
    p_ingest.add_argument("--collection", type=str, default=None, help="Custom collection name")
    p_ingest.set_defaults(func=cmd_ingest)

    # --- query ---
    p_query = subparsers.add_parser("query", help="Ask a question about a book")
    p_query.add_argument("collection", help="Collection name (use 'list' to see available)")
    p_query.add_argument("question", help="Your question")
    p_query.add_argument("--top-k", type=int, default=20, help="Top-K retrieval (default: 20)")
    p_query.add_argument("--top-n", type=int, default=5, help="Top-N after reranking (default: 5)")
    p_query.add_argument("--debug", action="store_true", help="Show retrieval & reranking details")
    p_query.set_defaults(func=cmd_query)

    # --- summarize ---
    p_summary = subparsers.add_parser("summarize", help="Generate full book summary")
    p_summary.add_argument("collection", help="Collection name")
    p_summary.set_defaults(func=cmd_summarize)

    # --- list ---
    p_list = subparsers.add_parser("list", help="List ingested books")
    p_list.set_defaults(func=cmd_list)

    # --- delete ---
    p_delete = subparsers.add_parser("delete", help="Delete a book collection")
    p_delete.add_argument("collection", help="Collection name to delete")
    p_delete.set_defaults(func=cmd_delete)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
