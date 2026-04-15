# RAG Book Summarizer

A Retrieval-Augmented Generation (RAG) pipeline that lets you ingest any PDF book, ask questions about it, and get grounded answers with page citations — built from scratch to learn every layer of a RAG system, from vector database setup to embedding, retrieval, reranking, and LLM generation.

---

## What This Project Covers

| Concept | Where it lives |
|---|---|
| PDF text extraction | `src/ingestion.py` |
| Text chunking strategies (fixed, sentence, recursive) | `src/chunker.py` |
| Dense embeddings with sentence-transformers | `src/embedder.py` |
| Vector database (ChromaDB + HNSW index) | `src/vector_store.py` |
| Bi-encoder retrieval (top-K nearest neighbor search) | `src/retriever.py` |
| Cross-encoder reranking | `src/reranker.py` |
| LLM generation with grounded prompting | `src/rag_agent.py` |
| CLI to tie everything together | `main.py` |

---

## Architecture

```
INGEST PHASE (run once per book)
─────────────────────────────────────────────────────────
PDF → ingestion.py → chunker.py → embedder.py → vector_store.py (ChromaDB)

QUERY PHASE (run on every question)
─────────────────────────────────────────────────────────
Question → embedder.py → vector_store.py → retriever.py (top-20)
                                                  ↓
                                           reranker.py (top-5)
                                                  ↓
                                           rag_agent.py → LLM → Answer
```

---

## Tech Stack

- **PDF parsing** — [PyMuPDF](https://pymupdf.readthedocs.io/)
- **Embeddings** — [`all-MiniLM-L6-v2`](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) via `sentence-transformers` (local, free, 384-dim)
- **Vector database** — [ChromaDB](https://www.trychroma.com/) with HNSW index, persisted to disk
- **Reranker** — [`cross-encoder/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/cross-encoder/ms-marco-MiniLM-L-6-v2) (local, free)
- **LLM** — OpenAI `gpt-4o-mini` via API
- **Language** — Python 3.11+

---

## Project Structure

```
bookSummarizer/
├── main.py              # CLI entry point
├── requirements.txt
├── books/               # Drop your PDFs here
├── chroma_db/           # ChromaDB persisted storage (auto-created)
└── src/
    ├── ingestion.py     # PDF → list of pages
    ├── chunker.py       # Pages → overlapping text chunks
    ├── embedder.py      # Text → 384-dim vectors
    ├── vector_store.py  # ChromaDB CRUD operations
    ├── retriever.py     # Query → top-K nearest chunks
    ├── reranker.py      # top-K → top-N reranked chunks
    └── rag_agent.py     # Chunks + query → LLM answer
```

---

## Setup

**1. Clone the repo**

```bash
git clone https://github.com/your-username/bookSummarizer.git
cd bookSummarizer
```

**2. Create and activate a virtual environment**

```bash
python3 -m venv venv
source venv/bin/activate      # macOS/Linux
# venv\Scripts\activate       # Windows
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

> First install downloads PyTorch and sentence-transformers (~500MB). Subsequent runs use the cache.

**4. Set your OpenAI API key**

```bash
export OPENAI_API_KEY='sk-...'
```

---

## Usage

### Ingest a book

```bash
python main.py ingest books/mybook.pdf
```

With options:

```bash
# Choose chunking strategy
python main.py ingest books/mybook.pdf --strategy fixed
python main.py ingest books/mybook.pdf --strategy sentence
python main.py ingest books/mybook.pdf --strategy recursive   # default

# Tune chunk size and overlap
python main.py ingest books/mybook.pdf --chunk-size 256 --overlap 30

# Custom collection name
python main.py ingest books/mybook.pdf --collection atomic-habits
```

### Ask a question

```bash
python main.py query mybook "What is the main argument of this book?"
```

With debug mode to see retrieval and reranking scores:

```bash
python main.py query mybook "What does the author say about habits?" --debug
```

### Generate a full summary

```bash
python main.py summarize mybook
```

### List all ingested books

```bash
python main.py list
```

### Delete a collection

```bash
python main.py delete mybook
```

---

## How It Works — Deep Dive

### 1. Ingestion (`src/ingestion.py`)
Uses PyMuPDF (`fitz`) to open a PDF and extract text page by page. Each page becomes a `PageContent` object carrying the text, page number, and source filename. Pages with fewer than 20 characters (blank pages, page numbers) are skipped.

### 2. Chunking (`src/chunker.py`)
The full book text is split into overlapping chunks. Three strategies are available:

- **Fixed-size** — slides a window of N characters, advancing by `N - overlap` each step. Simple but can cut sentences mid-word.
- **Sentence-based** — groups sentences together until a character limit is reached. Respects sentence boundaries.
- **Recursive** (default) — tries to split on `\n\n` (paragraph) first, then `\n`, then `. `, then spaces. Produces the most semantically coherent chunks.

Each chunk carries its text, a unique ID, the page number it came from, and which strategy created it.

### 3. Embeddings (`src/embedder.py`)
Each chunk's text is passed through `all-MiniLM-L6-v2`, a locally-running sentence-transformer model that outputs a 384-dimensional vector. Semantically similar texts produce vectors that point in similar directions in this 384-dimensional space. Similarity is measured with cosine similarity: `(a · b) / (||a|| × ||b||)`, ranging from 0 (unrelated) to 1 (identical meaning).

### 4. Vector Store (`src/vector_store.py`)
Chunk texts, embeddings, and metadata are stored in ChromaDB — a local vector database that persists data to `./chroma_db/`. Each book gets its own named collection. ChromaDB uses an HNSW (Hierarchical Navigable Small World) index for `O(log N)` approximate nearest-neighbor search.

### 5. Retrieval (`src/retriever.py`)
At query time, the user's question is embedded with the same model. ChromaDB finds the 20 chunks with the nearest vectors to the query vector. This is **bi-encoder retrieval** — fast but approximate, because query and chunks are encoded independently.

### 6. Reranking (`src/reranker.py`)
The 20 retrieved chunks are passed to a **cross-encoder** (`ms-marco-MiniLM-L-6-v2`), which reads each `(query, chunk)` pair together rather than independently. This captures fine-grained interaction between the query and document, producing much more accurate relevance scores. The top 5 are kept. The two-stage design is intentional: bi-encoder for speed over the full corpus, cross-encoder for precision over the small candidate set.

### 7. Generation (`src/rag_agent.py`)
The top 5 chunks are formatted as labelled excerpts with page numbers and passed to `gpt-4o-mini` with a strict system prompt that instructs the model to only use the provided excerpts, cite page numbers, and say "I don't have enough information" if the chunks don't cover the question. `temperature=0.3` keeps the output factual and grounded.

---

## Learning Experiments

These will help you understand each layer intuitively:

```bash
# See cosine similarity in action — watch similar sentences score high
python -m src.embedder

# Compare chunking strategies on your book
python -m src.chunker books/mybook.pdf

# See raw retrieval scores vs reranking scores side by side
python main.py query mybook "your question" --debug

# Try different chunk sizes and compare answer quality
python main.py ingest books/mybook.pdf --chunk-size 256 --collection mybook-small
python main.py ingest books/mybook.pdf --chunk-size 1024 --collection mybook-large
python main.py query mybook-small "your question"
python main.py query mybook-large "your question"
```

---

## Key Concepts Learned

- **Why chunking matters** — chunk size and overlap directly control retrieval quality
- **What embeddings are** — dense vectors that place semantically similar text nearby in high-dimensional space
- **How vector databases work** — HNSW index for fast approximate nearest-neighbor search
- **Bi-encoder vs cross-encoder** — speed vs accuracy tradeoff in the two-stage retrieval pipeline
- **Grounded prompting** — how to prevent LLM hallucination by constraining it to retrieved context

---

## Requirements

```
pymupdf
chromadb
sentence-transformers
openai
numpy
```

---

## License

MIT
