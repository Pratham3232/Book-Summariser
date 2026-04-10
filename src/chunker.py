"""
Text Chunking Module
Three strategies for splitting text into chunks suitable for embedding & retrieval.
"""

import re
import uuid
from dataclasses import dataclass, field
from src.ingestion import PageContent


@dataclass
class Chunk:
    chunk_id: str
    text: str
    page_number: int
    source_file: str
    strategy: str
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Strategy 1: Fixed-Size Chunking
# ---------------------------------------------------------------------------
# Simplest approach — slide a window of `chunk_size` characters, advancing
# by `chunk_size - overlap` each step. Overlap prevents cutting mid-sentence.

def fixed_size_chunking(
    pages: list[PageContent],
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    full_text = "\n\n".join(p.text for p in pages)
    chunks: list[Chunk] = []
    step = chunk_size - overlap
    page_map = _build_page_map(pages)

    for start in range(0, len(full_text), step):
        end = start + chunk_size
        text = full_text[start:end].strip()
        if len(text) < 20:
            continue
        chunks.append(Chunk(
            chunk_id=_make_id(),
            text=text,
            page_number=_offset_to_page(start, page_map),
            source_file=pages[0].source_file if pages else "",
            strategy="fixed_size",
        ))
    return chunks


# ---------------------------------------------------------------------------
# Strategy 2: Sentence-Based Chunking
# ---------------------------------------------------------------------------
# Splits text into sentences, then groups them until the group hits
# ~max_tokens characters. This respects sentence boundaries so chunks
# are more semantically coherent.

def sentence_chunking(
    pages: list[PageContent],
    max_chars: int = 500,
) -> list[Chunk]:
    full_text = "\n\n".join(p.text for p in pages)
    sentences = _split_sentences(full_text)
    page_map = _build_page_map(pages)
    chunks: list[Chunk] = []

    current_group: list[str] = []
    current_len = 0
    group_start_offset = 0
    offset = 0

    for sent in sentences:
        if current_len + len(sent) > max_chars and current_group:
            text = " ".join(current_group).strip()
            if len(text) >= 20:
                chunks.append(Chunk(
                    chunk_id=_make_id(),
                    text=text,
                    page_number=_offset_to_page(group_start_offset, page_map),
                    source_file=pages[0].source_file if pages else "",
                    strategy="sentence",
                ))
            current_group = []
            current_len = 0
            group_start_offset = offset

        current_group.append(sent)
        current_len += len(sent)
        offset += len(sent) + 1  # +1 for the space between sentences

    if current_group:
        text = " ".join(current_group).strip()
        if len(text) >= 20:
            chunks.append(Chunk(
                chunk_id=_make_id(),
                text=text,
                page_number=_offset_to_page(group_start_offset, page_map),
                source_file=pages[0].source_file if pages else "",
                strategy="sentence",
            ))

    return chunks


# ---------------------------------------------------------------------------
# Strategy 3: Recursive Character Splitting
# ---------------------------------------------------------------------------
# Tries to split on the most meaningful boundary first (\n\n = paragraph),
# then falls back to lesser boundaries (\n, ". ", " "). This preserves
# document structure better than fixed-size.

SEPARATORS = ["\n\n", "\n", ". ", " "]


def recursive_chunking(
    pages: list[PageContent],
    chunk_size: int = 512,
    overlap: int = 50,
) -> list[Chunk]:
    full_text = "\n\n".join(p.text for p in pages)
    page_map = _build_page_map(pages)
    raw_chunks = _recursive_split(full_text, chunk_size, SEPARATORS)

    chunks: list[Chunk] = []
    offset = 0
    for text in raw_chunks:
        text = text.strip()
        if len(text) < 20:
            offset += len(text)
            continue
        chunks.append(Chunk(
            chunk_id=_make_id(),
            text=text,
            page_number=_offset_to_page(offset, page_map),
            source_file=pages[0].source_file if pages else "",
            strategy="recursive",
        ))
        offset += len(text)

    return chunks


def _recursive_split(text: str, chunk_size: int, separators: list[str]) -> list[str]:
    if len(text) <= chunk_size:
        return [text]

    sep = separators[0] if separators else " "
    remaining_seps = separators[1:] if len(separators) > 1 else []

    parts = text.split(sep)
    results: list[str] = []
    current = ""

    for part in parts:
        candidate = current + sep + part if current else part
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                results.append(current)
            if len(part) > chunk_size and remaining_seps:
                results.extend(_recursive_split(part, chunk_size, remaining_seps))
            else:
                current = part
                continue
            current = ""

    if current:
        results.append(current)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_id() -> str:
    return uuid.uuid4().hex[:12]


def _split_sentences(text: str) -> list[str]:
    """Naive sentence splitter on period/question/exclamation followed by space."""
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]


def _build_page_map(pages: list[PageContent]) -> list[tuple[int, int]]:
    """Build a list of (cumulative_offset, page_number) for offset-to-page lookup."""
    mapping: list[tuple[int, int]] = []
    offset = 0
    for p in pages:
        mapping.append((offset, p.page_number))
        offset += len(p.text) + 2  # +2 for "\n\n" join separator
    return mapping


def _offset_to_page(char_offset: int, page_map: list[tuple[int, int]]) -> int:
    """Given a character offset into the combined text, return the page number."""
    page_num = 1
    for start, pn in page_map:
        if start > char_offset:
            break
        page_num = pn
    return page_num


if __name__ == "__main__":
    from src.ingestion import extract_text_from_pdf
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.chunker <path_to_pdf>")
        sys.exit(1)
    pages = extract_text_from_pdf(sys.argv[1])
    for strategy_name, fn in [("fixed", fixed_size_chunking), ("sentence", sentence_chunking), ("recursive", recursive_chunking)]:
        chunks = fn(pages)
        print(f"\n{strategy_name}: {len(chunks)} chunks")
        if chunks:
            print(f"  Sample: {chunks[0].text[:100]}...")
