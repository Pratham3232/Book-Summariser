"""
PDF Ingestion Module
Extracts text from PDF files page-by-page using PyMuPDF (fitz).
"""

import fitz  # PyMuPDF
from pathlib import Path
from dataclasses import dataclass


@dataclass
class PageContent:
    page_number: int
    text: str
    source_file: str


def extract_text_from_pdf(pdf_path: str) -> list[PageContent]:
    """
    Opens a PDF and extracts text from every page.
    Returns a list of PageContent objects with page number, text, and source file metadata.
    """
    path = Path(pdf_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")
    if path.suffix.lower() != ".pdf":
        raise ValueError(f"Not a PDF file: {pdf_path}")

    doc = fitz.open(pdf_path)
    pages: list[PageContent] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")

        # Skip pages with negligible text (headers/footers only)
        cleaned = text.strip()
        if len(cleaned) < 20:
            continue

        pages.append(PageContent(
            page_number=page_num + 1,  # 1-indexed for human readability
            text=cleaned,
            source_file=path.name,
        ))

    doc.close()
    print(f"Extracted {len(pages)} pages from '{path.name}'")
    return pages


def combine_pages(pages: list[PageContent]) -> str:
    """Joins all page texts into a single string (useful for full-book operations)."""
    return "\n\n".join(p.text for p in pages)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python -m src.ingestion <path_to_pdf>")
        sys.exit(1)
    pages = extract_text_from_pdf(sys.argv[1])
    for p in pages[:3]:
        print(f"\n--- Page {p.page_number} ---")
        print(p.text[:300], "...")
