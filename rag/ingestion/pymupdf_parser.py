"""
PyMuPDF parser. This is the fallback when GROBID is unavailable, and it's
also the source of accurate per-page text that we use to assign page
numbers to GROBID blocks.

The tradeoff is the mirror image of GROBID: PyMuPDF knows exactly which
page text came from, but it has no idea about sections, references, or
header metadata. So as a standalone parser the section field stays None and
the bibkey usually falls back to the file name.
"""

from __future__ import annotations

import re
from pathlib import Path

import fitz  # PyMuPDF
from loguru import logger

from rag.ingestion.schema import ParsedDocument, TextBlock


def _split_paragraphs(page_text: str) -> list[str]:
    """Break a page into paragraph-ish blocks on blank lines."""
    parts = re.split(r"\n\s*\n", page_text)
    return [re.sub(r"\s+", " ", p).strip() for p in parts if p.strip()]


def page_texts(pdf_path: str | Path) -> list[str]:
    """Return the raw text of each page, 0-indexed. Used by the page mapper."""
    with fitz.open(pdf_path) as doc:
        return [page.get_text("text") for page in doc]


def parse_pdf_pymupdf(pdf_path: str | Path) -> ParsedDocument:
    pdf_path = Path(pdf_path)
    blocks: list[TextBlock] = []
    with fitz.open(pdf_path) as doc:
        n_pages = doc.page_count
        meta_title = (doc.metadata or {}).get("title") or None
        meta_author = (doc.metadata or {}).get("author") or None
        for page_index, page in enumerate(doc):
            for para in _split_paragraphs(page.get_text("text")):
                # PyMuPDF pages are 0-indexed; citations use 1-indexed pages.
                blocks.append(TextBlock(text=para, section=None, page=page_index + 1))

    authors = [meta_author] if meta_author else []
    doc_obj = ParsedDocument(
        source_path=str(pdf_path),
        parser="pymupdf",
        title=meta_title,
        authors=authors,
        year=None,
        n_pages=n_pages,
        blocks=blocks,
    )
    logger.info(
        f"PyMuPDF parsed '{pdf_path.name}': {len(blocks)} blocks across "
        f"{n_pages} pages, bibkey={doc_obj.bibkey}"
    )
    return doc_obj
