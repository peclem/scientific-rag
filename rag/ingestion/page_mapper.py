"""
Page assignment for GROBID blocks.

GROBID gives us clean, reflowed paragraphs but no page numbers. PyMuPDF
gives us per-page text but messy structure. This module bridges them: for
each GROBID block we take a short probe from the start of the block and find
which page's text contains it.

Matching is done on whitespace-stripped lowercase text because GROBID
reflows paragraphs while PyMuPDF keeps the original line breaks, so the two
never match character-for-character. Stripping whitespace makes the probe
robust to that. Search advances monotonically through pages since blocks
come in document order, which keeps it cheap and avoids matching a repeated
phrase to an earlier page.
"""

from __future__ import annotations

import re

from loguru import logger

from rag.ingestion.schema import ParsedDocument

PROBE_CHARS = 50


def _strip(text: str) -> str:
    return re.sub(r"\s+", "", text).lower()


def assign_pages(doc: ParsedDocument, page_texts: list[str]) -> ParsedDocument:
    """Fill in block.page in place using PyMuPDF page texts. Returns doc."""
    stripped_pages = [_strip(t) for t in page_texts]
    doc.n_pages = len(page_texts)

    search_from = 0
    last_known = 1
    for block in doc.blocks:
        probe = _strip(block.text)[:PROBE_CHARS]
        found_page = None
        if probe:
            for page_idx in range(search_from, len(stripped_pages)):
                if probe in stripped_pages[page_idx]:
                    found_page = page_idx + 1  # 1-indexed for citations
                    search_from = page_idx     # next block starts here
                    break
        if found_page is not None:
            block.page = found_page
            last_known = found_page
        else:
            # Couldn't locate it (figures, tables, equations often mangle the
            # text). Attribute it to the last page we did find so the citation
            # is approximately right rather than missing.
            block.page = last_known

    matched = sum(1 for b in doc.blocks if b.page is not None)
    logger.debug(f"Page mapping: {matched}/{len(doc.blocks)} blocks placed")
    return doc
