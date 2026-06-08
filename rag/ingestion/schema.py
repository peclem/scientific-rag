"""
Data model for parsed documents.

The whole pipeline downstream of parsing (chunking, retrieval, citations)
only ever sees these objects, so the parser implementations (GROBID,
PyMuPDF) are interchangeable as long as they produce a ParsedDocument.

The citation we ultimately want looks like [Smith2024, p.7], so every
TextBlock carries the section it came from and a best-effort page number,
and the document carries enough bibliographic info to build the bibkey.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class TextBlock:
    """A contiguous piece of text (usually a paragraph) plus where it came from."""

    text: str
    section: str | None = None   # heading this block sits under, if known
    page: int | None = None      # 1-indexed page, best effort


@dataclass
class ParsedDocument:
    source_path: str
    parser: str                  # "grobid" or "pymupdf"
    title: str | None = None
    authors: list[str] = field(default_factory=list)
    year: int | None = None
    n_pages: int | None = None
    blocks: list[TextBlock] = field(default_factory=list)
    bibkey_override: str | None = None  # force a specific key (e.g. a dataset id)

    @property
    def bibkey(self) -> str:
        """
        Short citation key like 'Smith2024'. Falls back to the file name
        when author/year are missing, so a citation is always produceable
        even on a badly parsed PDF. An explicit override wins over both,
        which the eval corpus uses to guarantee unique keys per paper.
        """
        if self.bibkey_override:
            return self.bibkey_override
        surname = None
        if self.authors:
            # Take the last whitespace-separated token of the first author
            # as the surname. Good enough for "Jane Smith" style names.
            surname = self.authors[0].strip().split()[-1]
        if surname and self.year:
            clean = re.sub(r"[^A-Za-z]", "", surname)
            return f"{clean}{self.year}"
        # Fallback: file stem, stripped to something citation-ish.
        stem = self.source_path.rsplit("/", 1)[-1].rsplit(".", 1)[0]
        return re.sub(r"[^A-Za-z0-9]", "", stem) or "unknown"

    def non_empty_blocks(self) -> list[TextBlock]:
        return [b for b in self.blocks if b.text and b.text.strip()]


@dataclass
class Chunk:
    """
    A retrieval unit. Carries everything needed to build a citation and to
    trace an answer back to its source, which is the whole point of the
    citation-enforced design.
    """

    chunk_id: str
    text: str
    bibkey: str
    source_path: str
    title: str | None = None
    section: str | None = None
    page: int | None = None
    n_tokens: int | None = None
    parser: str | None = None   # which parser produced the source doc

    @property
    def citation(self) -> str:
        """Human-facing citation tag, e.g. '[Smith2024, p.7]'."""
        if self.page is not None:
            return f"[{self.bibkey}, p.{self.page}]"
        return f"[{self.bibkey}]"

    def metadata(self) -> dict:
        """Flat dict for the vector store. Chroma needs primitive values, so
        None is replaced with empty/zero to keep it happy."""
        return {
            "chunk_id": self.chunk_id,
            "bibkey": self.bibkey,
            "source_path": self.source_path,
            "title": self.title or "",
            "section": self.section or "",
            "page": self.page if self.page is not None else 0,
            "citation": self.citation,
        }
