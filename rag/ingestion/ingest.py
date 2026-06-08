"""
Ingestion pipeline: PDF -> parsed document -> chunks -> vector store.

This is the glue that the /ingest API route and the CLI both call. It keeps
the BM25 corpus in sync too by persisting chunk text alongside the vector
store, so the lexical retriever can be rebuilt without re-parsing PDFs.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger

from config import settings as default_settings
from rag.ingestion.chunker import chunk_document
from rag.ingestion.parser import parse_pdf
from rag.ingestion.schema import Chunk, ParsedDocument, TextBlock
from rag.retrieval.vector_store import VectorStore

# Formats the /ingest route accepts beyond PDF.
TEXT_SUFFIXES = {".txt", ".md", ".markdown"}


def _chunk_record(c: Chunk) -> dict:
    return {
        "chunk_id": c.chunk_id,
        "text": c.text,
        "bibkey": c.bibkey,
        "source_path": c.source_path,
        "title": c.title,
        "section": c.section,
        "page": c.page,
    }


def _append_corpus(chunks: list[Chunk], settings) -> None:
    """Persist chunk text to a JSONL the BM25 index is built from later."""
    corpus_path = Path(settings.processed_dir) / "corpus.jsonl"
    corpus_path.parent.mkdir(parents=True, exist_ok=True)
    with corpus_path.open("a", encoding="utf-8") as fh:
        for c in chunks:
            fh.write(json.dumps(_chunk_record(c), ensure_ascii=False) + "\n")


def _parse_text_file(path: Path) -> ParsedDocument:
    """Turn a .txt/.md file into a ParsedDocument. No structure to recover,
    so each blank-line-separated paragraph becomes a block and the title is
    the file name."""
    raw = path.read_text(encoding="utf-8", errors="ignore")
    paragraphs = [re.sub(r"\s+", " ", p).strip() for p in re.split(r"\n\s*\n", raw)]
    blocks = [TextBlock(text=p, section=None, page=None) for p in paragraphs if p]
    return ParsedDocument(
        source_path=str(path),
        parser="text",
        title=path.stem,
        blocks=blocks,
    )


def ingest_file(path: str | Path, store: VectorStore | None = None,
                settings=default_settings) -> list[Chunk]:
    """Ingest a single file, dispatching on extension (PDF vs text)."""
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        return ingest_pdf(path, store=store, settings=settings)
    if path.suffix.lower() in TEXT_SUFFIXES:
        store = store or VectorStore(settings=settings)
        doc = _parse_text_file(path)
        chunks = chunk_document(doc, settings)
        store.add_chunks(chunks)
        _append_corpus(chunks, settings)
        logger.info(f"Ingested {path} -> {len(chunks)} chunks (text)")
        return chunks
    raise ValueError(f"Unsupported file type: {path.suffix}")


def ingest_pdf(pdf_path: str | Path, store: VectorStore | None = None,
               settings=default_settings) -> list[Chunk]:
    store = store or VectorStore(settings=settings)
    doc = parse_pdf(pdf_path, settings)
    chunks = chunk_document(doc, settings)
    store.add_chunks(chunks)
    _append_corpus(chunks, settings)
    logger.info(f"Ingested {pdf_path} -> {len(chunks)} chunks ({doc.parser})")
    return chunks


def ingest_directory(pdf_dir: str | Path, settings=default_settings) -> int:
    pdf_dir = Path(pdf_dir)
    pdfs = sorted(pdf_dir.glob("*.pdf"))
    if not pdfs:
        logger.warning(f"No PDFs found in {pdf_dir}")
        return 0
    store = VectorStore(settings=settings)
    total = 0
    for pdf in pdfs:
        total += len(ingest_pdf(pdf, store=store, settings=settings))
    logger.info(f"Ingested {len(pdfs)} PDFs, {total} chunks total")
    return total
