"""
/ingest: upload a PDF, TXT, or Markdown file and add it to the corpus.

The file is saved under data/raw_pdfs, parsed and chunked, embedded into
Chroma, and appended to the BM25 corpus. We refresh the BM25 index after so
lexical search immediately sees the new content.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile

from api.deps import get_vector_store, refresh_bm25
from api.schemas.models import IngestResponse
from config import settings
from rag.ingestion.ingest import TEXT_SUFFIXES, ingest_file

router = APIRouter()

ALLOWED = {".pdf", *TEXT_SUFFIXES}


@router.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)) -> IngestResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED)}",
        )

    dest = Path(settings.raw_pdf_dir) / Path(file.filename).name
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(await file.read())

    store = get_vector_store()
    chunks = ingest_file(dest, store=store, settings=settings)
    refresh_bm25()

    return IngestResponse(
        filename=dest.name,
        parser=chunks[0].parser if chunks else "none",
        chunks_added=len(chunks),
        collection_size=store.count(),
    )
