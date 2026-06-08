"""
Top-level PDF parsing entry point.

Strategy:
  1. If GROBID is up, use it for structure + metadata, then back-fill page
     numbers from PyMuPDF's per-page text. This is the good path and gives
     proper sections and a real bibkey.
  2. If GROBID is down and fallback is allowed, use PyMuPDF alone. We lose
     section structure but keep accurate pages and still produce chunks.
  3. If GROBID is down and fallback is disabled, raise, so a misconfigured
     ingestion fails loudly instead of silently degrading.

Everything else in the pipeline calls parse_pdf and doesn't care which path
ran.
"""

from __future__ import annotations

from pathlib import Path

from loguru import logger

from config import settings as default_settings
from rag.ingestion import grobid_client, pymupdf_parser
from rag.ingestion.page_mapper import assign_pages
from rag.ingestion.schema import ParsedDocument


def parse_pdf(pdf_path: str | Path, settings=default_settings) -> ParsedDocument:
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise FileNotFoundError(pdf_path)

    use_grobid = grobid_client.is_grobid_alive(settings.grobid_url)

    if use_grobid:
        try:
            doc = grobid_client.parse_pdf_grobid(
                pdf_path, settings.grobid_url, settings.grobid_timeout_s
            )
            # Back-fill page numbers from PyMuPDF.
            pages = pymupdf_parser.page_texts(pdf_path)
            assign_pages(doc, pages)
            return doc
        except Exception as exc:  # noqa: BLE001
            logger.warning(f"GROBID parse failed for {pdf_path.name}: {exc}")
            if not settings.grobid_fallback_pymupdf:
                raise
            logger.info("Falling back to PyMuPDF.")

    elif not settings.grobid_fallback_pymupdf:
        raise RuntimeError(
            f"GROBID not reachable at {settings.grobid_url} and PyMuPDF "
            f"fallback is disabled."
        )
    else:
        logger.info("GROBID not reachable, using PyMuPDF fallback.")

    return pymupdf_parser.parse_pdf_pymupdf(pdf_path)
