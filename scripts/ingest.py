"""
CLI for ingesting PDFs into the vector store.

    PYTHONPATH=. .venv/bin/python scripts/ingest.py data/raw_pdfs
    PYTHONPATH=. .venv/bin/python scripts/ingest.py paper.pdf --reset
"""

from __future__ import annotations

import argparse
from pathlib import Path

from rag.ingestion.ingest import ingest_directory, ingest_pdf
from rag.retrieval.vector_store import VectorStore


def main() -> None:
    ap = argparse.ArgumentParser(description="Ingest PDFs into the vector store.")
    ap.add_argument("path", help="a PDF file or a directory of PDFs")
    ap.add_argument("--reset", action="store_true",
                    help="clear the collection before ingesting")
    args = ap.parse_args()

    if args.reset:
        VectorStore().reset()

    path = Path(args.path)
    if path.is_dir():
        ingest_directory(path)
    else:
        ingest_pdf(path)


if __name__ == "__main__":
    main()
