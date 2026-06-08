"""
BM25 lexical retriever.

The vector side is good at semantic similarity but can miss exact terms:
a specific method name, an acronym, a dataset, a gene symbol. BM25 covers
that blind spot by matching tokens directly, which is exactly why the
hybrid setup pairs the two.

This is an in-memory index built with rank_bm25 over the same chunks that
went into the vector store. It rebuilds from data/processed/corpus.jsonl,
so it can be reconstructed without re-parsing PDFs. That's fine at the scale
this project works at; a production system would push lexical search into
something like Elasticsearch, which the plan deliberately leaves out.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from loguru import logger
from rank_bm25 import BM25Okapi

from config import settings as default_settings

_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization. Simple on purpose: BM25 doesn't
    need anything clever, and keeping it plain makes results easy to reason
    about during the hybrid-vs-vector ablation."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    def __init__(self, records: list[dict]):
        # records: dicts with at least chunk_id, text, and the chunk metadata.
        self.records = records
        self._tokenized = [tokenize(r["text"]) for r in records]
        self.bm25 = BM25Okapi(self._tokenized)
        logger.info(f"BM25 index built over {len(records)} chunks")

    @classmethod
    def from_corpus_file(cls, path: str | Path | None = None,
                         settings=default_settings) -> "BM25Retriever":
        path = Path(path) if path else Path(settings.processed_dir) / "corpus.jsonl"
        records = []
        with path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return cls(records)

    @classmethod
    def from_chunks(cls, chunks) -> "BM25Retriever":
        records = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "bibkey": c.bibkey,
                "source_path": c.source_path,
                "title": c.title,
                "section": c.section,
                "page": c.page,
                "citation": c.citation,
            }
            for c in chunks
        ]
        return cls(records)

    def query(self, query_text: str, k: int | None = None) -> list[dict]:
        k = k or default_settings.bm25_top_k
        scores = self.bm25.get_scores(tokenize(query_text))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        hits = []
        for i in ranked:
            r = self.records[i]
            hits.append({
                "chunk_id": r["chunk_id"],
                "text": r["text"],
                "metadata": r,
                "score": float(scores[i]),
            })
        return hits
