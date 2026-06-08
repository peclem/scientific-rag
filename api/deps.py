"""
Shared component wiring for the API.

The heavy objects (vector store, reranker, LLM) are built once and reused.
They each load their model lazily on first real use, so importing this
module and starting the app stays fast; the first request pays the load
cost.

BM25 is special: its in-memory index has to reflect whatever has been
ingested, so it's (re)built from corpus.jsonl and refreshed after an ingest.

Per-request ablation toggles (use_hyde, top_n, ...) are applied by copying
the settings and building a lightweight pipeline around the cached heavy
components, so a request can flip HyDE off without disturbing global state.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from config import settings
from rag.generation.llm import get_llm
from rag.pipeline import RAGPipeline
from rag.reranking.reranker import Reranker
from rag.retrieval.bm25_retriever import BM25Retriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.vector_store import VectorStore


@lru_cache(maxsize=1)
def get_vector_store() -> VectorStore:
    return VectorStore(settings=settings)


@lru_cache(maxsize=1)
def get_reranker() -> Reranker:
    return Reranker(settings=settings)


_bm25: BM25Retriever | None = None


def get_bm25(force_reload: bool = False) -> BM25Retriever | None:
    global _bm25
    corpus = Path(settings.processed_dir) / "corpus.jsonl"
    if (_bm25 is None or force_reload) and corpus.exists():
        _bm25 = BM25Retriever.from_corpus_file(corpus, settings)
    return _bm25


def refresh_bm25() -> None:
    """Call after ingestion so lexical search sees the new chunks."""
    get_bm25(force_reload=True)


def build_pipeline(overrides: dict | None = None) -> RAGPipeline:
    s = settings.model_copy(update=overrides) if overrides else settings
    retriever = HybridRetriever(
        vector_store=get_vector_store(),
        bm25=get_bm25(),
        reranker=get_reranker(),
        settings=s,
    )
    return RAGPipeline(retriever=retriever, llm=get_llm(), settings=s)
