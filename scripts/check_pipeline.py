"""
Full pipeline smoke test: question in, grounded answer + citations out.

Runs the whole chain (HyDE, hybrid retrieval, rerank, generation) against
whatever is already in the vector store. Reports latency and peak VRAM so we
can confirm the LLM-plus-retrieval-models memory story on the 12GB card.

Run with retrieval models on CPU (the realistic serving config):
    PYTHONPATH=. LOW_VRAM_MODE=true .venv/bin/python scripts/check_pipeline.py
"""

from __future__ import annotations

import torch

from config import settings
from rag.generation.llm import get_llm
from rag.pipeline import RAGPipeline
from rag.reranking.reranker import Reranker
from rag.retrieval.bm25_retriever import BM25Retriever
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.vector_store import VectorStore


def main() -> None:
    print(f"low_vram_mode={settings.low_vram_mode} "
          f"(embedder/reranker on {'cpu' if settings.low_vram_mode else 'gpu'})")

    store = VectorStore(settings=settings)
    print(f"collection size: {store.count()}")
    bm25 = BM25Retriever.from_corpus_file(settings=settings)
    reranker = Reranker(settings=settings)
    retriever = HybridRetriever(store, bm25, reranker, settings=settings)
    pipeline = RAGPipeline(retriever, get_llm(settings), settings=settings)

    question = "What is multi-head attention and why is it used instead of single attention?"
    print(f"\nQ: {question}\n")

    result = pipeline.answer(question)

    print("----- answer -----")
    print(result["answer"])
    print("\n----- citations -----")
    for c in result["citations"]:
        print(f"  {c['citation']}  (section: {c.get('section')})")
    if result["fabricated_citations"]:
        print("  fabricated:", result["fabricated_citations"])
    print(f"\nlatency: {result['latency_s']}s | contexts used: {len(result['contexts'])}")

    if torch.cuda.is_available():
        print(f"peak VRAM (process): {torch.cuda.max_memory_allocated() / 1024**3:.2f} GB")


if __name__ == "__main__":
    main()
