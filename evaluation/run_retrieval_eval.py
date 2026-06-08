"""
Retrieval evaluation: the ablation grid the roadmap asks for.

Compares, on the same QASPER questions and gold labels:
    vector            - dense only
    hybrid            - dense + BM25, fused with RRF
    hybrid+rerank     - add the cross-encoder reranker
    hybrid+rerank+hyde- add HyDE query expansion

Why the ordering in this script matters: HyDE needs the LLM, the reranker
wants the GPU, and both don't fit in 12GB together. So we run in phases:

    1. chunk papers and align gold labels (CPU)
    2. if any config uses HyDE, load the LLM, generate every hypothetical
       answer up front, then free the LLM
    3. embed the corpus (GPU), then run all configs with the reranker on GPU

That keeps each GPU-heavy model resident alone, so the whole sweep stays fast
and never has to fall back to slow CPU reranking.

    PYTHONPATH=. .venv/bin/python evaluation/run_retrieval_eval.py --max-papers 50
"""

from __future__ import annotations

import argparse
import gc
import json
import time
from datetime import datetime
from pathlib import Path

import torch
from loguru import logger

from config import settings
from evaluation.corpus_builder import build_eval_corpus
from evaluation.benchmarks.qasper import load_qasper
from evaluation.metrics.retrieval import aggregate, evaluate_query
from rag.reranking.reranker import Reranker
from rag.retrieval.hybrid_retriever import HybridRetriever

CONFIGS = {
    "vector": dict(use_hybrid=False, use_reranker=False, use_hyde=False),
    "hybrid": dict(use_hybrid=True, use_reranker=False, use_hyde=False),
    "hybrid+rerank": dict(use_hybrid=True, use_reranker=True, use_hyde=False),
    "hybrid+rerank+hyde": dict(use_hybrid=True, use_reranker=True, use_hyde=True),
}
KS = (1, 3, 5, 10)


def _free_llm(llm) -> None:
    """Release the LLM so the GPU is free for embedder + reranker."""
    from rag.generation import llm as llm_mod

    llm._model = None
    llm._tokenizer = None
    llm_mod._LLM_SINGLETON = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _precompute_hyde(questions) -> dict[str, str]:
    from rag.generation.llm import get_llm
    from rag.retrieval.hyde import generate_hyde

    logger.info(f"Generating HyDE expansions for {len(questions)} questions...")
    llm = get_llm(settings)
    cache: dict[str, str] = {}
    t0 = time.time()
    for i, q in enumerate(questions, 1):
        cache[q.qid] = generate_hyde(q.question, llm)
        if i % 20 == 0:
            logger.info(f"  HyDE {i}/{len(questions)} ({time.time()-t0:.0f}s)")
    _free_llm(llm)
    logger.info(f"HyDE done in {time.time()-t0:.0f}s, LLM freed")
    return cache


def run(max_papers: int, eval_top_n: int, include_hyde: bool) -> dict:
    configs = {k: v for k, v in CONFIGS.items()
               if include_hyde or not v["use_hyde"]}

    papers = load_qasper("validation", max_papers=max_papers)
    corpus = build_eval_corpus(papers, settings, embed=False)
    questions = corpus.questions

    hyde_cache: dict[str, str] = {}
    if any(c["use_hyde"] for c in configs.values()):
        hyde_cache = _precompute_hyde(questions)

    # Embed the corpus now that the LLM is gone.
    corpus.vector_store.add_chunks(list(corpus.chunks_by_id.values()))
    reranker = Reranker(settings=settings)

    results: dict[str, dict] = {}
    for name, cfg in configs.items():
        s = settings.model_copy(update=cfg)
        retriever = HybridRetriever(corpus.vector_store, corpus.bm25, reranker, settings=s)
        per_query = []
        t0 = time.time()
        for q in questions:
            hyde_text = hyde_cache.get(q.qid) if cfg["use_hyde"] else None
            hits = retriever.retrieve(q.question, hyde_text, top_n=eval_top_n)
            retrieved_ids = [h["chunk_id"] for h in hits]
            per_query.append(evaluate_query(retrieved_ids, q.gold_ids, KS))
        results[name] = aggregate(per_query)
        logger.info(f"[{name}] done in {time.time()-t0:.0f}s")

    return {
        "corpus": {
            "papers": len(papers),
            "chunks": corpus.n_chunks,
            "questions": len(questions),
        },
        "settings": {
            "chunk_size_tokens": settings.chunk_size_tokens,
            "chunk_overlap_tokens": settings.chunk_overlap_tokens,
            "eval_top_n": eval_top_n,
            "embedding_model": settings.embedding_model,
            "reranker_model": settings.reranker_model,
        },
        "results": results,
    }


def _print_table(report: dict) -> None:
    results = report["results"]
    cols = ["recall@1", "recall@5", "recall@10", "mrr", "ndcg@10"]
    name_w = max(len(n) for n in results) + 2
    header = "config".ljust(name_w) + "".join(c.rjust(11) for c in cols)
    print("\n" + header)
    print("-" * len(header))
    for name, metrics in results.items():
        row = name.ljust(name_w) + "".join(f"{metrics[c]:.3f}".rjust(11) for c in cols)
        print(row)
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-papers", type=int, default=50)
    ap.add_argument("--top-n", type=int, default=20, help="ranked list length for metrics")
    ap.add_argument("--no-hyde", action="store_true", help="skip the HyDE config (faster)")
    args = ap.parse_args()

    report = run(args.max_papers, args.top_n, include_hyde=not args.no_hyde)
    _print_table(report)

    out_dir = Path(settings.root_dir) / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"retrieval_eval_{stamp}.json"
    out_path.write_text(json.dumps(report, indent=2))
    print(f"saved report -> {out_path}")


if __name__ == "__main__":
    main()
