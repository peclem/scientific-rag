"""
Generation evaluation: answer quality, not just retrieval.

For a sample of QASPER questions it runs the production retrieval config,
generates an answer, and scores it four ways:
  - citation precision (grounded vs fabricated citation tags)
  - faithfulness / hallucination rate (claim-level NLI vs the context)
  - ROUGE-L against the QASPER reference answer (secondary)
  - a local-model rubric judge

The hard part is VRAM. On a 12GB card we can only hold one heavy model at a
time, so the runner is strictly phased and frees each model before loading
the next:

  1. HyDE expansions      (LLM)        -> free LLM
  2. embed + retrieve + rerank (embedder + reranker on GPU) -> free both
  3. generate + judge     (LLM)        -> free LLM
  4. faithfulness          (NLI model)

    PYTHONPATH=. .venv/bin/python evaluation/run_generation_eval.py --max-questions 30
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
from evaluation.benchmarks.qasper import load_qasper
from evaluation.corpus_builder import build_eval_corpus
from evaluation.gpt_judge.judge import aggregate_judgments, judge_answer
from evaluation.metrics.faithfulness import FaithfulnessScorer
from evaluation.metrics.generation import citation_metrics, rouge_metrics
from rag.prompts.builder import build_messages
from rag.prompts.citations import resolve_citations
from rag.reranking.reranker import Reranker
from rag.retrieval.hybrid_retriever import HybridRetriever


def _cuda_free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _free_model_holder(obj) -> None:
    """Drop a lazily-loaded model wrapper's weights to reclaim VRAM."""
    for attr in ("_model", "_tokenizer"):
        if hasattr(obj, attr):
            setattr(obj, attr, None)
    _cuda_free()


def _free_llm() -> None:
    from rag.generation import llm as llm_mod

    if llm_mod._LLM_SINGLETON is not None:
        llm_mod._LLM_SINGLETON._model = None
        llm_mod._LLM_SINGLETON._tokenizer = None
        llm_mod._LLM_SINGLETON = None
    _cuda_free()


def run(max_papers: int, max_questions: int) -> dict:
    papers = load_qasper("validation", max_papers=max_papers)
    corpus = build_eval_corpus(papers, settings, embed=False)
    # Keep references for ROUGE.
    refs = {q.qid: q.reference for p in papers for q in p.questions}
    questions = corpus.questions[:max_questions]
    logger.info(f"Generation eval over {len(questions)} questions")

    # --- Phase 1: HyDE (LLM) ---
    from rag.generation.llm import get_llm
    from rag.retrieval.hyde import generate_hyde

    hyde_cache: dict[str, str] = {}
    if settings.use_hyde:
        llm = get_llm(settings)
        for q in questions:
            hyde_cache[q.qid] = generate_hyde(q.question, llm)
        _free_llm()
        logger.info("HyDE done, LLM freed")

    # --- Phase 2: retrieve + rerank (embedder + reranker on GPU) ---
    corpus.vector_store.add_chunks(list(corpus.chunks_by_id.values()))
    reranker = Reranker(settings=settings)
    retriever = HybridRetriever(corpus.vector_store, corpus.bm25, reranker, settings=settings)

    retrieved: dict[str, list[dict]] = {}
    for q in questions:
        hits = retriever.retrieve(q.question, hyde_cache.get(q.qid), top_n=settings.rerank_top_n)
        retrieved[q.qid] = hits
    _free_model_holder(corpus.vector_store.embedder)
    _free_model_holder(reranker)
    logger.info("Retrieval done, embedder + reranker freed")

    # --- Phase 3: generate + judge (LLM) ---
    llm = get_llm(settings)
    results = []
    for q in questions:
        hits = retrieved[q.qid]
        answer = llm.generate(build_messages(q.question, hits))
        cites = resolve_citations(answer, hits)
        judgment = judge_answer(llm, q.question, answer, [h["text"] for h in hits])
        results.append({
            "qid": q.qid,
            "question": q.question,
            "answer": answer,
            "reference": refs.get(q.qid, ""),
            "contexts": [h["text"] for h in hits],
            "citations": cites["cited"],
            "fabricated_citations": cites["fabricated"],
            "judgment": judgment,
        })
    _free_llm()
    logger.info("Generation + judging done, LLM freed")

    # --- Phase 4: faithfulness (NLI) ---
    scorer = FaithfulnessScorer(settings=settings, device="cuda")
    faith_per = []
    for r in results:
        f = scorer.score_answer(r["answer"], r["contexts"])
        r["faithfulness"] = f
        faith_per.append(f)

    # --- Aggregate ---
    report = {
        "n_questions": len(questions),
        "citation": citation_metrics(results),
        "faithfulness": scorer.aggregate(faith_per),
        "rouge": rouge_metrics(
            [r["answer"] for r in results], [r["reference"] for r in results]
        ),
        "judge": aggregate_judgments([r["judgment"] for r in results]),
    }
    return {"report": report, "details": results}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-papers", type=int, default=40)
    ap.add_argument("--max-questions", type=int, default=30)
    args = ap.parse_args()

    t0 = time.time()
    out = run(args.max_papers, args.max_questions)
    report = out["report"]

    print("\n===== GENERATION EVALUATION =====")
    print(json.dumps(report, indent=2))
    print(f"\ntotal time: {time.time()-t0:.0f}s")

    out_dir = Path(settings.root_dir) / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"generation_eval_{stamp}.json").write_text(json.dumps(out, indent=2))
    print(f"saved -> generation_eval_{stamp}.json")


if __name__ == "__main__":
    main()
