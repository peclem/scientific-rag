"""
The capstone experiment: a 2x2 comparison of fine-tuning x RAG.

    Base       - Qwen3-8B, no retrieval (parametric knowledge only)
    Base+RAG   - Qwen3-8B with the retrieval pipeline
    FT         - QLoRA-tuned Qwen3-8B, no retrieval
    FT+RAG     - QLoRA-tuned Qwen3-8B with retrieval

All four cells answer the same questions and are scored with the same metrics
(citation precision, NLI faithfulness, ROUGE, and a single fixed judge), so
the differences isolate what fine-tuning adds versus what RAG adds. The honest
hypothesis: RAG dominates accuracy/grounding; FT mainly tightens format and
citation habits.

Fairness + VRAM notes:
  - One fixed judge (the base model) scores every cell, so the judge isn't a
    confound. Judging happens after all answers exist.
  - Only one heavy model is resident at a time. Phased as:
      1. base LLM: HyDE + Base (no-RAG) answers           -> free
      2. retrieval: embed + retrieve contexts             -> free
      3. FT LLM: FT and FT+RAG answers                    -> free
      4. base LLM: Base+RAG answers, then judge all cells -> free
      5. NLI: faithfulness for all cells

    PYTHONPATH=. .venv/bin/python evaluation/run_2x2_comparison.py --max-questions 25
"""

from __future__ import annotations

import argparse
import gc
import json
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
from rag.generation.llm import LLM
from rag.prompts.builder import build_messages
from rag.prompts.citations import resolve_citations
from rag.reranking.reranker import Reranker
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.hyde import generate_hyde

ADAPTER_PATH = "adapters/qwen3-8b-scirag"


def _cuda_free() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _free(obj) -> None:
    for attr in ("_model", "_tokenizer"):
        if hasattr(obj, attr):
            setattr(obj, attr, None)
    _cuda_free()


def _generate_cell(llm: LLM, questions, contexts_by_qid: dict | None) -> list[dict]:
    """Answer every question with this model. contexts_by_qid=None means no RAG."""
    out = []
    for q in questions:
        hits = contexts_by_qid.get(q.qid, []) if contexts_by_qid else []
        answer = llm.generate(build_messages(q.question, hits))
        cites = resolve_citations(answer, hits)
        out.append({
            "qid": q.qid,
            "question": q.question,
            "answer": answer,
            "contexts": [h["text"] for h in hits],
            "citations": cites["cited"],
            "fabricated_citations": cites["fabricated"],
        })
    return out


def run(max_papers: int, max_questions: int) -> dict:
    papers = load_qasper("validation", max_papers=max_papers)
    corpus = build_eval_corpus(papers, settings, embed=False)
    refs = {q.qid: q.reference for p in papers for q in p.questions}
    questions = corpus.questions[:max_questions]
    logger.info(f"2x2 comparison over {len(questions)} questions")

    base_settings = settings.model_copy(update={"lora_adapter_path": ""})
    ft_settings = settings.model_copy(update={"lora_adapter_path": ADAPTER_PATH})

    # --- Phase 1: base LLM -> HyDE + Base (no-RAG) ---
    base = LLM(base_settings)
    hyde_cache = {}
    if settings.use_hyde:
        for q in questions:
            hyde_cache[q.qid] = generate_hyde(q.question, base)
    cells = {"base": _generate_cell(base, questions, None)}
    _free(base)
    logger.info("Phase 1 done (HyDE + Base)")

    # --- Phase 2: retrieval ---
    corpus.vector_store.add_chunks(list(corpus.chunks_by_id.values()))
    reranker = Reranker(settings=settings)
    retriever = HybridRetriever(corpus.vector_store, corpus.bm25, reranker, settings=settings)
    contexts = {
        q.qid: retriever.retrieve(q.question, hyde_cache.get(q.qid), top_n=settings.rerank_top_n)
        for q in questions
    }
    _free(corpus.vector_store.embedder)
    _free(reranker)
    logger.info("Phase 2 done (retrieval)")

    # --- Phase 3: FT LLM -> FT and FT+RAG ---
    ft = LLM(ft_settings)
    cells["ft"] = _generate_cell(ft, questions, None)
    cells["ft+rag"] = _generate_cell(ft, questions, contexts)
    _free(ft)
    logger.info("Phase 3 done (FT, FT+RAG)")

    # --- Phase 4: base LLM -> Base+RAG, then judge ALL cells with base ---
    judge = LLM(base_settings)
    cells["base+rag"] = _generate_cell(judge, questions, contexts)
    for name, results in cells.items():
        for r in results:
            r["judgment"] = judge_answer(judge, r["question"], r["answer"], r["contexts"])
    _free(judge)
    logger.info("Phase 4 done (Base+RAG + judging)")

    # --- Phase 5: NLI faithfulness for all cells ---
    scorer = FaithfulnessScorer(settings=settings, device="cuda")
    report = {}
    for name, results in cells.items():
        faith = [scorer.score_answer(r["answer"], r["contexts"]) for r in results]
        report[name] = {
            "citation": citation_metrics(results),
            "faithfulness_nli": scorer.aggregate(faith),
            "rouge": rouge_metrics([r["answer"] for r in results],
                                   [refs.get(r["qid"], "") for r in results]),
            "judge": aggregate_judgments([r["judgment"] for r in results]),
        }

    return {"n_questions": len(questions), "cells": report, "details": cells}


CELL_ORDER = ["base", "base+rag", "ft", "ft+rag"]


def _print_table(report: dict) -> None:
    cells = report["cells"]
    print(f"\n===== 2x2 COMPARISON (n={report['n_questions']}) =====")
    header = f"{'cell':<10}{'judge_overall':>14}{'judge_faith':>13}{'cite_prec':>11}{'nli_faith':>11}{'rougeL':>9}"
    print(header)
    print("-" * len(header))
    for name in CELL_ORDER:
        c = cells.get(name, {})
        j = c.get("judge", {})
        print(f"{name:<10}"
              f"{j.get('overall', 0):>14.2f}"
              f"{j.get('faithfulness', 0):>13.2f}"
              f"{c.get('citation', {}).get('micro_citation_precision', 0):>11.2f}"
              f"{c.get('faithfulness_nli', {}).get('faithfulness', 0):>11.2f}"
              f"{c.get('rouge', {}).get('rougeL_f', 0):>9.3f}")
    print()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-papers", type=int, default=40)
    ap.add_argument("--max-questions", type=int, default=25)
    args = ap.parse_args()

    if not Path(ADAPTER_PATH).exists():
        raise SystemExit(f"Adapter not found at {ADAPTER_PATH}. Train it first "
                         f"(training/qlora/train.py).")

    out = run(args.max_papers, args.max_questions)
    _print_table(out)

    out_dir = Path(settings.root_dir) / "evaluation" / "reports"
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out_dir / f"comparison_2x2_{stamp}.json").write_text(json.dumps(out, indent=2))
    print(f"saved -> comparison_2x2_{stamp}.json")


if __name__ == "__main__":
    main()
