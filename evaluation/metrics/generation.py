"""
Generation-quality metrics.

Citation metrics reuse the pipeline's own citation resolution (cited tags
that match a retrieved chunk vs fabricated tags that don't):
  - citation_presence: did the answer cite anything at all
  - citation_precision: of the tags it used, how many were grounded

ROUGE-L is kept as a secondary lexical-overlap check against QASPER's
reference answer. It's a weak signal for free-form scientific QA (a correct
answer worded differently scores low), so it's reported but not leaned on;
faithfulness and the judge carry more weight.
"""

from __future__ import annotations

from statistics import mean

from rouge_score import rouge_scorer

_rouge = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)


def rouge_l(prediction: str, reference: str) -> float:
    if not prediction or not reference:
        return 0.0
    return _rouge.score(reference, prediction)["rougeL"].fmeasure


def citation_metrics(results: list[dict]) -> dict[str, float]:
    """results: pipeline outputs, each with 'citations' and 'fabricated_citations'."""
    n = len(results)
    if n == 0:
        return {}
    with_cite = 0
    per_answer_precision: list[float] = []
    total_cited = total_fab = 0
    for r in results:
        cited = len(r.get("citations", []))
        fab = len(r.get("fabricated_citations", []))
        total_cited += cited
        total_fab += fab
        if cited + fab > 0:
            with_cite += 1
            per_answer_precision.append(cited / (cited + fab))
    denom = total_cited + total_fab
    return {
        "citation_presence": with_cite / n,
        "macro_citation_precision": mean(per_answer_precision) if per_answer_precision else 0.0,
        "micro_citation_precision": (total_cited / denom) if denom else 0.0,
        "n": n,
    }


def rouge_metrics(predictions: list[str], references: list[str]) -> dict[str, float]:
    scores = [rouge_l(p, r) for p, r in zip(predictions, references) if r]
    return {"rougeL_f": mean(scores) if scores else 0.0, "n_scored": len(scores)}
