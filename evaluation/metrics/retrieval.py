"""
Retrieval metrics: Recall@K, MRR, nDCG@K.

All three take a ranked list of retrieved chunk ids and a set of gold ids.

  - Recall@K: of the gold chunks, how many made the top K. Answers "did we
    pull the evidence in at all", which matters most for RAG since a chunk
    that isn't retrieved can never inform the answer.
  - MRR: 1 / rank of the first gold hit. Rewards putting a relevant chunk
    near the top.
  - nDCG@K: rank-discounted gain, normalized by the best achievable ordering.
    Rewards getting all the gold chunks high, not just the first.
"""

from __future__ import annotations

import math


def recall_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    hits = sum(1 for cid in retrieved[:k] if cid in gold)
    return hits / len(gold)


def mrr(retrieved: list[str], gold: set[str]) -> float:
    for rank, cid in enumerate(retrieved, start=1):
        if cid in gold:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: list[str], gold: set[str], k: int) -> float:
    if not gold:
        return 0.0
    dcg = 0.0
    for i, cid in enumerate(retrieved[:k]):
        if cid in gold:
            dcg += 1.0 / math.log2(i + 2)  # i is 0-indexed, so rank = i+1
    ideal_hits = min(len(gold), k)
    idcg = sum(1.0 / math.log2(i + 2) for i in range(ideal_hits))
    return dcg / idcg if idcg else 0.0


def evaluate_query(retrieved: list[str], gold: set[str],
                   ks: tuple[int, ...] = (1, 3, 5, 10)) -> dict[str, float]:
    out: dict[str, float] = {"mrr": mrr(retrieved, gold)}
    for k in ks:
        out[f"recall@{k}"] = recall_at_k(retrieved, gold, k)
        out[f"ndcg@{k}"] = ndcg_at_k(retrieved, gold, k)
    return out


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: sum(q[key] for q in per_query) / len(per_query) for key in keys}
