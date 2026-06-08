"""
Reciprocal Rank Fusion (RRF).

RRF merges several ranked lists into one. The key property, and the reason
it's a good default for hybrid search, is that it uses each item's *rank
position*, not its raw score. Vector cosine similarities and BM25 scores
live on completely different scales, so trying to add them directly is
meaningless. RRF sidesteps that entirely: an item's contribution from a list
is 1 / (k + rank), summed across the lists it appears in.

k (default 60) dampens the influence of top ranks so a single list can't
dominate. Items retrieved by both methods naturally float up because they
collect score from both.
"""

from __future__ import annotations

from config import settings as default_settings


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int | None = None,
    top_n: int | None = None,
) -> list[dict]:
    """
    Each result list is assumed already sorted best-first. Items are matched
    across lists by chunk_id. Returns a single list sorted by fused score,
    with the per-method ranks kept for debugging/logging.
    """
    k = k if k is not None else default_settings.rrf_k

    fused: dict[str, dict] = {}
    for result_list in result_lists:
        for rank, hit in enumerate(result_list):
            cid = hit["chunk_id"]
            contribution = 1.0 / (k + rank)
            if cid not in fused:
                fused[cid] = {
                    "chunk_id": cid,
                    "text": hit["text"],
                    "metadata": hit.get("metadata", {}),
                    "rrf_score": 0.0,
                    "ranks": [],
                }
            fused[cid]["rrf_score"] += contribution
            fused[cid]["ranks"].append(rank)

    ordered = sorted(fused.values(), key=lambda h: h["rrf_score"], reverse=True)
    return ordered[:top_n] if top_n else ordered
