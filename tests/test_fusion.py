"""Tests for Reciprocal Rank Fusion. Pure logic, no models."""

from rag.retrieval.fusion import reciprocal_rank_fusion


def _hit(cid: str):
    return {"chunk_id": cid, "text": cid, "metadata": {"chunk_id": cid}}


def test_rrf_rewards_agreement():
    # A chunk ranked highly by both lists should beat one ranked highly by
    # only one. With k=60: 'a' = 1/60 + 1/60, 'b' = 1/60 + 1/63.
    list_a = [_hit("a"), _hit("b"), _hit("c")]
    list_b = [_hit("a"), _hit("d"), _hit("c"), _hit("b")]
    fused = reciprocal_rank_fusion([list_a, list_b], k=60)
    assert fused[0]["chunk_id"] == "a"


def test_rrf_score_formula():
    # Single list, single item at rank 0 -> 1/(60+0).
    fused = reciprocal_rank_fusion([[_hit("x")]], k=60)
    assert abs(fused[0]["rrf_score"] - (1.0 / 60)) < 1e-9


def test_rrf_top_n_truncates():
    hits = [_hit(c) for c in "abcde"]
    fused = reciprocal_rank_fusion([hits], k=60, top_n=2)
    assert len(fused) == 2


def test_rrf_merges_by_chunk_id():
    fused = reciprocal_rank_fusion([[_hit("a")], [_hit("a")]], k=60)
    assert len(fused) == 1
    assert fused[0]["ranks"] == [0, 0]
