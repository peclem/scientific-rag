"""Tests for citation extraction and resolution. Pure logic, no models."""

from rag.prompts.citations import extract_tags, resolve_citations


def _hit(citation, bibkey="Smith2024", chunk_id="c0"):
    return {
        "chunk_id": chunk_id,
        "text": "some text",
        "metadata": {"citation": citation, "bibkey": bibkey, "chunk_id": chunk_id},
    }


def test_extract_tags_dedup_and_order():
    answer = "First [A2024, p.1]. Then [B2023]. Again [A2024, p.1]."
    assert extract_tags(answer) == ["[A2024, p.1]", "[B2023]"]


def test_resolve_marks_supported_citation():
    hits = [_hit("[Smith2024, p.7]")]
    answer = "Transformers use attention [Smith2024, p.7]."
    res = resolve_citations(answer, hits)
    assert len(res["cited"]) == 1
    assert res["cited"][0]["citation"] == "[Smith2024, p.7]"
    assert res["fabricated"] == []


def test_resolve_flags_fabricated_citation():
    hits = [_hit("[Smith2024, p.7]")]
    answer = "As shown in [Bogus1999, p.3], this is made up."
    res = resolve_citations(answer, hits)
    assert res["cited"] == []
    assert res["fabricated"] == ["[Bogus1999, p.3]"]


def test_resolve_no_citations():
    hits = [_hit("[Smith2024, p.7]")]
    res = resolve_citations("An answer with no tags.", hits)
    assert res["cited"] == []
    assert res["fabricated"] == []
