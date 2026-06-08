"""
Gold relevance labels: which chunks should a question retrieve?

QASPER gives evidence as paragraph text drawn from the paper. Our chunker
reflows and token-splits that same text, so an evidence paragraph may land
whole in one chunk or be split across two. We can't rely on exact substring
matching, so we match on word 5-grams: a chunk is relevant to an evidence
paragraph if they share enough 5-grams. Real prose 5-grams are distinctive,
so this catches the overlap across chunk boundaries while almost never firing
on unrelated chunks.

This is the alignment step the roadmap flagged as the hidden hard task. It's
deliberately conservative (drops table/figure markers upstream, requires real
overlap) so the retrieval metrics reflect genuine hits.
"""

from __future__ import annotations

from rag.ingestion.schema import Chunk
from rag.retrieval.bm25_retriever import tokenize

NGRAM_N = 5
MIN_SHARED_ABS = 4      # absolute shared 5-grams that count as a match
MIN_SHARED_RATIO = 0.3  # or this fraction of the evidence's 5-grams


def _ngrams(text: str, n: int = NGRAM_N) -> set[tuple[str, ...]]:
    words = tokenize(text)
    if len(words) < n:
        # Too short for n-grams; represent as a single tuple so short evidence
        # still matches via exact word-sequence containment.
        return {tuple(words)} if words else set()
    return {tuple(words[i:i + n]) for i in range(len(words) - n + 1)}


def chunk_matches_evidence(chunk_text: str, evidence: str) -> bool:
    ev = _ngrams(evidence)
    if not ev:
        return False
    ch = _ngrams(chunk_text)
    shared = len(ev & ch)
    return shared >= MIN_SHARED_ABS or shared >= MIN_SHARED_RATIO * len(ev)


def gold_chunk_ids(evidence_paragraphs: list[str], chunks: list[Chunk]) -> set[str]:
    """Chunk ids (within a single paper) that overlap any evidence paragraph."""
    relevant: set[str] = set()
    # Pre-compute chunk n-grams once since each is tested against every evidence.
    chunk_ngrams = [(c.chunk_id, _ngrams(c.text)) for c in chunks]
    for ev in evidence_paragraphs:
        ev_ng = _ngrams(ev)
        if not ev_ng:
            continue
        threshold = max(MIN_SHARED_ABS, MIN_SHARED_RATIO * len(ev_ng)) \
            if len(ev_ng) > 1 else 1
        for cid, ch_ng in chunk_ngrams:
            if len(ev_ng & ch_ng) >= threshold:
                relevant.add(cid)
    return relevant
