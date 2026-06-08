"""
Citation extraction and checking.

After generation we pull the [bibkey, p.N] tags out of the answer and match
them against the tags we actually put in the context. This serves two
purposes: it builds the citations list the API returns, and it gives the
citation evaluation a concrete signal, a cited tag that doesn't correspond
to any provided excerpt is a fabricated reference, which rule 3 forbids.
"""

from __future__ import annotations

import re

# Matches bracketed tags like [Smith2024, p.7] or [Smith2024].
_CITATION_RE = re.compile(r"\[[^\[\]]+\]")


def extract_tags(answer: str) -> list[str]:
    """Unique citation-looking tags in the answer, in order of appearance."""
    seen: list[str] = []
    for m in _CITATION_RE.findall(answer):
        tag = m.strip()
        if tag not in seen:
            seen.append(tag)
    return seen


def resolve_citations(answer: str, hits: list[dict]) -> dict:
    """
    Cross-reference the answer's tags against the retrieved chunks.

    Returns:
      cited      - retrieved chunks whose tag appears in the answer
      fabricated - tags in the answer that match no retrieved chunk
    """
    available = {}
    for h in hits:
        meta = h.get("metadata", {})
        tag = meta.get("citation")
        if tag:
            available[tag] = {
                "citation": tag,
                "bibkey": meta.get("bibkey"),
                "title": meta.get("title") or None,
                "section": meta.get("section") or None,
                "page": meta.get("page") or None,
                "source_path": meta.get("source_path"),
                "chunk_id": meta.get("chunk_id"),
            }

    tags = extract_tags(answer)
    cited = [available[t] for t in tags if t in available]
    fabricated = [t for t in tags if t not in available]
    return {"cited": cited, "fabricated": fabricated}
