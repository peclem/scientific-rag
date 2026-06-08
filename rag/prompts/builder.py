"""
Prompt construction.

The plan treats prompting as a first-class, evaluated component, so the
system prompt and the way context is laid out both live here where they're
easy to version and A/B.

The core idea behind the format: every retrieved chunk is presented with the
exact citation tag we want the model to reuse, e.g. [Smith2024, p.7]. The
model is told to cite using only those tags and to never invent a reference.
That gives us something concrete to check in the citation evaluation later:
a tag in the answer either matches a tag we put in the context or it doesn't.
"""

from __future__ import annotations

from rag.ingestion.schema import Chunk

SYSTEM_PROMPT = """You are a scientific assistant that answers questions using retrieved excerpts from scientific papers.

Follow these rules:
1. Base your answer on the retrieved context whenever it is available.
2. If the context does not contain the answer, say so plainly instead of guessing.
3. Never invent references, citations, numbers, or findings.
4. Support each claim with a citation tag copied verbatim from the context. Each excerpt begins with its tag (for example [Smith2024, p.7] or [1909.01234]). Reuse that exact tag, character for character. Do not add, remove, or change a page number, and do not cite excerpts by their position.
5. Prefer concise, precise scientific language.
6. Distinguish what the evidence states from your own inference, and flag uncertainty when it exists.
7. State findings directly. Do not narrate about the source (avoid phrases like "the context mentions" or "is referenced in"); state the fact and attach its citation tag.
"""


def format_context(chunks: list[dict | Chunk]) -> str:
    """Render retrieved chunks into a citation-tagged context block.

    Accepts either Chunk objects or the hit dicts the retriever returns, so
    the same formatter works in tests and in the live pipeline.
    """
    lines: list[str] = []
    for ch in chunks:
        if isinstance(ch, Chunk):
            citation = ch.citation
            section = ch.section
            text = ch.text
        else:
            meta = ch.get("metadata", {})
            citation = meta.get("citation") or f"[{meta.get('bibkey', 'unknown')}]"
            section = meta.get("section") or None
            text = ch["text"]
        # Lead each excerpt with ONLY its citation tag on its own line (no
        # "Excerpt N" index and no inline section, both of which the model
        # would otherwise pull into the brackets). The tag is the single thing
        # we want it to copy verbatim; section context goes on a separate line.
        parts = [citation]
        if section:
            parts.append(f"From section: {section}")
        parts.append(text.strip())
        lines.append("\n".join(parts))
    return "\n\n".join(lines)


def build_messages(query: str, chunks: list[dict | Chunk]) -> list[dict]:
    """Assemble the chat messages for the LLM."""
    if chunks:
        context = format_context(chunks)
        user_content = (
            f"Retrieved context:\n\n{context}\n\n"
            f"Question: {query}\n\n"
            f"Answer using the context above and cite supporting excerpts with "
            f"their tags."
        )
    else:
        # No retrieval hits: make the model's lack of grounding explicit
        # rather than letting it fall back to parametric memory silently.
        user_content = (
            f"Question: {query}\n\n"
            f"No retrieved context was available. If you cannot answer from "
            f"established scientific knowledge with confidence, say so."
        )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_content},
    ]
