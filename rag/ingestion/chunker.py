"""
Token-based chunking with metadata propagation.

Two deliberate choices here:

1. Chunks are measured in TOKENS using the embedding model's own tokenizer,
   not characters. That way chunk_size means the same thing the embedder
   sees, and the chunk-size sweep in the eval phase compares like with like.

2. Chunking happens within a section, never across section boundaries. The
   roadmap's reasoning is that scientific context is paragraph and section
   level, so we'd rather not glue the end of "Methods" onto the start of
   "Results". Each chunk inherits the section heading and the page of its
   first token, which is what the citation needs.

The sliding window keeps an overlap so a sentence split across a chunk
boundary still has a chance of landing whole in a neighbouring chunk.
"""

from __future__ import annotations

from functools import lru_cache

from loguru import logger
from transformers import AutoTokenizer

from config import settings as default_settings
from rag.ingestion.schema import Chunk, ParsedDocument, TextBlock


@lru_cache(maxsize=4)
def _get_tokenizer(model_name: str):
    # Loading just the tokenizer is cheap (no model weights). Cached so we
    # don't re-read it for every document.
    return AutoTokenizer.from_pretrained(model_name)


def _group_by_section(blocks: list[TextBlock]) -> list[tuple[str | None, list[TextBlock]]]:
    """Collapse consecutive blocks that share a section heading into groups."""
    groups: list[tuple[str | None, list[TextBlock]]] = []
    for b in blocks:
        if groups and groups[-1][0] == b.section:
            groups[-1][1].append(b)
        else:
            groups.append((b.section, [b]))
    return groups


def chunk_document(doc: ParsedDocument, settings=default_settings) -> list[Chunk]:
    tokenizer = _get_tokenizer(settings.embedding_model)
    size = settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens
    step = max(1, size - overlap)

    chunks: list[Chunk] = []
    counter = 0

    for section, blocks in _group_by_section(doc.non_empty_blocks()):
        # Build one token stream for the section, remembering which page each
        # token came from so a chunk can report the right page.
        token_ids: list[int] = []
        token_pages: list[int | None] = []
        for b in blocks:
            ids = tokenizer.encode(b.text, add_special_tokens=False)
            token_ids.extend(ids)
            token_pages.extend([b.page] * len(ids))

        if not token_ids:
            continue

        for start in range(0, len(token_ids), step):
            window = token_ids[start:start + size]
            if not window:
                break
            text = tokenizer.decode(window, skip_special_tokens=True).strip()
            if not text:
                continue
            # Page = first known page within the window.
            page = next((p for p in token_pages[start:start + size] if p), None)

            chunks.append(
                Chunk(
                    chunk_id=f"{doc.bibkey}::{counter}",
                    text=text,
                    bibkey=doc.bibkey,
                    source_path=doc.source_path,
                    title=doc.title,
                    section=section,
                    page=page,
                    n_tokens=len(window),
                    parser=doc.parser,
                )
            )
            counter += 1

            if start + size >= len(token_ids):
                break  # window already reached the end of the section

    logger.info(
        f"Chunked '{doc.bibkey}' into {len(chunks)} chunks "
        f"(size={size}, overlap={overlap} tokens)"
    )
    return chunks
