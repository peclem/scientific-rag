"""
Tests for the token-based chunker. Uses the real bge-m3 tokenizer (cached
after first download), since the whole point is that sizes are measured in
the embedder's tokens.
"""

from rag.ingestion.chunker import chunk_document
from rag.ingestion.schema import ParsedDocument, TextBlock


def _doc(blocks):
    return ParsedDocument(
        source_path="/tmp/Smith_2024.pdf",
        parser="test",
        title="A Paper",
        authors=["Jane Smith"],
        year=2024,
        blocks=blocks,
    )


def test_chunks_respect_token_cap():
    long_text = "attention mechanism " * 500  # well over one chunk
    doc = _doc([TextBlock(text=long_text, section="Methods", page=3)])
    chunks = chunk_document(doc)
    assert len(chunks) > 1
    assert all(c.n_tokens <= 512 for c in chunks)


def test_metadata_propagates():
    doc = _doc([TextBlock(text="short paragraph about transformers", section="Intro", page=2)])
    chunks = chunk_document(doc)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.bibkey == "Smith2024"
    assert c.section == "Intro"
    assert c.page == 2
    assert c.parser == "test"
    assert c.citation == "[Smith2024, p.2]"


def test_sections_do_not_merge():
    doc = _doc([
        TextBlock(text="intro text here", section="Intro", page=1),
        TextBlock(text="methods text here", section="Methods", page=2),
    ])
    chunks = chunk_document(doc)
    sections = {c.section for c in chunks}
    assert sections == {"Intro", "Methods"}
