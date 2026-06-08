"""
Pydantic request/response models.

These are the validation layer the plan asks for: reject empty or oversized
questions, keep retrieval/generation parameters in sane ranges, and give the
API a stable, documented contract. FastAPI turns these into the OpenAPI docs
for free.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from config import settings


class ChatRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=settings.max_question_chars)
    stream: bool = False  # accepted for forward-compat; serving is non-streaming

    # Optional per-request overrides of the retrieval knobs. None means "use
    # the configured default", which keeps benchmarking reproducible while
    # still allowing quick manual experiments.
    top_n: int | None = Field(default=None, ge=1, le=20)
    use_hyde: bool | None = None
    use_hybrid: bool | None = None
    use_reranker: bool | None = None


class Citation(BaseModel):
    citation: str
    bibkey: str | None = None
    title: str | None = None
    section: str | None = None
    page: int | None = None
    source_path: str | None = None
    chunk_id: str | None = None


class ContextChunk(BaseModel):
    text: str
    citation: str | None = None
    bibkey: str | None = None
    page: int | None = None
    section: str | None = None


class ChatResponse(BaseModel):
    answer: str
    citations: list[Citation]
    fabricated_citations: list[str] = []
    contexts: list[ContextChunk] = []
    latency_s: float | None = None
    request_id: str | None = None


class IngestResponse(BaseModel):
    filename: str
    parser: str
    chunks_added: int
    collection_size: int


class HealthResponse(BaseModel):
    status: str
    collection_size: int
    grobid_available: bool
    llm_model: str
    embedding_model: str
