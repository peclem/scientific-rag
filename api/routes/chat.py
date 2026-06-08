"""
/chat: the main question-answering endpoint.

Per-request ablation toggles map onto a settings override so a caller can,
say, turn HyDE off for one question without restarting anything. The heavy
models stay cached; only the lightweight pipeline wrapper is rebuilt.
"""

from __future__ import annotations

from fastapi import APIRouter, Request

from api.deps import build_pipeline
from api.schemas.models import ChatRequest, ChatResponse, Citation, ContextChunk

router = APIRouter()


def _overrides(req: ChatRequest) -> dict | None:
    o: dict = {}
    if req.top_n is not None:
        o["rerank_top_n"] = req.top_n
    if req.use_hyde is not None:
        o["use_hyde"] = req.use_hyde
    if req.use_hybrid is not None:
        o["use_hybrid"] = req.use_hybrid
    if req.use_reranker is not None:
        o["use_reranker"] = req.use_reranker
    return o or None


@router.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, request: Request) -> ChatResponse:
    pipeline = build_pipeline(_overrides(req))
    result = pipeline.answer(req.question)

    contexts = []
    for c in result["contexts"]:
        meta = c.get("metadata", {})
        contexts.append(ContextChunk(
            text=c["text"],
            citation=meta.get("citation"),
            bibkey=meta.get("bibkey"),
            page=meta.get("page") or None,
            section=meta.get("section") or None,
        ))

    return ChatResponse(
        answer=result["answer"],
        citations=[Citation(**c) for c in result["citations"]],
        fabricated_citations=result["fabricated_citations"],
        contexts=contexts,
        latency_s=result.get("latency_s"),
        request_id=getattr(request.state, "request_id", None),
    )
