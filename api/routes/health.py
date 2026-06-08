"""Health check. Cheap, no model loading, safe to poll."""

from __future__ import annotations

from fastapi import APIRouter

from api.deps import get_vector_store
from api.schemas.models import HealthResponse
from config import settings
from rag.ingestion.grobid_client import is_grobid_alive

router = APIRouter()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        collection_size=get_vector_store().count(),
        grobid_available=is_grobid_alive(settings.grobid_url),
        llm_model=settings.llm_model,
        embedding_model=settings.embedding_model,
    )
