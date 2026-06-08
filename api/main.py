"""
FastAPI application.

Wires the routes together and adds a small logging middleware: every request
gets an id and its latency is logged via Loguru. That's the level of
observability the plan wants, enough to debug and to track experiments,
without pulling in production monitoring stacks.

Run it with:
    PYTHONPATH=. .venv/bin/uvicorn api.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import sys
import time
import uuid

from fastapi import FastAPI, Request
from loguru import logger

from api.routes import chat, health, ingest
from config import settings


def _configure_logging() -> None:
    settings.ensure_dirs()
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    # Rotating file log for after-the-fact debugging and experiment tracking.
    logger.add(
        settings.log_dir / "api_{time:YYYY-MM-DD}.log",
        rotation="1 day",
        retention="14 days",
        level="DEBUG",
        enqueue=True,
    )


_configure_logging()

app = FastAPI(
    title="Scientific RAG",
    description="Retrieval-augmented QA over scientific papers with citations.",
    version="0.1.0",
)


@app.middleware("http")
async def log_requests(request: Request, call_next):
    request_id = uuid.uuid4().hex[:8]
    request.state.request_id = request_id
    t0 = time.time()
    with logger.contextualize(request_id=request_id):
        logger.info(f"{request.method} {request.url.path}")
        response = await call_next(request)
        dt = (time.time() - t0) * 1000
        logger.info(f"{request.method} {request.url.path} "
                    f"-> {response.status_code} in {dt:.0f}ms")
    response.headers["X-Request-ID"] = request_id
    return response


app.include_router(health.router, tags=["health"])
app.include_router(chat.router, tags=["chat"])
app.include_router(ingest.router, tags=["ingest"])


@app.get("/", tags=["health"])
def root() -> dict:
    return {"service": "scientific-rag", "docs": "/docs"}
