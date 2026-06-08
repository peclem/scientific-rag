"""
ChromaDB vector store wrapper.

Two things worth calling out:

1. The collection is created with hnsw:space=cosine. Chroma defaults to L2,
   so without this the "cosine similarity" claim in the design would be a
   lie. Our embeddings are normalized, so cosine and dot product agree, but
   we set it explicitly anyway.

2. We pass our own embeddings instead of letting Chroma call a built-in
   embedding function. That keeps the embedding model, device, and
   normalization under our control (and consistent with query-time).

Chroma returns cosine *distance* (0 = identical, 2 = opposite). We convert
to a similarity score (1 - distance) so higher always means more relevant,
which is what the fusion and reranking code expects.
"""

from __future__ import annotations

import logging
import os

# chromadb 0.6.x has a broken interaction with newer posthog: its telemetry
# client calls posthog.capture() with the old signature and logs a noisy
# "Failed to send telemetry event" error on every operation. The disable flag
# (env var and Settings) is ignored in this version, so we just mute the
# telemetry logger. We also set the env var in case a future bump respects it.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

import chromadb  # noqa: E402
from loguru import logger  # noqa: E402

from config import settings as default_settings  # noqa: E402
from rag.ingestion.schema import Chunk  # noqa: E402
from rag.retrieval.embedder import Embedder  # noqa: E402


class VectorStore:
    def __init__(self, embedder: Embedder | None = None, settings=default_settings,
                 collection_name: str = "scientific_papers"):
        self.settings = settings
        self.embedder = embedder or Embedder(settings)
        # Disable Chroma's anonymous telemetry: it's noisy (and currently
        # throws on a posthog version mismatch), and we don't want background
        # network calls from an offline research tool.
        self.client = chromadb.PersistentClient(
            path=str(settings.chroma_dir),
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": settings.distance_metric},
        )

    def add_chunks(self, chunks: list[Chunk]) -> None:
        if not chunks:
            return
        embeddings = self.embedder.embed_documents([c.text for c in chunks])
        self.collection.add(
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=embeddings,
            metadatas=[c.metadata() for c in chunks],
        )
        logger.info(f"Added {len(chunks)} chunks (collection now {self.count()})")

    def query(self, query_text: str, k: int | None = None) -> list[dict]:
        """Return the top-k chunks as dicts with a similarity score."""
        k = k or self.settings.vector_top_k
        q_emb = self.embedder.embed_query(query_text)
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
        hits = []
        for doc, meta, dist in zip(
            res["documents"][0], res["metadatas"][0], res["distances"][0]
        ):
            hits.append({
                "chunk_id": meta.get("chunk_id"),
                "text": doc,
                "metadata": meta,
                "score": 1.0 - dist,   # cosine distance -> similarity
            })
        return hits

    def count(self) -> int:
        return self.collection.count()

    def reset(self) -> None:
        """Drop and recreate the collection. Handy when re-ingesting."""
        name = self.collection.name
        self.client.delete_collection(name)
        self.collection = self.client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": self.settings.distance_metric},
        )
        logger.info(f"Reset collection '{name}'")
