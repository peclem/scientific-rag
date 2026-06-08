"""
Dense embedding model (bge-m3).

We use bge-m3 through sentence-transformers for plain dense vectors. bge-m3
can also produce sparse and ColBERT-style vectors via FlagEmbedding, but the
plan keeps lexical retrieval as a separate BM25 component, so we only need
the dense side here.

Embeddings are L2-normalized so that cosine similarity (what Chroma is told
to use) behaves sensibly. The model loads lazily on first use, and its
device comes from config so we can move it to CPU when the LLM needs the GPU
(low_vram_mode).
"""

from __future__ import annotations

from loguru import logger

from config import settings as default_settings


class Embedder:
    def __init__(self, settings=default_settings):
        self.model_name = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        # In low-VRAM mode the retrieval models give up the GPU to the LLM.
        self.device = "cpu" if settings.low_vram_mode else settings.embedding_device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            logger.info(f"Loading embedder {self.model_name} on {self.device}")
            self._model = SentenceTransformer(self.model_name, device=self.device)
        return self._model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        vecs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(texts) > 64,
        )
        return vecs.tolist()

    def embed_query(self, text: str) -> list[float]:
        # bge-m3 needs no special query prefix, so queries and documents are
        # embedded the same way.
        vec = self.model.encode([text], normalize_embeddings=True)[0]
        return vec.tolist()
