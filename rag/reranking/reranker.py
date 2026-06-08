"""
Cross-encoder reranker (bge-reranker-v2-m3).

The vector and BM25 stages are tuned for recall: cast a wide net (top 30ish)
so the right chunk is somewhere in the pile. The reranker is where precision
comes from. Unlike the bi-encoder embedder, a cross-encoder reads the query
and a candidate together and scores their actual relevance, which is much
more accurate but too slow to run over the whole corpus. Running it on only
the fused candidates is the sweet spot.

It loads lazily and respects low_vram_mode, since when the LLM is serving we
can't also keep a cross-encoder on the GPU.
"""

from __future__ import annotations

from loguru import logger

from config import settings as default_settings


class Reranker:
    def __init__(self, settings=default_settings):
        self.model_name = settings.reranker_model
        self.device = "cpu" if settings.low_vram_mode else settings.reranker_device
        self._model = None

    @property
    def model(self):
        if self._model is None:
            from FlagEmbedding import FlagReranker

            logger.info(f"Loading reranker {self.model_name} on {self.device}")
            use_fp16 = self.device != "cpu"
            self._model = FlagReranker(
                self.model_name, use_fp16=use_fp16, devices=self.device
            )
        return self._model

    def rerank(self, query: str, hits: list[dict], top_n: int | None = None) -> list[dict]:
        """Score each candidate against the query and keep the best top_n.
        Adds a 'rerank_score' to each returned hit."""
        top_n = top_n or default_settings.rerank_top_n
        if not hits:
            return []
        pairs = [[query, h["text"]] for h in hits]
        scores = self.model.compute_score(
            pairs,
            normalize=True,
            max_length=default_settings.rerank_max_length,
        )
        # compute_score returns a float for a single pair, a list otherwise.
        if not isinstance(scores, list):
            scores = [scores]
        for h, s in zip(hits, scores):
            h["rerank_score"] = float(s)
        ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
        return ranked[:top_n]
