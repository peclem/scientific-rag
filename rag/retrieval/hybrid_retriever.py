"""
Hybrid retriever: the full retrieval pipeline in one place.

Flow (each stage is switchable from config so we can ablate it):

    query (+ optional HyDE text)
        -> vector search        (semantic recall)
        -> BM25 search          (lexical recall)
        -> RRF fusion           (combine the ranked lists)
        -> cross-encoder rerank (precision)
        -> top-N chunks with citations

HyDE is folded in as just another ranked list rather than a special case:
if a hypothetical answer is provided, we run a second vector search with it
and let RRF merge it with the rest. That keeps this class unaware of how the
HyDE text was produced (it's the orchestrator's job to call the LLM), and it
means the HyDE ablation is a one-line change at the call site.

The retriever scores candidates for recall first, hands a wide candidate set
(rerank_pool) to the reranker, and returns the precise top-N.
"""

from __future__ import annotations

from loguru import logger

from config import settings as default_settings
from rag.retrieval.bm25_retriever import BM25Retriever
from rag.retrieval.fusion import reciprocal_rank_fusion
from rag.retrieval.vector_store import VectorStore


class HybridRetriever:
    def __init__(self, vector_store: VectorStore, bm25: BM25Retriever | None = None,
                 reranker=None, settings=default_settings):
        self.vector_store = vector_store
        self.bm25 = bm25
        self.reranker = reranker
        self.settings = settings

    def retrieve(self, query: str, hyde_text: str | None = None,
                 top_n: int | None = None) -> list[dict]:
        # top_n overrides the configured final count. Evaluation uses it to
        # get a longer ranked list (e.g. 20) so Recall@10 is computable.
        s = self.settings
        final_n = top_n or s.rerank_top_n
        ranked_lists = [self.vector_store.query(query, s.vector_top_k)]

        # HyDE: a second semantic search using the hypothetical answer.
        if s.use_hyde and hyde_text:
            ranked_lists.append(self.vector_store.query(hyde_text, s.vector_top_k))

        # Lexical search on the real query (not the HyDE text, which is
        # generated and would add lexical noise).
        if s.use_hybrid and self.bm25 is not None:
            ranked_lists.append(self.bm25.query(query, s.bm25_top_k))

        fused = reciprocal_rank_fusion(ranked_lists, k=s.rrf_k)

        if s.use_reranker and self.reranker is not None:
            # Rerank a bounded slice of the fused list, then keep top_n. The
            # pool is capped (rerank_pool_size) because cross-encoder scoring
            # is the slowest stage, especially on CPU.
            pool = fused[: max(s.rerank_pool_size, final_n)]
            final = self.reranker.rerank(query, pool, top_n=final_n)
        else:
            final = fused[:final_n]

        logger.debug(
            f"Retrieve: lists={len(ranked_lists)} fused={len(fused)} "
            f"-> returned {len(final)} (hyde={'on' if hyde_text else 'off'}, "
            f"rerank={'on' if (s.use_reranker and self.reranker) else 'off'})"
        )
        return final
