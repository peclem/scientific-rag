"""
Build a retrieval evaluation corpus from QASPER papers.

All papers are chunked and embedded into one shared collection, and every
question retrieves against that whole pool, not just its own paper. That's
the realistic setting: the system doesn't get told which paper the answer is
in, so good retrieval has to pick the right chunks out of many papers. Gold
labels stay scoped to the question's own paper (its evidence chunks).

Questions whose evidence didn't align to any chunk are dropped here, so the
metrics are computed only over questions that actually have a gold target.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from loguru import logger

from config import settings as default_settings
from evaluation.benchmarks.qasper import QasperPaper
from evaluation.gold_labels import gold_chunk_ids
from rag.ingestion.chunker import chunk_document
from rag.ingestion.schema import Chunk
from rag.retrieval.bm25_retriever import BM25Retriever
from rag.retrieval.vector_store import VectorStore


@dataclass
class EvalQuestion:
    qid: str
    question: str
    paper_id: str
    gold_ids: set[str]


@dataclass
class EvalCorpus:
    chunks_by_id: dict[str, Chunk]
    questions: list[EvalQuestion]
    vector_store: VectorStore
    bm25: BM25Retriever
    n_chunks: int = field(default=0)


def build_eval_corpus(papers: list[QasperPaper], settings=default_settings,
                      collection_name: str = "qasper_eval",
                      embed: bool = True) -> EvalCorpus:
    all_chunks: list[Chunk] = []
    questions: list[EvalQuestion] = []

    for p in papers:
        chunks = chunk_document(p.document, settings)
        all_chunks.extend(chunks)
        for q in p.questions:
            gold = gold_chunk_ids(q.evidence, chunks)
            if gold:
                questions.append(EvalQuestion(q.qid, q.question, p.paper_id, gold))

    chunks_by_id = {c.chunk_id: c for c in all_chunks}
    logger.info(
        f"Eval corpus: {len(papers)} papers, {len(all_chunks)} chunks, "
        f"{len(questions)} questions with gold labels"
    )

    vector_store = VectorStore(settings=settings, collection_name=collection_name)
    vector_store.reset()
    if embed:
        # add_chunks embeds internally; sentence-transformers batches the encode.
        vector_store.add_chunks(all_chunks)

    bm25 = BM25Retriever.from_chunks(all_chunks)

    return EvalCorpus(
        chunks_by_id=chunks_by_id,
        questions=questions,
        vector_store=vector_store,
        bm25=bm25,
        n_chunks=len(all_chunks),
    )
