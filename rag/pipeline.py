"""
The RAG pipeline, wired with LangChain.

This is the orchestration layer the plan calls for. The individual stages
(HyDE, retrieval, prompt building, generation, citation resolution) are
plain functions and classes; LangChain's LCEL composes them into one
runnable chain. Using RunnablePassthrough.assign threads a small state dict
through the steps, so each stage sees what the previous ones produced.

Keeping orchestration in LangChain (rather than a hand-rolled function) is
deliberate: it's one of the skills this project is meant to exercise, and it
makes the flow easy to trace and to swap stages in and out for ablations.
"""

from __future__ import annotations

import time

from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from loguru import logger

from config import settings as default_settings
from rag.generation.llm import LLM, get_llm
from rag.prompts.builder import build_messages
from rag.prompts.citations import resolve_citations
from rag.retrieval.hybrid_retriever import HybridRetriever
from rag.retrieval.hyde import generate_hyde


class RAGPipeline:
    def __init__(self, retriever: HybridRetriever, llm: LLM | None = None,
                 settings=default_settings):
        self.retriever = retriever
        self.llm = llm or get_llm(settings)
        self.settings = settings
        self.chain = self._build_chain()

    def _hyde_step(self, state: dict) -> str | None:
        if not self.settings.use_hyde:
            return None
        return generate_hyde(state["question"], self.llm)

    def _format_response(self, state: dict) -> dict:
        citations = resolve_citations(state["answer"], state["hits"])
        if citations["fabricated"]:
            # Not necessarily wrong (the model may cite background knowledge),
            # but it's the signal the citation eval cares about, so log it.
            logger.warning(f"Fabricated citation tags: {citations['fabricated']}")
        return {
            "question": state["question"],
            "answer": state["answer"].strip(),
            "citations": citations["cited"],
            "fabricated_citations": citations["fabricated"],
            "contexts": [
                {"text": h["text"], "metadata": h.get("metadata", {})}
                for h in state["hits"]
            ],
        }

    def _build_chain(self):
        return (
            RunnablePassthrough.assign(hyde=RunnableLambda(self._hyde_step))
            | RunnablePassthrough.assign(
                hits=RunnableLambda(
                    lambda s: self.retriever.retrieve(s["question"], s["hyde"])
                )
            )
            | RunnablePassthrough.assign(
                messages=RunnableLambda(
                    lambda s: build_messages(s["question"], s["hits"])
                )
            )
            | RunnablePassthrough.assign(
                answer=RunnableLambda(lambda s: self.llm.generate(s["messages"]))
            )
            | RunnableLambda(self._format_response)
        )

    def answer(self, question: str) -> dict:
        t0 = time.time()
        result = self.chain.invoke({"question": question})
        result["latency_s"] = round(time.time() - t0, 2)
        logger.info(
            f"Answered in {result['latency_s']}s, "
            f"{len(result['citations'])} citations, "
            f"{len(result['contexts'])} contexts"
        )
        return result
