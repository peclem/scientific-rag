"""
Build supervised fine-tuning data.

The honest premise (see the roadmap): QLoRA here mainly teaches *format and
habits*, not new knowledge. The system already gets its facts from retrieval.
So every SFT target models exactly the behavior we want at inference: given
retrieved context and a question, answer concisely and attach a citation tag
copied from the context.

To avoid distribution shift, each example is built with the SAME prompt shape
the live pipeline uses (build_messages), in TRL's prompt/completion format so
loss falls only on the answer. Three sources contribute, mixed roughly
60/20/20 per the roadmap:

  - QASPER   (train split): scientific QA with evidence + reference answers
  - PubMedQA (pqa_labeled): biomedical QA with a long-form reference answer
  - SciFact  (claims+corpus): claim verification (supported / refuted)

QASPER uses its TRAIN split, kept separate from the validation split used for
evaluation, so we never train on what we measure on. PubMedQA and SciFact are
not part of the eval set, so they pose no leakage with the QASPER eval.
"""

from __future__ import annotations

from datasets import Dataset, concatenate_datasets
from loguru import logger

from evaluation.benchmarks.qasper import load_qasper
from rag.prompts.builder import build_messages

MIN_REFERENCE_WORDS = 5   # skip terse yes/no or single-token answers


def _make_hits(citation_tag: str, texts: list[str], section: str = "") -> list[dict]:
    """Wrap context texts as retriever-style hits so build_messages formats
    them with citation tags, mirroring inference."""
    return [
        {"text": t, "metadata": {"citation": citation_tag, "bibkey": citation_tag.strip("[]"),
                                 "section": section}}
        for t in texts if t and t.strip()
    ]


def _example(question: str, hits: list[dict], answer: str) -> dict:
    prompt = build_messages(question, hits)
    completion = [{"role": "assistant", "content": answer}]
    return {"prompt": prompt, "completion": completion}


# --------------------------------------------------------------------------- #
# QASPER
# --------------------------------------------------------------------------- #
def build_qasper_sft(split: str = "train", max_examples: int | None = None) -> Dataset:
    papers = load_qasper(split=split)
    records: list[dict] = []
    for p in papers:
        tag = f"[{p.paper_id}]"
        for q in p.questions:
            if len(q.reference.split()) < MIN_REFERENCE_WORDS or not q.evidence:
                continue
            ref = q.reference.strip().rstrip(".")
            answer = ref + "." if tag in ref else f"{ref} {tag}."
            records.append(_example(q.question, _make_hits(tag, q.evidence), answer))
            if max_examples and len(records) >= max_examples:
                return Dataset.from_list(records)
    return Dataset.from_list(records)


# --------------------------------------------------------------------------- #
# PubMedQA
# --------------------------------------------------------------------------- #
def build_pubmedqa_sft(max_examples: int | None = None) -> Dataset:
    from datasets import load_dataset

    ds = load_dataset("qiaojin/PubMedQA", "pqa_labeled", split="train")
    records: list[dict] = []
    for ex in ds:
        long_answer = (ex.get("long_answer") or "").strip()
        if len(long_answer.split()) < MIN_REFERENCE_WORDS:
            continue
        tag = f"[PMID:{ex['pubid']}]"
        contexts = ex["context"]["contexts"]
        hits = _make_hits(tag, contexts)
        if not hits:
            continue
        ans = long_answer.rstrip(".")
        answer = f"{ans} {tag}."
        records.append(_example(ex["question"], hits, answer))
        if max_examples and len(records) >= max_examples:
            break
    return Dataset.from_list(records)


# --------------------------------------------------------------------------- #
# SciFact
# --------------------------------------------------------------------------- #
def build_scifact_sft(max_examples: int | None = None) -> Dataset:
    from datasets import load_dataset

    corpus = load_dataset("allenai/scifact", "corpus", split="train", trust_remote_code=True)
    abstracts = {row["doc_id"]: row["abstract"] for row in corpus}

    claims = load_dataset("allenai/scifact", "claims", split="train", trust_remote_code=True)
    records: list[dict] = []
    for ex in claims:
        doc_id = ex.get("evidence_doc_id")
        label = ex.get("evidence_label")
        if not doc_id or label not in ("SUPPORT", "CONTRADICT"):
            continue
        doc_id_int = int(doc_id)
        sentences = abstracts.get(doc_id_int)
        if not sentences:
            continue
        # Use the cited evidence sentences as context when available, else the
        # whole abstract.
        idxs = ex.get("evidence_sentences") or []
        context = [sentences[i] for i in idxs if i < len(sentences)] or sentences
        tag = f"[{doc_id_int}]"
        verdict = "supported" if label == "SUPPORT" else "refuted"
        question = f"Is the following claim supported or refuted by the evidence? Claim: {ex['claim']}"
        answer = f"The claim is {verdict} by the evidence. {tag}."
        records.append(_example(question, _make_hits(tag, context), answer))
        if max_examples and len(records) >= max_examples:
            break
    return Dataset.from_list(records)


# --------------------------------------------------------------------------- #
# Mixed
# --------------------------------------------------------------------------- #
def build_mixed_sft(qasper_n: int = 1200, pubmedqa_n: int = 400,
                    scifact_n: int = 400, seed: int = 42) -> Dataset:
    """Concatenate the three sources (~60/20/20 by default) and shuffle."""
    qasper = build_qasper_sft("train", max_examples=qasper_n)
    pubmedqa = build_pubmedqa_sft(max_examples=pubmedqa_n)
    scifact = build_scifact_sft(max_examples=scifact_n)
    logger.info(
        f"SFT mix: QASPER={len(qasper)}, PubMedQA={len(pubmedqa)}, "
        f"SciFact={len(scifact)}, total={len(qasper)+len(pubmedqa)+len(scifact)}"
    )
    return concatenate_datasets([qasper, pubmedqa, scifact]).shuffle(seed=seed)
