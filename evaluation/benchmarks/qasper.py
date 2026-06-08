"""
QASPER loader for evaluation.

QASPER is question answering over scientific papers, and crucially it ships
the full paper text plus, for each question, the evidence paragraphs an
annotator used to answer. That gives us two things for free:

  - a retrieval corpus built straight from full_text (no PDF parsing needed,
    so parsing quality doesn't confound the retrieval numbers), and
  - gold evidence to label which chunks *should* be retrieved.

We turn each paper into the same ParsedDocument the production pipeline uses,
so the chunks we evaluate are produced by the real chunker. Each question
keeps the set of evidence paragraph strings; the gold-label step matches
those against chunks.

Using the validation split keeps this data separate from anything the
fine-tuning phase will train on (which draws from the train split).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from datasets import load_dataset

from rag.ingestion.schema import ParsedDocument, TextBlock

# Evidence strings starting with these are table/figure float markers, not
# real prose paragraphs. They rarely correspond to a retrievable text chunk,
# so we drop them from the gold set to avoid unfairly penalizing retrieval.
_FLOAT_PREFIXES = ("FLOAT SELECTED", "Table TABREF", "Figure FIGREF")


@dataclass
class QasperQuestion:
    qid: str
    question: str
    evidence: list[str] = field(default_factory=list)  # evidence paragraph texts
    reference: str = ""   # a human reference answer, for ROUGE
    answerable: bool = True


@dataclass
class QasperPaper:
    paper_id: str
    title: str
    document: ParsedDocument
    questions: list[QasperQuestion]


def _build_document(paper_id: str, title: str, full_text: dict) -> ParsedDocument:
    blocks: list[TextBlock] = []
    for section, paragraphs in zip(full_text["section_name"], full_text["paragraphs"]):
        for para in paragraphs:
            if para and para.strip():
                blocks.append(TextBlock(text=para.strip(), section=section or None))
    return ParsedDocument(
        source_path=f"qasper://{paper_id}",
        parser="qasper",
        title=title,
        # Force the paper id as the bibkey so chunk_ids are unique across the
        # whole multi-paper eval corpus.
        bibkey_override=paper_id,
        blocks=blocks,
    )


def _collect_evidence(answers: dict) -> tuple[list[str], bool]:
    """Gather unique evidence paragraphs across all annotators for a question."""
    evidence: list[str] = []
    answerable = False
    for ann in answers["answer"]:
        if not ann.get("unanswerable", False):
            answerable = True
        for ev in ann.get("evidence", []) or []:
            ev = (ev or "").strip()
            if not ev or ev.startswith(_FLOAT_PREFIXES):
                continue
            if ev not in evidence:
                evidence.append(ev)
    return evidence, answerable


def _reference_answer(answers: dict) -> str:
    """Pick a human reference answer: prefer a free-form answer, then joined
    extractive spans, then yes/no. Used only for ROUGE."""
    for ann in answers["answer"]:
        if ann.get("unanswerable", False):
            continue
        if ann.get("free_form_answer"):
            return ann["free_form_answer"].strip()
        if ann.get("extractive_spans"):
            return " ".join(ann["extractive_spans"]).strip()
        if ann.get("yes_no") is not None:
            return "Yes" if ann["yes_no"] else "No"
    return ""


def load_qasper(split: str = "validation", max_papers: int | None = None,
                answerable_only: bool = True) -> list[QasperPaper]:
    ds = load_dataset("allenai/qasper", split=split)
    papers: list[QasperPaper] = []

    for ex in ds:
        document = _build_document(ex["id"], ex["title"], ex["full_text"])
        if not document.blocks:
            continue

        qas = ex["qas"]
        questions: list[QasperQuestion] = []
        for qid, question, answers in zip(
            qas["question_id"], qas["question"], qas["answers"]
        ):
            evidence, answerable = _collect_evidence(answers)
            if answerable_only and not evidence:
                continue
            questions.append(QasperQuestion(
                qid=qid, question=question, evidence=evidence,
                reference=_reference_answer(answers), answerable=answerable
            ))

        if not questions:
            continue
        papers.append(QasperPaper(
            paper_id=ex["id"], title=ex["title"], document=document, questions=questions
        ))
        if max_papers and len(papers) >= max_papers:
            break

    return papers
