"""
Local-model LLM judge.

A rubric-based judge that scores an answer on four axes (1-5): faithfulness
to the context, citation quality, completeness, and format adherence. It runs
on a local model (Qwen3 by default), so the whole project stays offline with
no external API.

Caveat worth stating plainly: when the judge model is the same model that
produced the answer, the evaluation is somewhat circular and a model may
favor its own style. We keep the rubric concrete and context-grounded to
reduce that, and the judge is treated as one signal alongside the objective
metrics (retrieval, citation precision, NLI faithfulness), not the sole word.
The folder name (gpt_judge) is historical; no GPT or external API is used.
"""

from __future__ import annotations

import json
import re
from statistics import mean

from loguru import logger

JUDGE_SYSTEM = """You are a strict evaluator of answers produced by a scientific question-answering system.

You are given a question, the retrieved context excerpts the system was allowed to use, and the system's answer. Score the answer on each axis from 1 (poor) to 5 (excellent):

- faithfulness: every claim is supported by the provided context, nothing invented.
- citation_quality: claims are backed by citation tags that point to relevant context.
- completeness: the answer actually addresses the question.
- format: concise, clear, scientific phrasing.

Respond with ONLY a JSON object, no other text:
{"faithfulness": <1-5>, "citation_quality": <1-5>, "completeness": <1-5>, "format": <1-5>, "rationale": "<one sentence>"}
"""

_AXES = ("faithfulness", "citation_quality", "completeness", "format")
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)


def _build_user(question: str, answer: str, contexts: list[str]) -> str:
    context_block = "\n\n".join(f"[{i+1}] {c}" for i, c in enumerate(contexts))
    return (
        f"Question:\n{question}\n\n"
        f"Retrieved context:\n{context_block}\n\n"
        f"System answer:\n{answer}\n\n"
        f"Score the answer as instructed."
    )


def _parse(raw: str) -> dict | None:
    m = _JSON_RE.search(raw)
    if not m:
        return None
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(a in data for a in _AXES):
        return None
    # Clamp to the 1-5 range in case the model drifts.
    for a in _AXES:
        try:
            data[a] = max(1, min(5, int(data[a])))
        except (ValueError, TypeError):
            return None
    return data


def judge_answer(llm, question: str, answer: str, contexts: list[str]) -> dict | None:
    messages = [
        {"role": "system", "content": JUDGE_SYSTEM},
        {"role": "user", "content": _build_user(question, answer, contexts)},
    ]
    raw = llm.generate(messages, max_new_tokens=256, temperature=0.0)
    parsed = _parse(raw)
    if parsed is None:
        logger.warning(f"Judge returned unparseable output: {raw[:120]}")
    return parsed


def aggregate_judgments(judgments: list[dict]) -> dict[str, float]:
    valid = [j for j in judgments if j]
    if not valid:
        return {}
    out = {axis: mean(j[axis] for j in valid) for axis in _AXES}
    out["overall"] = mean(out[a] for a in _AXES)
    out["n_judged"] = len(valid)
    out["parse_rate"] = len(valid) / len(judgments)
    return out
