"""
Faithfulness via claim-level NLI.

This is the hallucination metric, and the project's highest-priority quality
signal: of the things the answer asserts, how many are actually supported by
the retrieved context? We:

  1. split the answer into claims (sentences, with citation tags stripped),
  2. for each claim, run NLI against every retrieved chunk with the chunk as
     premise and the claim as hypothesis,
  3. count the claim as supported if any chunk entails it.

faithfulness = supported / total claims; hallucination_rate = 1 - that.
Checking against each chunk separately (rather than one concatenated blob)
keeps every premise within the NLI model's length limit and means a claim
only needs one piece of real evidence to count as grounded.

The NLI model (a DeBERTa cross-encoder) is small and runs fine on GPU or CPU.

A measured caveat from validation: this per-chunk NLI is a *conservative*
signal. It under-credits true claims that combine facts from several chunks
(no single chunk fully entails them) or that paraphrase heavily, so the
reported faithfulness reads lower than manual inspection or the LLM judge
suggests. We therefore treat it as a strict lower bound and use the LLM
judge's faithfulness axis as the headline faithfulness number. An LLM-based
per-claim verifier (RAGAS-style) would be a more accurate drop-in if needed.
"""

from __future__ import annotations

import re
from statistics import mean

from loguru import logger

from config import settings as default_settings

# Strip bracketed citation tags before splitting into claims; the tag itself
# isn't a factual assertion to verify.
_TAG_RE = re.compile(r"\[[^\[\]]+\]")
_SENT_RE = re.compile(r"(?<=[.!?])\s+")
_MARKDOWN_RE = re.compile(r"[*_`#]+")              # bold/italic/code/heading marks
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s*")  # leading bullet / number

NLI_MODEL = "cross-encoder/nli-deberta-v3-base"
MIN_CLAIM_WORDS = 4
# A claim counts as supported if some chunk entails it with at least this
# probability. NLI models are strict: requiring argmax==entailment marks
# paraphrased/summarized-but-true claims as unsupported (they land on
# "neutral"). A probability threshold is much better calibrated.
ENTAIL_THRESHOLD = 0.5


def split_claims(answer: str) -> list[str]:
    text = _TAG_RE.sub("", answer)
    text = _MARKDOWN_RE.sub("", text)
    claims = []
    # Split on newlines first (list items) then on sentence boundaries, so
    # bulleted answers don't collapse into one giant "sentence".
    for line in text.splitlines():
        line = _LIST_PREFIX_RE.sub("", line)
        for sent in _SENT_RE.split(line):
            sent = sent.strip()
            if len(sent.split()) >= MIN_CLAIM_WORDS:
                claims.append(sent)
    return claims


class FaithfulnessScorer:
    def __init__(self, settings=default_settings, device: str | None = None):
        self.device = device or ("cpu" if settings.low_vram_mode else "cuda")
        self._model = None
        self._entail_idx = 1  # corrected from config on load

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder

            logger.info(f"Loading NLI model {NLI_MODEL} on {self.device}")
            self._model = CrossEncoder(NLI_MODEL, device=self.device)
            id2label = getattr(self._model.model.config, "id2label", {})
            for idx, label in id2label.items():
                if str(label).lower().startswith("entail"):
                    self._entail_idx = int(idx)
        return self._model

    def score_answer(self, answer: str, context_texts: list[str]) -> dict:
        claims = split_claims(answer)
        if not claims or not context_texts:
            return {"faithfulness": 0.0, "hallucination_rate": 1.0, "n_claims": len(claims)}

        supported = 0
        for claim in claims:
            pairs = [(ctx, claim) for ctx in context_texts]
            # apply_softmax gives per-class probabilities; we want the highest
            # entailment probability any single chunk gives this claim.
            probs = self.model.predict(pairs, apply_softmax=True)  # (n_ctx, 3)
            max_entail = max(float(row[self._entail_idx]) for row in probs)
            if max_entail >= ENTAIL_THRESHOLD:
                supported += 1

        faithfulness = supported / len(claims)
        return {
            "faithfulness": faithfulness,
            "hallucination_rate": 1.0 - faithfulness,
            "n_claims": len(claims),
            "supported_claims": supported,
        }

    def aggregate(self, per_answer: list[dict]) -> dict[str, float]:
        scored = [a for a in per_answer if a["n_claims"] > 0]
        if not scored:
            return {"faithfulness": 0.0, "hallucination_rate": 1.0}
        return {
            "faithfulness": mean(a["faithfulness"] for a in scored),
            "hallucination_rate": mean(a["hallucination_rate"] for a in scored),
            "mean_claims_per_answer": mean(a["n_claims"] for a in scored),
        }
