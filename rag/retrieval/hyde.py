"""
HyDE (Hypothetical Document Embeddings) query expansion.

The idea: instead of embedding the user's question (which is phrased like a
question), ask the LLM to write a short passage that *answers* it, in the
style of a paper, and embed that. A hypothetical answer tends to sit closer
in embedding space to the real passages than the question does, which can
lift recall when the user's wording differs from the source text.

Worth being honest about: HyDE is not a guaranteed win. For queries that
already use precise terminology it can add noise, and it costs an extra LLM
call per query. That's exactly why it sits behind the use_hyde switch and
gets measured as an ablation rather than assumed.
"""

from __future__ import annotations

from loguru import logger

HYDE_SYSTEM = (
    "You write a single short passage, in the style of a scientific paper, "
    "that plausibly answers the user's question. Be specific and use precise "
    "technical language. Do not hedge, do not say you are unsure, and do not "
    "mention that this is hypothetical. Two or three sentences."
)


def generate_hyde(query: str, llm, max_new_tokens: int = 200) -> str:
    messages = [
        {"role": "system", "content": HYDE_SYSTEM},
        {"role": "user", "content": f"Question: {query}\n\nPassage:"},
    ]
    # A little sampling warmth helps the passage read like prose rather than a
    # restatement of the question.
    text = llm.generate(messages, max_new_tokens=max_new_tokens, temperature=0.7)
    logger.debug(f"HyDE expansion ({len(text)} chars): {text[:120]}...")
    return text.strip()
