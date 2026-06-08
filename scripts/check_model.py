"""
Smoke test for the LLM: load Qwen3-8B in 4-bit, run one generation, and
report how much VRAM it actually uses. The point is to confirm the model
fits the real budget on this 12GB card before we build anything on top of
it, and to sanity-check the thinking-mode handling.

Run from the repo root:
    PYTHONPATH=. .venv/bin/python scripts/check_model.py
"""

from __future__ import annotations

import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from config import settings


def gb(num_bytes: float) -> float:
    return num_bytes / (1024 ** 3)


def main() -> None:
    assert torch.cuda.is_available(), "CUDA not available, nothing to test"
    torch.cuda.reset_peak_memory_stats()

    print(f"Loading {settings.llm_model} in 4-bit NF4 ...")
    t0 = time.time()

    quant_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    tokenizer = AutoTokenizer.from_pretrained(settings.llm_model)
    model = AutoModelForCausalLM.from_pretrained(
        settings.llm_model,
        quantization_config=quant_config,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    load_s = time.time() - t0
    weights_vram = gb(torch.cuda.memory_allocated())
    print(f"Loaded in {load_s:.1f}s. Weights resident: {weights_vram:.2f} GB")

    # Qwen3 is a hybrid thinking model; enable_thinking toggles the <think>
    # block. We default it off for clean QA answers.
    messages = [
        {"role": "system", "content": "You are a concise scientific assistant."},
        {"role": "user", "content": "In one sentence, what problem does retrieval-augmented generation solve?"},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=settings.enable_thinking,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")

    t1 = time.time()
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=128,
            do_sample=True,
            temperature=settings.temperature,
            top_p=settings.top_p,
        )
    gen_s = time.time() - t1

    new_tokens = out[0][inputs["input_ids"].shape[1]:]
    answer = tokenizer.decode(new_tokens, skip_special_tokens=True)
    peak_vram = gb(torch.cuda.max_memory_allocated())

    print("\n----- generation -----")
    print(answer.strip())
    print("----------------------")
    print(f"\nGenerated {new_tokens.shape[0]} tokens in {gen_s:.1f}s "
          f"({new_tokens.shape[0] / gen_s:.1f} tok/s)")
    print(f"Peak VRAM (this process): {peak_vram:.2f} GB")
    print(f"Total card capacity: ~12 GB. Headroom matters because the "
          f"embedder + reranker also want GPU room.")


if __name__ == "__main__":
    main()
