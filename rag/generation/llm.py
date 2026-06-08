"""
Qwen3-8B generation wrapper.

This is the single place the model is loaded and called. It loads in 4-bit
(NF4) to fit the 12GB card, optionally stacks a QLoRA adapter on top (that's
how the base-vs-fine-tuned comparison is run: same class, different
lora_adapter_path), and exposes a plain generate(messages) -> text.

Qwen3 is a hybrid thinking model. With enable_thinking on, the output starts
with a <think> ... </think> block before the actual answer. We default
thinking off for QA, but strip the block defensively in case it's ever on,
so callers always get the clean answer.

Loading is expensive, so the model is held as a process-wide singleton via
get_llm().
"""

from __future__ import annotations

import torch
from loguru import logger

from config import settings as default_settings


class LLM:
    def __init__(self, settings=default_settings):
        self.settings = settings
        self._model = None
        self._tokenizer = None

    def _load(self):
        from transformers import (
            AutoModelForCausalLM,
            AutoTokenizer,
            BitsAndBytesConfig,
        )

        s = self.settings
        logger.info(f"Loading LLM {s.llm_model} (4bit={s.load_in_4bit})")

        quant_config = None
        if s.load_in_4bit:
            quant_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
            )

        self._tokenizer = AutoTokenizer.from_pretrained(s.llm_model)
        self._model = AutoModelForCausalLM.from_pretrained(
            s.llm_model,
            quantization_config=quant_config,
            device_map=s.llm_device,
            torch_dtype=torch.bfloat16,
        )

        # Optional fine-tuned adapter for the comparison experiment.
        if s.lora_adapter_path:
            from peft import PeftModel

            logger.info(f"Attaching LoRA adapter from {s.lora_adapter_path}")
            self._model = PeftModel.from_pretrained(self._model, s.lora_adapter_path)

        self._model.eval()

    @property
    def model(self):
        if self._model is None:
            self._load()
        return self._model

    @property
    def tokenizer(self):
        if self._tokenizer is None:
            self._load()
        return self._tokenizer

    @staticmethod
    def _strip_thinking(text: str) -> str:
        # Keep only what follows the reasoning block, if there is one.
        if "</think>" in text:
            return text.split("</think>", 1)[1].strip()
        return text.strip()

    def generate(self, messages: list[dict], max_new_tokens: int | None = None,
                 temperature: float | None = None) -> str:
        s = self.settings
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=s.enable_thinking,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens or s.max_new_tokens,
                do_sample=(temperature or s.temperature) > 0,
                temperature=temperature or s.temperature,
                top_p=s.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = out[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._strip_thinking(text)


_LLM_SINGLETON: LLM | None = None


def get_llm(settings=default_settings) -> LLM:
    """Process-wide LLM instance, loaded on first real use."""
    global _LLM_SINGLETON
    if _LLM_SINGLETON is None:
        _LLM_SINGLETON = LLM(settings)
    return _LLM_SINGLETON
