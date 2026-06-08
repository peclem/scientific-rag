"""
QLoRA fine-tuning of Qwen3-8B for the scientific RAG house style.

Loads the base model in 4-bit, attaches LoRA adapters, and trains with TRL's
SFTTrainer on the SFT data (chat-format examples). Only the assistant turn
contributes to the loss (assistant_only_loss), so the model learns to produce
the answer, not to reproduce the prompt.

VRAM notes for the 12GB card: gradient checkpointing is on, batch size is 1
with gradient accumulation for an effective batch of 16, and the base weights
stay 4-bit. Use --smoke for a 10-step run that just confirms it fits and the
loss moves before committing to a full run.

    PYTHONPATH=. .venv/bin/python training/qlora/train.py --smoke
    PYTHONPATH=. .venv/bin/python training/qlora/train.py
"""

from __future__ import annotations

import argparse
import os

# The system site-packages has an old wandb (0.16.1) that breaks on numpy 2.0
# (np.float_ was removed). trl imports wandb eagerly unless it's disabled, so
# turn it off here before trl is imported. We log to "none" anyway.
os.environ["WANDB_DISABLED"] = "true"
os.environ["WANDB_MODE"] = "disabled"

import torch
from loguru import logger
from peft import LoraConfig, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer

from training.qlora.config import QLoRAConfig
from training.sft_dataset import build_mixed_sft, build_qasper_sft


def load_base(cfg: QLoRAConfig):
    quant = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
    )
    tokenizer = AutoTokenizer.from_pretrained(cfg.base_model)
    model = AutoModelForCausalLM.from_pretrained(
        cfg.base_model,
        quantization_config=quant,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
    )
    model.config.use_cache = False  # required with gradient checkpointing
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    return model, tokenizer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true",
                    help="10-step run on a tiny dataset to check VRAM/loss")
    args = ap.parse_args()

    cfg = QLoRAConfig()

    if args.smoke:
        logger.info("Building tiny QASPER SFT set for smoke run...")
        dataset = build_qasper_sft(split="train", max_examples=40)
    else:
        logger.info("Building mixed SFT dataset (QASPER + PubMedQA + SciFact)...")
        dataset = build_mixed_sft()
    logger.info(f"SFT examples: {len(dataset)}")

    model, tokenizer = load_base(cfg)

    lora = LoraConfig(
        r=cfg.lora_r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=list(cfg.target_modules),
        bias="none",
        task_type="CAUSAL_LM",
    )

    sft_config = SFTConfig(
        output_dir=cfg.output_dir,
        per_device_train_batch_size=cfg.batch_size,
        gradient_accumulation_steps=cfg.grad_accum,
        learning_rate=cfg.learning_rate,
        warmup_ratio=cfg.warmup_ratio,
        max_length=cfg.max_seq_length,
        num_train_epochs=cfg.epochs,
        max_steps=10 if args.smoke else -1,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        bf16=True,
        logging_steps=cfg.logging_steps if not args.smoke else 1,
        save_strategy="no" if args.smoke else cfg.save_strategy,
        # Loss is on the completion only because the dataset is in
        # prompt/completion format (TRL masks the prompt automatically).
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=lora,
        processing_class=tokenizer,
    )

    trainer.train()

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1024**3
        logger.info(f"Peak training VRAM: {peak:.2f} GB")

    if not args.smoke:
        trainer.save_model(cfg.output_dir)
        logger.info(f"Adapter saved to {cfg.output_dir}")
    else:
        logger.info("Smoke run complete (adapter not saved).")


if __name__ == "__main__":
    main()
