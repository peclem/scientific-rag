"""
QLoRA training configuration.

Defaults follow the roadmap and are sized for a 12GB card: 4-bit base, LoRA
on all attention and MLP projections, batch size 1 with gradient
accumulation, and a capped sequence length. Sequence length is the main VRAM
lever during training (activations scale with it), so it's deliberately
modest.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QLoRAConfig:
    base_model: str = "Qwen/Qwen3-8B"
    output_dir: str = "adapters/qwen3-8b-scirag"

    # LoRA
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    target_modules: tuple[str, ...] = (
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    )

    # Training
    epochs: int = 2
    batch_size: int = 1
    grad_accum: int = 16          # effective batch 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.03
    max_seq_length: int = 1536    # main VRAM knob
    logging_steps: int = 10
    save_strategy: str = "epoch"

    # Data
    max_train_papers: int | None = None
    max_train_examples: int | None = None
