"""Model construction: Whisper-small + LoRA adapter, plus device/precision helpers."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from peft import LoraConfig, PeftModel, get_peft_model
from transformers import (
    WhisperForConditionalGeneration,
    WhisperProcessor,
)

from .config import Config


@dataclass
class ParamStats:
    """Trainable-vs-total parameter accounting for a LoRA model."""

    trainable: int
    total: int

    @property
    def percentage(self) -> float:
        return 100.0 * self.trainable / self.total if self.total else 0.0

    def __str__(self) -> str:
        return (
            f"trainable params: {self.trainable:,} || "
            f"all params: {self.total:,} || "
            f"trainable%: {self.percentage:.4f}"
        )


def get_device() -> str:
    """Pick the best available device: CUDA > MPS (Apple Silicon) > CPU."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def resolve_fp16(requested: bool | None, device: str) -> bool:
    """Resolve the fp16 flag.

    fp16 mixed precision is only enabled on CUDA (e.g. the Colab T4). On MPS and
    CPU we stay in fp32, which is both more stable and what the HF Trainer
    expects on those backends. ``requested=None`` means "auto".
    """
    if requested is None:
        return device == "cuda"
    return bool(requested) and device == "cuda"


def load_processor(cfg: Config) -> WhisperProcessor:
    """Load the Whisper processor configured for the target language/task."""
    return WhisperProcessor.from_pretrained(
        cfg.model.base_model,
        language=cfg.model.language,
        task=cfg.model.task,
    )


def count_parameters(model: torch.nn.Module) -> ParamStats:
    """Count trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return ParamStats(trainable=trainable, total=total)


def load_base_model(cfg: Config) -> WhisperForConditionalGeneration:
    """Load the base Whisper model and configure it for generation/training."""
    model = WhisperForConditionalGeneration.from_pretrained(cfg.model.base_model)

    # Whisper-specific generation setup: no hard-coded forced decoder ids /
    # suppressed tokens — language & task are supplied via generation_config.
    model.config.forced_decoder_ids = None
    model.config.suppress_tokens = []
    model.generation_config.language = cfg.model.language.lower()
    model.generation_config.task = cfg.model.task
    model.generation_config.forced_decoder_ids = None
    return model


def build_lora_model(cfg: Config) -> tuple[PeftModel, ParamStats]:
    """Load Whisper-small and wrap it with a LoRA adapter.

    Returns the PEFT-wrapped model and its parameter statistics. Only the LoRA
    adapter weights are trainable — expected to be well under 2% of the total.
    """
    base = load_base_model(cfg)

    # Gradient checkpointing + LoRA: the frozen base produces inputs that don't
    # require grad, so we must explicitly make embedding outputs require grad
    # for gradients to reach the adapter.
    if cfg.train.gradient_checkpointing:
        base.config.use_cache = False
        base.enable_input_require_grads()

    lora_config = LoraConfig(
        r=cfg.model.lora_r,
        lora_alpha=cfg.model.lora_alpha,
        target_modules=cfg.model.target_modules,
        lora_dropout=cfg.model.lora_dropout,
        bias="none",
    )
    model = get_peft_model(base, lora_config)
    stats = count_parameters(model)
    return model, stats


def load_finetuned_model(cfg: Config, adapter_dir: str) -> PeftModel:
    """Load the base model and attach a trained LoRA adapter for inference."""
    base = load_base_model(cfg)
    model = PeftModel.from_pretrained(base, adapter_dir)
    model.config.use_cache = True
    return model
