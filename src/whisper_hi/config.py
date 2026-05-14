"""Typed configuration loaded from YAML.

The whole project is driven by a single :class:`Config` object built from
``configs/whisper_small_hi.yaml``. Using dataclasses (instead of passing raw
dicts around) gives us IDE autocompletion, fail-fast validation of unknown
keys, and a single place to document every knob.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

import yaml


@dataclass
class DataConfig:
    """Dataset loading, filtering, and preprocessing options."""

    dataset_name: str = "mozilla-foundation/common_voice_17_0"
    dataset_config: str = "hi"
    train_split: str = "train"
    eval_split: str = "validation"
    test_split: str = "test"
    text_column: str = "sentence"
    audio_column: str = "audio"
    sampling_rate: int = 16_000
    min_duration_s: float = 1.0
    max_duration_s: float = 30.0
    max_train_samples: int | None = None
    max_eval_samples: int | None = None
    max_test_samples: int | None = None
    num_proc: int = 2


@dataclass
class ModelConfig:
    """Base model and LoRA adapter hyperparameters."""

    base_model: str = "openai/whisper-small"
    language: str = "Hindi"
    task: str = "transcribe"
    lora_r: int = 32
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    target_modules: list[str] = field(default_factory=lambda: ["q_proj", "v_proj"])

    def __post_init__(self) -> None:
        if self.lora_r <= 0:
            raise ValueError(f"lora_r must be positive, got {self.lora_r}")
        if self.lora_alpha <= 0:
            raise ValueError(f"lora_alpha must be positive, got {self.lora_alpha}")
        if not 0.0 <= self.lora_dropout < 1.0:
            raise ValueError(f"lora_dropout must be in [0, 1), got {self.lora_dropout}")
        if not self.target_modules:
            raise ValueError("target_modules must list at least one module")


@dataclass
class TrainConfig:
    """Trainer / optimization options."""

    output_dir: str = "artifacts/whisper-small-hi-lora"
    per_device_train_batch_size: int = 16
    per_device_eval_batch_size: int = 8
    gradient_accumulation_steps: int = 1
    learning_rate: float = 1e-3
    warmup_steps: int = 50
    num_train_epochs: float = 3.0
    max_steps: int = -1
    fp16: bool | None = None
    eval_strategy: str = "steps"
    eval_steps: int = 200
    save_steps: int = 200
    logging_steps: int = 25
    save_total_limit: int = 2
    generation_max_length: int = 225
    predict_with_generate: bool = True
    gradient_checkpointing: bool = True
    seed: int = 42
    report_to: str = "none"


@dataclass
class HubConfig:
    """Optional Hugging Face Hub push settings."""

    push_to_hub: bool = False
    repo_id: str | None = None
    private: bool = False


@dataclass
class Config:
    """Top-level config aggregating every section."""

    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    hub: HubConfig = field(default_factory=HubConfig)

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Config":
        """Build a :class:`Config` from a YAML file, validating all keys."""
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        if not isinstance(raw, dict):
            raise ValueError(f"Config file {path} must contain a mapping at the top level.")

        section_types = {
            "data": DataConfig,
            "model": ModelConfig,
            "train": TrainConfig,
            "hub": HubConfig,
        }
        unknown_sections = set(raw) - set(section_types)
        if unknown_sections:
            raise ValueError(f"Unknown config section(s): {sorted(unknown_sections)}")

        kwargs: dict[str, Any] = {}
        for name, dc_type in section_types.items():
            kwargs[name] = _build_section(dc_type, raw.get(name, {}) or {}, name)
        return cls(**kwargs)

    def apply_smoke_overrides(self) -> "Config":
        """Mutate this config in place for a fast CPU/MPS smoke test.

        Trims the dataset to a handful of samples, caps training to a few
        steps, disables fp16, and shrinks batch sizes so the full pipeline can
        run end-to-end on a laptop in a couple of minutes.
        """
        self.data.max_train_samples = 16
        self.data.max_eval_samples = 8
        self.data.max_test_samples = 8
        self.data.num_proc = 1
        self.train.per_device_train_batch_size = 2
        self.train.per_device_eval_batch_size = 2
        self.train.gradient_accumulation_steps = 1
        self.train.max_steps = 5
        self.train.num_train_epochs = 1.0
        self.train.warmup_steps = 1
        self.train.eval_steps = 5
        self.train.save_steps = 5
        self.train.logging_steps = 1
        self.train.fp16 = False
        self.train.gradient_checkpointing = False
        self.train.output_dir = "artifacts/smoke-whisper-small-hi-lora"
        return self

    def to_dict(self) -> dict[str, Any]:
        """Plain nested dict — handy for logging or writing a model card."""
        return dataclasses.asdict(self)


def _build_section(dc_type: type, values: dict[str, Any], section: str):
    """Instantiate one dataclass section, rejecting unknown keys."""
    valid = {f.name for f in fields(dc_type)}
    unknown = set(values) - valid
    if unknown:
        raise ValueError(f"Unknown key(s) in [{section}]: {sorted(unknown)}")
    return dc_type(**values)
