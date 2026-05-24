"""Fast CPU smoke tests.

Light tests (config parsing, text normalization, fp16 logic) run anywhere with
the dependencies installed. Heavy tests that download ``whisper-small`` are
gated behind the ``WHISPER_HEAVY_TESTS=1`` environment variable so the default
``pytest`` run stays offline and fast.

Run light tests:   pytest
Run everything:    WHISPER_HEAVY_TESTS=1 pytest
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make src/ importable without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from whisper_hi.config import Config  # noqa: E402
from whisper_hi.data import normalize_text  # noqa: E402
from whisper_hi.model import resolve_fp16  # noqa: E402

CONFIG_PATH = Path(__file__).resolve().parent.parent / "configs" / "whisper_small_hi.yaml"
HEAVY = os.environ.get("WHISPER_HEAVY_TESTS") == "1"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
def test_config_loads_from_yaml():
    cfg = Config.from_yaml(CONFIG_PATH)
    assert cfg.model.base_model == "openai/whisper-small"
    assert cfg.model.lora_r == 32
    assert cfg.model.lora_alpha == 64
    assert cfg.model.target_modules == ["q_proj", "v_proj"]
    assert cfg.data.sampling_rate == 16_000


def test_config_rejects_unknown_key(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  not_a_real_key: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown key"):
        Config.from_yaml(bad)


def test_config_rejects_unknown_section(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("nonsense:\n  foo: 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown config section"):
        Config.from_yaml(bad)


def test_smoke_overrides_shrink_everything():
    cfg = Config.from_yaml(CONFIG_PATH).apply_smoke_overrides()
    assert cfg.data.max_train_samples == 16
    assert cfg.train.max_steps == 5
    assert cfg.train.fp16 is False


def test_config_to_dict_is_nested():
    cfg = Config.from_yaml(CONFIG_PATH)
    d = cfg.to_dict()
    assert set(d) == {"data", "model", "train", "hub"}
    assert d["model"]["lora_r"] == 32
    assert d["data"]["sampling_rate"] == 16_000


def test_config_rejects_invalid_lora_rank(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text("model:\n  lora_r: 0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="lora_r must be positive"):
        Config.from_yaml(bad)


# --------------------------------------------------------------------------- #
# Text normalization
# --------------------------------------------------------------------------- #
def test_normalize_strips_and_collapses_whitespace():
    assert normalize_text("  नमस्ते   दुनिया  ") == "नमस्ते दुनिया"


def test_normalize_preserves_devanagari_and_case():
    # Devanagari is untouched; Latin text is NOT lowercased.
    assert normalize_text("हिंदी ASR Test") == "हिंदी ASR Test"


def test_normalize_handles_none():
    assert normalize_text(None) == ""


# --------------------------------------------------------------------------- #
# Precision logic
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "requested,device,expected",
    [
        (None, "cuda", True),
        (None, "mps", False),
        (None, "cpu", False),
        (True, "mps", False),  # fp16 never enabled off-CUDA
        (True, "cuda", True),
        (False, "cuda", False),
    ],
)
def test_resolve_fp16(requested, device, expected):
    assert resolve_fp16(requested, device) is expected


# --------------------------------------------------------------------------- #
# Heavy: requires downloading whisper-small
# --------------------------------------------------------------------------- #
@pytest.mark.skipif(not HEAVY, reason="set WHISPER_HEAVY_TESTS=1 to run")
def test_lora_trainable_fraction_under_two_percent():
    from whisper_hi.model import build_lora_model

    cfg = Config.from_yaml(CONFIG_PATH)
    cfg.train.gradient_checkpointing = False
    _, stats = build_lora_model(cfg)
    assert stats.percentage < 2.0, f"trainable% too high: {stats.percentage:.4f}"


@pytest.mark.skipif(not HEAVY, reason="set WHISPER_HEAVY_TESTS=1 to run")
def test_collator_pads_labels_with_minus_100():
    import torch

    from whisper_hi.collator import DataCollatorSpeechSeq2SeqWithPadding
    from whisper_hi.model import load_base_model, load_processor

    cfg = Config.from_yaml(CONFIG_PATH)
    processor = load_processor(cfg)
    model = load_base_model(cfg)
    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    n_mels = model.config.num_mel_bins
    features = [
        {"input_features": torch.zeros(n_mels, 3000), "labels": [1, 2, 3]},
        {"input_features": torch.zeros(n_mels, 3000), "labels": [4, 5]},
    ]
    batch = collator(features)
    assert batch["input_features"].shape[0] == 2
    assert batch["labels"].shape[0] == 2
    # The shorter sequence must be padded with -100.
    assert (batch["labels"] == -100).any()
