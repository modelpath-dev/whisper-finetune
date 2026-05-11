"""Single-file inference with a trained LoRA adapter.

A thin, reusable wrapper that loads the base model + adapter, featurizes one
audio file, and returns the decoded transcription. Used by
``scripts/transcribe.py`` and convenient for ad-hoc testing.
"""

from __future__ import annotations

import librosa
import torch

from .config import Config
from .model import get_device, load_finetuned_model, load_processor


def transcribe_file(cfg: Config, adapter_dir: str, audio_path: str) -> str:
    """Transcribe a single audio file and return the predicted text.

    Args:
        cfg: Project configuration.
        adapter_dir: Directory containing the trained LoRA adapter.
        audio_path: Path to an audio file (any format librosa can read).
    """
    device = get_device()
    processor = load_processor(cfg)
    model = load_finetuned_model(cfg, adapter_dir).to(device)
    model.eval()

    speech, _ = librosa.load(audio_path, sr=cfg.data.sampling_rate, mono=True)
    features = processor.feature_extractor(
        speech, sampling_rate=cfg.data.sampling_rate, return_tensors="pt"
    ).input_features.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            input_features=features,
            max_length=cfg.train.generation_max_length,
        )

    return processor.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
