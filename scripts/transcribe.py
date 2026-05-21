#!/usr/bin/env python
"""Transcribe a single audio file with a trained LoRA adapter.

Example:
    python scripts/transcribe.py \
        --config configs/whisper_small_hi.yaml \
        --adapter artifacts/whisper-small-hi-lora \
        --audio sample.wav
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from whisper_hi.config import Config
from whisper_hi.inference import transcribe_file


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument("--adapter", required=True, help="Trained LoRA adapter directory.")
    parser.add_argument("--audio", required=True, help="Audio file to transcribe.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    text = transcribe_file(cfg, args.adapter, args.audio)
    print(text)


if __name__ == "__main__":
    main()
