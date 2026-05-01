#!/usr/bin/env python
"""Fine-tune Whisper-small with LoRA on Common Voice 17 (Hindi).

Examples:
    # Full training (Colab T4)
    python scripts/run_train.py --config configs/whisper_small_hi.yaml

    # Fast end-to-end smoke test (laptop CPU/MPS, ~minutes)
    python scripts/run_train.py --config configs/whisper_small_hi.yaml --smoke
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from whisper_hi.config import Config
from whisper_hi.train import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Tiny dataset + few steps to validate the full pipeline quickly.",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.smoke:
        cfg.apply_smoke_overrides()
        print("[mode] SMOKE TEST — results are not meaningful, pipeline check only.")

    run_training(cfg)


if __name__ == "__main__":
    main()
