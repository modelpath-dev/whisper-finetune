#!/usr/bin/env python
"""Download, filter, and report stats for the Common Voice 17 (Hindi) splits.

This is a convenience / sanity-check step — training and evaluation load the
data themselves. Run it first to warm the Hugging Face cache and confirm your
dataset access works.

Examples:
    python scripts/prepare_data.py --config configs/whisper_small_hi.yaml
    python scripts/prepare_data.py --config configs/whisper_small_hi.yaml --smoke
"""

from __future__ import annotations

import argparse

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from whisper_hi.config import Config
from whisper_hi.data import describe, load_raw_datasets, prepare_datasets
from whisper_hi.model import load_processor


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="Subsample to a handful of examples for a quick sanity check.",
    )
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.smoke:
        cfg.apply_smoke_overrides()

    print(f"[data] dataset: {cfg.data.dataset_name} ({cfg.data.dataset_config})")
    raw = load_raw_datasets(cfg)
    print("[data] raw split sizes:")
    print(describe(raw))

    processor = load_processor(cfg)
    prepared = prepare_datasets(raw, processor, cfg)
    print("[data] prepared (filtered + featurized) split sizes:")
    print(describe(prepared))


if __name__ == "__main__":
    main()
