#!/usr/bin/env python
"""Evaluate baseline vs LoRA fine-tuned Whisper on the test split; write JSON.

Examples:
    python scripts/run_eval.py \
        --config configs/whisper_small_hi.yaml \
        --adapter artifacts/whisper-small-hi-lora \
        --output artifacts/eval_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import _bootstrap  # noqa: F401  (adds src/ to sys.path)

from whisper_hi.config import Config
from whisper_hi.evaluate import evaluate_models


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Path to a YAML config file.")
    parser.add_argument(
        "--adapter",
        required=True,
        help="Directory containing the trained LoRA adapter.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/eval_results.json",
        help="Where to write the JSON results.",
    )
    parser.add_argument("--smoke", action="store_true", help="Subsample the test split.")
    args = parser.parse_args()

    cfg = Config.from_yaml(args.config)
    if args.smoke:
        cfg.apply_smoke_overrides()

    results = evaluate_models(cfg, args.adapter)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")

    # Pretty console table.
    b, f, imp = results["baseline"], results["finetuned"], results["improvement"]
    print("\n" + "=" * 48)
    print(f"{'metric':<8}{'baseline':>12}{'finetuned':>12}{'Δ rel%':>14}")
    print("-" * 48)
    print(f"{'WER':<8}{b['wer']:>12.2f}{f['wer']:>12.2f}{imp['wer_rel_pct']:>13.1f}%")
    print(f"{'CER':<8}{b['cer']:>12.2f}{f['cer']:>12.2f}{imp['cer_rel_pct']:>13.1f}%")
    print("=" * 48)
    print(f"\n[done] results written to {out_path}")


if __name__ == "__main__":
    main()
