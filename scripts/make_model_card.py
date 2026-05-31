#!/usr/bin/env python
"""Fill the model card template with metrics from the training/eval artifacts.

Reads the evaluation results and training summary JSON, substitutes the
bracketed placeholders in the template, and writes a ready-to-publish model card.

Example:
    python scripts/make_model_card.py \
        --eval artifacts/eval_results.json \
        --summary artifacts/whisper-small-hi-lora/training_summary.json \
        --repo-id your-username/whisper-small-hi-lora
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", default="artifacts/MODEL_CARD_TEMPLATE.md")
    parser.add_argument("--eval", default="artifacts/eval_results.json")
    parser.add_argument(
        "--summary",
        default="artifacts/whisper-small-hi-lora/training_summary.json",
    )
    parser.add_argument("--output", default="artifacts/MODEL_CARD.md")
    parser.add_argument("--repo-id", default="<your-username>/whisper-small-hi-lora")
    args = parser.parse_args()

    template = Path(args.template).read_text(encoding="utf-8")
    ev = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    train_cfg = summary["config"]["train"]

    replacements = {
        "[BASE_WER]": f"{ev['baseline']['wer']:.2f}",
        "[FT_WER]": f"{ev['finetuned']['wer']:.2f}",
        "[WER_REL]": f"{ev['improvement']['wer_rel_pct']:.1f}",
        "[BASE_CER]": f"{ev['baseline']['cer']:.2f}",
        "[FT_CER]": f"{ev['finetuned']['cer']:.2f}",
        "[CER_REL]": f"{ev['improvement']['cer_rel_pct']:.1f}",
        "[TRAINABLE]": f"{summary['trainable_params']:,}",
        "[TRAINABLE_PCT]": f"{summary['trainable_percentage']}",
        "[EPOCHS]": str(train_cfg["num_train_epochs"]),
        "[WARMUP]": str(train_cfg["warmup_steps"]),
        "[YOUR_REPO_ID]": args.repo_id,
    }

    card = template
    for placeholder, value in replacements.items():
        card = card.replace(placeholder, value)

    out_path = Path(args.output)
    out_path.write_text(card, encoding="utf-8")
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
