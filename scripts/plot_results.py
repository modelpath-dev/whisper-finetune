#!/usr/bin/env python
"""Render a baseline-vs-finetuned WER/CER bar chart from eval results JSON.

Example:
    python scripts/plot_results.py \
        --results artifacts/eval_results.json \
        --output artifacts/wer_cer_comparison.png
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe (Colab / CI)
import matplotlib.pyplot as plt  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="artifacts/eval_results.json")
    parser.add_argument("--output", default="artifacts/wer_cer_comparison.png")
    args = parser.parse_args()

    results = json.loads(Path(args.results).read_text(encoding="utf-8"))
    baseline = results["baseline"]
    finetuned = results["finetuned"]

    metrics = ["WER", "CER"]
    base_vals = [baseline["wer"], baseline["cer"]]
    ft_vals = [finetuned["wer"], finetuned["cer"]]

    x = range(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(6, 4.5))
    bars_b = ax.bar([i - width / 2 for i in x], base_vals, width, label="Baseline", color="#9aa0a6")
    bars_f = ax.bar([i + width / 2 for i in x], ft_vals, width, label="LoRA fine-tuned", color="#1a73e8")

    ax.set_ylabel("Error rate (%)  — lower is better")
    ax.set_title("Whisper-small Hindi ASR: baseline vs LoRA fine-tuned")
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.bar_label(bars_b, fmt="%.1f", padding=2)
    ax.bar_label(bars_f, fmt="%.1f", padding=2)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150)
    print(f"[done] plot saved to {out_path}")


if __name__ == "__main__":
    main()
