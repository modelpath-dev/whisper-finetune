"""WER / CER metrics for Hindi ASR.

We report Word Error Rate and Character Error Rate as percentages. CER is
especially informative for Devanagari, where word segmentation can be noisy.
Predictions and references are run through the same light normalization used
during preprocessing so the comparison is apples-to-apples.
"""

from __future__ import annotations

from typing import Callable

import evaluate

from .data import normalize_text


def compute_wer_cer(predictions: list[str], references: list[str]) -> dict[str, float]:
    """Compute WER and CER (as percentages) for decoded string lists."""
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    preds = [normalize_text(p) for p in predictions]
    refs = [normalize_text(r) for r in references]
    return {
        "wer": 100.0 * wer_metric.compute(predictions=preds, references=refs),
        "cer": 100.0 * cer_metric.compute(predictions=preds, references=refs),
    }


def build_compute_metrics(processor) -> Callable:
    """Return a ``compute_metrics`` callable for the HF Seq2SeqTrainer.

    The trainer passes an ``EvalPrediction`` with generated token ids
    (``predictions``) and gold ``label_ids`` (with ``-100`` padding).
    """
    tokenizer = processor.tokenizer
    wer_metric = evaluate.load("wer")
    cer_metric = evaluate.load("cer")
    pad_id = tokenizer.pad_token_id

    def compute_metrics(pred) -> dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # Restore pad tokens so decoding doesn't choke on -100.
        label_ids[label_ids == -100] = pad_id

        pred_str = tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        pred_str = [normalize_text(p) for p in pred_str]
        label_str = [normalize_text(r) for r in label_str]

        return {
            "wer": 100.0 * wer_metric.compute(predictions=pred_str, references=label_str),
            "cer": 100.0 * cer_metric.compute(predictions=pred_str, references=label_str),
        }

    return compute_metrics
