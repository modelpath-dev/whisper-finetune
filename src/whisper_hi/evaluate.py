"""Evaluate the baseline Whisper-small against the LoRA fine-tuned model.

Both models are run over the held-out test split and scored with WER/CER. The
results are returned as a dict and (by the CLI wrapper) written to JSON so the
plotting script and README can consume them.
"""

from __future__ import annotations

import torch
from torch.utils.data import DataLoader

from .collator import DataCollatorSpeechSeq2SeqWithPadding
from .config import Config
from .data import load_raw_datasets, prepare_datasets
from .metrics import compute_wer_cer
from .model import get_device, load_base_model, load_finetuned_model, load_processor


@torch.no_grad()
def transcribe_dataset(
    model,
    processor,
    dataset,
    collator: DataCollatorSpeechSeq2SeqWithPadding,
    device: str,
    batch_size: int,
    generation_max_length: int,
    use_fp16: bool,
) -> tuple[list[str], list[str]]:
    """Generate transcriptions for an entire dataset; return (preds, refs)."""
    model.to(device)
    model.eval()
    if use_fp16:
        model.half()

    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator)
    tokenizer = processor.tokenizer
    pad_id = tokenizer.pad_token_id

    predictions: list[str] = []
    references: list[str] = []

    for batch in loader:
        input_features = batch["input_features"].to(device)
        if use_fp16:
            input_features = input_features.half()

        generated_ids = model.generate(
            input_features=input_features,
            max_length=generation_max_length,
        )

        labels = batch["labels"].clone()
        labels[labels == -100] = pad_id

        predictions.extend(tokenizer.batch_decode(generated_ids, skip_special_tokens=True))
        references.extend(tokenizer.batch_decode(labels, skip_special_tokens=True))

    return predictions, references


def evaluate_models(cfg: Config, adapter_dir: str) -> dict:
    """Score baseline and fine-tuned models on the test split.

    Returns a dict with per-model WER/CER and the absolute/relative improvement.
    """
    device = get_device()
    use_fp16 = device == "cuda"
    print(f"[eval] device={device}  fp16={use_fp16}")

    processor = load_processor(cfg)

    # Prepare only the test split (reuse the same pipeline as training).
    raw = load_raw_datasets(cfg)
    test_only = raw.__class__({"test": raw["test"]})
    prepared = prepare_datasets(test_only, processor, cfg)
    test_set = prepared["test"]
    print(f"[eval] test examples: {len(test_set):,}")

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=load_base_model(cfg).config.decoder_start_token_id,
    )
    batch_size = cfg.train.per_device_eval_batch_size
    gen_len = cfg.train.generation_max_length

    # --- Baseline -----------------------------------------------------------
    print("[eval] scoring baseline whisper-small…")
    baseline = load_base_model(cfg)
    base_preds, refs = transcribe_dataset(
        baseline, processor, test_set, collator, device, batch_size, gen_len, use_fp16
    )
    baseline_scores = compute_wer_cer(base_preds, refs)
    del baseline
    if device == "cuda":
        torch.cuda.empty_cache()

    # --- Fine-tuned ---------------------------------------------------------
    print(f"[eval] scoring fine-tuned adapter from {adapter_dir}…")
    finetuned = load_finetuned_model(cfg, adapter_dir)
    ft_preds, _ = transcribe_dataset(
        finetuned, processor, test_set, collator, device, batch_size, gen_len, use_fp16
    )
    finetuned_scores = compute_wer_cer(ft_preds, refs)

    def rel_improvement(before: float, after: float) -> float:
        return 100.0 * (before - after) / before if before else 0.0

    results = {
        "num_test_examples": len(test_set),
        "baseline": {
            "wer": round(baseline_scores["wer"], 4),
            "cer": round(baseline_scores["cer"], 4),
        },
        "finetuned": {
            "wer": round(finetuned_scores["wer"], 4),
            "cer": round(finetuned_scores["cer"], 4),
        },
        "improvement": {
            "wer_abs": round(baseline_scores["wer"] - finetuned_scores["wer"], 4),
            "cer_abs": round(baseline_scores["cer"] - finetuned_scores["cer"], 4),
            "wer_rel_pct": round(
                rel_improvement(baseline_scores["wer"], finetuned_scores["wer"]), 2
            ),
            "cer_rel_pct": round(
                rel_improvement(baseline_scores["cer"], finetuned_scores["cer"]), 2
            ),
        },
    }
    return results
