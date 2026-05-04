"""Training entrypoint: LoRA fine-tuning of Whisper-small with the HF Trainer."""

from __future__ import annotations

import json
from pathlib import Path

from transformers import (
    EarlyStoppingCallback,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)

from .collator import DataCollatorSpeechSeq2SeqWithPadding
from .config import Config
from .data import describe, load_raw_datasets, prepare_datasets
from .metrics import build_compute_metrics
from .model import build_lora_model, get_device, load_processor, resolve_fp16


def run_training(cfg: Config) -> dict:
    """Run the full fine-tuning loop and return a small results summary.

    Saves the trained LoRA adapter, the processor, and a ``training_summary.json``
    to ``cfg.train.output_dir``.
    """
    set_seed(cfg.train.seed)
    device = get_device()
    fp16 = resolve_fp16(cfg.train.fp16, device)

    print(f"[setup] device={device}  fp16={fp16}")
    print(f"[setup] base model: {cfg.model.base_model}")

    # --- Processor, model, adapter -----------------------------------------
    processor = load_processor(cfg)
    model, stats = build_lora_model(cfg)
    print(f"[lora] {stats}")
    if stats.percentage >= 2.0:
        print(f"[lora] WARNING: trainable% = {stats.percentage:.4f} (expected < 2%).")

    # --- Data ---------------------------------------------------------------
    print("[data] loading Common Voice 17 (hi)…")
    raw = load_raw_datasets(cfg)
    datasets = prepare_datasets(raw, processor, cfg)
    print("[data] prepared splits:")
    print(describe(datasets))

    collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )
    compute_metrics = build_compute_metrics(processor)

    # --- Trainer ------------------------------------------------------------
    output_dir = cfg.train.output_dir
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    training_args = Seq2SeqTrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=cfg.train.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.train.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.train.gradient_accumulation_steps,
        learning_rate=cfg.train.learning_rate,
        warmup_steps=cfg.train.warmup_steps,
        num_train_epochs=cfg.train.num_train_epochs,
        max_steps=cfg.train.max_steps,
        fp16=fp16,
        gradient_checkpointing=cfg.train.gradient_checkpointing,
        eval_strategy=cfg.train.eval_strategy,
        eval_steps=cfg.train.eval_steps,
        save_strategy="steps",
        save_steps=cfg.train.save_steps,
        save_total_limit=cfg.train.save_total_limit,
        logging_steps=cfg.train.logging_steps,
        predict_with_generate=cfg.train.predict_with_generate,
        generation_max_length=cfg.train.generation_max_length,
        load_best_model_at_end=True,
        metric_for_best_model="wer",
        greater_is_better=False,
        report_to=[cfg.train.report_to] if cfg.train.report_to != "none" else [],
        seed=cfg.train.seed,
        # PeftModel's forward signature is generic; without this the Trainer
        # would strip our input_features/labels columns.
        remove_unused_columns=False,
        label_names=["labels"],
    )

    callbacks = []
    if cfg.train.early_stopping_patience > 0:
        callbacks.append(
            EarlyStoppingCallback(early_stopping_patience=cfg.train.early_stopping_patience)
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        data_collator=collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=callbacks,
    )
    # use_cache is incompatible with gradient checkpointing during training.
    model.config.use_cache = False

    print("[train] starting…")
    train_result = trainer.train()

    # --- Save artifacts -----------------------------------------------------
    trainer.save_model(output_dir)  # saves the LoRA adapter (PeftModel)
    processor.save_pretrained(output_dir)

    print("[eval] evaluating best model on validation split…")
    eval_metrics = trainer.evaluate()

    summary = {
        "base_model": cfg.model.base_model,
        "trainable_params": stats.trainable,
        "total_params": stats.total,
        "trainable_percentage": round(stats.percentage, 4),
        "train_runtime_s": round(train_result.metrics.get("train_runtime", 0.0), 2),
        "train_loss": round(train_result.metrics.get("train_loss", 0.0), 4),
        "validation_wer": round(eval_metrics.get("eval_wer", float("nan")), 4),
        "validation_cer": round(eval_metrics.get("eval_cer", float("nan")), 4),
        "config": cfg.to_dict(),
    }
    summary_path = Path(output_dir) / "training_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[done] adapter + summary saved to {output_dir}")
    print(f"[done] validation WER={summary['validation_wer']}  CER={summary['validation_cer']}")

    # --- Optional Hub push --------------------------------------------------
    if cfg.hub.push_to_hub:
        if not cfg.hub.repo_id:
            print("[hub] push_to_hub=true but hub.repo_id is unset — skipping push.")
        else:
            print(f"[hub] pushing adapter to {cfg.hub.repo_id}…")
            model.push_to_hub(cfg.hub.repo_id, private=cfg.hub.private)
            processor.push_to_hub(cfg.hub.repo_id, private=cfg.hub.private)

    return summary
