# Architecture

A short tour of how the pieces fit together.

## Data flow

```
Common Voice 17 (hi)
        │  load_raw_datasets()           src/whisper_hi/data.py
        ▼
  resample to 16 kHz, drop empty transcripts
        │  prepare_datasets()
        ▼
  log-Mel input_features + tokenized labels   (filter 1–30 s)
        │  DataCollatorSpeechSeq2SeqWithPadding   src/whisper_hi/collator.py
        ▼
  Seq2SeqTrainer (LoRA adapter only)         src/whisper_hi/train.py
        │
        ▼
  artifacts/<run>/  →  adapter + training_summary.json
        │  evaluate_models()                src/whisper_hi/evaluate.py
        ▼
  baseline vs fine-tuned WER/CER  →  eval_results.json  →  plot
```

## Module responsibilities

| Module        | Responsibility                                                   |
|---------------|------------------------------------------------------------------|
| `config.py`   | YAML → typed dataclasses with validation; the single source of truth. |
| `data.py`     | Dataset loading, resampling, filtering, feature extraction, text normalization. |
| `collator.py` | Pad input features and labels into batched tensors for training. |
| `model.py`    | Build Whisper + LoRA, device/precision selection, parameter accounting. |
| `metrics.py`  | WER / CER computation for the Trainer and standalone evaluation. |
| `train.py`    | Wire everything into a `Seq2SeqTrainer` run; save artifacts.     |
| `evaluate.py` | Compare baseline vs fine-tuned on the test split.                |
| `inference.py`| Transcribe a single audio file with a trained adapter.           |

## Design notes

- **Config-driven:** scripts and the Colab notebook consume the same YAML, so a
  run is fully described by its config plus the dataset revision.
- **Adapter-only training:** the base Whisper weights stay frozen; only the LoRA
  adapter (< 2% of parameters) is trained and saved.
- **Precision:** fp16 is enabled only on CUDA; MPS/CPU stay in fp32 for stability.
