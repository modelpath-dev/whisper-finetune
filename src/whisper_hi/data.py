"""Common Voice 17 (Hindi) loading and preprocessing.

Pipeline per split:
    1. Stream-load the split from the Hugging Face Hub.
    2. Drop empty / whitespace-only transcripts.
    3. Resample audio to the target sample rate (16 kHz mono).
    4. Extract log-Mel input features and tokenize the (normalized) text.
    5. Drop clips shorter than ``min_duration_s`` or longer than ``max_duration_s``.

The result is a ``DatasetDict`` whose examples carry exactly two model-facing
columns: ``input_features`` (the log-Mel spectrogram) and ``labels`` (token ids).
"""

from __future__ import annotations

import re

from datasets import Audio, Dataset, DatasetDict, load_dataset

from .config import Config

_WHITESPACE_RE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Light normalization that preserves Devanagari script.

    Devanagari is caseless, so we deliberately do NOT lowercase. We only strip
    leading/trailing whitespace and collapse internal runs of whitespace to a
    single space — this keeps WER/CER fair without altering the script.
    """
    if text is None:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def load_raw_datasets(cfg: Config) -> DatasetDict:
    """Load the train/validation/test splits and resample audio to 16 kHz.

    Common Voice 17 is a gated dataset behind a HF loading script, so this
    requires ``huggingface-cli login`` (after accepting the dataset terms) and
    ``trust_remote_code=True``.
    """
    split_map = {
        "train": cfg.data.train_split,
        "validation": cfg.data.eval_split,
        "test": cfg.data.test_split,
    }
    raw = DatasetDict()
    for name, split in split_map.items():
        raw[name] = load_dataset(
            cfg.data.dataset_name,
            cfg.data.dataset_config,
            split=split,
            trust_remote_code=True,
        )
    # Decode + resample on the fly to the target rate.
    raw = raw.cast_column(cfg.data.audio_column, Audio(sampling_rate=cfg.data.sampling_rate))
    return raw


def prepare_datasets(raw: DatasetDict, processor, cfg: Config) -> DatasetDict:
    """Turn raw audio/text rows into model-ready ``input_features`` + ``labels``.

    ``processor`` is a ``WhisperProcessor`` (feature extractor + tokenizer).
    """
    feature_extractor = processor.feature_extractor
    tokenizer = processor.tokenizer
    text_col = cfg.data.text_column
    audio_col = cfg.data.audio_column

    prepared = DatasetDict()
    sample_caps = {
        "train": cfg.data.max_train_samples,
        "validation": cfg.data.max_eval_samples,
        "test": cfg.data.max_test_samples,
    }

    for split_name, dataset in raw.items():
        # 1. Drop empty / whitespace-only transcripts up front (cheap, avoids
        #    tokenizing junk).
        dataset = dataset.filter(
            lambda t: bool(normalize_text(t)),
            input_columns=[text_col],
        )

        # 2. Optionally subsample (smoke tests / quick experiments). Done before
        #    feature extraction so we don't waste compute on rows we'll discard.
        cap = sample_caps.get(split_name)
        if cap is not None:
            dataset = dataset.select(range(min(cap, len(dataset))))

        # 3. Feature extraction + tokenization.
        def _prepare(batch: dict) -> dict:
            audio = batch[audio_col]
            features = feature_extractor(
                audio["array"], sampling_rate=audio["sampling_rate"]
            )
            return {
                "input_features": features.input_features[0],
                "input_length": len(audio["array"]) / audio["sampling_rate"],
                "labels": tokenizer(normalize_text(batch[text_col])).input_ids,
            }

        dataset = dataset.map(
            _prepare,
            remove_columns=dataset.column_names,
            num_proc=cfg.data.num_proc,
            desc=f"Extracting features [{split_name}]",
        )

        # 4. Duration filter (now that we know each clip's length in seconds).
        dataset = dataset.filter(
            lambda length: cfg.data.min_duration_s <= length <= cfg.data.max_duration_s,
            input_columns=["input_length"],
        ).remove_columns(["input_length"])

        prepared[split_name] = dataset

    return prepared


def describe(dataset: Dataset | DatasetDict) -> str:
    """Return a short human-readable summary of split sizes."""
    if isinstance(dataset, DatasetDict):
        rows = [f"  {name:<11} {len(ds):>7,} examples" for name, ds in dataset.items()]
        return "\n".join(rows)
    return f"{len(dataset):,} examples"
