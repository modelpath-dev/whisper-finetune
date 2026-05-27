"""Data collator for Whisper sequence-to-sequence training.

Input features and label sequences have different lengths and are padded
independently. Label padding tokens are replaced with ``-100`` so they are
ignored by the cross-entropy loss, and a leading BOS token (added by the
tokenizer during preprocessing) is stripped because Whisper prepends it during
the forward pass.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad a batch of {input_features, labels} examples into tensors.

    Attributes:
        processor: A ``WhisperProcessor`` providing the feature extractor and
            tokenizer used for padding.
        decoder_start_token_id: The model's decoder start token id; used to
            detect and strip a redundant leading BOS token from the labels.
    """

    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        # Pad the log-Mel features (fixed-length for Whisper, but pad for safety).
        input_features = [{"input_features": f["input_features"]} for f in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # Pad the tokenized transcripts.
        label_features = [{"input_ids": f["labels"]} for f in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # Replace padding with -100 so the loss ignores it.
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # If every sequence starts with the decoder start token, drop it: the
        # model adds it back internally.
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels
        return batch
