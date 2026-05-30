---
language: hi
license: apache-2.0
library_name: peft
tags:
  - whisper
  - automatic-speech-recognition
  - lora
  - peft
  - hindi
datasets:
  - mozilla-foundation/common_voice_17_0
metrics:
  - wer
  - cer
base_model: openai/whisper-small
pipeline_tag: automatic-speech-recognition
---

# Whisper-small Hindi — LoRA adapter

LoRA adapter fine-tuning [`openai/whisper-small`](https://huggingface.co/openai/whisper-small)
for Hindi automatic speech recognition on Mozilla Common Voice 17.

> Fill in the bracketed values from `artifacts/eval_results.json` and
> `artifacts/training_summary.json` after training.

## Results (Common Voice 17 `hi` test split)

| Metric | Baseline whisper-small | LoRA fine-tuned | Relative improvement |
|-------:|:----------------------:|:---------------:|:--------------------:|
| WER    | [BASE_WER]             | [FT_WER]        | [WER_REL]%           |
| CER    | [BASE_CER]             | [FT_CER]        | [CER_REL]%           |

![WER/CER comparison](wer_cer_comparison.png)

## Training procedure

- **Base model:** `openai/whisper-small`
- **Method:** LoRA (PEFT), `target_modules=["q_proj", "v_proj"]`, `r=32`,
  `alpha=64`, `dropout=0.05`
- **Trainable params:** [TRAINABLE] ([TRAINABLE_PCT]% of total)
- **Precision:** fp16 (NVIDIA T4)
- **Epochs / steps:** [EPOCHS]
- **Learning rate:** 1e-3, [WARMUP] warmup steps
- **Hardware:** Google Colab free tier (single T4, 16 GB)

## Usage

```python
from peft import PeftModel
from transformers import WhisperForConditionalGeneration, WhisperProcessor

base = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
model = PeftModel.from_pretrained(base, "[YOUR_REPO_ID]")
processor = WhisperProcessor.from_pretrained("[YOUR_REPO_ID]", language="Hindi", task="transcribe")
```

## Limitations

Trained on a small (~5–8 hour) read-speech subset of Common Voice. Performance
on spontaneous speech, noisy audio, code-switching, or dialects outside the
training distribution may be lower.
