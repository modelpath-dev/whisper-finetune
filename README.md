# Whisper-small Hindi ASR — LoRA Fine-tuning

[![tests](https://github.com/modelpath-dev/whisper-finetune/actions/workflows/tests.yml/badge.svg)](https://github.com/modelpath-dev/whisper-finetune/actions/workflows/tests.yml)
[![python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)
[![license: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

Fine-tune [OpenAI Whisper-small](https://huggingface.co/openai/whisper-small) for **Hindi
automatic speech recognition** using **LoRA** (Low-Rank Adaptation via [PEFT](https://github.com/huggingface/peft))
on the [Mozilla Common Voice 17](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0)
dataset, then measure the WER/CER improvement over the un-tuned baseline.

The project is built to run two ways:

- **Locally** (incl. Apple Silicon, CPU/MPS) for fast smoke tests of the full pipeline.
- **Google Colab free tier** (single T4 GPU) for the real fp16 training run.

> Why LoRA? We train **< 2%** of the model's parameters (the adapter only), which keeps the
> Colab T4's 16 GB of VRAM comfortable and produces a tiny, shareable artifact instead of a
> full 1+ GB checkpoint.

---


## Project layout

```
whisper_finetune/
├── README.md
├── requirements.txt          # pinned dependencies
├── pyproject.toml            # editable-install metadata + pytest config
├── configs/
│   └── whisper_small_hi.yaml # single source of truth for all hyperparameters
├── src/whisper_hi/           # importable, type-hinted library code
│   ├── config.py             # YAML → typed dataclasses (fail-fast validation)
│   ├── data.py               # load / filter / resample / featurize CV17 (hi)
│   ├── collator.py           # DataCollatorSpeechSeq2SeqWithPadding
│   ├── model.py              # Whisper + LoRA, device/precision helpers
│   ├── metrics.py            # WER / CER via evaluate + jiwer
│   ├── train.py              # Seq2SeqTrainer loop
│   └── evaluate.py           # baseline vs fine-tuned comparison
├── scripts/                  # thin CLI wrappers around the library
│   ├── prepare_data.py
│   ├── run_train.py
│   ├── run_eval.py
│   └── plot_results.py
├── notebooks/
│   └── whisper_hi_colab.ipynb  # one-click end-to-end Colab notebook
├── tests/
│   └── test_smoke.py         # fast CPU tests (+ optional heavy model tests)
└── artifacts/                # outputs: adapter, JSON metrics, plot, model card
```

---

## Method

| Choice            | Value                                   |
|-------------------|-----------------------------------------|
| Base model        | `openai/whisper-small`                  |
| Adapter           | LoRA (PEFT)                             |
| `target_modules`  | `["q_proj", "v_proj"]`                  |
| `r` / `alpha`     | `32` / `64`                             |
| `lora_dropout`    | `0.05`                                  |
| Precision         | fp16 on CUDA (T4); fp32 on MPS/CPU      |
| Optimizer / LR    | AdamW / `1e-3`                          |
| Dataset           | Common Voice 17, config `hi`            |
| Audio             | 16 kHz mono, clips 1–30 s               |
| Text              | Devanagari preserved, **not** lowercased |
| Metrics           | WER, CER                                |

The trainable-parameter percentage is printed at startup (`[lora] trainable params: … || trainable%: …`)
so the < 2% claim is verifiable, not just asserted.

---

## Setup (local, Apple Silicon / macOS)

This machine is Apple Silicon, so we install into a virtual environment and use the
CPU/MPS PyTorch build for **smoke tests only** — full training happens on Colab.

```bash
cd whisper_finetune

# 1. Create and activate a venv (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# 2. Install pinned dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 3. (optional) editable install so `import whisper_hi` works from anywhere
pip install -e .
```

Common tasks are wrapped in a `Makefile`:

```bash
make install   # install pinned deps
make test      # fast offline unit tests
make smoke     # full pipeline on a tiny subset (no GPU)
make train     # full LoRA training run
make eval      # baseline vs fine-tuned on the test split
make plot      # render the WER/CER chart
```

> **Common Voice 17 is gated.** Accept the dataset terms on its
> [Hub page](https://huggingface.co/datasets/mozilla-foundation/common_voice_17_0), then
> authenticate: `huggingface-cli login`.

### Run the smoke test (no GPU needed)

Validates the entire pipeline — data → features → LoRA → train → eval — on a handful of
examples and a few steps. Results are meaningless; it's a wiring check.

```bash
python scripts/prepare_data.py --config configs/whisper_small_hi.yaml --smoke
python scripts/run_train.py    --config configs/whisper_small_hi.yaml --smoke
python scripts/run_eval.py     --config configs/whisper_small_hi.yaml \
    --adapter artifacts/smoke-whisper-small-hi-lora --smoke
```

### Run the unit tests

```bash
pytest                      # fast, offline (config / normalization / fp16 logic)
WHISPER_HEAVY_TESTS=1 pytest # also downloads whisper-small to check LoRA % + collator
```

---

## Full training (Google Colab, T4)

Open `notebooks/whisper_hi_colab.ipynb` in Colab (**Runtime → Change runtime type → T4 GPU**)
and run all cells. The notebook installs deps (without clobbering Colab's torch), logs in to
the Hub, trains, evaluates, plots, and optionally pushes the adapter.

Or, from a terminal on any CUDA box:

```bash
python scripts/run_train.py --config configs/whisper_small_hi.yaml
python scripts/run_eval.py  --config configs/whisper_small_hi.yaml \
    --adapter artifacts/whisper-small-hi-lora
python scripts/plot_results.py
```

Edit `configs/whisper_small_hi.yaml` to change any hyperparameter — it is the single source
of truth, consumed identically by the scripts and the notebook.

---

## Inference

Transcribe a single audio file with a trained adapter:

```bash
python scripts/transcribe.py \
    --config configs/whisper_small_hi.yaml \
    --adapter artifacts/whisper-small-hi-lora \
    --audio sample.wav
```

Or from Python:

```python
from whisper_hi.config import Config
from whisper_hi.inference import transcribe_file

cfg = Config.from_yaml("configs/whisper_small_hi.yaml")
print(transcribe_file(cfg, "artifacts/whisper-small-hi-lora", "sample.wav"))
```

---

## Publishing

1. Train + evaluate; copy `artifacts/MODEL_CARD_TEMPLATE.md` to your HF repo as `README.md`
   and fill in the bracketed metrics.
2. Set `hub.push_to_hub: true` and `hub.repo_id` in the config (or push the adapter manually).
3. Push this repo to GitHub — `artifacts/` heavy files are git-ignored except the plot and
   model card template.

---

## Notes & limitations

- A ~5–8 hour Common Voice subset is small; expect a solid relative WER/CER drop but not
  production-grade accuracy. Read speech only.
- fp16 is restricted to CUDA on purpose — MPS/CPU stay in fp32 for stability.
- No experiment-tracker / heavy-framework dependencies by design; the stack is intentionally
  minimal (transformers, peft, datasets, accelerate, evaluate, jiwer, librosa, soundfile).

## License

MIT (code). Common Voice data is CC0; Whisper is MIT.
