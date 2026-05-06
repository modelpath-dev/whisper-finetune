.PHONY: install dev test smoke train eval plot clean

CONFIG ?= configs/whisper_small_hi.yaml
ADAPTER ?= artifacts/whisper-small-hi-lora

install:  ## Install pinned dependencies
	pip install --upgrade pip
	pip install -r requirements.txt

dev: install  ## Editable install + dev extras
	pip install -e ".[dev]"

test:  ## Run fast offline unit tests
	pytest

smoke:  ## Run the full pipeline on a tiny subset (no GPU needed)
	python scripts/prepare_data.py --config $(CONFIG) --smoke
	python scripts/run_train.py --config $(CONFIG) --smoke
	python scripts/run_eval.py --config $(CONFIG) \
		--adapter artifacts/smoke-whisper-small-hi-lora --smoke

train:  ## Full LoRA fine-tuning run
	python scripts/run_train.py --config $(CONFIG)

eval:  ## Evaluate baseline vs fine-tuned on the test split
	python scripts/run_eval.py --config $(CONFIG) --adapter $(ADAPTER)

plot:  ## Render the WER/CER comparison chart
	python scripts/plot_results.py

clean:  ## Remove caches and Python build artifacts
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache *.egg-info build dist
