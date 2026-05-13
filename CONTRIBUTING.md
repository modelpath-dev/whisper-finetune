# Contributing

Thanks for your interest in the project. A few notes to keep things consistent.

## Development setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
pip install -e .
```

## Before opening a PR

- Run the unit tests: `make test` (or `pytest`).
- Run the smoke test if you touched the data/train/eval path: `make smoke`.
- Lint your changes: `make lint`.
- Keep functions type-hinted and add a short docstring for new public functions.
- Update `CHANGELOG.md` under **Unreleased**.

## Commit style

Small, focused commits with a short imperative subject line
(e.g. `Add seed override to training script`).
