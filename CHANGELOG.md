# Changelog

All notable changes to this project are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and this project adheres
to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- LoRA fine-tuning pipeline for Whisper-small on Common Voice 17 (Hindi).
- Typed YAML configuration with fail-fast validation.
- Data preprocessing (16 kHz resample, duration/transcript filtering).
- Training and evaluation entrypoints with WER/CER metrics.
- One-click Colab notebook and CPU smoke tests.
