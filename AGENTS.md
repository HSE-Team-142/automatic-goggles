# Repository Instructions

## Project Overview

This repository contains PAWN++, a Python/ML project for machine-generated text detection. The main model freezes one or two Hugging Face causal language models, extracts token-level and sequence-level features, and trains lightweight MLP/gating heads. The repo also includes a RoBERTa baseline, MAGE dataset utilities, saved experiment metrics, and a notebook report.

## Environment

- Use Python 3.13.
- Use `uv` for dependency management.
- Preferred setup command:

```bash
uv sync
```

- If sandboxed cache access fails, use a writable cache directory:

```bash
UV_CACHE_DIR=/tmp/uv-cache uv sync
```

## Important Paths

- `pawn++/model.py`: PAWN++ model definition.
- `pawn++/extract_features.py`: frozen LM feature extraction.
- `pawn++/train.py`: PAWN++ training entrypoint.
- `pawn++/inference.py`: PAWN++ inference entrypoint.
- `pawn++/run_pawn_ood.py`: PAWN++ OOD evaluation.
- `pawn++/bert_baseline/train_bert.py`: RoBERTa baseline training.
- `pawn++/bert_baseline/run_bert_ood.py`: RoBERTa OOD evaluation.
- `pawn++/configs.py`: Pydantic experiment config models.
- `pawn++/experiments/`: YAML configs and saved metrics.
- `report.ipynb`: experiment report and result tables.

## Commands

Run PAWN++ training:

```bash
uv run pawn++/train.py \
  --config pawn++/experiments/MAGE/configs/pawn/single_model/mage_llama_instruct.yaml \
  --train_dataset path/to/train.csv \
  --valid_dataset path/to/valid.csv \
  --test_dataset path/to/test.csv \
  --output_dir output/pawn \
  --report_to tensorboard
```

Run RoBERTa baseline training:

```bash
PYTHONPATH=pawn++ uv run pawn++/bert_baseline/train_bert.py \
  --base_model FacebookAI/roberta-base \
  --train_dataset path/to/train.csv \
  --valid_dataset path/to/valid.csv \
  --test_dataset path/to/test.csv \
  --output_dir output/roberta \
  --label_smoothing 0.2 \
  --pos_weight 0.413
```

Run a quick syntax check:

```bash
uv run --python 3.13 python -m py_compile \
  pawn++/configs.py \
  pawn++/mlp.py \
  pawn++/dataset_module.py \
  pawn++/extract_features.py \
  pawn++/model.py \
  pawn++/train.py \
  pawn++/inference.py \
  pawn++/run_pawn_ood.py \
  pawn++/bert_baseline/train_bert.py \
  pawn++/bert_baseline/run_bert_ood.py \
  pawn++/dataset/sample_dataset.py \
  pawn++/dataset/MAGE/prepare_testbeds.py
```

## Development Notes

- The source directory is `pawn++`. Use lowercase `pawn++/...` in commands for Linux compatibility.
- Many configs use gated Hugging Face models such as Llama. Expect to need `HF_TOKEN` in `.env`.
- Dataset CSVs are expected to contain `text` and `label` columns.
- The MAGE preparation utility documents the project convention as `0 = human-written` and `1 = machine-generated`.
- `pawn++/checkpoint-39884/` and other checkpoint artifacts can be large; avoid modifying or committing generated model artifacts unless explicitly requested.
- The working tree may contain user-created untracked files. Do not delete or revert them unless explicitly asked.

## Code Style

- Keep edits narrowly scoped.
- Prefer existing patterns over new abstractions.
- Use `polars` for CSV/dataframe work where the project already does.
- Use Hugging Face `Trainer` conventions for training changes.
- Avoid adding network-dependent tests for model downloads; prefer config parsing, metric logic, and small unit-level checks when possible.
