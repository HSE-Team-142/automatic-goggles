# PAWN++

PAWN++ is an extension of the [PAWN](https://www.sciencedirect.com/science/article/pii/S156625352500538X?ref=pdf_download&fr=RR-2&rr=a0595b571864e4a0) detector for identifying machine-generated text. It adds an optional second frozen language model, cross-model metrics, second-model token metrics, hidden-state fusion, and aggregated sequence-level features. The repository also includes a RoBERTa baseline and MAGE evaluation utilities.

## Setup

Python 3.14 and `uv`:

```bash
uv sync
```

## Trained Model

A trained PAWN++ checkpoint is available on Hugging Face:

https://huggingface.co/crayden/pawnplus

## Train PAWN++

```bash
uv run pawn++/train.py \
  --config pawn++/experiments/MAGE/configs/pawn/single_model/mage_llama_instruct.yaml \
  --train_dataset path/to/train.csv \
  --valid_dataset path/to/valid.csv \
  --test_dataset path/to/test.csv \
  --output_dir output/pawn \
  --report_to tensorboard
```

Experiment configurations are stored under `pawn++/experiments/`. Training selects the best checkpoint by validation AUROC and writes test metrics to `test_metrics.json`.

## Train RoBERTa Baseline

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

## OOD Evaluation

PAWN++:

```bash
uv run pawn++/run_pawn_ood.py \
  --config path/to/config.yaml \
  --checkpoint path/to/pytorch_model.bin \
  --datasets path/to/test_ood.csv \
  --output_dir output/ood
```

RoBERTa:

```bash
PYTHONPATH=pawn++ uv run pawn++/bert_baseline/run_bert_ood.py \
  --checkpoint path/to/checkpoint \
  --base_model FacebookAI/roberta-base \
  --datasets path/to/test_ood.csv \
  --output_dir output/bert_ood
```

## Structure

```text
pawn++/                         PAWN++ implementation and utilities
pawn++/bert_baseline/           RoBERTa training and OOD evaluation
pawn++/experiments/             Experiment configs and saved metrics
report.ipynb                    Experiment report and result tables
```
