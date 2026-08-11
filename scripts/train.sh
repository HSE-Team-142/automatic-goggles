#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

CONFIG="${CONFIG:-pawn++/experiments/MAGE/configs/pawn/two_models/mage_llama_instruct_llama_base_metrics_xppl_hs_uniform_agg_metrics_full.yaml}"
TRAIN_DATASET="${TRAIN_DATASET:-pawn++/mage/testbeds/cross_domains_cross_models/train.csv}"
VALID_DATASET="${VALID_DATASET:-pawn++/mage/testbeds/cross_domains_cross_models/valid.csv}"
TEST_DATASET="${TEST_DATASET:-pawn++/mage/testbeds/cross_domains_cross_models/test.csv}"
OUTPUT_DIR="${OUTPUT_DIR:-pawn++/experiments/MAGE/cross_domains_cross_models/runs/pawn/two_models/ood_retrain}"
REPORT_TO="${REPORT_TO:-tensorboard}"

uv run --python 3.13 python pawn++/train.py \
  --config "$CONFIG" \
  --train_dataset "$TRAIN_DATASET" \
  --valid_dataset "$VALID_DATASET" \
  --test_dataset "$TEST_DATASET" \
  --output_dir "$OUTPUT_DIR" \
  --report_to "$REPORT_TO"

echo "Training output written to: $OUTPUT_DIR"
