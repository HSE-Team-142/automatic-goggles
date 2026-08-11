#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

CONFIG="${CONFIG:-pawn++/experiments/MAGE/configs/pawn/two_models/mage_llama_instruct_llama_base_metrics_xppl_hs_uniform_agg_metrics_full.yaml}"
OUTPUT_DIR="${OUTPUT_DIR:-pawn++/experiments/MAGE/ood/runs/ood_retrain_eval}"
BATCH_SIZE="${BATCH_SIZE:-4}"

DEFAULT_CHECKPOINT="pawn++/experiments/MAGE/cross_domains_cross_models/runs/pawn/two_models/ood_retrain/pytorch_model.bin"
CHECKPOINT="${CHECKPOINT:-$DEFAULT_CHECKPOINT}"

if [[ ! -f "$CHECKPOINT" ]]; then
  shopt -s nullglob
  checkpoints=(pawn++/experiments/MAGE/cross_domains_cross_models/runs/pawn/two_models/ood_retrain/checkpoint-*/pytorch_model.bin)
  shopt -u nullglob

  if (( ${#checkpoints[@]} == 1 )); then
    CHECKPOINT="${checkpoints[0]}"
  elif (( ${#checkpoints[@]} > 1 )); then
    CHECKPOINT="$(printf '%s\n' "${checkpoints[@]}" | sort -V | tail -n 1)"
  else
    echo "Checkpoint not found: $CHECKPOINT" >&2
    echo "Set CHECKPOINT=/path/to/pytorch_model.bin or run scripts/train.sh first." >&2
    exit 1
  fi
fi

DATASETS=("$@")
if (( ${#DATASETS[@]} == 0 )); then
  DATASETS=(
    "pawn++/mage/testbeds/test_ood_gpt_para.csv"
    "pawn++/mage/testbeds/test_ood_gpt.csv"
  )
fi

uv run --python 3.13 python pawn++/run_pawn_ood.py \
  --config "$CONFIG" \
  --checkpoint "$CHECKPOINT" \
  --datasets "${DATASETS[@]}" \
  --output_dir "$OUTPUT_DIR" \
  --batch_size "$BATCH_SIZE"

echo "OOD evaluation output written to: $OUTPUT_DIR"
