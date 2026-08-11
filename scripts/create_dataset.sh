#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

export UV_CACHE_DIR="${UV_CACHE_DIR:-/tmp/uv-cache}"

DATA_DIR="${DATA_DIR:-pawn++/mage/testbeds}"

uv run --python 3.13 python pawn++/dataset/MAGE/prepare_testbeds.py "$DATA_DIR"

echo "MAGE testbeds written to: $DATA_DIR"
