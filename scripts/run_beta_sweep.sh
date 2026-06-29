#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SEEDS="${SEEDS:-101,102,103,104,105}"
OUTPUT_DIR="${OUTPUT_DIR:-runs/sweeps}"

cd "$ROOT"
python3 -m baby_model.cli sweep \
  --config configs/experiments/v02-sweep.json \
  --output-dir "$OUTPUT_DIR" \
  --seeds "$SEEDS"
