#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERIFY_MINIGRID_DIR=".tmp/verify-minigrid"
MINIGRID_EXTRA_CONFIG="${MINIGRID_EXTRA_CONFIG:-}"
MINIGRID_EXTRA_SEED="${MINIGRID_EXTRA_SEED:-201}"

if [[ -n "$MINIGRID_EXTRA_CONFIG" ]]; then
  if [[ ! "$MINIGRID_EXTRA_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
    echo "invalid MINIGRID_EXTRA_CONFIG: $MINIGRID_EXTRA_CONFIG" >&2
    exit 2
  fi
  if [[ ! "$MINIGRID_EXTRA_SEED" =~ ^[0-9]+$ ]]; then
    echo "invalid MINIGRID_EXTRA_SEED: $MINIGRID_EXTRA_SEED" >&2
    exit 2
  fi
fi

case "$VERIFY_MINIGRID_DIR" in
  .tmp/verify-minigrid) rm -rf -- "$VERIFY_MINIGRID_DIR" ;;
  *) echo "refusing to remove unexpected path: $VERIFY_MINIGRID_DIR" >&2; exit 2 ;;
esac

python3 -m baby_model.minigrid_probe --output-dir "$VERIFY_MINIGRID_DIR/probe"
python3 -m baby_model.minigrid_experiment \
  --config configs/experiments/minigrid-smoke.json \
  --output-dir "$VERIFY_MINIGRID_DIR/smoke" \
  --seed 101

if [[ -n "$MINIGRID_EXTRA_CONFIG" ]]; then
  python3 -m baby_model.minigrid_experiment \
    --config "$MINIGRID_EXTRA_CONFIG" \
    --output-dir "$VERIFY_MINIGRID_DIR/extra" \
    --seed "$MINIGRID_EXTRA_SEED"
fi
