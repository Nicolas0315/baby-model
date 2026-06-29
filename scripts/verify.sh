#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m compileall -q baby_model tests
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m baby_model.cli verify-config configs/experiments/v0-smoke.json
python3 -m baby_model.cli verify-config configs/experiments/v02-sweep.json
python3 -m baby_model.cli verify-config configs/experiments/v03-sweep.json
VERIFY_RUN_DIR=".tmp/verify-run"
case "$VERIFY_RUN_DIR" in
  .tmp/verify-run) rm -rf -- "$VERIFY_RUN_DIR" ;;
  *) echo "refusing to remove unexpected path: $VERIFY_RUN_DIR" >&2; exit 2 ;;
esac
python3 -m baby_model.cli run --config configs/experiments/v0-smoke.json --output-dir .tmp/verify-run --seed 11
VERIFY_SWEEP_DIR=".tmp/verify-sweep"
case "$VERIFY_SWEEP_DIR" in
  .tmp/verify-sweep) rm -rf -- "$VERIFY_SWEEP_DIR" ;;
  *) echo "refusing to remove unexpected path: $VERIFY_SWEEP_DIR" >&2; exit 2 ;;
esac
python3 -m baby_model.cli sweep --config configs/experiments/v02-sweep.json --output-dir "$VERIFY_SWEEP_DIR" --seeds 1,2
python3 -m baby_model.cli sweep --config configs/experiments/v03-sweep.json --output-dir "$VERIFY_SWEEP_DIR/v03" --seeds 1,2

echo "verify ok"
