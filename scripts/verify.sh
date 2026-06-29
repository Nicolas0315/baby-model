#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

python3 -m compileall -q baby_model tests
python3 -m unittest discover -s tests -p 'test_*.py'
python3 -m baby_model.cli verify-config configs/experiments/v0-smoke.json
python3 -m baby_model.cli verify-config configs/experiments/v02-sweep.json
python3 -m baby_model.cli verify-config configs/experiments/v03-sweep.json
python3 -m json.tool configs/experiments/minigrid-torch-unlock-smoke.json >/dev/null
python3 -m json.tool configs/experiments/minigrid-torch-adda-v11.json >/dev/null
python3 -m json.tool configs/experiments/minigrid-torch-adda-v12.json >/dev/null
python3 -m json.tool configs/experiments/minigrid-torch-adda-v13.json >/dev/null
python3 -m json.tool configs/experiments/minigrid-torch-adda-v14.json >/dev/null
python3 -m json.tool configs/fleet/gpu-wheel-policy.json >/dev/null
python3 -m baby_model.gpu_compat --config configs/fleet/gpu-wheel-policy.json >/dev/null
bash -n scripts/*.sh
MINIGRID_SETUP_DRY_RUN=1 \
MINIGRID_ENV_BACKEND=venv \
MINIGRID_PYTHON=python3 \
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
MINIGRID_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
MINIGRID_TORCH_CPU_FALLBACK=1 \
MINIGRID_TORCH_INSTALLER=pip \
MINIGRID_ENV_CLEAR=1 \
./scripts/setup_minigrid_env.sh >/dev/null
MINIGRID_SETUP_DRY_RUN=1 \
MINIGRID_ENV_BACKEND=uv \
MINIGRID_PYTHON=python3 \
MINIGRID_TORCH_SWEEP_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
MINIGRID_TORCH_INSTALLER=pip \
MINIGRID_ENV_CLEAR=1 \
./scripts/setup_minigrid_env.sh >/dev/null
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
