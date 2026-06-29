#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERIFY_MINIGRID_DIR=".tmp/verify-minigrid"
MINIGRID_EXTRA_CONFIG="${MINIGRID_EXTRA_CONFIG:-}"
MINIGRID_EXTRA_SEED="${MINIGRID_EXTRA_SEED:-201}"
MINIGRID_CURRICULUM_CONFIG="${MINIGRID_CURRICULUM_CONFIG:-}"
MINIGRID_CURRICULUM_SEED="${MINIGRID_CURRICULUM_SEED:-301}"
MINIGRID_LINEAR_CONFIG="${MINIGRID_LINEAR_CONFIG:-}"
MINIGRID_LINEAR_SEED="${MINIGRID_LINEAR_SEED:-401}"
MINIGRID_LINEAR_SWEEP_CONFIG="${MINIGRID_LINEAR_SWEEP_CONFIG:-}"
MINIGRID_LINEAR_SWEEP_SEEDS="${MINIGRID_LINEAR_SWEEP_SEEDS:-401,402,403}"
MINIGRID_NEURAL_CONFIG="${MINIGRID_NEURAL_CONFIG:-}"
MINIGRID_NEURAL_SEED="${MINIGRID_NEURAL_SEED:-501}"
MINIGRID_TORCH_CONFIG="${MINIGRID_TORCH_CONFIG:-}"
MINIGRID_TORCH_SEED="${MINIGRID_TORCH_SEED:-601}"
MINIGRID_TORCH_SWEEP_CONFIG="${MINIGRID_TORCH_SWEEP_CONFIG:-}"
MINIGRID_TORCH_SWEEP_SEEDS="${MINIGRID_TORCH_SWEEP_SEEDS:-601,602,603}"
MINIGRID_TORCH_DEVICE="${MINIGRID_TORCH_DEVICE:-auto}"

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

if [[ -n "$MINIGRID_CURRICULUM_CONFIG" ]]; then
  if [[ ! "$MINIGRID_CURRICULUM_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
    echo "invalid MINIGRID_CURRICULUM_CONFIG: $MINIGRID_CURRICULUM_CONFIG" >&2
    exit 2
  fi
  if [[ ! "$MINIGRID_CURRICULUM_SEED" =~ ^[0-9]+$ ]]; then
    echo "invalid MINIGRID_CURRICULUM_SEED: $MINIGRID_CURRICULUM_SEED" >&2
    exit 2
  fi
fi

if [[ -n "$MINIGRID_LINEAR_CONFIG" ]]; then
  if [[ ! "$MINIGRID_LINEAR_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
    echo "invalid MINIGRID_LINEAR_CONFIG: $MINIGRID_LINEAR_CONFIG" >&2
    exit 2
  fi
  if [[ ! "$MINIGRID_LINEAR_SEED" =~ ^[0-9]+$ ]]; then
    echo "invalid MINIGRID_LINEAR_SEED: $MINIGRID_LINEAR_SEED" >&2
    exit 2
  fi
fi

if [[ -n "$MINIGRID_LINEAR_SWEEP_CONFIG" ]]; then
  if [[ ! "$MINIGRID_LINEAR_SWEEP_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
    echo "invalid MINIGRID_LINEAR_SWEEP_CONFIG: $MINIGRID_LINEAR_SWEEP_CONFIG" >&2
    exit 2
  fi
  if [[ ! "$MINIGRID_LINEAR_SWEEP_SEEDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "invalid MINIGRID_LINEAR_SWEEP_SEEDS: $MINIGRID_LINEAR_SWEEP_SEEDS" >&2
    exit 2
  fi
fi

if [[ -n "$MINIGRID_NEURAL_CONFIG" ]]; then
  if [[ ! "$MINIGRID_NEURAL_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
    echo "invalid MINIGRID_NEURAL_CONFIG: $MINIGRID_NEURAL_CONFIG" >&2
    exit 2
  fi
  if [[ ! "$MINIGRID_NEURAL_SEED" =~ ^[0-9]+$ ]]; then
    echo "invalid MINIGRID_NEURAL_SEED: $MINIGRID_NEURAL_SEED" >&2
    exit 2
  fi
fi

if [[ -n "$MINIGRID_TORCH_CONFIG" ]]; then
  if [[ ! "$MINIGRID_TORCH_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
    echo "invalid MINIGRID_TORCH_CONFIG: $MINIGRID_TORCH_CONFIG" >&2
    exit 2
  fi
  if [[ ! "$MINIGRID_TORCH_SEED" =~ ^[0-9]+$ ]]; then
    echo "invalid MINIGRID_TORCH_SEED: $MINIGRID_TORCH_SEED" >&2
    exit 2
  fi
  if [[ ! "$MINIGRID_TORCH_DEVICE" =~ ^(auto|cpu|cuda|cuda:[0-9]+|mps)$ ]]; then
    echo "invalid MINIGRID_TORCH_DEVICE: $MINIGRID_TORCH_DEVICE" >&2
    exit 2
  fi
fi

if [[ -n "$MINIGRID_TORCH_SWEEP_CONFIG" ]]; then
  if [[ ! "$MINIGRID_TORCH_SWEEP_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
    echo "invalid MINIGRID_TORCH_SWEEP_CONFIG: $MINIGRID_TORCH_SWEEP_CONFIG" >&2
    exit 2
  fi
  if [[ ! "$MINIGRID_TORCH_SWEEP_SEEDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "invalid MINIGRID_TORCH_SWEEP_SEEDS: $MINIGRID_TORCH_SWEEP_SEEDS" >&2
    exit 2
  fi
  if [[ ! "$MINIGRID_TORCH_DEVICE" =~ ^(auto|cpu|cuda|cuda:[0-9]+|mps)$ ]]; then
    echo "invalid MINIGRID_TORCH_DEVICE: $MINIGRID_TORCH_DEVICE" >&2
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

if [[ -n "$MINIGRID_CURRICULUM_CONFIG" ]]; then
  python3 -m baby_model.minigrid_curriculum \
    --config "$MINIGRID_CURRICULUM_CONFIG" \
    --output-dir "$VERIFY_MINIGRID_DIR/curriculum" \
    --seed "$MINIGRID_CURRICULUM_SEED"
fi

if [[ -n "$MINIGRID_LINEAR_CONFIG" ]]; then
  python3 -m baby_model.minigrid_linear \
    --config "$MINIGRID_LINEAR_CONFIG" \
    --output-dir "$VERIFY_MINIGRID_DIR/linear" \
    --seed "$MINIGRID_LINEAR_SEED"
fi

if [[ -n "$MINIGRID_LINEAR_SWEEP_CONFIG" ]]; then
  python3 -m baby_model.minigrid_linear_sweep \
    --config "$MINIGRID_LINEAR_SWEEP_CONFIG" \
    --output-dir "$VERIFY_MINIGRID_DIR/linear-sweep" \
    --seeds "$MINIGRID_LINEAR_SWEEP_SEEDS"
fi

if [[ -n "$MINIGRID_NEURAL_CONFIG" ]]; then
  python3 -m baby_model.minigrid_neural \
    --config "$MINIGRID_NEURAL_CONFIG" \
    --output-dir "$VERIFY_MINIGRID_DIR/neural" \
    --seed "$MINIGRID_NEURAL_SEED"
fi

if [[ -n "$MINIGRID_TORCH_CONFIG" ]]; then
  python3 -m baby_model.minigrid_torch \
    --config "$MINIGRID_TORCH_CONFIG" \
    --output-dir "$VERIFY_MINIGRID_DIR/torch" \
    --seed "$MINIGRID_TORCH_SEED" \
    --device "$MINIGRID_TORCH_DEVICE"
fi

if [[ -n "$MINIGRID_TORCH_SWEEP_CONFIG" ]]; then
  python3 -m baby_model.minigrid_torch_sweep \
    --config "$MINIGRID_TORCH_SWEEP_CONFIG" \
    --output-dir "$VERIFY_MINIGRID_DIR/torch-sweep" \
    --seeds "$MINIGRID_TORCH_SWEEP_SEEDS" \
    --device "$MINIGRID_TORCH_DEVICE"
fi
