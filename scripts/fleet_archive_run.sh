#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-both}"
SEEDS="${SEEDS:-101,102,103,104,105}"
CONFIG="${CONFIG:-configs/experiments/v02-sweep.json}"
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
MINIGRID_TORCH_DEVICE="${MINIGRID_TORCH_DEVICE:-auto}"
MINIGRID_TORCH_INDEX_URL="${MINIGRID_TORCH_INDEX_URL:-}"
RUN_ID="${RUN_ID:-baby-model-fleet-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "$ROOT" rev-parse --short HEAD)}"
SESSION="${SESSION:-$RUN_ID}"

usage() {
  cat >&2 <<'EOF'
Usage:
  MODE=both SEEDS=101,102,103 ./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
  MODE=both CONFIG=configs/experiments/v03-sweep.json ./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
  MODE=minigrid MINIGRID_EXTRA_CONFIG=configs/experiments/minigrid-babyai-unlock.json ./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
  MODE=minigrid MINIGRID_CURRICULUM_CONFIG=configs/experiments/minigrid-curriculum-unlock.json ./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
  MODE=minigrid MINIGRID_LINEAR_CONFIG=configs/experiments/minigrid-linear-unlock.json ./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
  MODE=minigrid MINIGRID_LINEAR_SWEEP_CONFIG=configs/experiments/minigrid-linear-unlock.json MINIGRID_LINEAR_SWEEP_SEEDS=401,402,403 ./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
  MODE=minigrid MINIGRID_NEURAL_CONFIG=configs/experiments/minigrid-neural-unlock.json ./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
  MODE=minigrid MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json MINIGRID_TORCH_DEVICE=auto ./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
  BABY_MODEL_FLEET_HOSTS="mac:host-a wsl:host-b" ./scripts/fleet_archive_run.sh

MODE values:
  verify  run ./scripts/verify.sh
  sweep   run ./scripts/run_beta_sweep.sh
  minigrid create optional MiniGrid venv, then run ./scripts/verify_minigrid.sh
  both    run verify, then sweep
EOF
}

case "$MODE" in
  verify|sweep|minigrid|both) ;;
  *) echo "invalid MODE: $MODE" >&2; usage; exit 2 ;;
esac

if [[ ! "$SEEDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "invalid SEEDS: $SEEDS" >&2
  exit 2
fi

if [[ ! "$CONFIG" =~ ^[A-Za-z0-9._/-]+$ ]]; then
  echo "invalid CONFIG: $CONFIG" >&2
  exit 2
fi

if [[ -n "$MINIGRID_EXTRA_CONFIG" && ! "$MINIGRID_EXTRA_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
  echo "invalid MINIGRID_EXTRA_CONFIG: $MINIGRID_EXTRA_CONFIG" >&2
  exit 2
fi

if [[ ! "$MINIGRID_EXTRA_SEED" =~ ^[0-9]+$ ]]; then
  echo "invalid MINIGRID_EXTRA_SEED: $MINIGRID_EXTRA_SEED" >&2
  exit 2
fi

if [[ -n "$MINIGRID_CURRICULUM_CONFIG" && ! "$MINIGRID_CURRICULUM_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
  echo "invalid MINIGRID_CURRICULUM_CONFIG: $MINIGRID_CURRICULUM_CONFIG" >&2
  exit 2
fi

if [[ ! "$MINIGRID_CURRICULUM_SEED" =~ ^[0-9]+$ ]]; then
  echo "invalid MINIGRID_CURRICULUM_SEED: $MINIGRID_CURRICULUM_SEED" >&2
  exit 2
fi

if [[ -n "$MINIGRID_LINEAR_CONFIG" && ! "$MINIGRID_LINEAR_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
  echo "invalid MINIGRID_LINEAR_CONFIG: $MINIGRID_LINEAR_CONFIG" >&2
  exit 2
fi

if [[ ! "$MINIGRID_LINEAR_SEED" =~ ^[0-9]+$ ]]; then
  echo "invalid MINIGRID_LINEAR_SEED: $MINIGRID_LINEAR_SEED" >&2
  exit 2
fi

if [[ -n "$MINIGRID_LINEAR_SWEEP_CONFIG" && ! "$MINIGRID_LINEAR_SWEEP_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
  echo "invalid MINIGRID_LINEAR_SWEEP_CONFIG: $MINIGRID_LINEAR_SWEEP_CONFIG" >&2
  exit 2
fi

if [[ ! "$MINIGRID_LINEAR_SWEEP_SEEDS" =~ ^[0-9]+(,[0-9]+)*$ ]]; then
  echo "invalid MINIGRID_LINEAR_SWEEP_SEEDS: $MINIGRID_LINEAR_SWEEP_SEEDS" >&2
  exit 2
fi

if [[ -n "$MINIGRID_NEURAL_CONFIG" && ! "$MINIGRID_NEURAL_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
  echo "invalid MINIGRID_NEURAL_CONFIG: $MINIGRID_NEURAL_CONFIG" >&2
  exit 2
fi

if [[ ! "$MINIGRID_NEURAL_SEED" =~ ^[0-9]+$ ]]; then
  echo "invalid MINIGRID_NEURAL_SEED: $MINIGRID_NEURAL_SEED" >&2
  exit 2
fi

if [[ -n "$MINIGRID_TORCH_CONFIG" && ! "$MINIGRID_TORCH_CONFIG" =~ ^configs/experiments/[A-Za-z0-9._-]+\.json$ ]]; then
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

if [[ -n "$MINIGRID_TORCH_INDEX_URL" && ! "$MINIGRID_TORCH_INDEX_URL" =~ ^https://download\.pytorch\.org/whl/[A-Za-z0-9._/-]+$ ]]; then
  echo "invalid MINIGRID_TORCH_INDEX_URL: $MINIGRID_TORCH_INDEX_URL" >&2
  exit 2
fi

if [[ ! "$RUN_ID" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid RUN_ID: $RUN_ID" >&2
  exit 2
fi

if [[ ! "$SESSION" =~ ^[A-Za-z0-9._-]+$ ]]; then
  echo "invalid SESSION: $SESSION" >&2
  exit 2
fi

if [[ "$#" -gt 0 ]]; then
  HOSTS=("$@")
elif [[ -n "${BABY_MODEL_FLEET_HOSTS:-}" ]]; then
  read -r -a HOSTS <<< "$BABY_MODEL_FLEET_HOSTS"
elif [[ -f "$ROOT/.local/fleet-hosts" ]]; then
  mapfile -t HOSTS < "$ROOT/.local/fleet-hosts"
else
  usage
  exit 2
fi

case "$MODE" in
  verify)
    JOB_CMD='./scripts/verify.sh; status=$?; echo exit=$status; exec bash'
    ;;
  sweep)
    JOB_CMD="CONFIG=$CONFIG SEEDS=$SEEDS OUTPUT_DIR=runs/fleet-sweeps ./scripts/run_beta_sweep.sh; status=\$?; echo exit=\$status; exec bash"
    ;;
  minigrid)
    MINIGRID_ENV=""
    if [[ -n "$MINIGRID_EXTRA_CONFIG" ]]; then
      MINIGRID_ENV="MINIGRID_EXTRA_CONFIG=$(printf '%q' "$MINIGRID_EXTRA_CONFIG") MINIGRID_EXTRA_SEED=$(printf '%q' "$MINIGRID_EXTRA_SEED") "
    fi
    if [[ -n "$MINIGRID_CURRICULUM_CONFIG" ]]; then
      MINIGRID_ENV="${MINIGRID_ENV}MINIGRID_CURRICULUM_CONFIG=$(printf '%q' "$MINIGRID_CURRICULUM_CONFIG") MINIGRID_CURRICULUM_SEED=$(printf '%q' "$MINIGRID_CURRICULUM_SEED") "
    fi
    if [[ -n "$MINIGRID_LINEAR_CONFIG" ]]; then
      MINIGRID_ENV="${MINIGRID_ENV}MINIGRID_LINEAR_CONFIG=$(printf '%q' "$MINIGRID_LINEAR_CONFIG") MINIGRID_LINEAR_SEED=$(printf '%q' "$MINIGRID_LINEAR_SEED") "
    fi
    if [[ -n "$MINIGRID_LINEAR_SWEEP_CONFIG" ]]; then
      MINIGRID_ENV="${MINIGRID_ENV}MINIGRID_LINEAR_SWEEP_CONFIG=$(printf '%q' "$MINIGRID_LINEAR_SWEEP_CONFIG") MINIGRID_LINEAR_SWEEP_SEEDS=$(printf '%q' "$MINIGRID_LINEAR_SWEEP_SEEDS") "
    fi
    if [[ -n "$MINIGRID_NEURAL_CONFIG" ]]; then
      MINIGRID_ENV="${MINIGRID_ENV}MINIGRID_NEURAL_CONFIG=$(printf '%q' "$MINIGRID_NEURAL_CONFIG") MINIGRID_NEURAL_SEED=$(printf '%q' "$MINIGRID_NEURAL_SEED") "
    fi
    if [[ -n "$MINIGRID_TORCH_CONFIG" ]]; then
      MINIGRID_ENV="${MINIGRID_ENV}MINIGRID_TORCH_CONFIG=$(printf '%q' "$MINIGRID_TORCH_CONFIG") MINIGRID_TORCH_SEED=$(printf '%q' "$MINIGRID_TORCH_SEED") MINIGRID_TORCH_DEVICE=$(printf '%q' "$MINIGRID_TORCH_DEVICE") "
      if [[ -n "$MINIGRID_TORCH_INDEX_URL" ]]; then
        TORCH_INSTALL="python -m pip install --index-url $(printf '%q' "$MINIGRID_TORCH_INDEX_URL") torch"
      else
        TORCH_INSTALL="python -m pip install torch"
      fi
      JOB_CMD="PYTHON_BIN=; for candidate in python3.12 python3.13 python3.11 python3.10 python3; do if command -v \$candidate >/dev/null 2>&1; then PYTHON_BIN=\$candidate; break; fi; done; if [[ -z \$PYTHON_BIN ]]; then echo no_supported_python >&2; exit 2; fi; echo python_bin=\$PYTHON_BIN; \$PYTHON_BIN -m venv .venv-minigrid && . .venv-minigrid/bin/activate && python -m pip install --upgrade pip && python -m pip install minigrid && $TORCH_INSTALL && ${MINIGRID_ENV}./scripts/verify_minigrid.sh; status=\$?; echo exit=\$status; exec bash"
    else
      JOB_CMD="if command -v uv >/dev/null 2>&1; then ${MINIGRID_ENV}uv run --with minigrid bash scripts/verify_minigrid.sh; else python3 -m venv .venv-minigrid && . .venv-minigrid/bin/activate && python -m pip install --upgrade pip && python -m pip install minigrid && ${MINIGRID_ENV}./scripts/verify_minigrid.sh; fi; status=\$?; echo exit=\$status; exec bash"
    fi
    ;;
  both)
    JOB_CMD="./scripts/verify.sh && CONFIG=$CONFIG SEEDS=$SEEDS OUTPUT_DIR=runs/fleet-sweeps ./scripts/run_beta_sweep.sh; status=\$?; echo exit=\$status; exec bash"
    ;;
esac

run_remote() {
  local kind="$1"
  local host="$2"
  local session_q mode_q job_q run_id_q
  session_q="$(printf '%q' "$SESSION")"
  mode_q="$(printf '%q' "$MODE")"
  job_q="$(printf '%q' "$JOB_CMD")"
  run_id_q="$(printf '%q' "$RUN_ID")"

  local remote_command
  remote_command=$(cat <<EOF
set -euo pipefail
archive="/tmp/${RUN_ID}.tar"
run_dir="\$HOME/work/baby-model-fleet/${RUN_ID}"
mkdir -p "\$run_dir"
cat > "\$archive"
tar -xf "\$archive" -C "\$run_dir"
rm -f "\$archive"
cd "\$run_dir"
echo run_id=$run_id_q
echo commit=$(git -C "$ROOT" rev-parse HEAD)
if tmux has-session -t $session_q 2>/dev/null; then
  echo session_exists=$session_q
else
  tmux new-session -d -s $session_q -n $mode_q $job_q
  echo session_started=$session_q
fi
tmux list-panes -t $session_q -F '#{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} cmd=#{pane_current_command} cwd=#{pane_current_path}'
EOF
)

  printf '## %s\n' "$host"
  if [[ "$kind" == "mac" ]]; then
    git -C "$ROOT" archive HEAD | ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "bash -lc $(printf '%q' "$remote_command")"
  elif [[ "$kind" == "wsl" ]]; then
    git -C "$ROOT" archive HEAD | ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      "wsl.exe -d Ubuntu -e bash -lc $(printf '%q' "$remote_command")"
  else
    echo "unknown host kind '$kind' for '$host'; expected mac:host or wsl:host" >&2
    exit 2
  fi
  printf '\n'
}

for entry in "${HOSTS[@]}"; do
  [[ -z "$entry" || "$entry" == \#* ]] && continue
  if [[ "$entry" == *:* ]]; then
    kind="${entry%%:*}"
    host="${entry#*:}"
  else
    kind="wsl"
    host="$entry"
  fi
  run_remote "$kind" "$host"
done
