#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODE="${MODE:-both}"
SEEDS="${SEEDS:-101,102,103,104,105}"
CONFIG="${CONFIG:-configs/experiments/v02-sweep.json}"
RUN_ID="${RUN_ID:-baby-model-fleet-$(date -u +%Y%m%dT%H%M%SZ)-$(git -C "$ROOT" rev-parse --short HEAD)}"
SESSION="${SESSION:-$RUN_ID}"

usage() {
  cat >&2 <<'EOF'
Usage:
  MODE=both SEEDS=101,102,103 ./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
  MODE=both CONFIG=configs/experiments/v03-sweep.json ./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
  BABY_MODEL_FLEET_HOSTS="mac:host-a wsl:host-b" ./scripts/fleet_archive_run.sh

MODE values:
  verify  run ./scripts/verify.sh
  sweep   run ./scripts/run_beta_sweep.sh
  both    run verify, then sweep
EOF
}

case "$MODE" in
  verify|sweep|both) ;;
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
