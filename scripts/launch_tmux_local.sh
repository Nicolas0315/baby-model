#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SESSION="${SESSION:-baby-model}"
ITERATIONS="${ITERATIONS:-3}"

if tmux has-session -t "$SESSION" 2>/dev/null; then
  echo "session_exists=$SESSION"
  if ! tmux list-windows -t "$SESSION" -F '#{window_name}' | grep -qx 'monitor'; then
    tmux new-window -t "$SESSION:" -n "monitor" -c "$ROOT" \
      "watch -n 5 'find runs/local-loop -maxdepth 2 -name summary.md | sort | tail -5; echo; git status --short'; exec bash"
  fi
  tmux list-panes -t "$SESSION" -F '#{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} cmd=#{pane_current_command} cwd=#{pane_current_path}'
  exit 0
fi

tmux new-session -d -s "$SESSION" -n "local-loop" -c "$ROOT" \
  "bash scripts/run_local_loop.sh --iterations '$ITERATIONS'; echo done; exec bash"
tmux new-window -t "$SESSION:" -n "monitor" -c "$ROOT" \
  "watch -n 5 'find runs/local-loop -maxdepth 2 -name summary.md | sort | tail -5; echo; git status --short'; exec bash"

echo "session_started=$SESSION"
tmux list-panes -t "$SESSION" -F '#{session_name}:#{window_index}.#{pane_index} pid=#{pane_pid} cmd=#{pane_current_command} cwd=#{pane_current_path}'
