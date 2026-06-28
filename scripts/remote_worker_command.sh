#!/usr/bin/env bash
set -euo pipefail

ROOT="${1:-$HOME/work/baby-model}"
ITERATIONS="${ITERATIONS:-3}"

cat <<EOF
cd "$ROOT" && git status --short && ./scripts/verify.sh && SESSION=baby-model ITERATIONS=$ITERATIONS ./scripts/launch_tmux_local.sh
EOF

