#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ITERATIONS=3
SLEEP_SECONDS=5

while [[ $# -gt 0 ]]; do
  case "$1" in
    --iterations)
      ITERATIONS="$2"
      shift 2
      ;;
    --sleep)
      SLEEP_SECONDS="$2"
      shift 2
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

cd "$ROOT"
mkdir -p runs/local-loop

for i in $(seq 1 "$ITERATIONS"); do
  seed=$((100 + i))
  echo "loop=$i seed=$seed started=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  python3 -m baby_model.cli run \
    --config configs/experiments/v0-smoke.json \
    --output-dir runs/local-loop \
    --seed "$seed"
  echo "loop=$i done=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  if [[ "$i" != "$ITERATIONS" ]]; then
    sleep "$SLEEP_SECONDS"
  fi
done

