#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VERIFY_MINIGRID_DIR=".tmp/verify-minigrid"
case "$VERIFY_MINIGRID_DIR" in
  .tmp/verify-minigrid) rm -rf -- "$VERIFY_MINIGRID_DIR" ;;
  *) echo "refusing to remove unexpected path: $VERIFY_MINIGRID_DIR" >&2; exit 2 ;;
esac

python3 -m baby_model.minigrid_probe --output-dir "$VERIFY_MINIGRID_DIR"
