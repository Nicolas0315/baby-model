#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ "$#" -gt 0 ]]; then
  HOSTS=("$@")
elif [[ -n "${BABY_MODEL_FLEET_HOSTS:-}" ]]; then
  read -r -a HOSTS <<< "$BABY_MODEL_FLEET_HOSTS"
elif [[ -f "$ROOT/.local/fleet-hosts" ]]; then
  mapfile -t HOSTS < "$ROOT/.local/fleet-hosts"
else
  cat >&2 <<'EOF'
No fleet hosts provided.

Use one of:
  ./scripts/fleet_inventory.sh mac:host-a wsl:host-b
  BABY_MODEL_FLEET_HOSTS="mac:host-a wsl:host-b" ./scripts/fleet_inventory.sh
  cp configs/fleet/hosts.example.txt .local/fleet-hosts

The .local/fleet-hosts path is git-ignored so private fleet names do not enter
GitHub by default.
EOF
  exit 2
fi

for entry in "${HOSTS[@]}"; do
  if [[ "$entry" == *:* ]]; then
    kind="${entry%%:*}"
    host="${entry#*:}"
  else
    kind="wsl"
    host="$entry"
  fi
  printf '## %s\n' "$host"
  if [[ "$kind" == "mac" ]]; then
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      'hostname; uname -a; tmux -V 2>/dev/null || true; sysctl -n hw.memsize 2>/dev/null || true'
  elif [[ "$kind" == "wsl" ]]; then
    ssh -o BatchMode=yes -o ConnectTimeout=8 "$host" \
      'hostname; wsl.exe -l -v; wsl.exe -e bash -lc "tmux -V 2>/dev/null || true; nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || true"'
  else
    echo "unknown host kind '$kind' for '$entry'; expected mac:host or wsl:host" >&2
    exit 2
  fi
  printf '\n'
done
