#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TEAM="${AGMSG_TEAM:-}"
FROM="${AGMSG_FROM:-codex-lead}"
TO="${1:-codex-lead}"
MESSAGE="${2:-baby-model progress update}"

if [[ -z "$TEAM" && -f "$ROOT/.local/agmsg-team" ]]; then
  TEAM="$(<"$ROOT/.local/agmsg-team")"
fi

if [[ -z "$TEAM" ]]; then
  cat >&2 <<'EOF'
No agmsg team configured.

Use one of:
  AGMSG_TEAM="team-name" ./scripts/agmsg_progress.sh <to> "message"
  printf '%s\n' "team-name" > .local/agmsg-team
EOF
  exit 2
fi

"$HOME/.agents/skills/agmsg/scripts/send.sh" "$TEAM" "$FROM" "$TO" "$MESSAGE"
