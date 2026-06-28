# baby-model Agent Rules

This repository follows the home-level Katala OS shared CLI context in
`~/AGENTS.md`.

Repo-specific rules:

- Keep the first experiment loop runnable with Python standard library only.
- Treat `runs/` artifacts as experiment evidence; do not commit bulky generated
  outputs except curated summaries.
- Start remote work read-only. Remote writes, service changes, long-running
  GPU jobs, and process termination require an explicit operator-approved lane.
- Every new experiment behavior needs either a unit test or a smoke command in
  `scripts/verify.sh`.
- Use `docs/progress/STATUS.md` as the human-readable current state.
- Do not store API keys, SSH material, tokens, cookies, or personal data in this
  repository.

