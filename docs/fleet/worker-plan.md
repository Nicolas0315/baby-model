# Fleet Worker Plan

## Worker Classes

- local orchestrator: docs, verification, small runs.
- macOS worker: portability smoke runs.
- Windows/WSL GPU worker A: lightweight GPU dependency pilot.
- Windows/WSL GPU worker B: primary heavy training lane.
- Windows/WSL GPU worker C: replication and ablations.

Retired or stale nodes are excluded.

## Safe Order

1. Read-only inventory:

   ```sh
   ./scripts/fleet_inventory.sh mac:<host> wsl:<host>
   ```

2. Push GitHub repository.
3. On each worker, check out the repository into `~/work/baby-model`.
4. After the worker lane is explicitly approved, run on the worker itself:

   ```sh
   ./scripts/verify.sh
   SESSION=baby-model ITERATIONS=5 ./scripts/launch_tmux_local.sh
   ```

5. For credential-free remote execution from the orchestrator, stream the
   current commit as a git archive and start the worker job in tmux:

   ```sh
   MODE=both SEEDS=101,102,103,104,105 ./scripts/fleet_archive_run.sh mac:<host> wsl:<host>
   ```

6. Pull back only metrics summaries, not secrets or shell history.

## Private Host Mapping

Do not commit real hostnames, IPs, or GPU inventory. Store local mapping in the
git-ignored `.local/fleet-hosts` file or provide hosts through
`BABY_MODEL_FLEET_HOSTS`.

Example:

```sh
cp configs/fleet/hosts.example.txt .local/fleet-hosts
./scripts/fleet_inventory.sh
```

For agmsg progress, keep the team name local:

```sh
AGMSG_TEAM="<team-name>" ./scripts/agmsg_progress.sh <agent> "message"
```

## Current Guardrail

Remote process starts and long-running GPU jobs are explicit research lanes.
Keep each job in tmux, write run artifacts under `runs/`, and report the
approved lane, worker class, session name, and verification result in
`docs/progress/STATUS.md`. Keep exact hostnames in local-only evidence.
