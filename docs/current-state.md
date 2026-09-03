# baby-model Current State

Updated: 2026-08-23 JST

## Purpose

`baby-model` is a research harness for testing AD-first / DA-delayed learning
ideas before spending fleet GPU time. The default repository loop is designed
to stay runnable with the Python standard library only.

## Reproducibility

- Runtime pin: `.python-version` currently pins Python `3.11`.
- Python project metadata: `pyproject.toml` declares `requires-python = ">=3.10"`
  and `dependencies = []`.
- Lockfile status: no Python lockfile is required for the default loop because
  the default project metadata declares no external dependencies.
- Optional dependencies: MiniGrid/BabyAI and PyTorch are isolated optional
  lanes and are not part of the default project dependency set.
- Main verifier: `./scripts/verify.sh`.
- CI: `.github/workflows/verify.yml` runs the verifier on pushes and pull
  requests with Python `3.11`.

## Local Verification Surface

Use the full verifier when mutating behavior:

```sh
./scripts/verify.sh
```

Use narrower static checks for documentation-only or repo-health changes:

```sh
python3 -m unittest discover -s tests -p 'test_*.py'
bash -n scripts/*.sh
```

The full verifier writes bounded temporary outputs under `.tmp/verify-run` and
`.tmp/verify-sweep`, then manages only those expected paths.

## Decision Verification

Sweep decisions are read from `## Seed-Level Statistics` in a sweep's
`summary.md`: per-condition sd and bootstrap interval, plus a paired sign-flip
test against the control condition. See
`docs/experiments/seed-statistics-and-holdout.md`. Two rules follow from the
mechanism:

- The exact paired p-floor is `2 / 2**n`. Three- and five-seed gates cannot
  reach p < 0.05; use six or more seeds for a significance claim.
- `success_rate_last_window` is measured under epsilon-greedy exploration on
  training episodes. Configs that set `holdout_episodes` also report
  `holdout_success_rate` from a greedy, no-learning pass over held-out episode
  seeds, which is the metric to prefer when the two disagree.

## Research Lanes

- Core stdlib lane: v0, v0.2, and v0.3 toy-environment sweeps.
- Optional MiniGrid/BabyAI lane: dependency-isolated probes and curriculum
  experiments.
- Optional PyTorch/GPU lane: CPU-safe and CUDA/MPS-capable smoke tests with
  fleet evidence kept in local docs outside this repository.

## GPU Lane Environment

The CUDA venv is not committed and is not durable; rebuild it from the scripted
path rather than assuming a previous one survives:

```sh
MINIGRID_VENV_DIR=.venv-minigrid-cuda \
MINIGRID_PYTHON=3.12 \
MINIGRID_ENV_BACKEND=uv \
MINIGRID_TORCH_INSTALLER=uv \
MINIGRID_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
MINIGRID_TORCH_DEVICE=cuda \
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
./scripts/setup_minigrid_env.sh
```

The v2.43-v2.45 `rtx4090` venv was found in 2026-08 as a broken symlink into a
sibling clone that had been removed, so that environment was not reproducible
as recorded. Two CUDA workers are now provisioned this way.

### `rtx4090` is faulty: do not use it for these sweeps

Three sweeps on `rtx4090` died with a SIGSEGV/SIGABRT caught by pygame's
parachute handler, while `rtx5060ti` and the Mac ran the identical code, config,
and (for `rtx5060ti`) the identical torch build for hours without a single
failure.

| attempt | filesystem | torch threads | ran for | exit | crash site |
| --- | --- | ---: | ---: | ---: | --- |
| v2.49 #1 | `/mnt/c` via drvfs | 12 | 18 min | 134 | `minigrid/core/grid.py process_vis` |
| v2.50 #1 | ext4 | 12 | 72 min | 139 | C frame, no Python frame |
| v2.50 #2 | ext4 | 1 | 5 min | 134 | `minigrid_torch.py _scatter` |

**Two explanations were tried and both were wrong.** The filesystem was not the
cause: moving the clone and venv to ext4 did not prevent the second crash. The
thread count was not the cause either: pinning to one thread did not prevent the
third.

The third trace is what settles it. It lands inside a list comprehension that
builds a list of Python ints from dict keys:

```python
flat_indices = [
    row * feature_dim + index for row, features in enumerate(rows) for index in features
]
```

That cannot segfault from program logic, and the indices are provably in range
(the loop above it rejects any key outside `[0, feature_dim)`, so the maximum
flat index is `len(rows) * feature_dim - 1` against a tensor of exactly that
length). Three crashes at three unrelated sites — a numpy visibility routine, an
unnamed C frame, and a pure-Python integer comprehension — on one host and never
on two others is the signature of a faulty host, not of a bug.

This host has never been memtested, which is a standing note elsewhere in the
fleet docs. Running one is an operator action.

**Until then, do not schedule baby-model sweeps on `rtx4090`.** Use `rtx5060ti`
or the Mac.

### Pin `torch` to one thread

The networks are 1024 -> 64 -> 7 with batch 16, so multithreaded BLAS is pure
overhead. Measured on a two-seed sweep: **42.09s at six threads, 31.09s at one
thread, 1.35x faster, with bit-identical results.**

`BABY_MODEL_TORCH_THREADS` sets it, and the resolved `torch_num_threads` is
recorded in the artifact's `framework` block. Unset by default so nothing
historical changes.

## Research Lanes

- Core stdlib lane: v0, v0.2, and v0.3 toy-environment sweeps.
- Optional MiniGrid/BabyAI lane: dependency-isolated probes and curriculum
  experiments.
- Optional PyTorch/GPU lane: CPU-safe and CUDA/MPS-capable smoke tests with
  fleet evidence kept in local docs outside this repository.

## GPU Lane Environment

The CUDA venv is not committed and is not durable; rebuild it from the scripted
path rather than assuming a previous one survives:

```sh
MINIGRID_VENV_DIR=.venv-minigrid-cuda \
MINIGRID_PYTHON=3.12 \
MINIGRID_ENV_BACKEND=uv \
MINIGRID_TORCH_INSTALLER=uv \
MINIGRID_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128 \
MINIGRID_TORCH_DEVICE=cuda \
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
./scripts/setup_minigrid_env.sh
```

The v2.43-v2.45 `rtx4090` venv was found in 2026-08 as a broken symlink into a
sibling clone that had been removed, so that environment was not reproducible
as recorded. Two CUDA workers are now provisioned this way.

### Put the working copy on ext4, not on `/mnt/c`

On `rtx4090`, `$HOME/work` is a root-owned symlink to `/mnt/c/Users/ogosh/work`,
so a clone placed there runs off the Windows filesystem through WSL's drvfs
bridge. A v2.49 sweep from that path died 18 minutes in with

```
Fatal Python error: pygame_parachute: (pygame parachute) Segmentation Fault
  minigrid/core/grid.py line 310 in process_vis
exit=134
```

with 91 GB of memory free, while the identical sweep on `rtx5060ti` — whose
`$HOME/work` is real ext4 — ran to completion. Rebuilding the clone and its venv
under `$HOME/ext4/` on `rtx4090` cleared it.

Treat a `/mnt/c`-backed working copy as unusable for long runs on that host.
Check before launching:

```sh
readlink -f "$HOME/work"   # must not start with /mnt/
df -hT .                   # must say ext4
```

## Active Local Slice

The current local working tree includes an active v1.9 affordance-progress
PyTorch experiment slice:

- `baby_model/minigrid_torch.py`
- `tests/test_experiment.py`
- `configs/experiments/minigrid-torch-adda-v19.json`
- `docs/experiments/minigrid-torch-adda-v19.md`

Treat that slice as experiment work until its smoke result and decision are
recorded.

## Boundaries

- Do not commit bulky `runs/` outputs except curated summaries.
- Keep host-level fleet evidence, private hostnames, credentials, SSH material,
  cookies, and personal data outside this repository.
- Start remote work read-only; remote writes, service changes, long GPU runs,
  and process termination require an explicitly approved lane.
