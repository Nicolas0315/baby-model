# baby-model Current State

Updated: 2026-06-29 JST

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

## Research Lanes

- Core stdlib lane: v0, v0.2, and v0.3 toy-environment sweeps.
- Optional MiniGrid/BabyAI lane: dependency-isolated probes and curriculum
  experiments.
- Optional PyTorch/GPU lane: CPU-safe and CUDA/MPS-capable smoke tests with
  fleet evidence kept in local docs outside this repository.

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
