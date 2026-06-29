# MiniGrid PyTorch DQN Lane

Date: 2026-06-29 JST

## Purpose

Issue #12 opens a real optional deep-learning lane after the dependency-free
neural pilot in issue #11 failed to improve `BabyAI-Unlock-v0`. The lane uses
PyTorch only when explicitly requested, so the default verifier and base
research loop remain dependency-free.

## Source Check

Retrieved: 2026-06-29 JST

- PyTorch Start Locally: https://pytorch.org/get-started/locally/
- `torch.cuda.is_available`:
  https://docs.pytorch.org/docs/2.12/generated/torch.cuda.is_available.html
- MPS backend:
  https://docs.pytorch.org/docs/2.12/notes/mps.html
- Model-building tutorial:
  https://pytorch.org/tutorials/beginner/basics/buildmodel_tutorial.html
- Optimization tutorial:
  https://pytorch.org/tutorials/beginner/basics/optimization_tutorial.html
- `torch.Tensor.to`:
  https://docs.pytorch.org/docs/2.12/generated/torch.Tensor.to.html
- Astral uv docs via Context7 library `/astral-sh/uv`, plus local `uv venv
  --help`: `uv venv --python`, `uv venv --allow-existing`, `UV_LINK_MODE`,
  `uv pip install`, and `uv pip install --index-url` usage.

The official PyTorch install selector reported latest stable `2.12.1` at
retrieval time. Device checks use `torch.cuda.is_available()` for CUDA and
`torch.backends.mps.is_available()` for Apple MPS.

## Dependency Boundary

Default `./scripts/verify.sh` still has no project dependency additions.
PyTorch is imported only by `baby_model.minigrid_torch`, and that module is run
only when `MINIGRID_TORCH_CONFIG` is provided to the optional MiniGrid verifier.

CPU/macOS optional setup:

```sh
python3 -m venv .venv-minigrid-torch
. .venv-minigrid-torch/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install minigrid torch
```

Linux CPU optional setup from the official CPU wheel index:

```sh
python3 -m pip install minigrid
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Windows CUDA optional setup, choose the index that matches the worker:

```sh
python3 -m pip install minigrid
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu130
python3 -m pip install torch --index-url https://download.pytorch.org/whl/cu132
```

Fleet workers should use the repo setup wrapper so Python version selection,
`venv` vs `uv`, and optional torch installation stay consistent:

```sh
MINIGRID_ENV_BACKEND=uv \
MINIGRID_PYTHON=3.12 \
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
MINIGRID_TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu \
./scripts/setup_minigrid_env.sh ./scripts/verify_minigrid.sh
```

Supported setup controls:

- `MINIGRID_ENV_BACKEND=auto|venv|uv`; `auto` chooses `uv` when available.
- `MINIGRID_PYTHON=3.12` requests a stable Python for PyTorch wheels on WSL.
- `MINIGRID_TORCH_INDEX_URL` selects an official PyTorch wheel index.
- `MINIGRID_TORCH_CPU_FALLBACK=1` lets fleet runs retry the same smoke on CPU
  when an explicit CUDA device fails.

## Config

- Config: `configs/experiments/minigrid-torch-unlock-smoke.json`
- Env: `BabyAI-Unlock-v0`
- Seed: `601`
- Episodes per condition: `12`
- Max steps: `120`
- Agent: PyTorch DQN, `feature_dim=1024`, `hidden_dim=64`,
  `learning_rate=0.001`, replay capacity `256`, batch size `8`
- Device: `auto`, preferring CUDA, then MPS, then CPU

Command:

```sh
. .venv-minigrid-torch/bin/activate
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
MINIGRID_TORCH_SEED=601 \
MINIGRID_TORCH_DEVICE=auto \
./scripts/verify_minigrid.sh
```

Fleet command shape:

```sh
MODE=minigrid \
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
MINIGRID_TORCH_SEED=601 \
MINIGRID_TORCH_DEVICE=auto \
MINIGRID_ENV_BACKEND=auto \
MINIGRID_PYTHON=3.12 \
./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
```

For Windows CUDA workers, set `MINIGRID_TORCH_INDEX_URL` to the official wheel
index for that worker, for example
`https://download.pytorch.org/whl/cu132`.
Set `MINIGRID_TORCH_CPU_FALLBACK=1` only when the run should preserve a CPU
training summary after an explicit CUDA failure; keep it off for strict GPU
proof.

## Local Verification

Local optional venv created on 2026-06-29:

- `torch==2.12.1`
- `minigrid==3.1.0`
- `gymnasium==1.3.0`
- `numpy==2.5.0`

Device availability on the current Mac:

- `torch.cuda.is_available()`: `False`
- `torch.backends.mps.is_available()`: `True`

Observed command:

```sh
. .venv-minigrid-torch/bin/activate
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
MINIGRID_TORCH_SEED=601 \
MINIGRID_TORCH_DEVICE=cpu \
./scripts/verify_minigrid.sh
```

CPU smoke result:

| condition | success_all | success_last | return_last | mean_steps_success | updates | parameters |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only` | 0.083 | 0.083 | 0.079 | 32.00 | 1345 | 66055 |
| `B_torch_encoder_first` | 0.000 | 0.000 | 0.000 |  | 953 | 66055 |
| `E_torch_progress` | 0.000 | 0.000 | 0.000 |  | 953 | 66055 |

Winner by last-window success: `A_torch_hard_only`.

An `auto` device smoke selected MPS on the current Mac and also completed, but
all three conditions had `success_last=0.000`. Treat this as a device/backend
smoke only, not a scientific conclusion.

## Current State

The runner/config/verifier hook and local CPU/MPS smoke are implemented. The
bounded fleet run is partially proven, with host-level evidence kept outside
this repository.

The fleet environment setup path was hardened after the partial run and
re-tested at source commit `c25dc648a938629938a809de382e293541e407e3`:

- `scripts/setup_minigrid_env.sh` now supports `venv` and `uv` backends.
- `scripts/fleet_archive_run.sh` forwards setup controls and can perform an
  explicit CPU fallback after CUDA failure.
- `./scripts/verify.sh` includes shell syntax and setup dry-run checks while
  remaining dependency-free.
- A local non-torch setup smoke completed with `uv`, Python 3.12, and
  `minigrid==3.1.0`.
- The two previously blocked Windows/WSL workers now produce PyTorch training
  summaries through CPU fallback or CPU-only wheel setup.

## Fleet Verification

Retrieved: 2026-06-29 JST

Source commit: `c009eda8ceea8b0e96f62ce24df2e4f00ea67e80`

The optional PyTorch lane was replicated via `git archive | ssh | tmux` without
adding PyTorch to the default project dependencies.

Proven worker coverage:

- Current Mac CPU smoke: `torch==2.12.1`, `device=cpu`, winner
  `A_torch_hard_only`.
- Current Mac backend smoke: `device=mps`, completed as a backend check.
- One remote macOS worker: `torch==2.12.1`, `device=mps`, winner
  `A_torch_hard_only`.
- One remote Windows/WSL worker: `torch==2.12.1+cu126`, `device=cuda`, winner
  `A_torch_hard_only`.
- One additional remote Windows/WSL worker: CUDA was blocked by driver/wheel
  support, but CPU fallback completed with `torch==2.12.1+cu132`,
  `device=cpu`, winner `A_torch_hard_only`.

The completed remote CPU/MPS/CUDA smoke summaries all matched the local CPU
table: `A_torch_hard_only` reached `success_last=0.083`, while
`B_torch_encoder_first` and `E_torch_progress` stayed at `0.000`.

Follow-up worker coverage at commit
`c25dc648a938629938a809de382e293541e407e3`:

- The newer-GPU Windows/WSL worker still failed CUDA because its driver was too
  old for the cu132 wheel, but scripted CPU fallback completed with
  `torch==2.12.1+cu132`, `device=cpu`, and winner `A_torch_hard_only`.
- The remaining Windows/WSL worker still did not finish the cu126 CUDA wheel
  install within a bounded smoke window, but a CPU-only wheel run completed with
  `torch==2.12.1+cpu`, `device=cpu`, and winner `A_torch_hard_only`.
- Both completed follow-up smoke summaries matched the local CPU table:
  `A_torch_hard_only success_last=0.083`; `B_torch_encoder_first` and
  `E_torch_progress` stayed at `0.000`.

Remaining blockers:

- Full-fleet PyTorch training-summary coverage is complete, but not all of it
  is GPU-backed.
- One newer-GPU worker needs a compatible driver/wheel combination before CUDA
  can be treated as proven.
- One Windows/WSL worker's CUDA wheel install is too slow for a bounded smoke
  lane, so it is currently proven only through the CPU wheel path.

This is enough to prove the optional runner and device-selection path, but not
enough to claim a robust GPU training result.

## Rollback

Remove `baby_model/minigrid_torch.py`,
`configs/experiments/minigrid-torch-unlock-smoke.json`, and the
`MINIGRID_TORCH_*` verifier/fleet hooks. Because PyTorch is optional and not in
`pyproject.toml`, the default research loop remains unaffected.
