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
./scripts/fleet_archive_run.sh mac:host-a wsl:host-b
```

For Windows CUDA workers, set `MINIGRID_TORCH_INDEX_URL` to the official wheel
index for that worker, for example
`https://download.pytorch.org/whl/cu132`.

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
next proof step is a bounded fleet/GPU run with host-level evidence kept
outside this repository.

## Rollback

Remove `baby_model/minigrid_torch.py`,
`configs/experiments/minigrid-torch-unlock-smoke.json`, and the
`MINIGRID_TORCH_*` verifier/fleet hooks. Because PyTorch is optional and not in
`pyproject.toml`, the default research loop remains unaffected.
