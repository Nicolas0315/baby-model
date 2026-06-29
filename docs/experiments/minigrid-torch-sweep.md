# MiniGrid PyTorch Sweep

Date: 2026-06-29 JST

## Purpose

The strict CUDA smoke lane proves that the PyTorch runner can execute on GPU
worker classes, but it is still a single-seed smoke. This sweep turns the lane
into repeatable multi-seed evidence for the Baby-AD/DA hypothesis on
`BabyAI-Unlock-v0`.

## Command

Local or worker command after optional dependencies are installed:

```sh
MINIGRID_TORCH_SWEEP_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
MINIGRID_TORCH_SWEEP_SEEDS=601,602,603 \
MINIGRID_TORCH_DEVICE=cuda \
./scripts/verify_minigrid.sh
```

Fleet command shape:

```sh
MODE=minigrid \
MINIGRID_TORCH_SWEEP_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
MINIGRID_TORCH_SWEEP_SEEDS=601,602,603 \
MINIGRID_TORCH_DEVICE=cuda \
MINIGRID_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 \
MINIGRID_TORCH_INSTALLER=pip \
MINIGRID_ENV_BACKEND=uv \
MINIGRID_PYTHON=3.12 \
MINIGRID_VENV_DIR=.venv-minigrid-gpu132-pip \
MINIGRID_ENV_CLEAR=0 \
./scripts/fleet_archive_run.sh wsl:host
```

Use `MINIGRID_ENV_CLEAR=0` only after a worker has a validated venv from the
strict CUDA setup path. Use `MINIGRID_ENV_CLEAR=1` when rebuilding from scratch.

## Output

The sweep writes:

- `metrics.json`
- `summary.md`
- `latest` symlink

The summary reports:

- torch version and device set across seeds
- win count per condition
- mean and median last-window success
- per-seed winners

## Interpretation

The current smoke result is not enough to claim robust learning. A condition
should be treated as a real signal only if it survives at least a three-seed
sweep and beats the baseline on mean last-window success or win count.

## Current Result

Source commit: `8bd7583c3a351f6d81a1b2e6c28fdf997039102f`

The three-seed CUDA sweep completed on both CUDA-proven worker classes:

- `gpu-worker-a`: `torch==2.12.1+cu132`, `device=cuda`
- `gpu-worker-c`: `torch==2.12.1+cu132`, `device=cuda`

Both workers produced the same aggregate table:

| condition | wins | mean_success_all | mean_success_last | median_success_last | mean_return_last |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only` | 3 | 0.028 | 0.028 | 0.000 | 0.026 |
| `B_torch_encoder_first` | 0 | 0.000 | 0.000 | 0.000 | 0.000 |
| `E_torch_progress` | 0 | 0.000 | 0.000 | 0.000 | 0.000 |

Conclusion: the single-seed hard-only win survives the three-seed CUDA sweep.
The current AD/DA variants do not yet beat the hard-only PyTorch baseline on
`BabyAI-Unlock-v0`.
