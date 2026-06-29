# GPU Compatibility Plan

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/13

## Purpose

Issue #13 separates "PyTorch smoke exists everywhere" from "CUDA is proven on
every GPU worker." The repository records anonymous worker classes only; exact
hostnames, GPU model names, bus IDs, and raw `nvidia-smi` evidence stay outside
this repository.

## Source Check

Retrieved: 2026-06-29 JST

- PyTorch Start Locally: https://pytorch.org/get-started/locally/
- PyTorch CUDA availability:
  https://docs.pytorch.org/docs/2.12/generated/torch.cuda.is_available.html
- NVIDIA CUDA Toolkit Release Notes:
  https://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html
- NVIDIA CUDA Compatibility:
  https://docs.nvidia.com/deploy/cuda-compatibility/

Working interpretation:

- PyTorch wheel indices remain explicit: CPU, `cu126`, `cu130`, and `cu132`.
- CUDA 13-family wheels require a CUDA 13-capable driver branch; workers with a
  CUDA 12 UMD must use CPU fallback until the driver path is changed.
- A newer architecture that failed with `no kernel image is available` on a
  CUDA 12.6 wheel is treated as CUDA 13-wheel-only for this lane.

## Anonymous Worker Policy

Policy source: `configs/fleet/gpu-wheel-policy.json`

Generate the current policy report:

```sh
python3 -m baby_model.gpu_compat \
  --config configs/fleet/gpu-wheel-policy.json
```

Current anonymous policy:

| worker_class | primary | fallback | status |
| --- | --- | --- | --- |
| `gpu-worker-a` | `cu132` | `cpu` | GPU candidate |
| `gpu-worker-b` | `cu132` after driver compatibility work | `cpu` | fallback required |
| `gpu-worker-c` | `cu132` | `cpu` | GPU candidate |

## Bounded Smoke Rule

For strict GPU proof, run without CPU fallback:

```sh
MODE=minigrid \
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
MINIGRID_TORCH_SEED=601 \
MINIGRID_TORCH_DEVICE=cuda \
MINIGRID_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 \
MINIGRID_ENV_BACKEND=uv \
MINIGRID_PYTHON=3.12 \
MINIGRID_VENV_DIR=.venv-minigrid-gpu \
MINIGRID_ENV_CLEAR=1 \
./scripts/fleet_archive_run.sh wsl:host
```

For non-GPU evidence, allow CPU fallback:

```sh
MINIGRID_TORCH_CPU_FALLBACK=1
```

## Current State

- `gpu-worker-a` completed strict CUDA smoke with `cu132` at source commit
  `e14f8c763b8a0bba8fb956b377d6ef3f5954056a`. The result matched the local
  CPU table: `A_torch_hard_only success_last=0.083`; `B_torch_encoder_first`
  and `E_torch_progress` stayed at `0.000`.
- `gpu-worker-b` is blocked for CUDA by driver/wheel compatibility and should
  not be counted as GPU-proven until that external state changes.
- `gpu-worker-c` has a CUDA 13-capable driver path, but the strict `cu132`
  smoke did not reach training because `uv pip install torch` stalled in the
  bounded smoke window after preparing the wheel stack. A resume attempt reused
  a partial environment and reached `torch` import, but failed with missing
  `libtorch_global_deps.so`. It remains an install-path blocker, not a driver
  blocker. Use `MINIGRID_ENV_CLEAR=1` for future strict GPU retries so partial
  wheel installs are not trusted.

## Acceptance For Issue #13

- `gpu-worker-a` and `gpu-worker-c` complete strict CUDA smoke with `cu132`, or
  each has a command-output blocker. Current state: `gpu-worker-a` passed;
  `gpu-worker-c` has an install-path blocker.
- `gpu-worker-b` remains documented as driver-blocked unless a driver update is
  explicitly approved and verified separately.
- Default `./scripts/verify.sh` and GitHub Actions remain green.
- Host-level evidence stays outside this repository.
