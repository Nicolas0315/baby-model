# PyTorch AD/DA v1.2 Longer Window

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/16

## Motivation

The v1.0 three-seed CUDA sweep favored hard-only training, and the v1.1
short-delay auxiliary-progress smoke stayed at zero success for every
condition. Both designs used only 12 episodes, so a delayed decoder may have
spent too much of the run exploring before the sparse external task could be
learned.

## Hypothesis

Give every condition a 48-episode learning window while keeping the same
`BabyAI-Unlock-v0` task. If AD-first learning is useful but slower to surface,
the longer window should let delayed-decoder or auxiliary-progress conditions
recover after the early representation phase. The hard-only baseline is also
extended to 48 episodes so the comparison isolates condition design instead of
total training budget.

## Config

Config: `configs/experiments/minigrid-torch-adda-v12.json`

Conditions:

- `A_torch_hard_only_long`: long-window hard-only baseline.
- `I_torch_long_delay`: long-window delayed decoder without intrinsic reward.
- `J_torch_long_aux_progress`: raw features, delayed decoder, progress
  auxiliary head.
- `K_torch_long_coarse_aux`: coarse features, delayed decoder, progress
  auxiliary head.

The agent also uses a larger replay buffer and batch size than the 12-episode
smoke config so the longer run does not overfit a tiny early replay sample.

## Smoke Command

```sh
MODE=minigrid \
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-adda-v12.json \
MINIGRID_TORCH_SEED=801 \
MINIGRID_TORCH_DEVICE=cuda \
MINIGRID_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 \
MINIGRID_TORCH_INSTALLER=pip \
MINIGRID_TORCH_CPU_FALLBACK=0 \
MINIGRID_ENV_BACKEND=uv \
MINIGRID_PYTHON=3.12 \
MINIGRID_VENV_DIR=.venv-minigrid-gpu132-pip \
MINIGRID_ENV_CLEAR=0 \
./scripts/fleet_archive_run.sh wsl:host
```

## Sweep Command

```sh
MODE=minigrid \
MINIGRID_TORCH_SWEEP_CONFIG=configs/experiments/minigrid-torch-adda-v12.json \
MINIGRID_TORCH_SWEEP_SEEDS=801,802,803 \
MINIGRID_TORCH_DEVICE=cuda \
MINIGRID_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 \
MINIGRID_TORCH_INSTALLER=pip \
MINIGRID_TORCH_CPU_FALLBACK=0 \
MINIGRID_ENV_BACKEND=uv \
MINIGRID_PYTHON=3.12 \
MINIGRID_VENV_DIR=.venv-minigrid-gpu132-pip \
MINIGRID_ENV_CLEAR=0 \
./scripts/fleet_archive_run.sh wsl:host
```

## Decision Rule

Run a single bounded CUDA smoke first. Escalate to a three-seed CUDA sweep only
if an AD/DA condition beats `A_torch_hard_only_long` on last-window success, or
ties last-window success while improving all-window success or mean return.
Otherwise, record the result as negative evidence and move to a representation
objective change instead of spending more GPU time on the same condition family.

## Current Result

Source commit: `b739545f58e55ed40e044b1a5cd5b3b4083f0dd9`

A bounded CUDA smoke completed on `gpu-worker-c` with
`torch==2.12.1+cu132` and `device=cuda`.

Summary artifact:

`/home/ogosh/work/baby-model-fleet/v12-smoke-20260629T050900Z-b739545/.tmp/verify-minigrid/torch/20260629T052135Z/summary.md`

| condition | success_all | success_last | return_last | updates |
| --- | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | 0.000 | 0.000 | 0.000 | 5745 |
| `I_torch_long_delay` | 0.021 | 0.000 | 0.000 | 5166 |
| `J_torch_long_aux_progress` | 0.000 | 0.000 | 0.000 | 5265 |
| `K_torch_long_coarse_aux` | 0.021 | 0.000 | 0.000 | 5222 |

Conclusion: the longer window produced isolated all-window successes for two
AD/DA conditions, but no condition achieved last-window success or mean-return
improvement. This does not meet the escalation rule for a multi-seed GPU sweep.
The next condition family should change the representation objective instead of
only extending the same delayed-decoder setup.
