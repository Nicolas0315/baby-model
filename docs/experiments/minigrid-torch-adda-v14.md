# PyTorch AD/DA v1.4 Task-Signal Prediction

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/18

## Motivation

The v1.3 next-feature prediction head trained on every transition but did not
improve sparse unlock-task success. Predicting the whole hashed feature vector
may be too low-level and weakly connected to the task.

## Hypothesis

Use a smaller task-relevant predictive target. The v1.4 head predicts a compact
next-observation signal that tracks mission words and object/state counts for
door, key, goal, wall, and door state. The main DQN still learns from sparse
external reward; the AD-specific predictive head shapes the shared hidden
representation during the decoder-delay period and later task episodes.

## Config

Config: `configs/experiments/minigrid-torch-adda-v14.json`

Conditions:

- `A_torch_hard_only_long`: long-window hard-only baseline.
- `N_torch_task_signal_delay`: delayed decoder plus task-signal prediction.
- `O_torch_task_signal_aux_progress`: delayed decoder, task-signal prediction,
  and progress as an auxiliary action-selection head.

## Smoke Command

```sh
MODE=minigrid \
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-adda-v14.json \
MINIGRID_TORCH_SEED=1001 \
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

Escalate to a three-seed CUDA sweep only if a task-signal predictive condition
beats `A_torch_hard_only_long` on last-window success, or ties last-window
success while improving all-window success or mean return. Otherwise, record
the result as negative evidence and switch to a curriculum-backed design.

## Current Result

Source commit: `6105475cc025eead8669c8abb967bb79161bdf3a`

A local CPU smoke passed in the existing optional PyTorch venv. A bounded CUDA
smoke then completed on `gpu-worker-c` with `torch==2.12.1+cu132` and
`device=cuda`.

Summary artifact:

`/home/ogosh/work/baby-model-fleet/v14-smoke-20260629T055500Z-6105475/.tmp/verify-minigrid/torch/20260629T060713Z/summary.md`

| condition | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | 0.021 | 0.050 | 0.046 | 0.0000 | 0 | 5679 |
| `N_torch_task_signal_delay` | 0.000 | 0.000 | 0.000 | 0.0060 | 5760 | 5265 |
| `O_torch_task_signal_aux_progress` | 0.000 | 0.000 | 0.000 | 0.0060 | 5760 | 5265 |

Conclusion: the task-signal predictive head executed and trained, but it did
not improve sparse unlock-task success. The hard-only long-window baseline was
the only condition with last-window success. Do not escalate this condition
family to a multi-seed GPU sweep; switch to a curriculum-backed design next.
