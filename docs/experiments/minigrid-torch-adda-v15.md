# PyTorch AD/DA v1.5 Curriculum-Backed Training

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/19

## Motivation

The v1.4 task-signal prediction head trained, but it did not improve sparse
`BabyAI-Unlock-v0` success. The next test is whether that AD/DA signal needs a
staged task distribution before the final sparse unlock task.

## Hypothesis

A curriculum-backed PyTorch DQN should learn a more useful shared
representation if it sees easier navigation and object-interaction stages
before unlock evaluation. The decoder-delay period is counted across the active
stage sequence, so the model has an AD-first warmup before normal DQN updates.

## Config

Config: `configs/experiments/minigrid-torch-adda-v15.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 12 episodes.
- `goto_warmup`: `BabyAI-GoToRedBall-v0`, 24 episodes.
- `unlock_eval`: `BabyAI-Unlock-v0`, 48 episodes.

Conditions:

- `A_torch_hard_only_long`: final unlock stage only, hard-only baseline.
- `P_torch_curriculum_task_signal_delay`: full curriculum, delayed decoder,
  and next-task-signal representation prediction.
- `Q_torch_curriculum_task_signal_aux_progress`: full curriculum, delayed
  decoder, next-task-signal prediction, and progress as an auxiliary
  action-selection head.

## Smoke Command

```sh
MODE=minigrid \
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-adda-v15.json \
MINIGRID_TORCH_SEED=1101 \
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

Escalate to a three-seed CUDA sweep only if a curriculum-backed condition beats
`A_torch_hard_only_long` on final-stage last-window success, or ties final-stage
last-window success while improving all-window success or mean return.
Otherwise, record the result as negative evidence and redesign the AD/DA signal
or task family.

## Current Result

A local CPU smoke passed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v15-torch/20260629T062217Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0000 | 0 | 5745 |
| `P_torch_curriculum_task_signal_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0065 | 7511 | 7264 |
| `Q_torch_curriculum_task_signal_aux_progress` | `unlock_eval` | 0.021 | 0.050 | 0.042 | 0.0064 | 7441 | 7186 |

Local conclusion: `Q_torch_curriculum_task_signal_aux_progress` met the
decision rule on CPU by beating the hard-only baseline on final-stage
last-window success. Treat this as a positive smoke signal only; require bounded
CUDA smoke before escalating to a multi-seed GPU sweep.
