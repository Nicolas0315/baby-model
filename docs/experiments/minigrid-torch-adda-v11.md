# PyTorch AD/DA v1.1 Conditions

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/15

## Motivation

The v1.0 three-seed CUDA sweep showed a robust negative result for the current
PyTorch AD/DA conditions: `A_torch_hard_only` won all seeds, while the delayed
decoder and progress-reward variants stayed at zero last-window success.

The likely failure mode is that a 4-episode decoder freeze in a 12-episode smoke
is too expensive, and mixing prediction progress into the reward target dilutes
the sparse external task signal.

## Hypothesis

Keep external reward as the main DQN target, shorten the decoder freeze, and use
prediction progress only as an auxiliary action-selection head. This tests a
more literal AD/DA split:

- AD/representation phase: short delay and progress-sensitive auxiliary head.
- DA/task phase: main Q-network still learns from external reward.
- Comparison: hard-only baseline remains in the same config.

## Config

Config: `configs/experiments/minigrid-torch-adda-v11.json`

Conditions:

- `A_torch_hard_only`: prior hard-only baseline.
- `F_torch_short_delay`: shorter DA freeze without intrinsic reward.
- `G_torch_aux_progress_short`: raw features, short freeze, progress auxiliary.
- `H_torch_aux_progress_coarse`: coarse features, short freeze, progress
  auxiliary.

## Smoke Command

```sh
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-adda-v11.json \
MINIGRID_TORCH_SEED=701 \
MINIGRID_TORCH_DEVICE=cuda \
./scripts/verify_minigrid.sh
```

## Sweep Command

```sh
MODE=minigrid \
MINIGRID_TORCH_SWEEP_CONFIG=configs/experiments/minigrid-torch-adda-v11.json \
MINIGRID_TORCH_SWEEP_SEEDS=701,702,703 \
MINIGRID_TORCH_DEVICE=cuda \
MINIGRID_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu132 \
MINIGRID_TORCH_INSTALLER=pip \
MINIGRID_ENV_BACKEND=uv \
MINIGRID_PYTHON=3.12 \
MINIGRID_VENV_DIR=.venv-minigrid-gpu132-pip \
MINIGRID_ENV_CLEAR=0 \
./scripts/fleet_archive_run.sh wsl:host
```

## Decision Rule

Treat a revised AD/DA condition as promising only if it beats
`A_torch_hard_only` on mean last-window success or win count in a three-seed
CUDA sweep. Otherwise, record it as negative evidence and change the condition
design again.

## Current Result

Source commit: `3fe12c716fe125577414b9a57da7709f31ed312c`

A bounded CUDA smoke completed on `gpu-worker-c` with
`torch==2.12.1+cu132` and `device=cuda`.

Summary artifact:

`/home/ogosh/work/baby-model-fleet/debug-pip-20260629T041337Z-d2a1c60/.tmp/verify-minigrid/torch/20260629T045933Z/summary.md`

| condition | success_all | success_last | return_last | updates |
| --- | ---: | ---: | ---: | ---: |
| `A_torch_hard_only` | 0.000 | 0.000 | 0.000 | 1433 |
| `F_torch_short_delay` | 0.000 | 0.000 | 0.000 | 1193 |
| `G_torch_aux_progress_short` | 0.000 | 0.000 | 0.000 | 1193 |
| `H_torch_aux_progress_coarse` | 0.000 | 0.000 | 0.000 | 1193 |

Conclusion: the v1.1 condition set is executable on CUDA, but the smoke is not
promising. The auxiliary-progress design should not be escalated to a multi-seed
GPU sweep without another condition change.
