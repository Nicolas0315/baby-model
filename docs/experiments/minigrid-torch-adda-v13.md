# PyTorch AD/DA v1.3 Predictive Representation

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/17

## Motivation

The v1.2 longer-window smoke showed that simply giving delayed-decoder
conditions more episodes is not enough. Two AD/DA conditions found an isolated
success, but no condition retained last-window success.

## Hypothesis

The AD phase needs a separate predictive representation objective. In v1.3 the
main DQN receives an auxiliary next-feature prediction head attached to the
shared hidden representation. This head is trained during the decoder-delay
period as well as later task episodes, while the main Q target remains tied to
external sparse reward unless the condition explicitly uses the existing
auxiliary progress head.

## Config

Config: `configs/experiments/minigrid-torch-adda-v13.json`

Conditions:

- `A_torch_hard_only_long`: v1.2 long-window hard-only baseline.
- `L_torch_predictive_delay`: delayed decoder plus next-feature prediction.
- `M_torch_predictive_aux_progress`: delayed decoder, next-feature prediction,
  and progress as an auxiliary action-selection head.

## Smoke Command

```sh
MODE=minigrid \
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-adda-v13.json \
MINIGRID_TORCH_SEED=901 \
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

Escalate to a three-seed CUDA sweep only if a predictive AD/DA condition beats
`A_torch_hard_only_long` on last-window success, or ties last-window success
while improving all-window success or mean return. Otherwise, record the result
as negative evidence and revisit the representation target or task curriculum.

## Current Result

Source commit: `fd3ef5eb3fcb833d78f7c68652709d0d75567e0e`

A bounded CUDA smoke completed on `gpu-worker-c` with
`torch==2.12.1+cu132` and `device=cuda`.

Summary artifact:

`/home/ogosh/work/baby-model-fleet/v13-smoke-20260629T053200Z-fd3ef5e/.tmp/verify-minigrid/torch/20260629T054439Z/summary.md`

| condition | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | 0.000 | 0.000 | 0.000 | 0.0000 | 0 | 5745 |
| `L_torch_predictive_delay` | 0.000 | 0.000 | 0.000 | 0.0031 | 5760 | 5265 |
| `M_torch_predictive_aux_progress` | 0.000 | 0.000 | 0.000 | 0.0035 | 5760 | 5265 |

Conclusion: the predictive head executed and trained on every transition, but
neither predictive condition improved external task success. This does not meet
the escalation rule for a multi-seed GPU sweep. The next design should revisit
the representation target or pair the predictive objective with a task
curriculum.
