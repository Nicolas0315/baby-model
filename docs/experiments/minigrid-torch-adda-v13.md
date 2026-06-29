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
