# PyTorch AD/DA v1.6 Action-Prior Signal

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/20

## Motivation

The v1.5 curriculum-backed `next_task_signal` objective recovered an isolated
success signal, but it did not robustly beat the hard-only baseline on bounded
CUDA. The next test is whether the AD-side representation needs a signal that
is closer to DA action selection than passive next-observation prediction.

## Hypothesis

An action-prior representation head should make the shared encoder more useful
for sparse BabyAI tasks because it predicts a heuristic next action from the
current observation. This keeps the setup label-free with respect to task
reward, but moves the auxiliary target closer to the decoder/action interface.

## Config

Config: `configs/experiments/minigrid-torch-adda-v16.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 12 episodes.
- `goto_warmup`: `BabyAI-GoToRedBall-v0`, 24 episodes.
- `unlock_eval`: `BabyAI-Unlock-v0`, 48 episodes.

Conditions:

- `A_torch_hard_only_long`: final unlock stage only, hard-only baseline.
- `R_torch_action_prior_delay`: full curriculum, delayed decoder, and
  action-prior representation prediction.
- `S_torch_action_prior_policy_mix`: full curriculum, delayed decoder,
  action-prior representation prediction, and a small action-prior bonus mixed
  into action selection after the decoder delay.

## Smoke Command

```sh
MODE=minigrid \
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-adda-v16.json \
MINIGRID_TORCH_SEED=1201 \
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

Escalate to a three-seed CUDA sweep only if an action-prior condition beats
`A_torch_hard_only_long` on final-stage last-window success, or ties final-stage
last-window success while improving all-window success or mean return.
Otherwise, record the result as negative evidence and redesign the AD/DA signal
or task family.

## Current Result

A local CPU smoke passed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v16-torch-fixed/20260629T071609Z/summary.md`

| condition | final_stage | prior | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0 | 5745 |
| `R_torch_action_prior_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.000 | 0.0313 | 7497 | 7244 |
| `S_torch_action_prior_policy_mix` | `unlock_eval` | 0.250 | 0.021 | 0.000 | 0.000 | 0.0329 | 6972 | 6723 |

Local conclusion: after correcting the MiniGrid view-coordinate used by the
action-prior label, the action-prior family did not beat the hard-only baseline
on CPU. `S_torch_action_prior_policy_mix` recovered one all-window success, but
not in the final window and not enough to meet the decision rule.

A bounded CUDA smoke for the corrected action-prior label completed on
`gpu-worker-c` with `torch==2.12.1+cu132`, `torch_cuda_available=True`, and
`device=cuda`.

| condition | final_stage | prior | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.000 | 0.0000 | 0 | 5745 |
| `R_torch_action_prior_delay` | `unlock_eval` | 0.000 | 0.021 | 0.050 | 0.044 | 0.0455 | 7475 | 7222 |
| `S_torch_action_prior_policy_mix` | `unlock_eval` | 0.250 | 0.000 | 0.000 | 0.000 | 0.0239 | 6975 | 6726 |

CUDA conclusion: `R_torch_action_prior_delay` met the v1.6 escalation rule by
beating the hard-only baseline on final-stage last-window success (`0.050` vs
`0.000`) and return (`0.044` vs `0.000`). This triggered the three-seed CUDA
sweep below before treating the result as robust.

## Three-Seed CUDA Sweep

A three-seed CUDA sweep completed on `gpu-worker-c` at source commit
`b66e1c591dea7afdddfeb362e49303de9b051e5b` with
`torch==2.12.1+cu132`, `torch_cuda_available=True`, and `device=cuda`.

Seeds: `1201,1202,1203`

| condition | wins | mean_success_all | mean_success_last | median_success_last | mean_return_last | median_return_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | 1 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 |
| `R_torch_action_prior_delay` | 1 | 0.007 | 0.017 | 0.000 | 0.015 | 0.000 |
| `S_torch_action_prior_policy_mix` | 1 | 0.007 | 0.017 | 0.000 | 0.016 | 0.000 |

Per-seed winners:

- `1201`: `R_torch_action_prior_delay`
- `1202`: `A_torch_hard_only_long`
- `1203`: `S_torch_action_prior_policy_mix`

Sweep conclusion: `R_torch_action_prior_delay` remained in the top group by
mean final-window success, but it did not stay ahead across seeds. `R` and `S`
tied on mean final-window success (`0.017`), all conditions had median
final-window success and median return of `0.000`, and each condition won one
seed. This completes the v1.6 follow-up as non-robust evidence; do not escalate
the current action-prior design further without a stronger signal or task
redesign.
