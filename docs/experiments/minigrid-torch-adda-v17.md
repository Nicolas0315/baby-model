# PyTorch AD/DA v1.7 Controllability Signal

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/22

## Motivation

The v1.6 action-prior objective was too brittle: the single-seed CUDA smoke was
positive, but the three-seed CUDA sweep split one win per condition and all
median final-window success and return values stayed at `0.000`. A direct
hand-coded next-action label is likely too narrow to become a robust AD-side
developmental signal.

## Hypothesis

A controllability representation target should be a better AD/DA bridge than a
direct action-prior label. Instead of predicting which action should be taken,
the encoder predicts whether the chosen action changes the sparse observation
signature. This keeps the auxiliary target reward-free while making the AD-side
representation sensitive to which observations are action-changeable.

## Config

Config: `configs/experiments/minigrid-torch-adda-v17.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 12 episodes.
- `goto_warmup`: `BabyAI-GoToRedBall-v0`, 24 episodes.
- `unlock_eval`: `BabyAI-Unlock-v0`, 48 episodes.

Conditions:

- `A_torch_hard_only_long`: final unlock stage only, hard-only baseline.
- `T_torch_controllability_delay`: full curriculum, delayed decoder, and
  action-conditioned controllability representation prediction.
- `U_torch_controllability_aux_progress`: full curriculum, delayed decoder,
  controllability representation prediction, and auxiliary progress reward kept
  out of the main reward target.

## Decision Rule

Run a bounded local smoke first. Escalate v1.7 to CUDA only if a
controllability condition beats `A_torch_hard_only_long` on final-stage
last-window success, or ties final-stage last-window success while improving
final-stage return and representation updates are non-zero.

If the bounded smoke does not meet that rule, record v1.7 as negative evidence
and redesign the task ladder or freeze/thaw schedule before spending GPU time.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v17-torch/20260629T075339Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0000 | 0 | 5745 |
| `T_torch_controllability_delay` | `unlock_eval` | 0.021 | 0.050 | 0.048 | 0.0098 | 7455 | 7200 |
| `U_torch_controllability_aux_progress` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0149 | 7628 | 7373 |

Local conclusion: `T_torch_controllability_delay` met the v1.7 escalation rule
by beating the hard-only baseline on final-stage last-window success (`0.050`
vs `0.000`) and return (`0.048` vs `0.000`) with non-zero representation
updates. Escalate this condition family to a bounded CUDA smoke before treating
the signal as useful.

A bounded CUDA smoke completed on `gpu-worker-c` at source commit
`556308831bafe60c99f5767e4e2c9a1b2199702f` with `torch==2.12.1+cu132`,
`torch_cuda_available=True`, and `device=cuda`.

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0000 | 0 | 5745 |
| `T_torch_controllability_delay` | `unlock_eval` | 0.021 | 0.050 | 0.048 | 0.0098 | 7455 | 7200 |
| `U_torch_controllability_aux_progress` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0112 | 7628 | 7373 |

CUDA smoke conclusion: `T_torch_controllability_delay` reproduced the local
positive smoke on strict CUDA. This triggered a three-seed CUDA sweep before
treating the controllability signal as robust.

## Three-Seed CUDA Sweep

A three-seed CUDA sweep completed on `gpu-worker-c` at source commit
`556308831bafe60c99f5767e4e2c9a1b2199702f` with
`torch==2.12.1+cu132`, `torch_cuda_available=True`, and `device=cuda`.

Seeds: `1301,1302,1303`

| condition | wins | mean_success_all | mean_success_last | median_success_last | mean_return_last | median_return_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | 2 | 0.007 | 0.000 | 0.000 | 0.000 | 0.000 |
| `T_torch_controllability_delay` | 1 | 0.007 | 0.017 | 0.000 | 0.016 | 0.000 |
| `U_torch_controllability_aux_progress` | 0 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 |

Per-seed winners:

- `1301`: `T_torch_controllability_delay`
- `1302`: `A_torch_hard_only_long`
- `1303`: `A_torch_hard_only_long`

Sweep conclusion: `T_torch_controllability_delay` stayed above the hard-only
baseline by mean final-window success (`0.017` vs `0.000`) and return (`0.016`
vs `0.000`), but the signal was not robust across seeds. Median final-window
success and median return were `0.000`, and `A_torch_hard_only_long` had the
higher seed win count (`2` vs `1`). Treat v1.7 as weak positive but not robust
enough to escalate without changing the task ladder or representation target.
