# PyTorch AD/DA v2.1 Explicit Subgoal-Progress Representation

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/26

## Motivation

v2.0 showed that predicting a generic semantic change mask is still too broad:
the transition-group conditions tied the baselines on final-stage last-window
success and did not improve final-stage return. The next target should encode
task-relevant subgoal progress more directly.

## Hypothesis

An explicit subgoal-progress target should be a better AD/DA bridge because it
asks the encoder to predict whether the action advanced the key-door-goal
chain, not only whether a visible affordance changed.

The representation target is a 10-dimensional event vector computed from the
current and next observations:

1. mission-conditioned key disappeared proxy
2. mission-conditioned key became visible
3. mission-conditioned locked-door count decreased
4. mission-conditioned open-door count increased
5. goal became visible
6. front cell became key
7. front cell became locked door
8. front cell became open door
9. front cell became goal
10. any unlock-chain progress event

The target remains reward-free. It uses only egocentric image and mission text
already present in the observation.

## Config

Config: `configs/experiments/minigrid-torch-adda-v21.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 12 episodes.
- `goto_warmup`: `BabyAI-GoToRedBall-v0`, 24 episodes.
- `unlock_eval`: `BabyAI-Unlock-v0`, 48 episodes.

Conditions:

- `A_torch_hard_only_long`: final unlock stage only, hard-only baseline.
- `T_torch_controllability_delay`: v1.7-style sparse ladder and binary
  controllability representation baseline.
- `ZB_torch_subgoal_progress_delay`: sparse ladder and action-conditioned
  subgoal-progress representation prediction.
- `ZC_torch_subgoal_progress_aux_progress`: sparse ladder, subgoal-progress
  representation prediction, and auxiliary progress reward kept out of the main
  reward target.

## Decision Rule

Run a bounded local smoke first. Escalate v2.1 to CUDA only if a
subgoal-progress condition beats both `A_torch_hard_only_long` and
`T_torch_controllability_delay` on final-stage last-window success, or ties both
while improving final-stage return with non-zero representation updates.

If subgoal-progress fails to beat or tie both baselines under that rule, record
v2.1 as negative target evidence and redesign around a richer state-plus-delta
target or a denser task ladder with more observable key/door transitions.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v21-torch/20260629T090656Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0000 | 0 | 5745 |
| `T_torch_controllability_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0168 | 7643 | 7397 |
| `ZB_torch_subgoal_progress_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0006 | 7493 | 7246 |
| `ZC_torch_subgoal_progress_aux_progress` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0004 | 7481 | 7226 |

Local conclusion: v2.1 did not meet the CUDA escalation rule. The
subgoal-progress conditions tied both baselines on final-stage last-window
success (`0.000`) but did not improve final-stage return (`0.000`). Treat this
as negative evidence for a sparse explicit subgoal event target under the
current ladder; the next target should encode richer state-plus-delta context or
move to a denser task ladder with more observable key/door transitions.
