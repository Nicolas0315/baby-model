# PyTorch AD/DA v2.0 Transition-Group Representation

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/25

## Motivation

v1.9 showed that predicting next-observation affordance bits is not enough: the
affordance-progress conditions tied the hard-only baseline but lost to the
v1.7-style controllability baseline. The next target predicts action effects
directly.

## Hypothesis

A transition-group target should be a better AD/DA bridge because it asks the
encoder to predict which semantic groups changed after an action. The target is
the elementwise change mask between the current and next affordance-progress
vectors: direction, mission-conditioned bits, front-cell affordances, visible
key/door/goal presence, door locked state, and missing-object proxies.

This keeps the target reward-free while making the representation explicitly
action-effect oriented.

## Config

Config: `configs/experiments/minigrid-torch-adda-v20.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 12 episodes.
- `goto_warmup`: `BabyAI-GoToRedBall-v0`, 24 episodes.
- `unlock_eval`: `BabyAI-Unlock-v0`, 48 episodes.

Conditions:

- `A_torch_hard_only_long`: final unlock stage only, hard-only baseline.
- `T_torch_controllability_delay`: v1.7-style sparse ladder and binary
  controllability representation baseline.
- `Z_torch_transition_group_delay`: sparse ladder and action-conditioned
  transition-group representation prediction.
- `ZA_torch_transition_group_aux_progress`: sparse ladder, transition-group
  representation prediction, and auxiliary progress reward kept out of the main
  reward target.

## Decision Rule

Run a bounded local smoke first. Escalate v2.0 to CUDA only if a
transition-group condition beats both `A_torch_hard_only_long` and
`T_torch_controllability_delay` on final-stage last-window success, or ties both
while improving final-stage return with non-zero representation updates.

If transition-group fails to beat or tie both baselines under that rule, record
v2.0 as negative target evidence and redesign around explicit subgoal progress.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v20-torch/20260629T085511Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0000 | 0 | 5745 |
| `T_torch_controllability_delay` | `unlock_eval` | 0.021 | 0.000 | 0.000 | 0.0075 | 7335 | 7107 |
| `Z_torch_transition_group_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0055 | 7514 | 7259 |
| `ZA_torch_transition_group_aux_progress` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0085 | 7466 | 7211 |

Local conclusion: v2.0 did not meet the CUDA escalation rule. The
transition-group conditions tied both baselines on final-stage last-window
success (`0.000`) but did not improve final-stage return (`0.000`). Treat this
as negative evidence for a plain transition change-mask target; the next target
should encode explicit subgoal progress or a richer state-plus-delta objective.
