# PyTorch AD/DA v2.2 State-Plus-Delta Representation

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/27

## Motivation

v2.1 showed that sparse explicit subgoal events are not enough under the current
ladder. The next target should keep the event labels but add the surrounding
state context so the encoder can distinguish where the event occurred and what
changed.

## Chosen Path

Use Option A from #27: a reward-free state-plus-delta target. Do not change the
ladder in v2.2. This isolates whether the representation target needs both
state context and transition labels before introducing a denser curriculum.

## Hypothesis

A state-plus-delta target should be a better AD/DA bridge because it predicts
the current affordance state, next affordance state, generic transition mask,
and explicit subgoal event mask from the same action-conditioned hidden state.

The target is a 58-dimensional vector:

- current affordance-progress vector: 16 dims
- next affordance-progress vector: 16 dims
- transition-group change mask: 16 dims
- explicit subgoal-progress event vector: 10 dims

This keeps the target reward-free while giving the auxiliary representation head
both state context and action-effect labels.

## Config

Config: `configs/experiments/minigrid-torch-adda-v22.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 12 episodes.
- `goto_warmup`: `BabyAI-GoToRedBall-v0`, 24 episodes.
- `unlock_eval`: `BabyAI-Unlock-v0`, 48 episodes.

Conditions:

- `A_torch_hard_only_long`: final unlock stage only, hard-only baseline.
- `T_torch_controllability_delay`: v1.7-style sparse ladder and binary
  controllability representation baseline.
- `ZD_torch_state_plus_delta_delay`: sparse ladder and action-conditioned
  state-plus-delta representation prediction.
- `ZE_torch_state_plus_delta_aux_progress`: sparse ladder, state-plus-delta
  representation prediction, and auxiliary progress reward kept out of the main
  reward target.

## Decision Rule

Run a bounded local smoke first. Escalate v2.2 to CUDA only if a
state-plus-delta condition beats both `A_torch_hard_only_long` and
`T_torch_controllability_delay` on final-stage last-window success, or ties both
while improving final-stage return with non-zero representation updates.

If state-plus-delta fails to beat or tie both baselines under that rule, record
v2.2 as negative target evidence and move to the denser key-door ladder option
from #27.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v22-torch/20260629T091701Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0000 | 0 | 5745 |
| `T_torch_controllability_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0076 | 7341 | 7086 |
| `ZD_torch_state_plus_delta_delay` | `unlock_eval` | 0.021 | 0.000 | 0.000 | 0.0120 | 7546 | 7291 |
| `ZE_torch_state_plus_delta_aux_progress` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0134 | 7469 | 7223 |

Local conclusion: v2.2 did not meet the CUDA escalation rule. The
state-plus-delta delay condition produced one all-window final-stage success
(`success_all=0.021`), but both state-plus-delta conditions tied both baselines
on final-stage last-window success (`0.000`) and did not improve final-stage
return (`0.000`). Treat this as weak but non-escalating evidence; the next
experiment should move to a denser key-door ladder where the target receives
more observable key pickup, locked-door, open-door, and goal transitions before
final unlock evaluation.
