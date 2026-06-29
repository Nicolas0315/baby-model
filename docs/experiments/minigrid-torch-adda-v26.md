# PyTorch AD/DA v2.6 Matched GoToObj Representation Isolation

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/31

## Motivation

v2.5 produced the strongest AD/DA signal so far after moving from BabyAI Unlock
to `BabyAI-GoToObj-v0`, but the three-seed CUDA sweep split seed winners across
controllability, hard-only, and state-plus-delta. The next experiment separates
the GoToObj task-family/curriculum effect from the representation objective
effect.

## Chosen Path

Use Option B from #31: isolate representation objective effect.

The GoToObj task family, stage ladder, decoder delay, episode budget, intrinsic
settings, and optimizer settings are held fixed for the primary comparison.
The only primary variable is whether the matched curriculum uses no
representation objective, controllability, or state-plus-delta.

## Hypothesis

If v2.5 was mostly a task-family or curriculum effect, the matched
no-representation curriculum should tie or beat the representation objectives.
If AD-side representation is contributing, either controllability or
state-plus-delta should beat the matched no-representation curriculum under the
same stage and delay schedule.

## Config

Config: `configs/experiments/minigrid-torch-adda-v26.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 12 episodes.
- `goto_red_ball_warmup`: `BabyAI-GoToRedBall-v0`, 24 episodes.
- `goto_obj_eval`: `BabyAI-GoToObj-v0`, 48 episodes.

Conditions:

- `A_torch_gotoobj_hard_only`: final GoToObj stage only, reference baseline.
- `ZK_torch_gotoobj_curriculum_no_repr_delay`: matched GoToObj curriculum,
  decoder delay, no representation objective.
- `ZL_torch_gotoobj_controllability_matched_delay`: matched GoToObj curriculum,
  decoder delay, binary controllability representation.
- `ZM_torch_gotoobj_state_plus_delta_matched_delay`: matched GoToObj
  curriculum, decoder delay, joint state-plus-delta representation.

## Decision Rule

Run a bounded local smoke first. Escalate v2.6 to CUDA only if a representation
condition beats `ZK_torch_gotoobj_curriculum_no_repr_delay` on final-stage
last-window success, or ties it while improving final-stage return with
non-zero representation updates.

If both representation conditions lose to or merely tie the matched
no-representation curriculum without return improvement, record v2.6 as
evidence that the v2.5 lift was mostly task-family/curriculum effect and move
to a non-DQN representation probe.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v26-torch/20260629T101758Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_gotoobj_hard_only` | `goto_obj_eval` | 0.292 | 0.300 | 0.209 | 0.0000 | 0 | 2514 |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | `goto_obj_eval` | 0.646 | 0.650 | 0.366 | 0.0000 | 0 | 3028 |
| `ZL_torch_gotoobj_controllability_matched_delay` | `goto_obj_eval` | 0.458 | 0.450 | 0.309 | 0.0000 | 3991 | 3738 |
| `ZM_torch_gotoobj_state_plus_delta_matched_delay` | `goto_obj_eval` | 0.583 | 0.650 | 0.461 | 0.0072 | 3639 | 3384 |

Local conclusion: `ZM_torch_gotoobj_state_plus_delta_matched_delay` met the
v2.6 escalation rule by tying the matched no-representation curriculum on
final-stage last-window success (`0.650`) while improving return (`0.461` vs
`0.366`) with non-zero representation updates. Escalate to a bounded CUDA smoke.

A bounded CUDA smoke completed on `gpu-worker-c` with `torch==2.12.1+cu132`,
`torch_cuda_available=True`, and `device=cuda`. The remote source snapshot used
commit `fa4077a7f1dd64ff715cbab22f92a5de7863f50f` plus the v2.6 config in this
change.

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_gotoobj_hard_only` | `goto_obj_eval` | 0.292 | 0.300 | 0.209 | 0.0000 | 0 | 2514 |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | `goto_obj_eval` | 0.646 | 0.650 | 0.366 | 0.0000 | 0 | 3028 |
| `ZL_torch_gotoobj_controllability_matched_delay` | `goto_obj_eval` | 0.458 | 0.450 | 0.309 | 0.0000 | 3991 | 3738 |
| `ZM_torch_gotoobj_state_plus_delta_matched_delay` | `goto_obj_eval` | 0.583 | 0.650 | 0.461 | 0.0072 | 3639 | 3384 |

CUDA smoke conclusion: v2.6 reproduced the local result. The matched
no-representation curriculum was still `winner_last_window`, so the v2.5 lift
is mostly a GoToObj task-family/curriculum effect under this bounded smoke.
`ZM_torch_gotoobj_state_plus_delta_matched_delay` remains the best
representation candidate by return, but it is not a stable success winner yet.
