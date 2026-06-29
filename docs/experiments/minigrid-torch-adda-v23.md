# PyTorch AD/DA v2.3 Denser Key-Door Ladder

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/28

## Motivation

v2.2 showed that a richer state-plus-delta target can produce a weak all-window
signal, but it still did not improve final-stage last-window success or return.
The next experiment changes the developmental ladder so the representation head
sees more observable key pickup, locked-door, open-door, and goal transitions
before final `BabyAI-Unlock-v0` evaluation.

## Chosen Path

Use a denser key-door ladder with the existing `state_plus_delta` target. This
keeps the representation objective fixed and changes only the pre-final
experience distribution.

The ladder extends the v1.8 dense ladder by adding `MiniGrid-UnlockPickup-v0`,
which exercises a key-door-pickup chain before final unlock evaluation.

## Hypothesis

The state-plus-delta target may need denser observable transitions before it can
transfer to final unlock evaluation. If this is true, dense key-door conditions
should beat the sparse controllability baseline on final-stage last-window
success or return.

## Config

Config: `configs/experiments/minigrid-torch-adda-v23.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 12 episodes.
- `goto_warmup`: `BabyAI-GoToRedBall-v0`, 24 episodes.
- `goto_door_warmup`: `MiniGrid-GoToDoor-5x5-v0`, 12 episodes.
- `open_door_warmup`: `BabyAI-OpenDoor-v0`, 12 episodes.
- `doorkey_warmup`: `MiniGrid-DoorKey-5x5-v0`, 16 episodes.
- `unlock_local_warmup`: `BabyAI-UnlockLocal-v0`, 20 episodes.
- `unlock_pickup_warmup`: `MiniGrid-UnlockPickup-v0`, 20 episodes.
- `unlock_eval`: `BabyAI-Unlock-v0`, 48 episodes.

Conditions:

- `A_torch_hard_only_long`: final unlock stage only, hard-only baseline.
- `T_torch_controllability_delay`: v1.7-style sparse ladder and binary
  controllability representation baseline.
- `ZF_torch_dense_keydoor_state_plus_delta_delay`: dense key-door ladder and
  state-plus-delta representation prediction.
- `ZG_torch_dense_keydoor_state_plus_delta_aux_progress`: dense key-door ladder,
  state-plus-delta representation prediction, and auxiliary progress reward
  kept out of the main reward target.

## Decision Rule

Run a bounded local smoke first. Escalate v2.3 to CUDA only if a dense key-door
condition beats both `A_torch_hard_only_long` and
`T_torch_controllability_delay` on final-stage last-window success, or ties both
while improving final-stage return with non-zero representation updates.

If dense key-door state-plus-delta fails to beat or tie both baselines under
that rule, record v2.3 as negative ladder evidence and redesign around either a
task-family change or a non-DQN representation learner.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v23-torch/20260629T092714Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0000 | 0 | 5745 |
| `T_torch_controllability_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0066 | 7617 | 7365 |
| `ZF_torch_dense_keydoor_state_plus_delta_delay` | `unlock_eval` | 0.021 | 0.000 | 0.000 | 0.0133 | 14194 | 13939 |
| `ZG_torch_dense_keydoor_state_plus_delta_aux_progress` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0136 | 14471 | 14227 |

Local conclusion: v2.3 did not meet the CUDA escalation rule. The dense
key-door state-plus-delta delay condition produced one all-window final-stage
success (`success_all=0.021`), but both dense conditions tied both baselines on
final-stage last-window success (`0.000`) and did not improve final-stage return
(`0.000`). The warmup stages show learnable intermediate tasks, but this ladder
did not transfer to stable final unlock success. Treat this as negative ladder
evidence and redesign around either a task-family change or a non-DQN
representation learner.
