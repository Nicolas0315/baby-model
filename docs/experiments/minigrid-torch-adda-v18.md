# PyTorch AD/DA v1.8 Dense Controllability Ladder

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/23

## Motivation

The v1.7 controllability objective produced a repeatable single-seed CPU/CUDA
signal, but its three-seed CUDA sweep was not robust. The representation target
may be directionally useful, but the developmental ladder may still be too
sparse before `BabyAI-Unlock-v0`.

## Hypothesis

A denser task ladder should make the controllability representation more
reusable. Instead of jumping from `BabyAI-GoToRedBall-v0` directly to
`BabyAI-Unlock-v0`, v1.8 inserts door navigation, door opening, key-door, and
local unlock stages before the final unlock evaluation.

## Config

Config: `configs/experiments/minigrid-torch-adda-v18.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 12 episodes.
- `goto_warmup`: `BabyAI-GoToRedBall-v0`, 24 episodes.
- `goto_door_warmup`: `MiniGrid-GoToDoor-5x5-v0`, 12 episodes.
- `open_door_warmup`: `BabyAI-OpenDoor-v0`, 12 episodes.
- `doorkey_warmup`: `MiniGrid-DoorKey-5x5-v0`, 16 episodes.
- `unlock_local_warmup`: `BabyAI-UnlockLocal-v0`, 20 episodes.
- `unlock_eval`: `BabyAI-Unlock-v0`, 48 episodes.

Conditions:

- `A_torch_hard_only_long`: final unlock stage only, hard-only baseline.
- `T_torch_controllability_delay`: v1.7-style sparse ladder and
  controllability representation baseline.
- `V_torch_dense_ladder_controllability_delay`: dense ladder and
  controllability representation.
- `W_torch_dense_ladder_controllability_aux_progress`: dense ladder,
  controllability representation, and auxiliary progress reward kept out of the
  main reward target.

## Decision Rule

Run a bounded local smoke first. Escalate v1.8 to CUDA only if a dense-ladder
condition beats both `A_torch_hard_only_long` and
`T_torch_controllability_delay` on final-stage last-window success, or ties both
while improving final-stage return with non-zero representation updates.

If the dense ladder only matches or loses to the v1.7-style `T` condition,
record v1.8 as negative ladder evidence and redesign the representation target
instead of spending GPU time.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v18-torch/20260629T083203Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | `unlock_eval` | 0.021 | 0.050 | 0.042 | 0.0000 | 0 | 5724 |
| `T_torch_controllability_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0122 | 7586 | 7331 |
| `V_torch_dense_ladder_controllability_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0034 | 11977 | 11731 |
| `W_torch_dense_ladder_controllability_aux_progress` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0190 | 11898 | 11643 |

Local conclusion: v1.8 did not meet the CUDA escalation rule. The hard-only
baseline won final-stage last-window success (`0.050`) and return (`0.042`),
while both dense-ladder controllability conditions stayed at `0.000` final-stage
last-window success and return. Treat this as negative ladder evidence: adding
door/key/local-unlock stages did not make the current controllability target
transfer to `BabyAI-Unlock-v0` in a bounded local smoke.
