# PyTorch AD/DA v2.4 Two-Phase Frozen Encoder Protocol

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/29

## Motivation

v2.3 showed that adding more key-door warmup tasks still did not transfer to
stable final `BabyAI-Unlock-v0` success under the current joint DQN plus
auxiliary-head setup. The next experiment should make the baby-model AD/DA
asymmetry explicit: first train perception/representation, then train the
decoder while the learned encoder is held fixed.

## Chosen Path

Use Option C from #29: a two-phase AD then DA protocol.

- AD phase: random-action representation learning only.
- Phase boundary: sync the DQN target network and freeze the shared encoder.
- DA phase: stop representation updates and train the DQN decoder/head on final
  unlock behavior.

This differs from earlier `decoder_delay_episodes`, which delayed DQN updates
but kept representation updates running after the delay and did not freeze the
encoder.

## Hypothesis

If the current failures come from DA updates overwriting or destabilizing the
AD representation, then freezing the encoder after an explicit AD-only phase
should improve final-stage sample efficiency or last-window success.

## Config

Config: `configs/experiments/minigrid-torch-adda-v24.json`

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
- `ZF_torch_dense_keydoor_state_plus_delta_delay`: v2.3 dense key-door ladder
  and joint state-plus-delta representation baseline.
- `ZH_torch_two_phase_state_plus_delta_frozen`: dense key-door ladder with
  116-episode AD-only state-plus-delta training, then frozen-encoder DA
  training on final unlock.

## Decision Rule

Run a bounded local smoke first. Escalate v2.4 to CUDA only if
`ZH_torch_two_phase_state_plus_delta_frozen` beats both
`A_torch_hard_only_long` and `ZF_torch_dense_keydoor_state_plus_delta_delay` on
final-stage last-window success, or ties both while improving final-stage return
with non-zero AD representation updates and `encoder_frozen=true`.

If two-phase frozen-encoder training fails under that rule, record v2.4 as
negative evidence for the current PyTorch DQN family and move to a task-family
change or a separate non-DQN representation learner.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v24-torch/20260629T094008Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_updates | updates | encoder_frozen |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `A_torch_hard_only_long` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0 | 5745 | false |
| `T_torch_controllability_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 7483 | 7233 | false |
| `ZF_torch_dense_keydoor_state_plus_delta_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 14639 | 14385 | false |
| `ZH_torch_two_phase_state_plus_delta_frozen` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 8480 | 5745 | true |

Local conclusion: v2.4 did not meet the CUDA escalation rule. The two-phase
condition correctly performed AD representation updates and froze the encoder
before DA training (`encoder_frozen=true`), but it tied both baselines on
final-stage last-window success (`0.000`) and did not improve final-stage return
(`0.000`). Treat this as negative evidence for the current PyTorch DQN family:
making AD then DA explicit is not enough without changing the task family or
moving representation learning outside this joint DQN architecture.
