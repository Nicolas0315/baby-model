# PyTorch AD/DA v1.9 Affordance-Progress Representation

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/24

## Motivation

The v1.7 controllability target was weakly positive but not robust, and v1.8
showed that adding a denser task ladder did not make the same binary target
transfer to `BabyAI-Unlock-v0`. The next test changes the AD-side target rather
than the ladder.

## Research Notes

The target design follows three external anchors:

- BabyAI frames these environments as a sample-efficiency testbed for grounded
  instruction learning, so bounded smoke tests should be treated as early
  evidence rather than proof of robust learning:
  https://arxiv.org/abs/1810.08272
- UNREAL-style auxiliary tasks use reward-free auxiliary signals to shape shared
  representations in sparse-reward RL:
  https://openreview.net/forum?id=SJ6yPD5xg
- Gibson's affordance framing suggests encoding what the local environment
  affords an acting body, rather than only predicting passive visual features:
  https://cs.brown.edu/courses/cs137/2017/readings/Gibson-AFF.pdf

## Hypothesis

An action-conditioned affordance-progress target should be more useful than
binary controllability because it preserves structured information about local
action opportunities and task progress: front-cell key/door/goal/wall status,
door state, visible key/door/goal presence, and mission-conditioned missing
object proxies.

## Config

Config: `configs/experiments/minigrid-torch-adda-v19.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 12 episodes.
- `goto_warmup`: `BabyAI-GoToRedBall-v0`, 24 episodes.
- `unlock_eval`: `BabyAI-Unlock-v0`, 48 episodes.

Conditions:

- `A_torch_hard_only_long`: final unlock stage only, hard-only baseline.
- `T_torch_controllability_delay`: v1.7-style sparse ladder and binary
  controllability representation baseline.
- `X_torch_affordance_progress_delay`: sparse ladder and action-conditioned
  affordance-progress representation prediction.
- `Y_torch_affordance_progress_aux_progress`: sparse ladder, affordance-progress
  representation prediction, and auxiliary progress reward kept out of the main
  reward target.

## Decision Rule

Run a bounded local smoke first. Escalate v1.9 to CUDA only if an
affordance-progress condition beats both `A_torch_hard_only_long` and
`T_torch_controllability_delay` on final-stage last-window success, or ties both
while improving final-stage return with non-zero representation updates.

If affordance-progress fails to beat or tie both baselines under that rule,
record v1.9 as negative target evidence and redesign the target around
transition groups or explicit subgoal progress.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v19-torch/20260629T084331Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_hard_only_long` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0000 | 0 | 5745 |
| `T_torch_controllability_delay` | `unlock_eval` | 0.021 | 0.050 | 0.045 | 0.0165 | 7280 | 7025 |
| `X_torch_affordance_progress_delay` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0193 | 7514 | 7259 |
| `Y_torch_affordance_progress_aux_progress` | `unlock_eval` | 0.000 | 0.000 | 0.000 | 0.0175 | 7420 | 7165 |

Local conclusion: v1.9 did not meet the CUDA escalation rule. The
v1.7-style `T_torch_controllability_delay` baseline won final-stage
last-window success (`0.050`) and return (`0.045`), while both
affordance-progress conditions stayed at `0.000` final-stage last-window
success and return. Treat this as negative evidence for the current
affordance-progress vector; the next target should model transition groups or
explicit subgoal progress rather than only next-observation affordance bits.
