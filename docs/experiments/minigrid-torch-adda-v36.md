# PyTorch AD/DA v2.16 Semantic Transition RL Integration

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/41

## Motivation

v2.15 produced positive multi-seed non-DQN evidence for
`target_visibility_transition` under scripted-policy collection. v2.16 is the
first bounded RL integration check: add the semantic transition objective to
the existing GoToObj PyTorch lane and compare it with the matched
no-representation curriculum and the prior best state-plus-delta candidate.

## Chosen Path

Use option 1 from #41: add a semantic-transition auxiliary representation
objective to the existing GoToObj PyTorch RL lane.

The new objective predicts a 49-way one-hot vector for the mission target's
before/after relation to the agent:

- `absent`
- `left_near`, `left_far`
- `center_near`, `center_far`
- `right_near`, `right_far`

This reuses the existing PyTorch vector-regression representation head. It is
not considered RL transfer evidence until a real RL smoke improves task
metrics under the documented rule.

## Config

Config: `configs/experiments/minigrid-torch-adda-v36.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 6 episodes.
- `goto_red_ball_warmup`: `BabyAI-GoToRedBall-v0`, 12 episodes.
- `goto_obj_eval`: `BabyAI-GoToObj-v0`, 24 episodes.

Conditions:

- `ZK_torch_gotoobj_curriculum_no_repr_delay`
- `ZM_torch_gotoobj_state_plus_delta_matched_delay`
- `ZN_torch_gotoobj_target_visibility_matched_delay`

## Decision Rule

Run a bounded local CPU smoke first. Treat v2.16 as positive only if
`ZN_torch_gotoobj_target_visibility_matched_delay` beats
`ZK_torch_gotoobj_curriculum_no_repr_delay` on final-stage last-window
success, or ties it while improving final-stage return, with non-zero
representation updates. Also compare against
`ZM_torch_gotoobj_state_plus_delta_matched_delay`; if semantic transition loses
to both matched baselines, do not escalate to CUDA.

The RL runner does not directly evaluate mission-object or mission-color probe
accuracy, so mission preservation is not a direct probe gate in v2.16. For this
CPU integration smoke, final-stage external success and return on
`BabyAI-GoToObj-v0` are the preservation proxy: a semantic objective must not
trade away the actual mission-following task metrics relative to the matched
no-representation curriculum.

Do not run CUDA unless the CPU smoke meets this rule.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch/MiniGrid
venv with `torch==2.12.1` and `device=cpu`.

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v36.json --output-dir .tmp/local-v36-torch --seed 3001`

Summary artifact:

`.tmp/local-v36-torch/20260629T131151Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | `goto_obj_eval` | 0.167 | 0.100 | 0.031 | 0.0000 | 0 | 2065 |
| `ZM_torch_gotoobj_state_plus_delta_matched_delay` | `goto_obj_eval` | 0.083 | 0.100 | 0.066 | 0.0077 | 2334 | 2205 |
| `ZN_torch_gotoobj_target_visibility_matched_delay` | `goto_obj_eval` | 0.375 | 0.400 | 0.274 | 0.0047 | 1997 | 1862 |

Decision: v2.16 met the local CPU rule. The semantic-transition condition
beat the matched no-representation curriculum on final-stage last-window
success (`0.400` vs `0.100`) and final-stage return (`0.274` vs `0.031`) with
non-zero representation updates, satisfying the mission-preservation proxy. It
also beat the state-plus-delta candidate on both metrics in this bounded smoke.

Conclusion: this is positive CPU-first RL integration evidence for the
semantic transition signal. It still does not prove CUDA or multi-seed
stability; the next step should be a bounded CUDA replication lane before
promoting the objective as a stable RL result.
