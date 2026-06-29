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

## CUDA Replication

A bounded single-seed CUDA smoke completed on `gpu-worker-c` at source commit
`8f628a562324fdcd7ea19209edf665a0fb027f0b` with `torch==2.12.1+cu132`,
`torch_cuda_available=True`, and `device=cuda`.

Summary artifact on the worker:

`.tmp/verify-minigrid/torch/20260629T132251Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | `goto_obj_eval` | 0.167 | 0.100 | 0.031 | 0.0000 | 0 | 2065 |
| `ZM_torch_gotoobj_state_plus_delta_matched_delay` | `goto_obj_eval` | 0.083 | 0.100 | 0.066 | 0.0077 | 2334 | 2205 |
| `ZN_torch_gotoobj_target_visibility_matched_delay` | `goto_obj_eval` | 0.375 | 0.400 | 0.274 | 0.0047 | 1997 | 1862 |

Conclusion: v2.16 now has positive CPU-first RL integration evidence and a
matching bounded CUDA replication on `gpu-worker-c`. It still does not prove
multi-seed stability; the next step should be a CUDA multi-seed matched sweep
before promoting the objective as a stable RL result.

## CUDA Multi-Seed Sweep

A bounded three-seed CUDA sweep completed on `gpu-worker-c` at source commit
`b3472a04c91205b263c5f7ea308cecc94da0f69e` with `torch==2.12.1+cu132`,
`torch_cuda_available=True`, and `devices=cuda`.

Summary artifact on the worker:

`.tmp/verify-minigrid/torch-sweep/20260629T133005Z/summary.md`

Seeds: `3001,3002,3003`

| condition | wins | mean_success_all | mean_success_last | median_success_last | mean_return_last | median_return_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 0 | 0.111 | 0.100 | 0.100 | 0.059 | 0.063 |
| `ZM_torch_gotoobj_state_plus_delta_matched_delay` | 2 | 0.264 | 0.283 | 0.300 | 0.158 | 0.179 |
| `ZN_torch_gotoobj_target_visibility_matched_delay` | 1 | 0.250 | 0.283 | 0.300 | 0.185 | 0.221 |

Per-seed winners:

- `3001`: `ZN_torch_gotoobj_target_visibility_matched_delay`
- `3002`: `ZM_torch_gotoobj_state_plus_delta_matched_delay`
- `3003`: `ZM_torch_gotoobj_state_plus_delta_matched_delay`

Decision: the semantic-transition condition passed the multi-seed rule against
the matched no-representation curriculum: it improved mean final-window
success and return, and its median final-window success was not worse. It did
not become a clear single-condition winner against `state_plus_delta`: `ZN`
tied `ZM` on mean and median final-window success, improved return, but lost
seed win count (`1` vs `2`).

Conclusion: v2.18 is positive representation-vs-no-repr multi-seed CUDA
evidence for the semantic transition signal, but not proof that this semantic
objective is the stable best representation target. The next lane should test
whether `state_plus_delta` and `target_visibility_transition` are
complementary, or add a direct mission-preservation probe to separate reward
return from mission-signal retention.
