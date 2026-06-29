# PyTorch AD/DA v2.19 Combined State Plus Semantic Smoke

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/44

## Motivation

v2.18 showed that `target_visibility_transition` beats the matched
no-representation curriculum across CUDA seeds, but it did not clearly beat
`state_plus_delta`: success tied, return favored semantic transition, and seed
wins favored state-plus-delta. v2.19 tests whether the two signals are
complementary before spending more CUDA.

## Chosen Path

Use the combined-objective path from #44. The new
`state_plus_target_visibility` representation target concatenates:

- `state_plus_delta` (`58` dimensions)
- `target_visibility_transition` (`49` dimensions)

The resulting vector has `107` dimensions and reuses the existing PyTorch
vector-regression auxiliary representation head.

## Config

Config: `configs/experiments/minigrid-torch-adda-v37.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 6 episodes.
- `goto_red_ball_warmup`: `BabyAI-GoToRedBall-v0`, 12 episodes.
- `goto_obj_eval`: `BabyAI-GoToObj-v0`, 24 episodes.

Conditions:

- `ZK_torch_gotoobj_curriculum_no_repr_delay`
- `ZM_torch_gotoobj_state_plus_delta_matched_delay`
- `ZN_torch_gotoobj_target_visibility_matched_delay`
- `ZO_torch_gotoobj_state_plus_target_visibility_delay`

## Decision Rule

Run a bounded local CPU smoke first. Escalate to CUDA only if
`ZO_torch_gotoobj_state_plus_target_visibility_delay` beats or ties both `ZM`
and `ZN` on final-stage last-window success while not lowering final-stage
return against either of them, improving final-stage return over at least one,
and running non-zero representation updates.

Do not claim semantic superiority unless it beats `ZM` on success or preserves
equal success with better return and median behavior across seeds.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch/MiniGrid
venv with `torch==2.12.1` and `device=cpu`.

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v37.json --output-dir .tmp/local-v37-torch --seed 3101`

Summary artifact:

`.tmp/local-v37-torch/20260629T133602Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | `goto_obj_eval` | 0.417 | 0.350 | 0.219 | 0.0000 | 0 | 1882 |
| `ZM_torch_gotoobj_state_plus_delta_matched_delay` | `goto_obj_eval` | 0.292 | 0.300 | 0.157 | 0.0044 | 2185 | 2050 |
| `ZN_torch_gotoobj_target_visibility_matched_delay` | `goto_obj_eval` | 0.083 | 0.100 | 0.062 | 0.0031 | 2331 | 2196 |
| `ZO_torch_gotoobj_state_plus_target_visibility_delay` | `goto_obj_eval` | 0.542 | 0.600 | 0.376 | 0.0060 | 1760 | 1625 |

Decision: v2.19 met the stricter CPU escalation rule. `ZO` beat both `ZM` and
`ZN` on final-stage last-window success and return, and ran non-zero
representation updates. It also beat the matched no-representation curriculum
in this smoke.

Conclusion: the combined `state_plus_target_visibility` objective is promising
enough for a bounded CUDA replication lane. This is still single-seed CPU
evidence, not CUDA or multi-seed proof.

## CUDA Replication

A bounded single-seed CUDA smoke completed on `gpu-worker-c` at source commit
`b94f765331d807f89fde8f119e02461641e9218d` with `torch==2.12.1+cu132`,
`torch_cuda_available=True`, and `device=cuda`.

Summary artifact on the worker:

`.tmp/verify-minigrid/torch/20260629T134438Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | `goto_obj_eval` | 0.417 | 0.350 | 0.219 | 0.0000 | 0 | 1882 |
| `ZM_torch_gotoobj_state_plus_delta_matched_delay` | `goto_obj_eval` | 0.292 | 0.300 | 0.157 | 0.0044 | 2185 | 2050 |
| `ZN_torch_gotoobj_target_visibility_matched_delay` | `goto_obj_eval` | 0.083 | 0.100 | 0.062 | 0.0031 | 2331 | 2196 |
| `ZO_torch_gotoobj_state_plus_target_visibility_delay` | `goto_obj_eval` | 0.542 | 0.600 | 0.376 | 0.0060 | 1760 | 1625 |

Decision: v2.20 met the CUDA replication rule. `ZO` beat both `ZM` and `ZN`
on final-stage last-window success and return, beat the matched
no-representation curriculum, and ran non-zero representation updates.

Conclusion: v2.20 is positive single-seed CUDA replication evidence for the
combined `state_plus_target_visibility` objective. It still does not prove
multi-seed stability; the next step should be a bounded CUDA multi-seed sweep.

## CUDA Multi-Seed Sweep

A bounded three-seed CUDA sweep completed on `gpu-worker-c` at source commit
`41389b1a24b46ff7385c9fedeeb2ed9f2c3d3b5b` with `torch==2.12.1+cu132`,
`torch_cuda_available=True`, and `devices=cuda`.

Summary artifact on the worker:

`.tmp/verify-minigrid/torch-sweep/20260629T135815Z/summary.md`

Seeds: `3101,3102,3103`

| condition | wins | mean_success_all | mean_success_last | median_success_last | mean_return_last | median_return_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 0 | 0.361 | 0.333 | 0.350 | 0.215 | 0.219 |
| `ZM_torch_gotoobj_state_plus_delta_matched_delay` | 1 | 0.444 | 0.450 | 0.300 | 0.275 | 0.180 |
| `ZN_torch_gotoobj_target_visibility_matched_delay` | 1 | 0.403 | 0.417 | 0.450 | 0.289 | 0.269 |
| `ZO_torch_gotoobj_state_plus_target_visibility_delay` | 1 | 0.319 | 0.350 | 0.350 | 0.208 | 0.207 |

Per-seed winners:

- `3101`: `ZO_torch_gotoobj_state_plus_target_visibility_delay`
- `3102`: `ZN_torch_gotoobj_target_visibility_matched_delay`
- `3103`: `ZM_torch_gotoobj_state_plus_delta_matched_delay`

Decision: v2.21 did not meet the multi-seed rule. `ZO` won one seed and tied
or beat `ZK` on final-window success, but it lost mean final-window success and
mean final-window return to both `ZM` and `ZN`.

Conclusion: the combined objective has positive single-seed CPU/CUDA evidence,
but the signal did not generalize across this three-seed CUDA sweep. The next
lane should stop escalating the simple concatenation target and instead add a
direct mission-preservation probe or redesign the auxiliary objective.
