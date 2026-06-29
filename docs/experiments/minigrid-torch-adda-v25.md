# PyTorch AD/DA v2.5 Task-Family Change to BabyAI GoToObj

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/30

## Motivation

v2.4 made the AD then DA protocol explicit but still failed on final
`BabyAI-Unlock-v0`. Before adding another representation mechanism, this
experiment checks whether the current AD/DA variants can show signal on a less
brittle BabyAI task family.

## Chosen Path

Use Option A from #30: task-family change.

The final evaluation moves from sparse unlock behavior to `BabyAI-GoToObj-v0`.
This keeps language-conditioned object navigation and sparse success, but avoids
the key-door unlock chain that has dominated recent failures.

## Hypothesis

If the current failures are mostly caused by the sparse unlock benchmark, then
AD/DA variants should show clearer signal on GoToObj. If all variants still
fail or only hard-only wins, the next step should be a separate non-DQN
representation learner or a supervised representation probe.

## Config

Config: `configs/experiments/minigrid-torch-adda-v25.json`

Stages:

- `empty_warmup`: `MiniGrid-Empty-5x5-v0`, 12 episodes.
- `goto_red_ball_warmup`: `BabyAI-GoToRedBall-v0`, 24 episodes.
- `goto_obj_eval`: `BabyAI-GoToObj-v0`, 48 episodes.

Conditions:

- `A_torch_gotoobj_hard_only`: final GoToObj stage only, hard-only baseline.
- `T_torch_gotoobj_controllability_delay`: GoToObj family curriculum with
  binary controllability representation.
- `ZI_torch_gotoobj_state_plus_delta_delay`: GoToObj family curriculum with
  joint state-plus-delta representation.
- `ZJ_torch_gotoobj_two_phase_state_plus_delta_frozen`: GoToObj family
  curriculum with 36-episode AD-only representation learning, then frozen
  encoder DA training on final GoToObj.

## Decision Rule

Run a bounded local smoke first. Escalate v2.5 to CUDA only if an AD/DA
condition beats `A_torch_gotoobj_hard_only` on final-stage last-window success,
or ties it while improving final-stage return with non-zero representation
updates. For `ZJ_torch_gotoobj_two_phase_state_plus_delta_frozen`,
`encoder_frozen=true` must also be present.

If no AD/DA condition beats or ties the hard-only baseline under that rule,
record v2.5 as negative task-family evidence and move to a separate non-DQN
representation learner or supervised representation probe.

## Current Result

A bounded local CPU smoke completed in the existing optional PyTorch venv with
`torch==2.12.1` and `device=cpu`.

Summary artifact:

`.tmp/local-v25-torch/20260629T095110Z/summary.md`

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_gotoobj_hard_only` | `goto_obj_eval` | 0.292 | 0.250 | 0.128 | 0.0000 | 0 | 2637 |
| `T_torch_gotoobj_controllability_delay` | `goto_obj_eval` | 0.396 | 0.550 | 0.331 | 0.0001 | 4261 | 4006 |
| `ZI_torch_gotoobj_state_plus_delta_delay` | `goto_obj_eval` | 0.208 | 0.300 | 0.201 | 0.0080 | 4564 | 4316 |
| `ZJ_torch_gotoobj_two_phase_state_plus_delta_frozen` | `goto_obj_eval` | 0.208 | 0.200 | 0.132 | 0.0000 | 1481 | 2740 |

Local conclusion: `T_torch_gotoobj_controllability_delay` met the v2.5
escalation rule by beating the hard-only baseline on final-stage last-window
success (`0.550` vs `0.250`) and return (`0.331` vs `0.128`) with non-zero
representation updates. Escalate this task-family condition set to a bounded
CUDA smoke before treating the signal as useful.

A bounded CUDA smoke completed on `gpu-worker-c` with `torch==2.12.1+cu132`,
`torch_cuda_available=True`, and `device=cuda`. The remote source snapshot used
commit `f14a56d3d18d90485a15fed989d681b6493e0f82` plus the v2.5 config in this
change.

| condition | final_stage | success_all | success_last | return_last | rep_loss | rep_updates | updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_gotoobj_hard_only` | `goto_obj_eval` | 0.292 | 0.250 | 0.128 | 0.0000 | 0 | 2637 |
| `T_torch_gotoobj_controllability_delay` | `goto_obj_eval` | 0.396 | 0.550 | 0.331 | 0.0001 | 4261 | 4006 |
| `ZI_torch_gotoobj_state_plus_delta_delay` | `goto_obj_eval` | 0.208 | 0.300 | 0.201 | 0.0080 | 4564 | 4316 |
| `ZJ_torch_gotoobj_two_phase_state_plus_delta_frozen` | `goto_obj_eval` | 0.208 | 0.200 | 0.132 | 0.0000 | 1481 | 2740 |

CUDA smoke conclusion: `T_torch_gotoobj_controllability_delay` reproduced the
local positive smoke on strict CUDA. Treat v2.5 as the first clear task-family
positive signal, but not yet robust until a multi-seed CUDA sweep confirms it.

## Three-Seed CUDA Sweep

A three-seed CUDA sweep completed on `gpu-worker-c` with
`torch==2.12.1+cu132`, `torch_cuda_available=True`, and `device=cuda`.

Seeds: `2101,2102,2103`

| condition | wins | mean_success_all | mean_success_last | median_success_last | mean_return_last | median_return_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `A_torch_gotoobj_hard_only` | 1 | 0.326 | 0.450 | 0.500 | 0.280 | 0.260 |
| `T_torch_gotoobj_controllability_delay` | 1 | 0.326 | 0.483 | 0.550 | 0.276 | 0.293 |
| `ZI_torch_gotoobj_state_plus_delta_delay` | 1 | 0.465 | 0.533 | 0.500 | 0.342 | 0.370 |
| `ZJ_torch_gotoobj_two_phase_state_plus_delta_frozen` | 0 | 0.319 | 0.400 | 0.300 | 0.269 | 0.190 |

Per-seed winners:

- `2101`: `T_torch_gotoobj_controllability_delay`
- `2102`: `A_torch_gotoobj_hard_only`
- `2103`: `ZI_torch_gotoobj_state_plus_delta_delay`

Sweep conclusion: v2.5 gives the strongest AD/DA evidence so far on the
GoToObj task family. `ZI_torch_gotoobj_state_plus_delta_delay` beat the
hard-only baseline by mean final-window success (`0.533` vs `0.450`) and mean
return (`0.342` vs `0.280`), while `T_torch_gotoobj_controllability_delay`
had the best median final-window success (`0.550`). The seed win count is still
split one win each across `T`, hard-only, and `ZI`, so treat this as positive
task-family evidence rather than a fully stable condition winner.
