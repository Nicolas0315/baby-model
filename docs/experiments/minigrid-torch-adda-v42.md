# PyTorch AD/DA v2.31 True Two-Head Objective

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/55

## Motivation

v2.29 made `ZU_torch_gotoobj_state_plus_target_visibility_b0075` the strongest
bounded CUDA beta-neighborhood candidate. v2.30 showed that a longer horizon did
not cleanly preserve that edge against the no-representation curriculum. v2.31
therefore tests whether separating the combined target into two explicit heads
helps more than further scalar beta tuning.

## Protocol

Config: `configs/experiments/minigrid-torch-adda-v42.json`

The new objective is `state_delta_and_target_visibility`. It uses two separate
prediction heads over the same encoded state plus action input:

- `state_plus_delta`: 58-dimensional state/transition/subgoal target
- `target_visibility_transition`: 49-dimensional mission-target visibility
  transition target

Compared conditions:

- `ZK_torch_gotoobj_curriculum_no_repr_delay`
- `ZU_torch_gotoobj_state_plus_target_visibility_b0075`
- `ZX_torch_gotoobj_two_head_state050_visibility025`
- `ZY_torch_gotoobj_two_head_state0375_visibility0375`

## CPU Smoke

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v42.json --output-dir .tmp/local-v55-two-head --seed 3701`

Summary artifact:

`.tmp/local-v55-two-head/20260629T153027Z/summary.md`

| condition | objective | state_beta | visibility_beta | success_last | return_last | target_visible_last | target_center_last | target_near_last | rep_updates |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | none | 0.0 | 0.0 | 0.300 | 0.204 | 0.550 | 0.300 | 0.350 | 0 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | single combined | 0.0 | 0.0 | 0.500 | 0.367 | 0.750 | 0.550 | 0.550 | 2033 |
| `ZX_torch_gotoobj_two_head_state050_visibility025` | two-head | 0.05 | 0.025 | 0.200 | 0.156 | 0.550 | 0.200 | 0.200 | 2199 |
| `ZY_torch_gotoobj_two_head_state0375_visibility0375` | two-head | 0.0375 | 0.0375 | 0.300 | 0.151 | 0.700 | 0.300 | 0.400 | 2192 |

## Decision

The two-head contract is implemented and verified, but this CPU gate is
negative for escalation. Both two-head variants underperformed the existing
single combined `ZU` baseline on final-window success and return.

Do not launch CUDA for v2.31. The next branch should change the protocol rather
than just split the same target: for example, add per-head loss diagnostics and
annealing, or test a phase-gated representation schedule that protects the
decoder from early auxiliary loss.
