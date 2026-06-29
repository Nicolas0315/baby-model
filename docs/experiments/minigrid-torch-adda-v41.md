# PyTorch AD/DA v2.30 Longer-Horizon ZU Probe Gate

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/54

## Motivation

v2.29 made `ZU_torch_gotoobj_state_plus_target_visibility_b0075` the strongest
beta-neighborhood candidate in a bounded three-seed CUDA sweep. v2.30 checks
whether that signal survives a longer CPU horizon before spending more CUDA.

This protocol doubles the stage lengths from the short beta sweeps:

- `empty_warmup`: 6 -> 12 episodes
- `goto_red_ball_warmup`: 12 -> 24 episodes
- `goto_obj_eval`: 24 -> 48 episodes
- decoder delay: 4 -> 8 episodes

## Config

- `configs/experiments/minigrid-torch-adda-v41.json`

Compared conditions:

- `ZK_torch_gotoobj_curriculum_no_repr_delay_long`
- `ZT_torch_gotoobj_state_plus_target_visibility_b005_long`
- `ZU_torch_gotoobj_state_plus_target_visibility_b0075_long`
- `ZR_torch_gotoobj_state_plus_target_visibility_b010_long`

## CPU Smoke

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v41.json --output-dir .tmp/local-v54-long-horizon --seed 3601`

Summary artifact:

`.tmp/local-v54-long-horizon/20260629T152021Z/summary.md`

| condition | beta | success_last | return_last | target_visible_last | target_center_last | target_near_last | rep_updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay_long` | 0.0 | 0.550 | 0.262 | 0.750 | 0.550 | 0.550 | 0 |
| `ZT_torch_gotoobj_state_plus_target_visibility_b005_long` | 0.05 | 0.250 | 0.172 | 0.600 | 0.250 | 0.250 | 4331 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075_long` | 0.075 | 0.500 | 0.367 | 0.750 | 0.550 | 0.600 | 3765 |
| `ZR_torch_gotoobj_state_plus_target_visibility_b010_long` | 0.1 | 0.100 | 0.030 | 0.250 | 0.150 | 0.100 | 4618 |

## Decision

The longer-horizon CPU gate is mixed, not positive for CUDA escalation.
`ZU` beat `ZK` on final-window return and `target_near_last`, and tied
`target_visible_last` and `target_center_last`, but `ZK` won final-window
success (`0.550` vs `0.500`).

Do not open a CUDA follow-up from v2.30 alone. The next useful branch is a true
two-head objective or a protocol that can keep ZU's return/probe benefits
without losing final-window success.
