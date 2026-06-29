# PyTorch AD/DA v2.24 State-Plus-Target-Visibility Beta Sweep

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/48

## Motivation

v2.21 showed that the simple combined `state_plus_target_visibility` target
was not stable across CUDA seeds at `representation_beta=0.3`. v2.23 showed
that masking the semantic target with a mission-conditioned zero vector was too
aggressive and failed the CPU gate.

v2.24 tests the smaller protocol change suggested by auxiliary-loss balancing:
keep the combined target, but reduce or increase its loss weight before
changing the model contract to true two-head training.

## Config

- `configs/experiments/minigrid-torch-adda-v39.json`

Compared conditions:

- `ZK_torch_gotoobj_curriculum_no_repr_delay`
- `ZM_torch_gotoobj_state_plus_delta_matched_delay`
- `ZN_torch_gotoobj_target_visibility_matched_delay`
- `ZR_torch_gotoobj_state_plus_target_visibility_b010`
- `ZO_torch_gotoobj_state_plus_target_visibility_b030`
- `ZS_torch_gotoobj_state_plus_target_visibility_b050`

## CPU Smoke

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v39.json --output-dir .tmp/local-v49-beta-sweep --seed 3401`

Summary artifact:

`.tmp/local-v49-beta-sweep/20260629T142051Z/summary.md`

| condition | beta | success_last | return_last | target_visible_last | target_center_last | target_near_last | rep_updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 0.0 | 0.150 | 0.063 | 0.250 | 0.150 | 0.200 | 0 |
| `ZM_torch_gotoobj_state_plus_delta_matched_delay` | 0.3 | 0.100 | 0.050 | 0.350 | 0.100 | 0.200 | 2321 |
| `ZN_torch_gotoobj_target_visibility_matched_delay` | 0.3 | 0.100 | 0.083 | 0.450 | 0.100 | 0.300 | 2339 |
| `ZR_torch_gotoobj_state_plus_target_visibility_b010` | 0.1 | 0.450 | 0.247 | 0.900 | 0.550 | 0.700 | 1991 |
| `ZO_torch_gotoobj_state_plus_target_visibility_b030` | 0.3 | 0.200 | 0.118 | 0.550 | 0.250 | 0.250 | 2344 |
| `ZS_torch_gotoobj_state_plus_target_visibility_b050` | 0.5 | 0.350 | 0.207 | 0.650 | 0.450 | 0.400 | 2172 |

## Decision

`ZR` passed the CPU gate. It beat the no-representation curriculum and both
single-objective representation baselines on:

- final-window success
- final-window return
- mission-target visible, center, and near probe rates

This is sufficient to open a bounded CUDA replication issue for the beta 0.1
combined objective. It is not multi-seed proof.
