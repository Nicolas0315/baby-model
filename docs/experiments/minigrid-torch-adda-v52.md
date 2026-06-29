# PyTorch AD/DA v2.41 Longer-Horizon Check for ZE

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/65

## Motivation

v2.40 confirmed `ZE_torch_gotoobj_state_plus_mission_target_b005` on the short
`6+12+24` curriculum across five CUDA seeds. v2.41 changes the evaluation
surface to a longer `12+24+48` curriculum with decoder delay `8`, using `ZE`
as the baseline to beat.

## Source State

- Config: `configs/experiments/minigrid-torch-adda-v47.json`
- Local runner: current Mac
- Torch: `2.12.1`
- Device: `cpu`
- Seeds: `4201,4202,4203`

## CPU Sweep

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch_sweep --config configs/experiments/minigrid-torch-adda-v47.json --output-dir .tmp/local-v65-ze-long --seeds 4201,4202,4203 --device cpu`

Summary artifact:

`.tmp/local-v65-ze-long/20260629T170916Z/summary.md`

| condition | wins | mean_success_last | median_success_last | mean_return_last | median_return_last | target_visible_last | target_center_last | target_near_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZE_torch_gotoobj_state_plus_mission_target_b005_long` | 1 | 0.400 | 0.250 | 0.239 | 0.162 | 0.633 | 0.433 | 0.450 |
| `ZF_torch_gotoobj_state_plus_mission_target_b0075_long` | 0 | 0.300 | 0.400 | 0.176 | 0.200 | 0.483 | 0.317 | 0.333 |
| `ZK_torch_gotoobj_curriculum_no_repr_delay_long` | 2 | 0.433 | 0.500 | 0.277 | 0.328 | 0.667 | 0.483 | 0.483 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075_long` | 0 | 0.300 | 0.400 | 0.201 | 0.262 | 0.533 | 0.300 | 0.317 |

Per-seed winners:

- `4201`: `ZK_torch_gotoobj_curriculum_no_repr_delay_long`
- `4202`: `ZE_torch_gotoobj_state_plus_mission_target_b005_long`
- `4203`: `ZK_torch_gotoobj_curriculum_no_repr_delay_long`

## Decision

Do not promote `ZE` as a universal baseline from the short-protocol evidence
alone.

The longer-horizon CPU gate preserved part of the `ZE` signal: `ZE` beat `ZU`
on mean final-window success, mean return, target-visible, target-center, and
target-near. However, the no-representation long curriculum `ZK_long` won two
of three seeds and led the aggregate success, return, and mission-probe
columns.

This shifts the next research question from "replicate `ZE` again" to "why does
representation help the short protocol but lose to no-representation under the
longer horizon?" The next branch should isolate whether the longer horizon
needs lower beta, delayed or stopped representation updates, or a curriculum
reference that hands off representation pressure after warmup.
