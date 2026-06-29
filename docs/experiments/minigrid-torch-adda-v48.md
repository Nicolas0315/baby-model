# PyTorch AD/DA v2.37 Mission-Preservation-First CPU Precheck

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/61

## Motivation

v2.36 rejected `ZB`/`ZD` as replacements for the `ZU` baseline. `ZD` preserved
target-visible and target-near probes better than `ZB`, but still lost to `ZU`
on success and return.

v2.37 returns to the existing mission-conditioned combined objective at a much
lower beta. The earlier mission-conditioned gate used beta `0.3` and collapsed
task reward; this precheck asks whether a lighter `state_plus_mission_target`
loss can keep the mission probe gains without losing `ZU`'s task performance.

## Source State

- Config: `configs/experiments/minigrid-torch-adda-v46.json`
- Local runner: current Mac
- Torch: `2.12.1`
- Device: `cpu`
- Seeds: `4101,4102,4103,4104,4105`

## CPU Sweep

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch_sweep --config configs/experiments/minigrid-torch-adda-v46.json --output-dir .tmp/local-v61-mission-preservation --seeds 4101,4102,4103,4104,4105 --device cpu`

Summary artifact:

`.tmp/local-v61-mission-preservation/20260629T162559Z/summary.md`

| condition | wins | mean_success_last | median_success_last | mean_return_last | median_return_last | target_visible_last | target_center_last | target_near_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZD_torch_gotoobj_two_head_state010_visibility065_state_anneal` | 0 | 0.180 | 0.150 | 0.093 | 0.068 | 0.470 | 0.190 | 0.220 |
| `ZE_torch_gotoobj_state_plus_mission_target_b005` | 4 | 0.290 | 0.350 | 0.177 | 0.222 | 0.620 | 0.310 | 0.390 |
| `ZF_torch_gotoobj_state_plus_mission_target_b0075` | 0 | 0.140 | 0.100 | 0.064 | 0.059 | 0.560 | 0.180 | 0.230 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | 1 | 0.210 | 0.250 | 0.132 | 0.177 | 0.470 | 0.240 | 0.270 |

Per-seed winners:

- `4101`: `ZE_torch_gotoobj_state_plus_mission_target_b005`
- `4102`: `ZE_torch_gotoobj_state_plus_mission_target_b005`
- `4103`: `ZE_torch_gotoobj_state_plus_mission_target_b005`
- `4104`: `ZU_torch_gotoobj_state_plus_target_visibility_b0075`
- `4105`: `ZE_torch_gotoobj_state_plus_mission_target_b005`

## Decision

`ZE_torch_gotoobj_state_plus_mission_target_b005` passed the CPU gate.

It beat `ZU` on win count, mean and median final-window success, mean and
median final-window return, target-visible, target-center, and target-near
averages. The matched beta `ZF` underperformed, so the useful signal is not
simply replacing the target vector; it is the lighter beta `0.05`
mission-conditioned combined objective.

Promote `ZE` to a bounded CUDA replication before treating it as a baseline
replacement. Do not replace `ZU` until CUDA reproduces the same-seed direction
and a later multi-seed gate confirms the signal.
