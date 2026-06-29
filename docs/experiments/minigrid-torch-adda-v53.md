# PyTorch AD/DA v2.42 Reduced Representation Pressure for Long-Horizon ZE

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/66

## Motivation

v2.41 showed that `ZE_torch_gotoobj_state_plus_mission_target_b005_long` beat
`ZU_long` under the longer `12+24+48` horizon, but still lost to the
no-representation `ZK_long` reference. v2.42 tests whether reducing
representation pressure can preserve the useful mission-target signal without
hurting longer-horizon task learning.

## Source State

- Config: `configs/experiments/minigrid-torch-adda-v48.json`
- Local runner: current Mac
- Torch: `2.12.1`
- Device: `cpu`
- Seeds: `4301,4302,4303`

## CPU Sweep

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch_sweep --config configs/experiments/minigrid-torch-adda-v48.json --output-dir .tmp/local-v66-ze-long-pressure --seeds 4301,4302,4303 --device cpu`

Summary artifact:

`.tmp/local-v66-ze-long-pressure/20260629T171626Z/summary.md`

| condition | wins | mean_success_last | median_success_last | mean_return_last | median_return_last | target_visible_last | target_center_last | target_near_last | mean_updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZE_torch_gotoobj_state_plus_mission_target_b005_long` | 1 | 0.367 | 0.250 | 0.232 | 0.156 | 0.550 | 0.400 | 0.433 | 3834.0 |
| `ZG_torch_gotoobj_state_plus_mission_target_b0025_long` | 0 | 0.250 | 0.250 | 0.147 | 0.117 | 0.550 | 0.300 | 0.383 | 3931.7 |
| `ZH_torch_gotoobj_state_plus_mission_target_b00375_long` | 0 | 0.317 | 0.250 | 0.193 | 0.114 | 0.517 | 0.333 | 0.350 | 4110.0 |
| `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop` | 2 | 0.500 | 0.500 | 0.381 | 0.380 | 0.700 | 0.517 | 0.567 | 3557.7 |
| `ZK_torch_gotoobj_curriculum_no_repr_delay_long` | 0 | 0.267 | 0.350 | 0.136 | 0.168 | 0.550 | 0.283 | 0.267 | 4008.0 |

Per-seed winners:

- `4301`: `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop`
- `4302`: `ZE_torch_gotoobj_state_plus_mission_target_b005_long`
- `4303`: `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop`

## Decision

`ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop` passed the CPU
gate. It won two of three seeds and beat both current `ZE_long` and
`ZK_long` on aggregate final-window success, return, target-visible,
target-center, and target-near.

Lower scalar beta alone did not fix the longer-horizon issue: `ZG` and `ZH`
underperformed `ZE`. The useful change is stopping representation updates
after the AD-only / decoder-delay phase, which keeps early representation
pressure but avoids continuing it through the long evaluation window.

Promote `ZI` to bounded CUDA replication before treating it as the new
long-horizon baseline.
