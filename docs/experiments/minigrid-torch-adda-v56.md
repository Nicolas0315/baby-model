# PyTorch AD/DA v2.45 CUDA Five-Seed Gate for Long-Horizon ZI AD-Stop

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/69

## Motivation

v2.44 passed the bounded three-seed CUDA gate for
`ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop`. v2.45 extends
the same long-horizon protocol to five CUDA seeds before treating the branch as
stable enough for the next validation axis.

## Source State

- Source commit: `42421c68e4d22183adb1f4ec928ad872cc962ef9`
- Config: `configs/experiments/minigrid-torch-adda-v48.json`
- Worker: `rtx4090` / WSL Ubuntu
- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `610.62`
- Torch: `2.11.0+cu128`
- Device: `cuda`
- Seeds: `4301,4302,4303,4304,4305`

## CUDA Sweep

Command on `rtx4090`:

`./.venv-minigrid-cuda/bin/python -m baby_model.minigrid_torch_sweep --config configs/experiments/minigrid-torch-adda-v48.json --output-dir .tmp/rtx4090-v69-zi-cuda-5seed --seeds 4301,4302,4303,4304,4305 --device cuda`

Summary artifact on `rtx4090`:

`~/work/baby-model-cuda-42421c6/.tmp/rtx4090-v69-zi-cuda-5seed/20260629T182133Z/summary.md`

| condition | wins | mean_success_last | median_success_last | mean_return_last | median_return_last | target_visible_last | target_center_last | target_near_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZE_torch_gotoobj_state_plus_mission_target_b005_long` | 1 | 0.360 | 0.450 | 0.234 | 0.275 | 0.590 | 0.390 | 0.430 |
| `ZG_torch_gotoobj_state_plus_mission_target_b0025_long` | 0 | 0.290 | 0.350 | 0.194 | 0.229 | 0.490 | 0.320 | 0.350 |
| `ZH_torch_gotoobj_state_plus_mission_target_b00375_long` | 1 | 0.320 | 0.250 | 0.200 | 0.169 | 0.510 | 0.340 | 0.350 |
| `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop` | 3 | 0.530 | 0.550 | 0.386 | 0.390 | 0.690 | 0.540 | 0.580 |
| `ZK_torch_gotoobj_curriculum_no_repr_delay_long` | 0 | 0.300 | 0.350 | 0.176 | 0.174 | 0.550 | 0.320 | 0.330 |

Per-seed winners:

- `4301`: `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop`
- `4302`: `ZH_torch_gotoobj_state_plus_mission_target_b00375_long`
- `4303`: `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop`
- `4304`: `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop`
- `4305`: `ZE_torch_gotoobj_state_plus_mission_target_b005_long`

## Decision

`ZI` passed the five-seed CUDA extension. It won three of five seeds and led
aggregate final-window success, return, target-center, and target-near. Its
target-visible rate was also the strongest aggregate value in this sweep.

Treat `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop` as the
current strongest long-horizon representation-driven baseline. The next gate
should test whether the same AD-stop direction survives a different validation
axis, such as another CUDA-capable worker or a longer horizon, rather than
adding more same-worker seeds immediately.
