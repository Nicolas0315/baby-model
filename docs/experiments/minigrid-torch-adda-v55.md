# PyTorch AD/DA v2.44 CUDA Multi-Seed Gate for Long-Horizon ZI AD-Stop

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/68

## Motivation

v2.43 replicated the `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop`
direction on CUDA seed `4301`. v2.44 runs the bounded multi-seed CUDA gate
before treating `ZI` as the long-horizon baseline candidate.

## Source State

- Source commit: `42421c68e4d22183adb1f4ec928ad872cc962ef9`
- Config: `configs/experiments/minigrid-torch-adda-v48.json`
- Worker: `rtx4090` / WSL Ubuntu
- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `610.62`
- Torch: `2.11.0+cu128`
- Device: `cuda`
- Seeds: `4301,4302,4303`

## CUDA Sweep

Command on `rtx4090`:

`./.venv-minigrid-cuda/bin/python -m baby_model.minigrid_torch_sweep --config configs/experiments/minigrid-torch-adda-v48.json --output-dir .tmp/rtx4090-v68-zi-cuda-sweep --seeds 4301,4302,4303 --device cuda`

Summary artifact on `rtx4090`:

`~/work/baby-model-cuda-42421c6/.tmp/rtx4090-v68-zi-cuda-sweep/20260629T180341Z/summary.md`

| condition | wins | mean_success_last | median_success_last | mean_return_last | median_return_last | target_visible_last | target_center_last | target_near_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZE_torch_gotoobj_state_plus_mission_target_b005_long` | 0 | 0.267 | 0.250 | 0.173 | 0.156 | 0.500 | 0.300 | 0.350 |
| `ZG_torch_gotoobj_state_plus_mission_target_b0025_long` | 0 | 0.300 | 0.400 | 0.184 | 0.229 | 0.550 | 0.350 | 0.383 |
| `ZH_torch_gotoobj_state_plus_mission_target_b00375_long` | 1 | 0.317 | 0.250 | 0.193 | 0.114 | 0.517 | 0.333 | 0.350 |
| `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop` | 2 | 0.500 | 0.500 | 0.381 | 0.380 | 0.700 | 0.517 | 0.567 |
| `ZK_torch_gotoobj_curriculum_no_repr_delay_long` | 0 | 0.267 | 0.350 | 0.136 | 0.168 | 0.550 | 0.283 | 0.267 |

Per-seed winners:

- `4301`: `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop`
- `4302`: `ZH_torch_gotoobj_state_plus_mission_target_b00375_long`
- `4303`: `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop`

## Decision

`ZI` passed the bounded multi-seed CUDA gate. It won two of three seeds and led
aggregate final-window success, return, target-visible, target-center, and
target-near.

Treat `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop` as the
current strongest long-horizon representation-driven baseline candidate. The
next gate should broaden confirmation to five CUDA seeds before calling the
long-horizon branch stable.
