# PyTorch AD/DA v2.43 CUDA Replication for Long-Horizon ZI AD-Stop

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/67

## Motivation

v2.42 found that stopping representation updates after the AD-only /
decoder-delay phase fixed the longer-horizon weakness for low-beta
`state_plus_mission_target`. v2.43 checks whether the CPU-positive
`ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop` direction
reproduces on a CUDA worker.

## Source State

- Source commit: `42421c68e4d22183adb1f4ec928ad872cc962ef9`
- Config: `configs/experiments/minigrid-torch-adda-v48.json`
- Worker: `rtx4090` / WSL Ubuntu
- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `610.62`
- Torch: `2.11.0+cu128`
- CUDA available: `True`
- Seed: `4301`

## CUDA Smoke

Command on `rtx4090`:

`./.venv-minigrid-cuda/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v48.json --output-dir .tmp/rtx4090-v67-zi-cuda --seed 4301 --device cuda`

Summary artifact on `rtx4090`:

`~/work/baby-model-cuda-42421c6/.tmp/rtx4090-v67-zi-cuda/20260629T175150Z/summary.md`

| condition | success_last | return_last | target_visible_last | target_center_last | target_near_last | rep_updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay_long` | 0.350 | 0.174 | 0.650 | 0.350 | 0.350 | 0 |
| `ZE_torch_gotoobj_state_plus_mission_target_b005_long` | 0.250 | 0.156 | 0.400 | 0.300 | 0.350 | 3903 |
| `ZG_torch_gotoobj_state_plus_mission_target_b0025_long` | 0.400 | 0.229 | 0.500 | 0.400 | 0.400 | 4047 |
| `ZH_torch_gotoobj_state_plus_mission_target_b00375_long` | 0.250 | 0.114 | 0.350 | 0.250 | 0.250 | 4338 |
| `ZI_torch_gotoobj_state_plus_mission_target_b005_long_ad_stop` | 0.450 | 0.340 | 0.600 | 0.450 | 0.500 | 240 |

## Decision

CUDA reproduced the CPU-positive direction for `ZI`: it won final-window
success and return, and beat the representation candidates on target-center and
target-near. `ZK_long` retained a higher target-visible rate on this single
seed, so this is not a final baseline replacement.

The low representation-update count (`240` vs thousands for continuous
representation candidates) confirms that the intended AD-stop pressure
reduction is active. Promote `ZI` to a bounded multi-seed CUDA gate before
treating it as the long-horizon baseline.
