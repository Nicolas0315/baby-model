# PyTorch AD/DA v2.40 Five-Seed CUDA Extension for ZE

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/64

## Motivation

v2.39 made `ZE_torch_gotoobj_state_plus_mission_target_b005` the current
strongest representation-driven candidate after a three-seed CUDA gate. v2.40
extends the same protocol to five CUDA seeds before moving to a different
horizon or objective family.

## Source State

- Source commit: `03d58b7ddbfd843414ebd9e57617b60c5346ba9f`
- Config: `configs/experiments/minigrid-torch-adda-v46.json`
- Worker: `rtx4090` / WSL Ubuntu
- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `610.62`
- Torch: `2.11.0+cu128`
- Device: `cuda`
- Seeds: `4101,4102,4103,4104,4105`

## CUDA Sweep

Command on `rtx4090`:

`./.venv-minigrid-cuda/bin/python -m baby_model.minigrid_torch_sweep --config configs/experiments/minigrid-torch-adda-v46.json --output-dir .tmp/rtx4090-v64-ze-cuda-5seed --seeds 4101,4102,4103,4104,4105 --device cuda`

Summary artifact on `rtx4090`:

`~/work/baby-model-cuda-03d58b7/.tmp/rtx4090-v64-ze-cuda-5seed/20260629T170242Z/summary.md`

| condition | wins | mean_success_last | median_success_last | mean_return_last | median_return_last | target_visible_last | target_center_last | target_near_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZD_torch_gotoobj_two_head_state010_visibility065_state_anneal` | 0 | 0.180 | 0.150 | 0.093 | 0.068 | 0.490 | 0.210 | 0.220 |
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

The five-seed CUDA extension supports the v2.39 conclusion. `ZE` won four of
five seeds and beat `ZU` on aggregate final-window success, return,
target-visible, target-center, and target-near metrics.

Keep `ZE_torch_gotoobj_state_plus_mission_target_b005` as the current strongest
representation-driven baseline. The next gate should change the evaluation
surface, not retest the same short protocol again: run a longer-horizon check
with `ZE` as the baseline to beat.
