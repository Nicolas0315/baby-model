# PyTorch AD/DA v2.39 CUDA Multi-Seed Gate for Low-Beta Mission Target ZE

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/63

## Motivation

v2.38 reproduced the CPU-positive `ZE` direction on CUDA seed `4101`, but the
single-seed run still left a return concern. v2.39 runs a bounded multi-seed
CUDA gate before treating `ZE_torch_gotoobj_state_plus_mission_target_b005` as
the new strongest baseline candidate.

## Source State

- Source commit: `03d58b7ddbfd843414ebd9e57617b60c5346ba9f`
- Config: `configs/experiments/minigrid-torch-adda-v46.json`
- Worker: `rtx4090` / WSL Ubuntu
- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `610.62`
- Torch: `2.11.0+cu128`
- Device: `cuda`
- Seeds: `4101,4102,4103`

## CUDA Sweep

Command on `rtx4090`:

`./.venv-minigrid-cuda/bin/python -m baby_model.minigrid_torch_sweep --config configs/experiments/minigrid-torch-adda-v46.json --output-dir .tmp/rtx4090-v63-ze-cuda-sweep --seeds 4101,4102,4103 --device cuda`

Summary artifact on `rtx4090`:

`~/work/baby-model-cuda-03d58b7/.tmp/rtx4090-v63-ze-cuda-sweep/20260629T165215Z/summary.md`

| condition | wins | mean_success_last | median_success_last | mean_return_last | median_return_last | target_visible_last | target_center_last | target_near_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZD_torch_gotoobj_two_head_state010_visibility065_state_anneal` | 0 | 0.150 | 0.150 | 0.079 | 0.068 | 0.483 | 0.200 | 0.217 |
| `ZE_torch_gotoobj_state_plus_mission_target_b005` | 3 | 0.333 | 0.350 | 0.209 | 0.222 | 0.633 | 0.367 | 0.450 |
| `ZF_torch_gotoobj_state_plus_mission_target_b0075` | 0 | 0.117 | 0.100 | 0.048 | 0.059 | 0.583 | 0.167 | 0.200 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | 0 | 0.150 | 0.100 | 0.096 | 0.043 | 0.433 | 0.167 | 0.200 |

Per-seed winners:

- `4101`: `ZE_torch_gotoobj_state_plus_mission_target_b005`
- `4102`: `ZE_torch_gotoobj_state_plus_mission_target_b005`
- `4103`: `ZE_torch_gotoobj_state_plus_mission_target_b005`

## Decision

`ZE` passed the bounded multi-seed CUDA gate. It won all three seeds and beat
`ZU` on aggregate final-window success, return, target-visible, target-center,
and target-near metrics.

Treat `ZE_torch_gotoobj_state_plus_mission_target_b005` as the current
strongest representation-driven baseline candidate. The next gate should
broaden confirmation rather than continue `ZB`/`ZD`: either a five-seed CUDA
extension or a longer-horizon check using `ZE` as the baseline to beat.
