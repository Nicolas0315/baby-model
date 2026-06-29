# PyTorch AD/DA v2.38 CUDA Replication for Low-Beta Mission Target ZE

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/62

## Motivation

v2.37 found a CPU-positive signal for
`ZE_torch_gotoobj_state_plus_mission_target_b005`: low-beta
mission-conditioned combined prediction beat the `ZU` baseline on five-seed
CPU success, return, and mission-probe aggregates. v2.38 checks whether the
same protocol reproduces on a CUDA worker before spending GPU time on a
multi-seed CUDA gate.

## Source State

- Source commit: `03d58b7ddbfd843414ebd9e57617b60c5346ba9f`
- Config: `configs/experiments/minigrid-torch-adda-v46.json`
- Worker: `rtx4090` / WSL Ubuntu
- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `610.62`
- Torch: `2.11.0+cu128`
- CUDA available: `True`
- Seed: `4101`

## CUDA Smoke

Command on `rtx4090`:

`./.venv-minigrid-cuda/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v46.json --output-dir .tmp/rtx4090-v62-ze-cuda --seed 4101 --device cuda`

Summary artifact on `rtx4090`:

`~/work/baby-model-cuda-03d58b7/.tmp/rtx4090-v62-ze-cuda/20260629T164512Z/summary.md`

| condition | success_last | return_last | target_visible_last | target_center_last | target_near_last | rep_updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | 0.300 | 0.202 | 0.650 | 0.300 | 0.350 | 2101 |
| `ZE_torch_gotoobj_state_plus_mission_target_b005` | 0.350 | 0.157 | 0.800 | 0.400 | 0.550 | 2101 |
| `ZF_torch_gotoobj_state_plus_mission_target_b0075` | 0.200 | 0.070 | 0.500 | 0.200 | 0.250 | 2240 |
| `ZD_torch_gotoobj_two_head_state010_visibility065_state_anneal` | 0.150 | 0.068 | 0.550 | 0.200 | 0.200 | 2391 |

## Decision

CUDA reproduced the CPU-positive direction on the same representative seed:
`ZE` won final-window success and beat `ZU` on target-visible, target-center,
and target-near probe rates.

The replication is not a full baseline replacement signal yet. `ZE` trailed
`ZU` on final-window return (`0.157` vs `0.202`), so the next gate must be a
multi-seed CUDA sweep that evaluates success, return, and mission-probe metrics
together before replacing `ZU`.
