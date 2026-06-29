# PyTorch AD/DA v2.27 Tight Beta Neighborhood Around ZR

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/51

## Motivation

v2.26 made `ZR_torch_gotoobj_state_plus_target_visibility_b010` the strongest
representation-driven AD/DA candidate so far. It passed CPU, single-seed CUDA,
and bounded three-seed CUDA gates.

v2.27 returns to CPU-first refinement and asks whether the beta `0.1` point is
actually optimal, or whether a lighter representation loss better preserves
task reward and mission-target probes.

## Config

- `configs/experiments/minigrid-torch-adda-v40.json`

Compared conditions:

- `ZK_torch_gotoobj_curriculum_no_repr_delay`
- `ZT_torch_gotoobj_state_plus_target_visibility_b005`
- `ZU_torch_gotoobj_state_plus_target_visibility_b0075`
- `ZR_torch_gotoobj_state_plus_target_visibility_b010`
- `ZV_torch_gotoobj_state_plus_target_visibility_b0125`
- `ZW_torch_gotoobj_state_plus_target_visibility_b015`

## CPU Smoke

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v40.json --output-dir .tmp/local-v51-tight-beta --seed 3501`

Summary artifact:

`.tmp/local-v51-tight-beta/20260629T144540Z/summary.md`

| condition | beta | success_last | return_last | target_visible_last | target_center_last | target_near_last | rep_updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 0.0 | 0.200 | 0.139 | 0.500 | 0.200 | 0.300 | 0 |
| `ZT_torch_gotoobj_state_plus_target_visibility_b005` | 0.05 | 0.650 | 0.472 | 0.850 | 0.650 | 0.700 | 1624 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | 0.075 | 0.550 | 0.399 | 0.700 | 0.600 | 0.600 | 1846 |
| `ZR_torch_gotoobj_state_plus_target_visibility_b010` | 0.1 | 0.250 | 0.122 | 0.700 | 0.300 | 0.350 | 2306 |
| `ZV_torch_gotoobj_state_plus_target_visibility_b0125` | 0.125 | 0.250 | 0.167 | 0.650 | 0.250 | 0.250 | 2231 |
| `ZW_torch_gotoobj_state_plus_target_visibility_b015` | 0.15 | 0.300 | 0.175 | 0.750 | 0.400 | 0.400 | 2237 |

## Decision

`ZT` passed the CPU gate and beat the current `ZR` baseline on final-window
success, final-window return, and all mission-preservation probe columns.

This is enough to justify a bounded CUDA replication issue for beta `0.05`,
but not enough to replace `ZR` as the stable baseline until CUDA evidence
exists.

## CUDA Replication

Issue: https://github.com/Nicolas0315/baby-model/issues/52

A bounded CUDA smoke completed on `gpu-worker-c` at source commit
`ebfcab3299cfe48e0d989ec168c31b80d56465d9`. Setup proved:

- `torch_version=2.12.1+cu132`
- `torch_cuda_available=True`
- `torch_cuda_device_count=1`
- `device=cuda`

Host-level evidence is kept outside the repository at:

`/Users/s30519/work/docs/research/baby-model/fleet-torch-adda-v40-2026-06-29.md`

CUDA artifact:

`.tmp/verify-minigrid/torch/20260629T145114Z/summary.md`

| condition | beta | success_last | return_last | target_visible_last | target_center_last | target_near_last | rep_updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 0.0 | 0.200 | 0.139 | 0.500 | 0.200 | 0.300 | 0 |
| `ZT_torch_gotoobj_state_plus_target_visibility_b005` | 0.05 | 0.650 | 0.472 | 0.850 | 0.650 | 0.700 | 1624 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | 0.075 | 0.550 | 0.399 | 0.700 | 0.600 | 0.600 | 1846 |
| `ZR_torch_gotoobj_state_plus_target_visibility_b010` | 0.1 | 0.250 | 0.122 | 0.700 | 0.300 | 0.350 | 2306 |
| `ZV_torch_gotoobj_state_plus_target_visibility_b0125` | 0.125 | 0.250 | 0.167 | 0.650 | 0.250 | 0.250 | 2231 |
| `ZW_torch_gotoobj_state_plus_target_visibility_b015` | 0.15 | 0.300 | 0.175 | 0.750 | 0.400 | 0.400 | 2237 |

CUDA decision: `ZT` reproduced the CPU-positive direction on the same seed.
This justifies a bounded multi-seed CUDA sweep before replacing `ZR`.
