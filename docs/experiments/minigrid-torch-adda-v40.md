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

## CUDA Multi-Seed Sweep

Issue: https://github.com/Nicolas0315/baby-model/issues/53

A three-seed CUDA sweep completed on `gpu-worker-c` at source commit
`3a9b729050600246c85c9bbd73013a0c46425754` with seeds `3501,3502,3503`.
Setup proved:

- `torch_version=2.12.1+cu132`
- `torch_cuda_available=True`
- `torch_cuda_device_count=1`
- `devices=cuda`

Host-level evidence is kept outside the repository at:

`/Users/s30519/work/docs/research/baby-model/fleet-torch-adda-v40-sweep-2026-06-29.md`

CUDA sweep artifact:

`.tmp/verify-minigrid/torch-sweep/20260629T150027Z/summary.md`

| condition | wins | mean_success_last | median_success_last | mean_return_last | median_return_last | target_visible_last | target_center_last | target_near_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 0 | 0.317 | 0.300 | 0.221 | 0.206 | 0.633 | 0.317 | 0.400 |
| `ZR_torch_gotoobj_state_plus_target_visibility_b010` | 0 | 0.233 | 0.250 | 0.138 | 0.122 | 0.550 | 0.267 | 0.333 |
| `ZT_torch_gotoobj_state_plus_target_visibility_b005` | 1 | 0.400 | 0.350 | 0.266 | 0.223 | 0.667 | 0.417 | 0.433 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | 2 | 0.550 | 0.550 | 0.372 | 0.360 | 0.700 | 0.567 | 0.583 |
| `ZV_torch_gotoobj_state_plus_target_visibility_b0125` | 0 | 0.350 | 0.400 | 0.193 | 0.176 | 0.600 | 0.350 | 0.367 |
| `ZW_torch_gotoobj_state_plus_target_visibility_b015` | 0 | 0.217 | 0.250 | 0.143 | 0.175 | 0.600 | 0.267 | 0.300 |

Per-seed winners:

- `3501`: `ZT_torch_gotoobj_state_plus_target_visibility_b005`
- `3502`: `ZU_torch_gotoobj_state_plus_target_visibility_b0075`
- `3503`: `ZU_torch_gotoobj_state_plus_target_visibility_b0075`

Decision: beta `0.05` beat the prior `ZR` baseline, but beta `0.075` became
the actual multi-seed CUDA winner. `ZU` is the current strongest
beta-neighborhood candidate.
