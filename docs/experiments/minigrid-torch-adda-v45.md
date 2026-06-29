# PyTorch AD/DA v2.34 CUDA Replication for Visibility-First ZB

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/58

## Motivation

v2.33 produced a positive CPU gate for the visibility-first two-head condition
`ZB_torch_gotoobj_two_head_state010_visibility065`. v2.34 replicates the same
seed on a CUDA worker before spending more GPU time on a multi-seed sweep.

## Source State

- Commit: `6fc5a5923eb3374bb8468a198120fd67677edb4e`
- Config: `configs/experiments/minigrid-torch-adda-v44.json`
- Worker: `rtx4090` / WSL Ubuntu
- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `610.62`
- Torch: `2.11.0+cu128`
- CUDA available: `True`

## CUDA Smoke

Command on `rtx4090`:

`./.venv-cuda/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v44.json --output-dir .tmp/rtx4090-v58-zb-cuda --seed 3901 --device cuda`

Summary artifact on `rtx4090`:

`~/work/baby-model-cuda-6fc5a59/.tmp/rtx4090-v58-zb-cuda/20260629T155900Z/summary.md`

| condition | success_last | return_last | visible_last | center_last | near_last | state_beta | visibility_beta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 0.050 | 0.035 | 0.350 | 0.100 | 0.050 | 0.0000 | 0.0000 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | 0.200 | 0.158 | 0.450 | 0.250 | 0.250 | 0.0000 | 0.0000 |
| `ZY_torch_gotoobj_two_head_state0375_visibility0375` | 0.150 | 0.078 | 0.450 | 0.150 | 0.250 | 0.0375 | 0.0375 |
| `ZB_torch_gotoobj_two_head_state010_visibility065` | 0.250 | 0.172 | 0.650 | 0.250 | 0.400 | 0.0100 | 0.0650 |
| `ZC_torch_gotoobj_two_head_state005_visibility070` | 0.200 | 0.118 | 0.600 | 0.350 | 0.300 | 0.0050 | 0.0700 |
| `ZD_torch_gotoobj_two_head_state010_visibility065_state_anneal` | 0.250 | 0.137 | 0.500 | 0.250 | 0.300 | 0.0000 | 0.0650 |

## Decision

CUDA replicated the CPU seed exactly enough for the next gate: `ZB` again won
final-window success, return, target visibility, and target-near rate over
same-seed `ZU`.

Promote `ZB` to a bounded three-seed CUDA sweep. Do not replace the current
baseline until multi-seed CUDA confirms the signal.
