# PyTorch AD/DA v2.35 CUDA Multi-Seed Sweep for Visibility-First ZB

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/59

## Motivation

v2.34 replicated the single-seed CPU signal for
`ZB_torch_gotoobj_two_head_state010_visibility065` on CUDA. v2.35 checks
whether that signal is stable across three CUDA seeds before promoting the
visibility-first protocol.

## Source State

- Commit: `6fc5a5923eb3374bb8468a198120fd67677edb4e`
- Config: `configs/experiments/minigrid-torch-adda-v44.json`
- Worker: `rtx4090` / WSL Ubuntu
- GPU: `NVIDIA GeForce RTX 4090`
- Driver: `610.62`
- Torch: `2.11.0+cu128`
- Device: `cuda`
- Seeds: `3901,3902,3903`

## CUDA Sweep

Command on `rtx4090`:

`./.venv-cuda/bin/python -m baby_model.minigrid_torch_sweep --config configs/experiments/minigrid-torch-adda-v44.json --output-dir .tmp/rtx4090-v59-zb-cuda-sweep --seeds 3901,3902,3903 --device cuda`

Summary artifact on `rtx4090`:

`~/work/baby-model-cuda-6fc5a59/.tmp/rtx4090-v59-zb-cuda-sweep/20260629T160825Z/summary.md`

| condition | wins | mean_success_last | median_success_last | mean_return_last | median_return_last | target_visible_last | target_center_last | target_near_last |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZB_torch_gotoobj_two_head_state010_visibility065` | 1 | 0.267 | 0.250 | 0.176 | 0.172 | 0.583 | 0.300 | 0.333 |
| `ZC_torch_gotoobj_two_head_state005_visibility070` | 0 | 0.217 | 0.200 | 0.127 | 0.118 | 0.517 | 0.267 | 0.300 |
| `ZD_torch_gotoobj_two_head_state010_visibility065_state_anneal` | 1 | 0.283 | 0.250 | 0.178 | 0.140 | 0.583 | 0.300 | 0.317 |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 1 | 0.217 | 0.200 | 0.132 | 0.140 | 0.567 | 0.267 | 0.300 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | 0 | 0.233 | 0.250 | 0.152 | 0.158 | 0.617 | 0.300 | 0.350 |
| `ZY_torch_gotoobj_two_head_state0375_visibility0375` | 0 | 0.167 | 0.150 | 0.084 | 0.078 | 0.433 | 0.217 | 0.250 |

Per-seed winners:

- `3901`: `ZB_torch_gotoobj_two_head_state010_visibility065`
- `3902`: `ZK_torch_gotoobj_curriculum_no_repr_delay`
- `3903`: `ZD_torch_gotoobj_two_head_state010_visibility065_state_anneal`

## Decision

Do not promote `ZB` as a stable baseline yet. It beat `ZU` on mean
final-window success and return, but it won only one of three seeds and lost to
`ZU` on target-visible and target-near averages.

The broader visibility-first family remains promising: `ZD` led mean
final-window success and return, while `ZB` retained the better median return
and target-near score among the two visibility-first candidates. The next branch
should focus on `ZB` vs `ZD` with a narrower protocol, not replace the current
baseline from this sweep alone.
