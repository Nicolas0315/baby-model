# PyTorch AD/DA v2.33 Visibility-First Two-Head Weighting

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/57

## Motivation

v2.32 showed that equal-pressure two-head training was likely overloading the
state-delta head: the state-delta raw loss was about 2x the target-visibility
raw loss. v2.33 tests whether keeping target-visibility pressure high while
downweighting state-delta improves the two-head protocol.

## Protocol

Config: `configs/experiments/minigrid-torch-adda-v44.json`

Compared conditions:

- `ZK_torch_gotoobj_curriculum_no_repr_delay`
- `ZU_torch_gotoobj_state_plus_target_visibility_b0075`
- `ZY_torch_gotoobj_two_head_state0375_visibility0375`
- `ZB_torch_gotoobj_two_head_state010_visibility065`
- `ZC_torch_gotoobj_two_head_state005_visibility070`
- `ZD_torch_gotoobj_two_head_state010_visibility065_state_anneal`

## CPU Smoke

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v44.json --output-dir .tmp/local-v57-visibility-first --seed 3901 --device cpu`

Summary artifact:

`.tmp/local-v57-visibility-first/20260629T154332Z/summary.md`

| condition | success_last | return_last | visible_last | center_last | near_last | state_loss | visibility_loss | state_beta | visibility_beta | rep_updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 0.050 | 0.035 | 0.350 | 0.100 | 0.050 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | 0.200 | 0.158 | 0.450 | 0.250 | 0.250 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 2288 |
| `ZY_torch_gotoobj_two_head_state0375_visibility0375` | 0.150 | 0.078 | 0.450 | 0.150 | 0.250 | 0.0161 | 0.0113 | 0.0375 | 0.0375 | 2391 |
| `ZB_torch_gotoobj_two_head_state010_visibility065` | 0.250 | 0.172 | 0.650 | 0.250 | 0.400 | 0.0343 | 0.0157 | 0.0100 | 0.0650 | 2118 |
| `ZC_torch_gotoobj_two_head_state005_visibility070` | 0.200 | 0.118 | 0.600 | 0.350 | 0.300 | 0.0279 | 0.0141 | 0.0050 | 0.0700 | 2303 |
| `ZD_torch_gotoobj_two_head_state010_visibility065_state_anneal` | 0.250 | 0.137 | 0.500 | 0.250 | 0.300 | 0.0378 | 0.0141 | 0.0000 | 0.0650 | 2167 |

## Decision

This is a positive CPU gate for the visibility-first hypothesis, but not yet a
baseline replacement. `ZB` beat the same-seed `ZU` baseline on final-window
success (`0.250` vs `0.200`), return (`0.172` vs `0.158`), target visibility
(`0.650` vs `0.450`), and target-near rate (`0.400` vs `0.250`).

Open a bounded CUDA replication issue for `ZB` before treating it as stable.
The absolute success rates are still low, so a single CPU seed is evidence for
the protocol direction, not a final claim.
