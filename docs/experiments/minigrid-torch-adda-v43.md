# PyTorch AD/DA v2.32 Two-Head Loss Diagnostics

Date: 2026-06-30 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/56

## Motivation

v2.31 implemented a true two-head representation objective, but both two-head
variants underperformed the single combined `ZU` baseline. v2.32 adds enough
diagnostics to explain that failure mode and tests whether annealing or stopping
the auxiliary representation loss protects DA learning.

## Protocol

Config: `configs/experiments/minigrid-torch-adda-v43.json`

New runtime diagnostics:

- `mean_representation_state_loss_last_window`
- `mean_representation_target_visibility_loss_last_window`
- `mean_representation_state_beta_last_window`
- `mean_representation_target_visibility_beta_last_window`

Compared conditions:

- `ZK_torch_gotoobj_curriculum_no_repr_delay`
- `ZU_torch_gotoobj_state_plus_target_visibility_b0075`
- `ZY_torch_gotoobj_two_head_state0375_visibility0375`
- `ZZ_torch_gotoobj_two_head_anneal_to_zero`
- `ZA_torch_gotoobj_two_head_ad_only_stop`

## CPU Smoke

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v43.json --output-dir .tmp/local-v56-diagnostics --seed 3801 --device cpu`

Summary artifact:

`.tmp/local-v56-diagnostics/20260629T153831Z/summary.md`

| condition | success_last | return_last | visible_last | center_last | near_last | state_loss | visibility_loss | state_beta | visibility_beta | rep_updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 0.200 | 0.110 | 0.650 | 0.300 | 0.350 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0 |
| `ZU_torch_gotoobj_state_plus_target_visibility_b0075` | 0.400 | 0.208 | 0.550 | 0.400 | 0.400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 1996 |
| `ZY_torch_gotoobj_two_head_state0375_visibility0375` | 0.300 | 0.230 | 0.600 | 0.400 | 0.400 | 0.0326 | 0.0157 | 0.0375 | 0.0375 | 2158 |
| `ZZ_torch_gotoobj_two_head_anneal_to_zero` | 0.100 | 0.068 | 0.650 | 0.100 | 0.200 | 0.0454 | 0.0162 | 0.0000 | 0.0000 | 2364 |
| `ZA_torch_gotoobj_two_head_ad_only_stop` | 0.200 | 0.178 | 0.450 | 0.200 | 0.200 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 120 |

## Decision

This CPU gate is negative for CUDA escalation. `ZU` still wins final-window
success, and the annealed/AD-only variants do not recover the decoder.

The diagnostic signal is useful: the two-head state-delta raw loss is roughly
2x the visibility-transition raw loss in the active two-head condition
(`0.0326` vs `0.0157`). The next branch should avoid treating the two heads as
equal-pressure objectives. A better next protocol is either visibility-first
weighting, state-head downweighting, or a separate non-DQN representation probe
before attaching the auxiliary loss to the decoder.
