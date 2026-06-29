# PyTorch AD/DA v2.23 Mission-Conditioned Target Objective

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/48

## Motivation

v2.21 showed that the naive `state_plus_target_visibility` concatenation was
not stable across CUDA seeds. v2.22 added a direct mission-preservation probe
and showed that the probe did not reveal a hidden winner among the existing
candidates.

v2.23 tests the smallest redesign: keep the existing single auxiliary MSE head,
but condition the semantic target on a known mission object and color. This
avoids training the semantic head to over-predict the uninformative
`absent->absent` transition when the mission target is not visible in either
state.

## Literature Routing

References checked on 2026-06-29:

- BabyAI grounds the task in compositional language-conditioned object goals:
  https://arxiv.org/abs/1810.08272
- UNREAL motivates auxiliary tasks that shape a shared RL representation:
  https://arxiv.org/abs/1611.05397
- GradNorm motivates treating multi-objective loss balance as a first-class
  risk before expanding to multiple heads:
  https://arxiv.org/abs/1711.02257
- SPR motivates predictive representations for data-efficient RL:
  https://arxiv.org/abs/2007.05929

This run deliberately chooses the lower-risk single-head target first. A true
two-head objective remains a separate follow-up because it changes the agent
loss contract and reporting schema.

## Objective

New representation objectives:

- `mission_target_transition`: same 49-dimensional relation transition space
  as `target_visibility_transition`, but returns all zeros when the mission
  target is unknown or when both current and next relation are `absent`.
- `state_plus_mission_target`: concatenates `state_plus_delta` with
  `mission_target_transition`, preserving the 107-dimensional target shape of
  the previous combined objective while filtering the semantic slice.

Config:

- `configs/experiments/minigrid-torch-adda-v38.json`

Compared conditions:

- `ZK_torch_gotoobj_curriculum_no_repr_delay`
- `ZM_torch_gotoobj_state_plus_delta_matched_delay`
- `ZN_torch_gotoobj_target_visibility_matched_delay`
- `ZO_torch_gotoobj_state_plus_target_visibility_delay`
- `ZP_torch_gotoobj_mission_target_visibility_delay`
- `ZQ_torch_gotoobj_state_plus_mission_target_delay`

## CPU Smoke

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v38.json --output-dir .tmp/local-v48-mission-conditioned --seed 3301`

Summary artifact:

`.tmp/local-v48-mission-conditioned/20260629T141537Z/summary.md`

| condition | success_last | return_last | target_visible_last | target_center_last | target_near_last | rep_updates |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 0.500 | 0.302 | 0.600 | 0.500 | 0.550 | 0 |
| `ZM_torch_gotoobj_state_plus_delta_matched_delay` | 0.350 | 0.188 | 0.700 | 0.450 | 0.450 | 2172 |
| `ZN_torch_gotoobj_target_visibility_matched_delay` | 0.200 | 0.140 | 0.450 | 0.250 | 0.300 | 2175 |
| `ZO_torch_gotoobj_state_plus_target_visibility_delay` | 0.400 | 0.215 | 0.650 | 0.400 | 0.500 | 1964 |
| `ZP_torch_gotoobj_mission_target_visibility_delay` | 0.000 | 0.000 | 0.600 | 0.150 | 0.150 | 2425 |
| `ZQ_torch_gotoobj_state_plus_mission_target_delay` | 0.000 | 0.000 | 0.300 | 0.000 | 0.000 | 2293 |

## Decision

The v2.23 single-head mission-conditioned targets are negative on this CPU
gate. Both new candidates produced representation updates, but neither
preserved task reward, and `ZQ` also degraded the direct mission-preservation
probe.

Do not escalate v2.23 to CUDA. The next useful move is a true two-head
objective or a loss-weight sweep, with the mission-preservation probe retained
as a gate.
