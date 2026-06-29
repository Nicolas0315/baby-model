# PyTorch AD/DA v2.22 Mission-Preservation Probe

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/47

## Motivation

Earlier GoToObj RL runs used final-stage external success and return as a
mission-preservation proxy. That was too indirect: a representation objective
could improve reward while still weakening the mission object/color signal.
v2.22 adds direct probe fields to the PyTorch RL runner so future objectives
can be compared on mission-target visibility, not just reward.

## Probe Definition

At the end of each episode, the runner probes the final observation using the
existing mission parser and corrected MiniGrid `image[x][y][channel]` view:

- `mission_target_known_rate`: mission contains a known object and color.
- `mission_target_visible_rate_all`: target object/color is visible at episode
  end across all episodes.
- `mission_target_visible_rate_last_window`: same metric over the final
  20-episode window.
- `mission_target_center_rate_last_window`: target is visible in the center
  column over the final window.
- `mission_target_near_rate_last_window`: target is visible within near
  Manhattan range over the final window.

The summary table shows the three last-window probe columns:
`target_visible_last`, `target_center_last`, and `target_near_last`.

## Implementation

Implemented in:

- `baby_model/minigrid_torch.py`
- `baby_model/minigrid_torch_sweep.py`
- `tests/test_experiment.py`

The probe is dependency-free for synthetic observations and is also included
in PyTorch sweep aggregates so multi-seed runs can report mean probe values.

## CPU Smoke

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_torch --config configs/experiments/minigrid-torch-adda-v37.json --output-dir .tmp/local-v47-mission-probe --seed 3201`

Summary artifact:

`.tmp/local-v47-mission-probe/20260629T140615Z/summary.md`

| condition | success_last | return_last | target_visible_last | target_center_last | target_near_last |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ZK_torch_gotoobj_curriculum_no_repr_delay` | 0.400 | 0.190 | 0.550 | 0.400 | 0.450 |
| `ZM_torch_gotoobj_state_plus_delta_matched_delay` | 0.250 | 0.170 | 0.400 | 0.250 | 0.250 |
| `ZN_torch_gotoobj_target_visibility_matched_delay` | 0.200 | 0.094 | 0.450 | 0.250 | 0.300 |
| `ZO_torch_gotoobj_state_plus_target_visibility_delay` | 0.450 | 0.295 | 0.600 | 0.500 | 0.500 |

## Decision

The probe fields worked in both `metrics.json` and `summary.md`. In this CPU
smoke, `ZO` had the best reward metrics and the best mission-target visibility
metrics, while `ZK` was second on both. The probe did not reveal a hidden
mission-preservation winner that success/return missed, so no CUDA issue is
opened from v2.22 alone.

Next: redesign the auxiliary objective or training protocol before spending
more CUDA. The probe should remain a reporting gate for future candidate
objectives.
