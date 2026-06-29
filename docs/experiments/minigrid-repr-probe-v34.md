# MiniGrid Representation Probe v2.14

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/39

## Motivation

v2.12 and v2.13 showed that `next_signature_bucket` is not a useful next
diagnostic for the non-DQN scripted-policy lane. The hashed bucket target can
be separable in raw features, but the learned encoder did not add downstream
transition-label lift. v2.14 replaces the hashed target with an interpretable
semantic transition label.

## Chosen Path

Use option 1 from #39: `target_visibility_transition`.

For each transition, extract the mission target object/color and classify the
target's visible relation to the agent before and after the action:

- `absent`
- `left_near`, `left_far`
- `center_near`, `center_far`
- `right_near`, `right_far`

The transition label is `before->after`, for example
`absent->center_near`. This uses the corrected MiniGrid
`image[x][y][channel]` axis handling from v2.11 and stays standard-library in
the repository core loop.

## Config

Config: `configs/experiments/minigrid-repr-probe-v34.json`

Dataset:

- `BabyAI-GoToRedBall-v0`: 6 scripted-policy episodes.
- `BabyAI-GoToObj-v0`: 10 scripted-policy episodes.

Feature sets:

- `raw_current`
- `predictive_target_visibility`

Probe labels:

- `mission_object`
- `mission_color`
- `target_visibility_transition`

## Decision Rule

Run a bounded local CPU smoke. Treat v2.14 as positive only if
`predictive_target_visibility` improves `target_visibility_transition` probe
lift over `raw_current` by at least `0.010`, while keeping `mission_object`
and `mission_color` accuracy within `0.050` of `raw_current`. Both feature
sets must have at least 10 test examples.

Do not escalate v2.14 to CUDA. If this semantic label fails, the next step
should either use a richer semantic object/color delta label or return to the
RL lane with the positive GoToObj controllability/state-plus-delta evidence.

## Current Result

A bounded local CPU smoke completed in the existing optional MiniGrid venv.

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_repr_probe --config configs/experiments/minigrid-repr-probe-v34.json --output-dir .tmp/local-v34-repr-probe --seed 2901`

Summary artifact:

`.tmp/local-v34-repr-probe-rerun/20260629T125413Z/summary.md`

Transitions: `905`

Predictive encoder held-out training result:

| encoder | target | test_accuracy | majority | lift |
| --- | --- | ---: | ---: | ---: |
| `predictive_target_visibility` | `target_visibility_transition` | 0.663 | 0.287 | 0.376 |

Probe results:

| feature_set | label | accuracy | majority | lift | test |
| --- | --- | ---: | ---: | ---: | ---: |
| `raw_current` | `mission_object` | 0.757 | 0.718 | 0.039 | 181 |
| `raw_current` | `mission_color` | 0.757 | 0.431 | 0.326 | 181 |
| `raw_current` | `target_visibility_transition` | 0.674 | 0.287 | 0.387 | 181 |
| `predictive_target_visibility` | `mission_object` | 0.757 | 0.718 | 0.039 | 181 |
| `predictive_target_visibility` | `mission_color` | 0.757 | 0.431 | 0.326 | 181 |
| `predictive_target_visibility` | `target_visibility_transition` | 0.680 | 0.287 | 0.392 | 181 |

Decision: v2.14 did not meet the relative-to-baseline rule. The semantic
encoder preserved mission-object and mission-color accuracy, and improved
`target_visibility_transition` lift by `0.0055`, but the documented threshold
was `0.010`.

The result is still stronger than the hashed bucket probes: the semantic target
itself is learnable with held-out lift `0.376`, and the downstream probe moved
in the right direction without erasing mission information.

Conclusion: treat v2.14 as a negative near-miss. The next step should run a
small multi-seed semantic probe or refine the semantic relation buckets before
returning to RL/CUDA.
