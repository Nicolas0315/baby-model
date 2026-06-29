# MiniGrid Representation Probe v2.11

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/36

## Motivation

v2.10 showed that making the predictive objective harder
(`next_signature_bucket`) did not help under the random-policy dataset. v2.11
therefore changes the data source first: collect GoToObj-family transitions
with a lightweight scripted policy that approaches visible target-like objects
before adding more encoder complexity.

## Chosen Path

Use Option A from #36: scripted object-approach policy.

The policy stays standard-library in this repository. At runtime it reads the
MiniGrid agent-relative observation image, extracts mission object/color tokens,
turns toward a visible matching object when one is present, moves forward when
the target is centered ahead, and otherwise follows a deterministic exploration
pattern. Random policy remains available for earlier configs.

## Config

Config: `configs/experiments/minigrid-repr-probe-v31.json`

Dataset:

- `BabyAI-GoToRedBall-v0`: 6 scripted-policy episodes.
- `BabyAI-GoToObj-v0`: 10 scripted-policy episodes.

Feature sets:

- `raw_current`
- `predictive_changed`

Probe labels:

- `mission_object`
- `mission_color`
- `changed`

## Decision Rule

Run a bounded local CPU smoke. Treat v2.11 as positive only if
`predictive_changed` on the scripted-policy dataset improves `changed` probe
lift over the v2.10 random-policy `predictive_changed` result (`0.209`) by at
least `0.010`, while keeping `mission_object` and `mission_color` accuracy
within `0.050` of `raw_current` on the same scripted dataset. Both feature sets
must have at least 10 test examples.

Do not escalate v2.11 to CUDA. If the rule passes, the next step is to decide
whether scripted-policy representation pretraining should feed back into the RL
auxiliary-head lane. If it fails, redesign transition labels before adding more
policy logic.

## Current Result

A bounded local CPU smoke completed in the existing optional MiniGrid venv.

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_repr_probe --config configs/experiments/minigrid-repr-probe-v31.json --output-dir .tmp/local-v31-repr-probe --seed 2601`

Summary artifact:

`.tmp/local-v31-repr-probe/20260629T113255Z/summary.md`

Transitions: `964`

Predictive encoder held-out training result:

| encoder | target | test_accuracy | majority | lift |
| --- | --- | ---: | ---: | ---: |
| `predictive_changed` | `changed` | 1.000 | 0.917 | 0.083 |

Probe results:

| feature_set | label | accuracy | majority | lift | test |
| --- | --- | ---: | ---: | ---: | ---: |
| `raw_current` | `mission_object` | 0.729 | 0.469 | 0.260 | 192 |
| `raw_current` | `mission_color` | 0.776 | 0.401 | 0.375 | 192 |
| `raw_current` | `changed` | 0.870 | 0.917 | -0.047 | 192 |
| `raw_current` | `next_signature_bucket` | 0.755 | 0.109 | 0.646 | 192 |
| `predictive_changed` | `mission_object` | 0.724 | 0.469 | 0.255 | 192 |
| `predictive_changed` | `mission_color` | 0.776 | 0.401 | 0.375 | 192 |
| `predictive_changed` | `changed` | 0.906 | 0.917 | -0.010 | 192 |
| `predictive_changed` | `next_signature_bucket` | 0.755 | 0.109 | 0.646 | 192 |

Decision: v2.11 did not meet the external-transition-baseline rule after
correcting the MiniGrid observation-axis handling. The `predictive_changed`
feature improved `changed` lift over `raw_current` by `0.036`, but its absolute
`changed` lift was `-0.010`, far below the v2.10 random-policy external
baseline `0.209`.

The scripted policy did improve the fixed-feature dataset structure:
`raw_current` `next_signature_bucket` lift rose to `0.646`, far above the v2.10
random-policy result `0.134`. This means the scripted dataset is useful, but
not for the current `changed` predictive objective.

Conclusion: do not move the scripted `changed` encoder back into RL/CUDA as-is.
The next step should use the corrected scripted-policy dataset with a transition
label that benefits from the new data distribution, especially
`next_signature_bucket` or a semantic object/color transition label.
