# MiniGrid Representation Probe v2.10

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/35

## Motivation

v2.9 showed that a trained non-DQN encoder for `changed` adds transition
information without destroying the mission signal. v2.10 checks whether a
richer predictive objective can improve a harder transition label before
returning to RL or CUDA.

## Chosen Path

Use Option A from #35: richer predictive objective.

The runner trains two standard-library frozen predictive encoders on the same
GoToObj-family random-policy transitions:

- `predictive_changed`: the v2.9-style encoder trained on `changed`.
- `predictive_next_signature`: a richer encoder trained on
  `next_signature_bucket`.

Both encoders use current sparse observation features plus the sampled action,
derive class vocabularies from the train split only, and freeze normalized score
channels before the centroid probes.

## Config

Config: `configs/experiments/minigrid-repr-probe-v30.json`

Dataset:

- `BabyAI-GoToRedBall-v0`: 6 random-policy episodes.
- `BabyAI-GoToObj-v0`: 10 random-policy episodes.

Feature sets:

- `raw_current`
- `predictive_changed`
- `predictive_next_signature`

Probe labels:

- `mission_object`
- `mission_color`
- `next_signature_bucket`

## Decision Rule

Run a bounded local CPU smoke. Treat v2.10 as positive only if
`predictive_next_signature` improves `next_signature_bucket` lift over both
`raw_current` and `predictive_changed` by at least `0.010`, while not reducing
`mission_object` or `mission_color` accuracy by more than `0.050` relative to
`raw_current`. All compared feature sets must have at least 10 test examples.

Do not escalate v2.10 to CUDA. If the rule passes, the next step is to decide
whether the richer non-DQN signal should be moved back into the RL auxiliary
head or first tested on a scripted-policy dataset. If it fails, prefer the
scripted-policy dataset path before adding more encoder complexity.

## Current Result

A bounded local CPU smoke completed in the existing optional MiniGrid venv.

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_repr_probe --config configs/experiments/minigrid-repr-probe-v30.json --output-dir .tmp/local-v30-repr-probe --seed 2501`

Summary artifact:

`.tmp/local-v30-repr-probe/20260629T111600Z/summary.md`

Transitions: `936`

Predictive encoder held-out training results:

| encoder | target | test_accuracy | majority | lift |
| --- | --- | ---: | ---: | ---: |
| `predictive_changed` | `changed` | 0.904 | 0.690 | 0.214 |
| `predictive_next_signature` | `next_signature_bucket` | 0.037 | 0.112 | -0.075 |

Probe results:

| feature_set | label | accuracy | majority | lift | test |
| --- | --- | ---: | ---: | ---: | ---: |
| `raw_current` | `mission_object` | 0.684 | 0.722 | -0.037 | 187 |
| `raw_current` | `mission_color` | 0.428 | 0.508 | -0.080 | 187 |
| `raw_current` | `next_signature_bucket` | 0.246 | 0.112 | 0.134 | 187 |
| `predictive_changed` | `mission_object` | 0.695 | 0.722 | -0.027 | 187 |
| `predictive_changed` | `mission_color` | 0.428 | 0.508 | -0.080 | 187 |
| `predictive_changed` | `next_signature_bucket` | 0.262 | 0.112 | 0.150 | 187 |
| `predictive_next_signature` | `mission_object` | 0.701 | 0.722 | -0.021 | 187 |
| `predictive_next_signature` | `mission_color` | 0.401 | 0.508 | -0.107 | 187 |
| `predictive_next_signature` | `next_signature_bucket` | 0.251 | 0.112 | 0.139 | 187 |

Decision: v2.10 did not meet the relative-to-reference rule. The
`predictive_next_signature` feature improved `next_signature_bucket` lift over
`raw_current` by only `0.005`, below the required `0.010`, and it was worse
than `predictive_changed` by `0.011`. The next-signature training objective
itself had held-out lift `-0.075`, so the richer bucket objective is not yet a
better non-DQN representation signal under this random-policy dataset.

Conclusion: do not return this richer bucket objective to RL/CUDA as-is. The
next step should prefer a more informative scripted-policy dataset or a
different transition label surface before adding more encoder complexity.
