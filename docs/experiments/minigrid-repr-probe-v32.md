# MiniGrid Representation Probe v2.12

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/37

## Motivation

v2.11 showed that the corrected scripted-policy GoToObj dataset made the
fixed-feature `next_signature_bucket` label much more separable than the
random-policy dataset, while the binary `changed` predictive encoder remained
negative. v2.12 therefore keeps the scripted dataset and changes only the
predictive target to `next_signature_bucket`.

## Chosen Path

Use option 1 from #37: train a predictive encoder against
`next_signature_bucket` on the corrected scripted-policy dataset.

This is the narrowest test of whether the encoder can add value beyond the raw
scripted-policy feature signal. It does not add semantic labels, multi-label
targets, CUDA, or new policy logic.

## Config

Config: `configs/experiments/minigrid-repr-probe-v32.json`

Dataset:

- `BabyAI-GoToRedBall-v0`: 6 scripted-policy episodes.
- `BabyAI-GoToObj-v0`: 10 scripted-policy episodes.

Feature sets:

- `raw_current`
- `predictive_next_signature`

Probe labels:

- `mission_object`
- `mission_color`
- `next_signature_bucket`

## Decision Rule

Run a bounded local CPU smoke. Treat v2.12 as positive only if
`predictive_next_signature` improves `next_signature_bucket` probe lift over
`raw_current` on the same corrected scripted-policy dataset by at least
`0.010`, while keeping `mission_object` and `mission_color` accuracy within
`0.050` of `raw_current`. Both feature sets must have at least 10 test
examples.

Because v2.11 already found a strong fixed-feature
`next_signature_bucket` lift of `0.646`, also report whether the learned
encoder adds value beyond that raw scripted-policy signal. Do not escalate
v2.12 to CUDA.

## Current Result

A bounded local CPU smoke completed in the existing optional MiniGrid venv.

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_repr_probe --config configs/experiments/minigrid-repr-probe-v32.json --output-dir .tmp/local-v32-repr-probe --seed 2701`

Summary artifact:

`.tmp/local-v32-repr-probe/20260629T114010Z/summary.md`

Transitions: `781`

Predictive encoder held-out training result:

| encoder | target | test_accuracy | majority | lift |
| --- | --- | ---: | ---: | ---: |
| `predictive_next_signature` | `next_signature_bucket` | 0.282 | 0.167 | 0.115 |

Probe results:

| feature_set | label | accuracy | majority | lift | test |
| --- | --- | ---: | ---: | ---: | ---: |
| `raw_current` | `mission_object` | 0.814 | 0.506 | 0.308 | 156 |
| `raw_current` | `mission_color` | 0.519 | 0.494 | 0.026 | 156 |
| `raw_current` | `next_signature_bucket` | 0.628 | 0.167 | 0.462 | 156 |
| `predictive_next_signature` | `mission_object` | 0.814 | 0.506 | 0.308 | 156 |
| `predictive_next_signature` | `mission_color` | 0.519 | 0.494 | 0.026 | 156 |
| `predictive_next_signature` | `next_signature_bucket` | 0.628 | 0.167 | 0.462 | 156 |

Decision: v2.12 did not meet the relative-to-baseline rule. The
`predictive_next_signature` encoder learned a weak held-out classifier signal
for `next_signature_bucket`, but the resulting representation did not improve
the downstream `next_signature_bucket` probe over `raw_current`; lift delta was
`0.000` against the required `0.010`.

Compared with the v2.11 fixed-feature lift of `0.646`, this seed also produced
a weaker raw scripted-policy fixed-feature lift of `0.462`, so the current
bucketed signature target is sensitive to sampled trajectories and still does
not add value beyond raw features.

Conclusion: do not move the bucketed next-signature predictive encoder back
into RL/CUDA as-is. The next step should remove raw passthrough for a purer
representation diagnostic, or replace hashed buckets with a semantic
object/color transition label.
