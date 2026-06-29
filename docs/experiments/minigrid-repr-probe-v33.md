# MiniGrid Representation Probe v2.13

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/38

## Motivation

v2.12 showed that a `next_signature_bucket` predictive encoder learned a weak
held-out classifier signal, but the downstream representation probe matched
`raw_current` exactly because the encoder embedding included raw passthrough
features. v2.13 tests whether the learned signal is hidden by raw passthrough.

## Chosen Path

Use option 1 from #38: pure representation diagnostic.

Train two encoders on the same corrected scripted-policy dataset:

- `predictive_next_signature`: `next_signature_bucket` target with raw
  passthrough enabled, matching v2.12.
- `predictive_next_signature_pure`: same target and training settings, but
  with raw passthrough disabled.

This keeps the experiment CPU-bounded, standard-library in the core loop, and
does not add a new semantic label or CUDA lane.

## Config

Config: `configs/experiments/minigrid-repr-probe-v33.json`

Dataset:

- `BabyAI-GoToRedBall-v0`: 6 scripted-policy episodes.
- `BabyAI-GoToObj-v0`: 10 scripted-policy episodes.

Feature sets:

- `raw_current`
- `predictive_next_signature`
- `predictive_next_signature_pure`

Probe labels:

- `mission_object`
- `mission_color`
- `next_signature_bucket`

## Decision Rule

Run a bounded local CPU smoke. Treat v2.13 as positive only if
`predictive_next_signature_pure` improves `next_signature_bucket` probe lift
over both `raw_current` and the raw-passthrough
`predictive_next_signature` reference by at least `0.010`, while keeping
`mission_object` and `mission_color` accuracy within `0.050` of `raw_current`.
All compared feature sets must have at least 10 test examples.

Do not escalate v2.13 to CUDA. If the pure representation fails, the next
probe should replace hashed buckets with a semantic object/color transition
label on the corrected scripted-policy dataset.

## Current Result

A bounded local CPU smoke completed in the existing optional MiniGrid venv.

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_repr_probe --config configs/experiments/minigrid-repr-probe-v33.json --output-dir .tmp/local-v33-repr-probe --seed 2801`

Summary artifact:

`.tmp/local-v33-repr-probe/20260629T120222Z/summary.md`

Transitions: `904`

Predictive encoder held-out training result:

| encoder | passthrough | test_accuracy | majority | lift |
| --- | --- | ---: | ---: | ---: |
| `predictive_next_signature` | yes | 0.361 | 0.122 | 0.239 |
| `predictive_next_signature_pure` | no | 0.361 | 0.122 | 0.239 |

Probe results:

| feature_set | label | accuracy | majority | lift | test |
| --- | --- | ---: | ---: | ---: | ---: |
| `raw_current` | `mission_object` | 0.600 | 0.572 | 0.028 | 180 |
| `raw_current` | `mission_color` | 0.611 | 0.567 | 0.044 | 180 |
| `raw_current` | `next_signature_bucket` | 0.622 | 0.122 | 0.500 | 180 |
| `predictive_next_signature` | `mission_object` | 0.661 | 0.572 | 0.089 | 180 |
| `predictive_next_signature` | `mission_color` | 0.683 | 0.567 | 0.117 | 180 |
| `predictive_next_signature` | `next_signature_bucket` | 0.622 | 0.122 | 0.500 | 180 |
| `predictive_next_signature_pure` | `mission_object` | 0.511 | 0.572 | -0.061 | 180 |
| `predictive_next_signature_pure` | `mission_color` | 0.483 | 0.567 | -0.083 | 180 |
| `predictive_next_signature_pure` | `next_signature_bucket` | 0.511 | 0.122 | 0.389 | 180 |

Decision: v2.13 did not meet the relative-to-reference rule.
`predictive_next_signature_pure` trailed both `raw_current` and the
raw-passthrough reference on `next_signature_bucket` lift by `-0.111`, and its
mission-object and mission-color accuracies dropped by more than the allowed
`0.050`.

The pure encoder did not reveal a hidden representation advantage. The
passthrough encoder did improve mission-object and mission-color probe accuracy
over `raw_current`, but not the transition label.

Conclusion: do not move the pure bucketed next-signature encoder into RL/CUDA.
The next non-DQN probe should replace hashed buckets with a semantic
object/color transition label on the corrected scripted-policy dataset.
