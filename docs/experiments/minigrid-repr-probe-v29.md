# MiniGrid Representation Probe v2.9

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/34

## Motivation

v2.8 showed that fixed current-observation features were not enough to
separate both mission labels and the transition-change label in a non-DQN
probe. Before returning to RL or CUDA, v2.9 tests whether a small trained
predictive encoder adds transition information while preserving the mission
signal already present in `raw_current`.

## Chosen Path

Use Option A from #34: trained transition-prediction encoder.

The encoder is CPU-safe and standard-library only. It trains a deterministic
multiclass linear predictor on the train split to predict `changed` from
current sparse features plus the sampled action. The frozen embedding appends
normalized learned class scores and the predicted class indicator to a raw
current-observation passthrough, then reuses the existing centroid probes.

This is intentionally not DQN, not PyTorch, and not CUDA. It is a cheap
diagnostic for whether an explicitly trained predictive signal improves the
representation probe before spending more GPU time.

## Config

Config: `configs/experiments/minigrid-repr-probe-v29.json`

Dataset:

- `BabyAI-GoToRedBall-v0`: 6 random-policy episodes.
- `BabyAI-GoToObj-v0`: 10 random-policy episodes.

Feature sets:

- `raw_current`: sparse hashed current-observation features.
- `predictive_encoder`: raw passthrough plus frozen predictive encoder score
  channels trained on `changed`.

Probe labels:

- `mission_object`
- `mission_color`
- `changed`

## Decision Rule

Run a bounded local CPU smoke. Treat v2.9 as positive only if
`predictive_encoder` beats `raw_current` on `changed` lift by at least `0.050`
and does not reduce `mission_object` or `mission_color` accuracy by more than
`0.050`. Both feature sets must have at least 10 test examples.

Do not escalate v2.9 to CUDA. If the rule passes, the next step is a stronger
non-DQN predictive objective or scripted-policy dataset. If it fails, treat the
trained linear changed objective as insufficient and redesign the data/label
surface before returning to RL.

## Current Result

A bounded local CPU smoke completed in the existing optional MiniGrid venv.

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_repr_probe --config configs/experiments/minigrid-repr-probe-v29.json --output-dir .tmp/local-v29-repr-probe --seed 2401`

Summary artifact:

`.tmp/local-v29-repr-probe/20260629T110807Z/summary.md`

Transitions: `821`

Predictive encoder target: `changed`

Predictive encoder held-out accuracy: `0.945`

| feature_set | label | accuracy | majority | lift | test |
| --- | --- | ---: | ---: | ---: | ---: |
| `raw_current` | `mission_object` | 0.750 | 0.628 | 0.122 | 164 |
| `raw_current` | `mission_color` | 0.591 | 0.555 | 0.037 | 164 |
| `raw_current` | `changed` | 0.518 | 0.500 | 0.018 | 164 |
| `predictive_encoder` | `mission_object` | 0.744 | 0.628 | 0.116 | 164 |
| `predictive_encoder` | `mission_color` | 0.567 | 0.555 | 0.012 | 164 |
| `predictive_encoder` | `changed` | 0.933 | 0.500 | 0.433 | 164 |

Decision: v2.9 met the relative decision rule. The trained predictive encoder
improved `changed` lift by `0.415` over `raw_current`, while mission-object
accuracy dropped by only `0.006` and mission-color accuracy dropped by `0.024`,
both within the allowed `0.050` maximum drop.

Conclusion: a simple trained non-DQN predictive signal can inject transition
information into the representation probe without destroying the mission
signal. This supports the next research step: test a stronger predictive
objective or a more informative scripted-policy dataset before returning to
RL/CUDA.
