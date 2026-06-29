# MiniGrid Representation Probe v2.8

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/33

## Motivation

v2.7 showed that GoToObj curriculum explains more of the current DQN
auxiliary-head success than the representation objective itself. Before
spending more GPU on RL, this experiment checks whether fixed observation
features already make the relevant probe labels separable.

## Chosen Path

Use Option B from #33: supervised diagnostic only.

This is intentionally non-DQN and CPU-safe. It collects random-policy
transitions from GoToObj-family tasks, extracts fixed current-observation
features, and evaluates lightweight centroid probes for mission object,
mission color, and whether the transition changed the observation signature.

## Hypothesis

If fixed features already separate mission and transition labels, the next
research step should train a non-DQN encoder objective and probe the frozen
embedding before returning to RL. If fixed features do not beat majority
baselines, the issue is likely data/labels/feature extraction rather than the
DQN optimizer.

## Config

Config: `configs/experiments/minigrid-repr-probe-v28.json`

Dataset:

- `BabyAI-GoToRedBall-v0`: 6 random-policy episodes.
- `BabyAI-GoToObj-v0`: 10 random-policy episodes.

Feature sets:

- `raw_current`: sparse hashed current-observation features.
- `affordance_current`: compact hand-built affordance vector from the current
  observation.

Probe labels:

- `mission_object`
- `mission_color`
- `changed`

## Decision Rule

Run a bounded local CPU smoke. Treat v2.8 as passing only if one fixed feature
set reaches at least `0.600` accuracy, at least `0.050` lift over majority
baseline, and at least 10 test examples for all three decision labels:
`mission_object`, `mission_color`, and `changed`.

If the rule passes, move to a trained non-DQN encoder probe. If the rule fails,
redesign the probe labels or data collection before returning to RL. v2.8 does
not escalate to CUDA.

## Current Result

A bounded local CPU smoke completed in the existing optional MiniGrid venv.

Summary artifact:

`.tmp/local-v28-repr-probe/20260629T104047Z/summary.md`

Transitions: `986`

| feature_set | label | accuracy | majority | lift | test |
| --- | --- | ---: | ---: | ---: | ---: |
| `raw_current` | `mission_object` | 0.690 | 0.569 | 0.122 | 197 |
| `raw_current` | `mission_color` | 0.553 | 0.386 | 0.168 | 197 |
| `raw_current` | `changed` | 0.574 | 0.584 | -0.010 | 197 |
| `raw_current` | `next_signature_bucket` | 0.223 | 0.102 | 0.122 | 197 |
| `affordance_current` | `mission_object` | 0.624 | 0.569 | 0.056 | 197 |
| `affordance_current` | `mission_color` | 0.396 | 0.386 | 0.010 | 197 |
| `affordance_current` | `changed` | 0.604 | 0.584 | 0.020 | 197 |
| `affordance_current` | `next_signature_bucket` | 0.142 | 0.102 | 0.041 | 197 |

Decision: v2.8 did not meet the rule. `raw_current` was the best feature set
with mean decision-label accuracy `0.606` and mean lift `0.093`, but it missed
the per-label thresholds because `mission_color` accuracy was `0.553` and
`changed` lift was negative. `affordance_current` passed neither the color nor
transition-change labels.

Conclusion: fixed current-observation features are not enough to make all probe
labels cleanly separable under this random-policy dataset. Before returning to
RL or CUDA, redesign the non-DQN probe around stronger transition labels or a
trained predictive encoder.
