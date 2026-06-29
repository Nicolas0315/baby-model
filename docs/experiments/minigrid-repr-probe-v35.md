# MiniGrid Representation Probe v2.15

Date: 2026-06-29 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/40

## Motivation

v2.14 made the first semantic transition label directionally useful but did
not meet its single-seed positive threshold. v2.15 tests whether that near-miss
is stable across seeds before deciding whether to refine the relation buckets
or return the signal to the RL/CUDA lane.

## Chosen Path

Use option 1 from #40: multi-seed semantic probe.

Run the v2.14 `target_visibility_transition` representation probe over three
fixed seeds and aggregate:

- semantic transition lift delta vs `raw_current`
- mission-object accuracy delta
- mission-color accuracy delta
- count of seeds with non-negative semantic lift delta

This keeps the experiment CPU-bounded and does not add CUDA or new label logic.

## Config

Config: `configs/experiments/minigrid-repr-probe-v35.json`

Runner:

`python -m baby_model.minigrid_repr_probe_sweep`

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

Run a bounded local CPU sweep with seeds `2901,2902,2903`. Treat v2.15 as
positive only if:

- mean `target_visibility_transition` lift delta over `raw_current` is at
  least `0.010`;
- no seed has mission-object or mission-color accuracy delta worse than
  `-0.050`;
- at least 2 of 3 seeds have non-negative semantic lift delta.

Do not escalate v2.15 to CUDA. If the rule fails, refine relation buckets or
return to the already-positive GoToObj RL evidence rather than feeding this
semantic probe into RL/CUDA.

## Current Result

A bounded local CPU sweep completed in the existing optional MiniGrid venv.

Command:

`./.venv-minigrid-torch/bin/python -m baby_model.minigrid_repr_probe_sweep --config configs/experiments/minigrid-repr-probe-v35.json --output-dir .tmp/local-v35-repr-probe-sweep --seeds 2901,2902,2903`

Summary artifact:

`.tmp/local-v35-repr-probe-sweep-rerun/20260629T130506Z/summary.md`

Aggregate:

| metric | value |
| --- | ---: |
| `decision_met` | true |
| `mean_transition_lift_delta` | 0.045099 |
| `nonnegative_transition_lift_delta_count` | 3 |
| `min_mission_object_accuracy_delta` | 0.000000 |
| `min_mission_color_accuracy_delta` | 0.000000 |
| `min_transition_test_examples` | 131 |

Per seed:

| seed | seed_decision | raw_lift | candidate_lift | lift_delta | object_delta | color_delta | test |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2901 | false | 0.387 | 0.392 | 0.006 | 0.000 | 0.000 | 181 |
| 2902 | true | 0.115 | 0.244 | 0.130 | 0.046 | 0.000 | 131 |
| 2903 | false | 0.240 | 0.240 | 0.000 | 0.000 | 0.000 | 167 |

Decision: v2.15 met the multi-seed semantic probe rule. Mean semantic lift
delta over `raw_current` was above `0.010`, all three seeds were non-negative,
and no seed dropped mission-object or mission-color accuracy below the allowed
`-0.050` gate.

Conclusion: the v2.14 semantic near-miss is stable enough to treat the
semantic transition label as a positive non-DQN representation diagnostic. The
next step should be a bounded RL/CUDA integration proposal, not more hashed
bucket probing.
