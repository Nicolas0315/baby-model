# MiniGrid Linear Function Approximation On BabyAI Unlock

Date: 2026-06-29 JST

## Purpose

Issue #9 tests whether a small CPU-safe function approximator changes the
negative `BabyAI-Unlock-v0` pattern from the tabular hard-task and curriculum
runs. The pilot keeps dependencies optional and uses a sparse hashed linear
Q-function over MiniGrid observations.

## Config

- Config: `configs/experiments/minigrid-linear-unlock.json`
- Env: `BabyAI-Unlock-v0`
- Seed: `401`
- Episodes per condition: `60`
- Max steps: `160`
- Agent: linear Q-learning, `feature_dim=4096`, `alpha=0.06`, `gamma=0.92`,
  `epsilon=0.2`
- Output: `.tmp/verify-minigrid/linear/20260629T014353Z/`

Command:

```sh
. .venv-minigrid/bin/activate
MINIGRID_LINEAR_CONFIG=configs/experiments/minigrid-linear-unlock.json \
MINIGRID_LINEAR_SEED=401 \
./scripts/verify_minigrid.sh
```

## Result

| condition | success_all | success_last | return_last | mean_steps_success | nonzero_weights |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A_linear_hard_only` | 0.017 | 0.000 | 0.000 | 45.00 | 4108 |
| `B_linear_encoder_first` | 0.050 | 0.050 | 0.044 | 47.33 | 3883 |
| `E_linear_progress` | 0.000 | 0.000 | 0.000 |  | 4126 |

Winner by last-window success: `B_linear_encoder_first`.

## Comparison

Previous `BabyAI-Unlock-v0` runs were negative for the Baby-AD/DA variant:

| run | best Baby-AD/DA condition | best last-window success |
| --- | --- | ---: |
| tabular hard task | `B_encoder_first` | 0.000 |
| curriculum final stage | `B_curriculum_encoder_first` | 0.000 |
| linear function approximation | `B_linear_encoder_first` | 0.050 |

This is the first local `BabyAI-Unlock-v0` run where the encoder-first variant
retained last-window success. It is still a tiny pilot, so the result should be
treated as a signal to replicate, not as a conclusion.

## Next

Replicate through fleet tmux. If all workers match, the next question is
whether the same effect holds across seeds or with a stronger optional neural
encoder.
