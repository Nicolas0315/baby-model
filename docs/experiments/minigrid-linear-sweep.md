# MiniGrid Linear Multi-Seed Sweep

Date: 2026-06-29 JST

## Purpose

Issue #10 checks whether the issue #9 single-seed positive signal survives
more than one seed. The sweep reuses the same sparse hashed linear Q-learning
config on `BabyAI-Unlock-v0`.

## Config

- Config: `configs/experiments/minigrid-linear-unlock.json`
- Seeds: `401,402,403`
- Env: `BabyAI-Unlock-v0`
- Episodes per condition per seed: `60`
- Max steps: `160`
- Output: `.tmp/verify-minigrid/linear-sweep/20260629T015612Z/`

Command:

```sh
. .venv-minigrid/bin/activate
MINIGRID_LINEAR_SWEEP_CONFIG=configs/experiments/minigrid-linear-unlock.json \
MINIGRID_LINEAR_SWEEP_SEEDS=401,402,403 \
./scripts/verify_minigrid.sh
```

## Result

| condition | wins | mean_success_all | mean_success_last | median_success_last | mean_return_last |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A_linear_hard_only` | 2 | 0.033 | 0.050 | 0.050 | 0.042 |
| `B_linear_encoder_first` | 1 | 0.039 | 0.050 | 0.050 | 0.046 |
| `E_linear_progress` | 0 | 0.006 | 0.000 | 0.000 | 0.000 |

Per-seed winners:

| seed | winner |
| ---: | --- |
| 401 | `B_linear_encoder_first` |
| 402 | `A_linear_hard_only` |
| 403 | `A_linear_hard_only` |

Winner by mean last-window success plus win-count tie-break:
`A_linear_hard_only`.

## Interpretation

The issue #9 single-seed result is not robust enough to treat as a conclusion.
`B_linear_encoder_first` still matched `A_linear_hard_only` on mean and median
last-window success and had slightly higher mean all-window success and return,
but it won only one of three seeds. The progress variant remained weak.

The next step should either broaden this sweep or move to a stronger optional
representation model. This result does not justify claiming that linear
function approximation alone solves the Baby-AD/DA hard-task problem.
