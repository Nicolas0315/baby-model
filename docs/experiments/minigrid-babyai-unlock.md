# BabyAI Unlock Harder Sparse Task

Date: 2026-06-29 JST

## Purpose

Issue #7 moves the MiniGrid lane beyond the easy
`MiniGrid-Empty-5x5-v0` smoke. This run uses `BabyAI-Unlock-v0`, which keeps
the same sparse-reward API surface but requires a harder key/door interaction.

## Config

- Config: `configs/experiments/minigrid-babyai-unlock.json`
- Environment: `BabyAI-Unlock-v0`
- Max steps: `160`
- Episodes per condition: `60`
- Seed: `201`
- Output: `runs/minigrid-hard/20260629T011647Z/`

Command:

```sh
. .venv-minigrid/bin/activate
python -m baby_model.minigrid_experiment \
  --config configs/experiments/minigrid-babyai-unlock.json \
  --output-dir runs/minigrid-hard \
  --seed 201
```

The same config also passes through the optional verifier:

```sh
. .venv-minigrid/bin/activate
MINIGRID_EXTRA_CONFIG=configs/experiments/minigrid-babyai-unlock.json \
MINIGRID_EXTRA_SEED=201 \
./scripts/verify_minigrid.sh
```

## Result

| condition | success_all | success_last | return_last | mean_steps_success | intrinsic_last |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A_end_to_end` | 0.033 | 0.050 | 0.040 | 107.50 | 0.000 |
| `B_encoder_first` | 0.050 | 0.000 | 0.000 | 26.00 | 0.000 |
| `E_progress_anneal` | 0.000 | 0.000 | 0.000 |  | 0.177 |

Winner by last-window success: `A_end_to_end`.

## Interpretation

This is a useful negative result for the current Baby-AD/DA implementation.
The encoder-first condition found a few early successes, but did not retain
last-window success on this harder sparse task. The progress-annealed intrinsic
variant explored more states but did not convert that signal into task reward.

The next change should focus on algorithm capacity or curriculum, not just
intrinsic beta tuning in the same tabular Q-learning setup.
