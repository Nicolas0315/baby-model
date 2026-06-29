# MiniGrid Neural Encoder On BabyAI Unlock

Date: 2026-06-29 JST

## Purpose

Issue #11 tests a stronger optional learned representation after the linear
multi-seed sweep showed that the issue #9 single-seed encoder-first signal was
not robust. This pilot keeps the default verifier dependency-free and uses a
small CPU-safe one-hidden-layer Q-network implemented in this repository.

## Config

- Config: `configs/experiments/minigrid-neural-unlock.json`
- Env: `BabyAI-Unlock-v0`
- Seed: `501`
- Episodes per condition: `60`
- Max steps: `160`
- Agent: one-hidden-layer Q-network, `feature_dim=1024`, `hidden_dim=24`,
  `alpha_output=0.05`, `alpha_hidden=0.005`, `gamma=0.92`, `epsilon=0.2`
- Output: `.tmp/verify-minigrid/neural/20260629T021250Z/`

Command:

```sh
. .venv-minigrid/bin/activate
MINIGRID_NEURAL_CONFIG=configs/experiments/minigrid-neural-unlock.json \
MINIGRID_NEURAL_SEED=501 \
./scripts/verify_minigrid.sh
```

## Result

| condition | success_all | success_last | return_last | mean_steps_success | nonzero_parameters |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A_neural_hard_only` | 0.017 | 0.000 | 0.000 | 27.00 | 12343 |
| `B_neural_encoder_first` | 0.017 | 0.000 | 0.000 | 12.00 | 168 |
| `E_neural_progress` | 0.017 | 0.000 | 0.000 | 101.00 | 12967 |

Winner by last-window success: `A_neural_hard_only` by tie order. All
conditions had `success_last=0.000`.

## Interpretation

This CPU-safe neural pilot did not improve the Baby-AD/DA hard-task result.
The learned hidden layer is active, but all conditions lost last-window
success.

## Fleet Replication

The neural config was replicated on all four configured worker classes at
commit `c08b295014ec24489585b6a3798b9834c6c1597e`. Every worker produced the
same result, with `A_neural_hard_only` winning only by tie order. Exact
host-level evidence is kept outside this repository in local docs.

The next decision is whether to tune the neural architecture in the
dependency-free runner or move to a real deep-learning dependency/GPU lane.
