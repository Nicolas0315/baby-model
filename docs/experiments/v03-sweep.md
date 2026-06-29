# v0.3 Annealed/Auxiliary Progress Sweep

## Purpose

Test whether prediction progress works better when it is annealed, gated, or
kept out of the Q target.

## Conditions

- `B_encoder_first`: coarse encoder, delayed decoder, no intrinsic reward.
- `D_progress_b005`: v0.2 best direct reward-shaping beta.
- `E_progress_anneal`: beta `0.05` linearly annealed to `0.0` after decoder
  unlock.
- `F_progress_gate`: beta `0.05`, skipped when external reward is positive.
- `G_progress_aux`: beta `0.05` trains a separate auxiliary action-value used
  as an action-selection bonus; the external Q target remains task reward only.

## Local Result

Run command:

```sh
SESSION=baby-model-v03-aux CONFIG=configs/experiments/v03-sweep.json OUTPUT_DIR=runs/v03-sweeps SEEDS=101,102,103,104,105 ./scripts/launch_tmux_sweep.sh
```

Local raw artifact on the current Mac:
`runs/v03-sweeps/20260629T005034Z/summary.md`. The `runs/` directory is
gitignored, so the durable result is transcribed below.

| condition | wins | mean_success_last | mean_success_all | mean_steps_success | intrinsic_last |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B_encoder_first` | 4 | 0.800 | 0.675 | 36.02 | 0.000 |
| `D_progress_b005` | 0 | 0.720 | 0.530 | 38.60 | 0.437 |
| `E_progress_anneal` | 1 | 0.760 | 0.582 | 39.74 | 0.000 |
| `F_progress_gate` | 0 | 0.690 | 0.495 | 35.84 | 0.384 |
| `G_progress_aux` | 0 | 0.430 | 0.365 | 40.78 | 0.384 |

## Interpretation

Direct progress reward still underperforms the encoder-first baseline. Annealed
progress is the best intrinsic variant in this sweep, but still trails
`B_encoder_first`. The auxiliary exploration-pressure variant is worse than
direct reward shaping in this tiny grid, so auxiliary progress needs a weaker
weight, a decay schedule, or a representation-learning target before spending
fleet/GPU time on it.

## Verification

```sh
./scripts/verify.sh
```

Latest local verification on 2026-06-29 passed and included v0.2 and v0.3
two-seed sweep smoke tests.
