# v0.2 Beta Sweep

## Purpose

Test whether prediction-progress intrinsic reward helps after the task is made
less trivial than v0.1.

## Environment Delta

- Grid size: `9`
- Obstacles: `14`
- Toys: `5`
- Max steps: `100`
- Observation now marks walls as `1`, agent as `2`, goal as `3`, toys as `4`,
  and agent-on-goal as `5`.

## Conditions

- `B_encoder_first`: coarse encoder, delayed decoder, no intrinsic reward.
- `D_progress_b005`: prediction-progress reward with beta `0.05`.
- `D_progress_b010`: prediction-progress reward with beta `0.10`.
- `D_progress_b020`: prediction-progress reward with beta `0.20`.
- `D_progress_b040`: prediction-progress reward with beta `0.40`.

## Local Result

Run command:

```sh
SESSION=baby-model-sweep SEEDS=101,102,103,104,105 ./scripts/launch_tmux_sweep.sh
```

Local raw artifact on the current Mac:
`runs/sweeps/20260628T030149Z/summary.md`. The `runs/` directory is
gitignored, so the durable result is transcribed below.

| condition | wins | mean_success_last | mean_success_all | mean_steps_success | intrinsic_last |
| --- | ---: | ---: | ---: | ---: | ---: |
| `B_encoder_first` | 5 | 0.800 | 0.675 | 36.02 | 0.000 |
| `D_progress_b005` | 0 | 0.720 | 0.530 | 38.60 | 0.437 |
| `D_progress_b010` | 0 | 0.540 | 0.403 | 42.32 | 0.831 |
| `D_progress_b020` | 0 | 0.370 | 0.327 | 42.84 | 1.613 |
| `D_progress_b040` | 0 | 0.360 | 0.317 | 41.76 | 2.794 |

## Interpretation

The progress reward is directionally useful only at the smallest beta tested.
Larger beta values over-weight intrinsic return and reduce sparse-goal success.
The next experiment should either anneal beta or make intrinsic reward a
gating/auxiliary update signal instead of adding it directly to the Q target.

## Fleet Replication

The same sweep was replicated on all four configured worker classes on
2026-06-29 via `git archive | ssh | tmux` at commit
`6b822baadff77cab6086584450507923d14437b1`. Every worker produced the same
aggregate winner, `B_encoder_first`. Exact host-level evidence is kept outside
this repository in local docs.

## Verification

```sh
./scripts/verify.sh
```

Latest local verification on 2026-06-29 passed and included a two-seed sweep
smoke test for `configs/experiments/v02-sweep.json`.
