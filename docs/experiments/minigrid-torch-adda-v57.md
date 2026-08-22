# PyTorch AD/DA v2.46 Cross-Axis Validation of Long-Horizon ZI AD-Stop

Date: 2026-08-23 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/70

## Motivation

v2.45 preserved the `ZI` AD-stop edge across five CUDA seeds on `rtx4090`.
Issue #70 asked for validation on a new axis, preferring a second CUDA worker.
This run changes two things at once: the worker, and the seed count from five to
eight — eight being the first count at which the paired sign-flip test can reach
p < 0.05, since its exact floor is `2 / 2**n` (0.062 at n = 5).

## Source State

- Source commit: `56f0c3d` plus the seed-statistics and greedy-holdout branch
- Config: `configs/experiments/minigrid-torch-adda-v49.json` (v48 plus
  `holdout_episodes: 60` and an explicit `baseline_condition`)
- Workers: `rtx4090` and `rtx5060ti`, both WSL Ubuntu
- GPUs: `NVIDIA GeForce RTX 4090` (driver `610.74`) and
  `NVIDIA GeForce RTX 5060 Ti` (driver `576.88`)
- Torch: `2.11.0+cu128` on both, rebuilt from
  `scripts/setup_minigrid_env.sh` with
  `MINIGRID_TORCH_INDEX_URL=https://download.pytorch.org/whl/cu128`
- Seeds: `5301,5302,5303,5304,5305,5306,5307,5308`
- Artifacts: `.tmp/v246-zi-4090-8seed/20260822T221034Z/summary.md` and
  `.tmp/v246-zi-crossaxis-8seed/20260822T221117Z/summary.md`

The `rtx4090` CUDA venv used for v2.43-v2.45 was found to be a broken symlink
into a sibling clone that had been deleted, so both environments were rebuilt
from scratch for this run.

## Result 1: the worker axis is not an axis

The two GPUs produced **bit-identical** results: every per-seed, per-condition
success rate, return, and holdout value matches, and the whole `aggregate` and
`statistics` blocks match. The only differing leaves in the entire artifact are
timestamps and `mean_representation_loss_last_window` at the 1e-10 level, which
never flips an `argmax`.

The same eight seeds were then run with `--device cpu` in the same venv on
`rtx4090`. **CPU is also bit-identical.**

| device | total | per seed |
| --- | ---: | ---: |
| CUDA `rtx4090` | 8.2 min | 70.0 s |
| CUDA `rtx5060ti` | 10.7 min | 91.3 s |
| CPU `rtx4090` | 9.4 min | 80.8 s |

The mechanism is in the code: weight initialization runs on CPU under
`torch.manual_seed` before `.to(device)`, there is no dropout, no CUDA RNG is
drawn, and action selection and replay sampling share one `random.Random(seed)`.

The "CUDA replication" gates in v2.34, v2.38, and v2.43 therefore carried no
independent evidence, and the preferred gate in issue #70 cannot produce any.
An RTX 4090 buys 1.15x over the same host's CPU on this loop.

## Result 2: the ZI edge does not survive a new seed set

| metric | v2.45 (seeds 4301-4305) | v2.46 (seeds 5301-5308) |
| --- | --- | --- |
| `ZI` Δ success vs `ZK` | +0.230, CI [+0.140, +0.340], p 0.062, 5/0/0 | +0.050, CI [-0.044, +0.125], p 0.375, 5/2/1 |
| `ZI` Δ return vs `ZK` | +0.210, CI [+0.153, +0.285] | +0.029, CI [-0.047, +0.106], p 0.500 |

Re-running the original seeds `4301-4305` on the rebuilt environment reproduced
v2.45 exactly (0.360 / 0.290 / 0.320 / 0.530 / 0.300), so the environment
rebuild, the driver change from `610.62` to `610.74`, and the branch's code
changes account for none of the difference.

The cause is the control, not `ZI`:

| condition | v2.45 | v2.46 | move |
| --- | ---: | ---: | ---: |
| `ZK` (no representation) | 0.300 | 0.431 | **+0.131** |
| `ZI` (AD-stop) | 0.530 | 0.481 | -0.049 |

The v2.45 effect was the control drawing a weak seed set. `ZI` barely moved.

## Result 3: every condition loses to a random policy on held-out evaluation

Uniform-random success on `BabyAI-GoToObj-v0`, measured on the exact episode
seeds each metric uses: **0.283** on the holdout range, **0.269** on the
reported training window. The two ranges are equally hard, so there is no
episode-difficulty confound.

| condition | repr updates | training window (eps 0.2) | greedy holdout | vs random 0.283 |
| --- | ---: | ---: | ---: | --- |
| `ZK` | 0 | 0.431 | 0.254 | below |
| `ZI` | 235 | 0.481 | 0.192 | below |
| `ZH` | 4098 | 0.431 | 0.158 | below |
| `ZG` | 4286 | 0.319 | 0.125 | below |
| `ZE` | 4228 | 0.256 | 0.079 | below |

`ZK`, `ZI`, and `ZH` clear the floor with the epsilon-greedy behaviour policy;
no condition clears it with its own `argmax`. That is the signature of a
Q-function whose greedy policy is worse than chance, with the 20% random
component doing the work. `epsilon` is never annealed, so the greedy policy was
never exercised during training and this was invisible until the holdout existed.

## Decision

`ZI` is **not** the long-horizon representation-driven baseline. It is not
separable from the no-representation control on a new seed set, and on held-out
greedy evaluation it is below both the control and a random policy.

Two further facts constrain how `ZI` should be read at all. Its
`decoder_delay_episodes` of 8 is a *global* episode index, and stage 1
(`empty_warmup`, `MiniGrid-Empty-5x5-v0`, 12 episodes) covers all of it, so the
entire AD-only phase happens in a room with no objects and no mission — while
the objective is `state_plus_mission_target`. Measured over this run, `ZI`
performs 235 representation updates against `ZE`'s 4228, all of them in that
stage. `ZI` is the no-representation control plus eight forced-random warmup
episodes and a prediction head trained only on target-free observations.

Issue #70's preferred gate is closed as unable to produce evidence. The next
gate is v2.47, which separates the representation objective from the shared
optimizer schedule.

Full audit: `docs/experiments/verification-audit-20260823.md`.
