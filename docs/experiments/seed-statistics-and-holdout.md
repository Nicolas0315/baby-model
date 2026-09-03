# Seed Statistics and Greedy Holdout Evaluation

Date: 2026-08-23 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/70

## Why

Through v2.45 every condition decision was made from point estimates only:
`mean_success_rate_last_window`, `median`, and a per-seed win count. Two
properties of the protocol make that unsafe:

1. `success_rate_last_window` is the mean over the final 20 training episodes.
   At p = 0.5 a single seed's estimate has a binomial standard error of about
   0.11, so seed-level values move by ±0.10 for free.
2. Those 20 episodes are collected with epsilon = 0.2 exploration on the very
   episodes the agent learned from. The metric mixes policy quality with
   exploration noise and has no held-out component.

Differences of 0.02-0.06 were nevertheless read as directional evidence in
several earlier decisions. This lane adds the two mechanisms needed to tell a
real effect from protocol noise.

## Mechanism 1: paired seed-level statistics

`baby_model/stats.py`, standard library only and deterministic.

Every condition runs on the same seed set, so the per-seed differences are
paired. Under the null "the condition label carries no information" the sign of
each seed's difference is a fair coin, which makes the exact test a sign-flip
randomization test over all `2**n` assignments (enumerated for n <= 20).
Intervals are percentile bootstrap with a fixed seed, so a report is
reproducible.

`run_minigrid_torch_sweep` now attaches a `statistics` block to every sweep
report and renders a `## Seed-Level Statistics` section into `summary.md`. The
baseline defaults to the first condition declared in the config, which is the
no-representation control; `baseline_condition` in the config overrides it.

Existing artifacts can be re-analyzed offline without a GPU:

```sh
python3 -m baby_model.stats path/to/metrics.json
```

### Hard ceiling at five seeds

The exact two-sided p-value floor of the sign-flip test is `2 / 2**n`: 0.250 at
n = 3, 0.062 at n = 5, 0.031 at n = 6, 0.008 at n = 8. **No three-seed or
five-seed gate in this repository can reach p < 0.05**, no matter how large the
effect. Gates that want conventional significance need at least six seeds, and
eight is the first comfortable number.

## Mechanism 2: greedy holdout evaluation

`run_greedy_holdout` in `baby_model/minigrid_torch.py`, enabled by
`holdout_episodes` at the top level of a torch config (default `0`, off).

After training completes it runs the frozen policy at epsilon = 0 with no
`update`, no representation update, and no auxiliary action bonus, on episode
seeds offset by `+99_000` so they cannot overlap the training range. It reports
`holdout_success_rate`, `holdout_mean_return`, and `holdout_mean_steps` per
condition, which then flow into the aggregate, the statistics block, and a
`winner_by_mean_holdout_success` line.

`winner_by_mean_success_last_window` is unchanged, so historical comparisons
still line up. The holdout winner is reported alongside it rather than
replacing it.

## Re-analysis of the existing CUDA evidence

Applied to the untouched v2.44 and v2.45 artifacts on `rtx4090`
(`.tmp/rtx4090-v68-zi-cuda-sweep/20260629T180341Z/metrics.json` and
`.tmp/rtx4090-v69-zi-cuda-5seed/20260629T182133Z/metrics.json`). Baseline is
the no-representation control `ZK_torch_gotoobj_curriculum_no_repr_delay_long`.

v2.45, five CUDA seeds, `success_rate_last_window`:

| condition | mean | sd | Δ vs `ZK` | Δ 95% CI | p | W/L/T |
| --- | ---: | ---: | ---: | :---: | ---: | :---: |
| `ZE_..._b005_long` | 0.360 | 0.182 | +0.060 | [-0.020, +0.140] | 0.500 | 3/1/1 |
| `ZG_..._b0025_long` | 0.290 | 0.195 | -0.010 | [-0.230, +0.210] | 1.000 | 2/2/1 |
| `ZH_..._b00375_long` | 0.320 | 0.222 | +0.020 | [-0.090, +0.170] | 0.938 | 2/3/0 |
| `ZI_..._b005_long_ad_stop` | 0.530 | 0.057 | **+0.230** | **[+0.140, +0.340]** | **0.062** | **5/0/0** |
| `ZK_..._no_repr_delay_long` | 0.300 | 0.112 | — | — | — | — |

v2.45, `mean_return_last_window`, same direction: `ZI` at +0.210,
CI [+0.153, +0.285], p = 0.062, 5/0/0; every other condition p >= 0.125 with a
CI that includes or nearly includes zero.

v2.44, three CUDA seeds: `ZI` is +0.233 with CI [+0.100, +0.450] and 3/0/0, at
the n = 3 p-floor of 0.250.

### What this changes

- **`ZI` survives.** It beat the control on every seed in both sweeps, its
  interval excludes zero on success and on return, and its seed-to-seed sd
  (0.057) is half the control's (0.112) — it is the only condition in the set
  that is also *stable*. At p = 0.062 it sits exactly on the n = 5 floor, so it
  is as significant as five seeds can express.
- **The beta-neighborhood conclusions do not survive.** `ZE`, `ZG`, and `ZH`
  are statistically indistinguishable from the no-representation control
  (p = 0.5-1.0, intervals straddling zero). Earlier decisions that ranked these
  against each other, and the v2.39/v2.40 reading of `ZE` as "the current
  strongest representation-driven candidate", were reading protocol noise. The
  documents stay as the record of what was run; the ranking between `ZE`, `ZG`,
  and `ZH` should not be carried forward as evidence.
- **The `ZI` result is now the one claim worth spending seeds on**, and it needs
  n >= 6 to clear p < 0.05 and a second worker to clear the single-axis risk.

## Verification

- `./scripts/verify.sh` (includes the `baby_model.stats` self-check and the
  `holdout_episodes` config gate).
- `python3 -m unittest discover -s tests -p 'test_*.py'`: 97 tests.
- Real-environment smoke on the local Mac with `minigrid` 3.1.0 and
  `torch` 2.12.1 (CPU): a three-seed sweep with `holdout_episodes` produced a
  `summary.md` carrying the statistics section, the `holdout_success_rate`
  block, and `winner_by_mean_holdout_success`.

## Not covered

- The v0/v0.2/v0.3 stdlib sweeps, the linear sweep, and the representation
  probe sweep still report point estimates only. The torch lane is where
  decisions are made, so only it was wired up.
- `holdout_episodes` defaults to off, so no existing config changes behavior.
  `configs/experiments/minigrid-torch-adda-v49.json` is the first config to
  enable it.
