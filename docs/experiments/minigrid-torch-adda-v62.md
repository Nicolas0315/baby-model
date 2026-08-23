# PyTorch AD/DA v2.53: the Hypothesis Passes, at 22 Seeds

Date: 2026-08-23 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/70

## Motivation

At eight seeds the representation objective at `lr = 1e-4` beat the
no-representation control by `+0.130` with p `0.1328` — the best evidence the
project had ever produced, and not significant. The paired differences had
sd `0.216`, so the power calculation asked for about 22 seeds. This is that run.

## Source State

- Config: `configs/experiments/minigrid-torch-adda-v57.json` — `ZK`,
  `ZE lr 1e-5`, `ZE lr 1e-4`, `ZE lr 1e-3` at 3200 eval episodes,
  `holdout_episodes: 120`, `separate_representation_optimizer: true`,
  **`common_random_numbers: true`**
- Seeds `5301-5322` (22), split across the Mac (torch `2.12.1`) and
  `rtx5060ti` (torch `2.11.0+cu128`), `device=cpu`, one thread
- 284,768 episode rows logged
- `rtx4090` excluded: three unexplained segfaults, see `docs/current-state.md`

## Result

Greedy holdout, 120 held-out episodes per seed. Random-policy floor **0.259**.
The exact paired p-floor at n=22 is 4.8e-07, so the test is no longer the
limiting factor.

| condition | holdout | sd | vs floor | train | steps to goal | clears own floor | worst seed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK` (no representation) | 0.503 | 0.315 | +0.244 | 0.718 | 34.3 | 15/22 | 0.000 |
| `ZE` lr 1e-5 | 0.543 | 0.321 | +0.284 | 0.720 | 32.0 | 14/22 | 0.000 |
| **`ZE` lr 1e-4** | **0.735** | **0.118** | **+0.476** | **0.980** | **20.7** | **22/22** | **0.483** |
| `ZE` lr 1e-3 | 0.031 | 0.047 | -0.228 | 0.184 | 62.1 | 0/22 | 0.000 |

| pair | Δ | 95% CI | p | W/L/T |
| --- | ---: | :---: | ---: | :---: |
| `ZE` lr 1e-5 - `ZK` | +0.040 | [-0.103, +0.189] | 0.6117 | 11/11/0 |
| **`ZE` lr 1e-4 - `ZK`** | **+0.232** | **[+0.089, +0.384]** | **0.0065** | **14/8/0** |
| `ZE` lr 1e-3 - `ZK` | -0.472 | [-0.601, -0.340] | 0.0002 | 2/19/1 |

Steps to goal, paired, lower is better: lr 1e-4 is **13.6 steps faster**
(p 0.0065, faster on 14 of 22 seeds); lr 1e-3 is 27.8 steps slower (p 0.0002).

Averaged learning curve, binned per seed across all 22:

| condition | verdict | curve | peak | steps |
| --- | --- | --- | ---: | ---: |
| `ZE` lr 1e-4 | improved | `▅▇▇▇████████` | 0.972 | 15.9 |
| `ZK` | improved | `▄▅▆▆▆▆▆▆▆▆▆▆` | 0.748 | 28.9 |
| `ZE` lr 1e-5 | improved | `▄▅▅▆▆▆▆▆▆▆▆▆` | 0.727 | 29.3 |
| `ZE` lr 1e-3 | declined | `▃▂▂▂▂▂▂▂▂▂▂▂` | 0.396 | 53.4 |

## What actually changed between 8 and 22 seeds

The effect **grew**: Δ went `+0.130` at n=8 to `+0.126` at n=15 to `+0.232` at
n=22. That is not the treatment improving. It is the control's failure mode
being sampled properly.

| n | `ZK` holdout | `ZE` lr 1e-4 holdout |
| ---: | ---: | ---: |
| 8 | 0.657 | 0.787 |
| 15 | 0.606 | 0.731 |
| 22 | 0.503 | 0.735 |

`ZE` lr 1e-4 barely moved. `ZK` fell by 0.154, because the added seeds contained
more of its collapses.

**That is the shape of the result.** `ZK` is bimodal: on 7 of 22 seeds its greedy
policy finishes at or below a random policy, including several at exactly 0.000.
`ZE` lr 1e-4 does this on **0 of 22**, with a worst seed of 0.483 and sd 0.118
against `ZK`'s 0.315.

So the representation objective at the right rate is not mainly making a good
agent better. It is **removing the failure mode**.

## Multiplicity, disclosed

I analysed this seed set four times — at n=4, 8, 15, and 22. That is optional
stopping and it inflates type-I error, so the p-value needs correcting.

The n=22 target was fixed in advance by the power calculation at n=8, and the
result at that pre-specified point is p `0.0065`. A Bonferroni correction across
four looks gives alpha `0.0125`, which `0.0065` clears. The conclusion survives
the correction, but a future gate should fix n before the first look rather than
rely on that.

One prediction of mine was also wrong in passing: at n=15 I estimated 34 seeds
would be needed. That assumed the observed Δ was stable, and Δ grew instead, so
significance arrived at 22.

## Decision

**The AD/DA hypothesis has passed a fair test.** With the representation head on
its own optimizer, at a learning rate two orders of magnitude below the policy's,
the representation objective:

- beats the no-representation control by `+0.232` on held-out greedy success,
  p `0.0065`, 14 of 22 seeds;
- reaches the goal 13.6 steps faster, same p;
- clears the random-policy floor on **every** seed, where the control fails on 7.

Three findings bound how far this should be read:

1. **It is rate-critical.** At the policy's own learning rate the same objective
   is catastrophic (0/22 above floor, p 0.0002). The window between useless
   (1e-5, p 0.61) and destructive (1e-3) is narrow.
2. **The task is near its ceiling for the winner.** `ZE` lr 1e-4 trains to 0.980
   and its averaged curve sits at 0.972 from early on. There is little headroom
   left on `GoToObj`, which is BabyAI's easiest tier, so this measures "removes
   failures" better than it measures "raises the ceiling".
3. **There is still no external upper reference.** The floor is measured; the
   ceiling is not. `0.735` has nothing above it to be compared against.

## Read again with rliable's tools — and corrected after adversarial review

`rliable` (Agarwal et al., NeurIPS 2021) exists for exactly this situation: few
runs, heavy tails. Two of its elements were reimplemented on the standard
library in `baby_model.stats`.

**The first version of this section overclaimed, and an adversarial review was
right to reject it.** It compared the profiles on nine hand-picked thresholds
and called the result "stochastic dominance". A coarse grid can hide a crossing
— a constructed counterexample now lives in the self-check, where `dominates()`
returns True on a three-point grid and the all-boundary check returns "crosses".
What follows is the corrected analysis.

### Checked at every unique observed value

| comparison | boundaries | verdict | gap range | crossings |
| --- | ---: | --- | :---: | --- |
| `ZE` lr 1e-4 vs `ZK` | 31 | **lr 1e-4 empirically dominates** | [+0.000, +0.409] | none |
| `ZE` lr 1e-5 vs `ZK` | 26 | **crosses** | [-0.045, +0.182] | at 0.267 |
| `ZE` lr 1e-3 vs `ZK` | 24 | `ZK` empirically dominates | [-0.773, +0.000] | none |

The substantive claim survives the stricter test: lr 1e-4's survival function is
at or above `ZK`'s at all 31 points where either can step, and strictly above
somewhere. The **wording** does not survive. This is a statement about 22
observed seeds, so it is *empirical* dominance; "stochastic dominance" is a
claim about the population that this sample cannot make. Corrected throughout.

The lr 1e-5 crossing sits at 0.267 — inside the interval the abbreviated table
below skips, which is why the earlier table could not reproduce its own "cross"
verdict. That was a presentation defect, not just wording.

### Profile with bootstrap CIs

| condition | IQM | mean | >0.0 | >0.2 | >0.4 | >0.6 | >0.8 |
| --- | ---: | ---: | :---: | :---: | :---: | :---: | :---: |
| `ZK` | 0.565 | 0.503 | 0.91 [0.77, 1.00] | 0.77 [0.59, 0.95] | 0.64 [0.45, 0.82] | 0.59 [0.36, 0.77] | 0.14 [0.00, 0.27] |
| `ZE` lr 1e-5 | 0.601 | 0.543 | 0.91 [0.77, 1.00] | 0.77 [0.59, 0.95] | 0.64 [0.41, 0.82] | 0.64 [0.41, 0.82] | 0.18 [0.05, 0.36] |
| `ZE` lr 1e-4 | 0.731 | 0.735 | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 1.00 [1.00, 1.00] | 0.86 [0.73, 1.00] | 0.32 [0.14, 0.50] |
| `ZE` lr 1e-3 | 0.017 | 0.031 | 0.45 | 0.00 | 0.00 | 0.00 | 0.00 |

**The CIs separate at only two of the five thresholds** — 0.2 and 0.4, where
lr 1e-4 is at 1.00 with a degenerate interval and `ZK` tops out at 0.95 and 0.82.
At 0.0, 0.6, and 0.8 the intervals overlap. So the profile supports "lr 1e-4 has
no seeds below 0.4 and `ZK` has several" much more strongly than it supports any
claim about the upper end.

### On the IQM-to-mean gap

The earlier text called the gap a tail-weight indicator and said lr 1e-4 has "no
tail to hide". That is more than the statistic supports: IQM trims both ends, so
the gap's sign does not identify which tail moved it. What is defensible is
narrower — `ZK`'s IQM (0.565) exceeds its mean (0.503) while lr 1e-4's IQM
(0.731) and mean (0.735) agree to 0.004, which is *consistent with* `ZK` having
low outliers, a reading independently supported by the seven seeds at or below
its own random floor.

### Not taken from rliable

`probability_of_improvement` uses Mann-Whitney and is unpaired, while common
random numbers make the paired sign-flip test both valid and more powerful here.
`StratifiedBootstrap` stratifies over (runs x tasks) and degenerates to an
ordinary bootstrap over runs on a single-task setup — it becomes necessary the
moment this spreads to several BabyAI levels.

## Next

- Measure the ceiling with a standard baseline on the same level, so 0.735 has a
  scale.
- Move to a harder level (`GoToLocal`, `PickupLoc`, `Unlock`). The result here is
  about removing collapses, and a harder task is where that should show more,
  not less.
- Anneal `epsilon`. The train-to-holdout gap is still 0.980 to 0.735 for the
  winner, and the greedy policy is never exercised during training.
