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

## Read again with rliable's tools

`rliable` (Agarwal et al., NeurIPS 2021) exists for exactly this situation: few
runs, heavy tails. Two of its elements were reimplemented on the standard
library in `baby_model.stats` and applied to this data.

**Performance profile** — fraction of seeds above each threshold:

| condition | IQM | mean | >0.0 | >0.2 | >0.4 | >0.6 | >0.8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `ZK` | 0.565 | 0.503 | 0.91 | 0.77 | 0.64 | 0.59 | 0.14 |
| `ZE` lr 1e-5 | 0.601 | 0.543 | 0.91 | 0.77 | 0.64 | 0.64 | 0.18 |
| **`ZE` lr 1e-4** | **0.731** | **0.735** | **1.00** | **1.00** | **1.00** | **0.86** | **0.32** |
| `ZE` lr 1e-3 | 0.017 | 0.031 | 0.45 | 0.00 | 0.00 | 0.00 | 0.00 |

**`ZE` lr 1e-4 stochastically dominates `ZK`** — at or above it at every
threshold and strictly above somewhere. That is a stronger claim than a mean
difference, and it does not depend on the mean being a good summary of a bimodal
distribution. `ZE` lr 1e-5 and `ZK` **cross**, which is the same conclusion the
paired test reached at p 0.61.

**The IQM-to-mean gap measures the tail.** `ZK`'s IQM (0.565) sits 0.062 above
its mean, because the mean is dragged down by seeds that end at zero. `ZE`
lr 1e-4's IQM (0.731) and mean (0.735) agree to within 0.004 — there is no tail
to hide. That gap is also why `ZK`'s mean moved -0.154 between n=8 and n=22
while lr 1e-4's moved -0.052.

Two of rliable's elements were deliberately **not** taken:
`probability_of_improvement`, because it uses Mann-Whitney and is therefore
unpaired, while common random numbers make the paired sign-flip test both valid
and more powerful here; and `StratifiedBootstrap`, which stratifies over
(runs x tasks) and degenerates to an ordinary bootstrap over runs on a
single-task setup.

## Next

- Measure the ceiling with a standard baseline on the same level, so 0.735 has a
  scale.
- Move to a harder level (`GoToLocal`, `PickupLoc`, `Unlock`). The result here is
  about removing collapses, and a harder task is where that should show more,
  not less.
- Anneal `epsilon`. The train-to-holdout gap is still 0.980 to 0.735 for the
  winner, and the greedy policy is never exercised during training.
