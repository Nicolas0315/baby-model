# PyTorch AD/DA v2.50-v2.52: `beta` Is a No-Op, the Dial Is the Learning Rate, and the Objective Has an Interior Optimum

Date: 2026-08-23 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/70

Covers three runs that belong together: the sweep that failed because it moved
the wrong knob, the measurement that explained why, and the sweep on the knob
that works.

## v2.50: sweeping `representation_beta` changed nothing

With the separate optimizer, `beta` at 0.005 / 0.05 / 0.5 — a 100x range — gave
holdout 0.079 / 0.065 / 0.102, all far below the random floor of 0.296, all
0/4 against the control.

That is not weak sensitivity. It is cancellation, and it is a property of Adam:
the update is `lr * m_hat / (sqrt(v_hat) + eps)`, and a constant factor on the
loss scales `m` by `beta` and `v` by `beta**2`, leaving `m/sqrt(v)` unchanged.

Measured directly, with a non-stationary target so the predictor cannot converge
and the gradient stays live:

| knob | range | encoder step |
| --- | --- | --- |
| `representation_beta` 0.005 -> 0.5 | 100x | 2.4063e-03 -> 2.4056e-03, **1.00x** |
| representation `lr` 1e-05 -> 1e-02 | 1000x | 9.6560e-06 -> 2.4057e-03, **249x** |

**`representation_beta` is an on/off switch, not a dial.** Every
beta-neighbourhood sweep before this — v2.24, v2.29, v2.37, and `ZG`/`ZH`/`ZE`
in v2.45 — was turning the switch as if it were the dial. That is
mechanistically why the v2.45 re-analysis found those conditions statistically
indistinguishable from one another: there was nothing there to distinguish.

Designing v2.50 around `beta` was a mistake on my part. The run's value is
entirely in having exposed it.

`agent`-level `representation_learning_rate` was added, then moved to the
condition level so a single sweep can hold several rates on one seed set —
cross-run comparison of absolute levels is not safe at this horizon.

## v2.51: an interior optimum appears

Four seeds, 3200 eval episodes, floor 0.296.

| condition | holdout | vs floor | train | Δ vs `ZK` | p | W/L/T |
| --- | ---: | ---: | ---: | ---: | ---: | :---: |
| `ZK` (none) | 0.619 | +0.323 | 0.787 | — | — | — |
| `ZE` lr 1e-5 | 0.554 | +0.258 | 0.812 | -0.065 | 0.750 | 2/2/0 |
| `ZE` lr 1e-4 | 0.698 | +0.402 | 0.950 | +0.079 | 0.500 | 3/1/0 |
| `ZE` lr 1e-3 | 0.023 | -0.273 | 0.150 | -0.596 | 0.125 | 0/4/0 |

Two readings. The v2.49 catastrophe is specific to `lr` = the Q learning rate:
1e-3 reproduces it (0.023 against 0.032) and two orders lower does not. So the
earlier claim "the representation objective prevents learning" narrows to
"learning the representation as fast as the policy prevents learning".

And lr 1e-4 was the first condition in this project to beat the control on
held-out greedy evaluation — at n=4, p 0.500, which is nothing. It bought an
eight-seed run, not a conclusion.

## v2.52: everything in place, and the benefit still is not significant

Eight seeds, `common_random_numbers` on so the pairing is real for the first
time, episode logging on, one thread, full provenance. Floor 0.279.

| condition | holdout | sd | vs floor | train | holdout steps |
| --- | ---: | ---: | ---: | ---: | ---: |
| `ZK` (none) | 0.657 | 0.208 | +0.378 | 0.869 | 25.4 |
| `ZE` lr 1e-5 | 0.605 | 0.298 | +0.326 | 0.787 | 28.3 |
| `ZE` lr 1e-4 | **0.787** | **0.089** | **+0.508** | **0.975** | **17.8** |
| `ZE` lr 1e-3 | 0.011 | 0.021 | -0.268 | 0.188 | 63.3 |

| pair | Δ | 95% CI | p | W/L/T |
| --- | ---: | :---: | ---: | :---: |
| `ZE` lr 1e-4 - `ZK` | +0.130 | [-0.004, +0.275] | **0.1328** | 5/3/0 |
| `ZE` lr 1e-3 - `ZK` | -0.646 | [-0.769, -0.487] | **0.0078** | **0/8/0** |

**The harm replicates and is significant. The benefit does not reach
significance** — five wins of eight. The CI on the benefit nearly excludes zero,
but the meta-review measured this bootstrap at 0.870 coverage against a nominal
0.95, so the CI is not the thing to lean on. The p-value is.

Both halves agree (Δ +0.125 on the Mac, +0.135 on `rtx5060ti`), so the
cross-build merge is not manufacturing or hiding the effect.

### A difference that is not about the mean

| condition | clears its own seed's floor | worst seed | sd |
| --- | ---: | ---: | ---: |
| `ZK` | 7/8 | 0.208 | 0.208 |
| `ZE` lr 1e-4 | **8/8** | **0.683** | **0.089** |

`ZK` has a catastrophic seed; lr 1e-4 has none, and less than half the spread.
This is descriptive, not a test, but it is the kind of difference a mean
comparison is badly suited to see.

### The averaged learning curve

Binned per seed and averaged across all eight:

| condition | verdict | curve | peak | steps |
| --- | --- | --- | ---: | ---: |
| `ZE` lr 1e-4 | improved | `▅▇▇▇████████` | 0.978 | 15.9 |
| `ZK` | improved | `▅▆▇▇▇▇▇▇▇▇▇▇` | 0.864 | 22.3 |
| `ZE` lr 1e-5 | improved | `▄▅▆▆▆▆▆▆▆▆▆▆` | 0.801 | 26.5 |
| `ZE` lr 1e-3 | declined | `▃▃▂▂▂▂▂▂▂▂▂▂` | 0.410 | 52.0 |

lr 1e-4 rises fastest, settles highest, and reaches the goal in about a third
fewer steps.

## Where this leaves the hypothesis

The AD/DA hypothesis now has its best evidence in the project's history and it
is **still not significant**. The paired differences have sd 0.216, so detecting
+0.130 at 80% power needs about **22 seeds**. Seeds 5309-5322 are running to get
there; at 22 the exact p-floor is about 5e-7, so the test will not be the
limiting factor.

What is already established, and did not need more seeds:

- The control learns: +0.378 above a measured random floor.
- Learning the representation at the policy's learning rate destroys learning,
  8/8 seeds, p 0.0078.
- `representation_beta` cannot control any of this.

## Tooling defects found by these runs

Both were caught by the tools' own first real use, and both are now pinned by
the self-checks:

1. The curve classifier judged against the peak **bin**. At ~67 episodes per bin
   the binomial SE near p=0.8 is 0.049, the entire noise band, so any end-of-run
   wobble read as a collapse. It now compares quarter means.
2. `binned_curve` binned the **concatenated** episode sequence, so with eight
   seeds every bin spanned parts of different seeds. It reported `ZK` as declined
   and lr 1e-3 as improved, both visibly wrong. It now bins per seed and averages
   bin-wise.
