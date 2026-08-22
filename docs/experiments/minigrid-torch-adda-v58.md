# PyTorch AD/DA v2.47 Null Control: Objective or Optimizer Schedule?

Date: 2026-08-23 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/70

## Motivation

`update_representation` and `update` call `step()` on the **same** Adam
instance, and the encoder is shared by both losses. Measured with 50 Q updates
and 50 interleaved representation updates:

| condition | Q updates | Adam steps on encoder | Adam steps on Q head |
| --- | ---: | ---: | ---: |
| no representation | 49 | 49 | 49 |
| representation on | 49 | **99** | 49 |

`beta` scales the gradient, not the step count, so **`beta -> 0` does not
recover the control**. Every representation condition ever run carried a
doubled optimizer schedule on the shared encoder, and there was no continuous
path from `ZE` to `ZK`. Worse, that control could not even be written:
`parse_minigrid_torch_config` rejected `representation_beta <= 0` whenever an
objective was set.

This run adds it. `ZN` uses the same objective, the same call sites, and the
same number of optimizer steps, with `beta = 0` so the gradient carries no
information. The pre-registered readings were: `ZN` matching `ZE` means the
objective contributes nothing; `ZN` matching `ZK` means the schedule is
harmless and the objective is what hurts.

## Source State

- Source commit: `56f0c3d` plus the seed-statistics, holdout, and
  `representation_null_control` branch
- Config: `configs/experiments/minigrid-torch-adda-v50.json`
- Worker: `rtx5060ti` / WSL Ubuntu, `NVIDIA GeForce RTX 5060 Ti`, driver
  `576.88`, torch `2.11.0+cu128`, device `cuda`
- Seeds: `5301,5302,5303,5304,5305,5306,5307,5308`
- Artifact: `.tmp/v247-null-control-8seed/20260822T223714Z/summary.md`

Per v2.46 the device is not an axis, so `cuda` here is a runtime detail, not a
condition.

## Result

`holdout_success_rate`, eight seeds, baseline `ZK`:

| condition | repr updates | holdout | Δ vs `ZK` | Δ 95% CI | p |
| --- | ---: | ---: | ---: | :---: | ---: |
| `ZK` (no representation) | 0 | 0.254 | — | — | — |
| `ZI` (beta 0.05, AD-stop) | 235 | 0.215 | -0.040 | [-0.235, +0.156] | 0.734 |
| **`ZN` (null control, beta 0)** | **4203** | **0.119** | **-0.135** | [-0.310, +0.046] | 0.203 |
| `ZE` (beta 0.05) | 4128 | 0.083 | -0.171 | [-0.263, -0.088] | **0.016** |

Direct paired comparisons:

| pair | Δ | 95% CI | p | W/L/T |
| --- | ---: | :---: | ---: | :---: |
| `ZN` vs `ZE` | +0.035 | [-0.067, +0.148] | 0.594 | 4/3/1 |
| `ZN` vs `ZI` | -0.096 | [-0.183, -0.004] | 0.109 | 1/7/0 |
| `ZN` vs `ZK` | -0.135 | [-0.310, +0.046] | 0.203 | 2/5/1 |

Decomposition of the `ZK` to `ZE` gap of 0.171:

| component | size | share |
| --- | ---: | ---: |
| optimizer schedule alone (`ZK` to `ZN`) | 0.135 | **79%** |
| objective content (`ZN` to `ZE`) | 0.035 | 21% |

`ZN` is statistically indistinguishable from `ZE` (p 0.594) and separates from
`ZI` in the same direction and magnitude as `ZE` does. Holdout success is
monotone decreasing in representation *update count*, not in `beta`:
0 → 0.254, 235 → 0.215, ~4130 → 0.083, ~4200 → 0.119.

## Mechanism, measured directly

With `beta = 0` the representation loss is `mse * 0.0`, so `backward()`
populates the encoder gradient with an **exact zero tensor** — not `None`. Adam
skips parameters whose gradient is `None`, but not parameters whose gradient is
zero: it decays the moment estimates and still applies
`-lr * m_hat / (sqrt(v_hat) + eps)`, which is nonzero while leftover
Q-learning momentum remains.

Measured after priming Adam with 6 real Q updates, then 20 representation-only
updates at `beta = 0`:

| quantity | value |
| --- | --- |
| encoder gradient from the representation loss | `0.000e+00` (zero tensor, not `None`) |
| encoder weight drift | `7.282e-02` |
| Q-head weight drift | `0.000e+00` (head is not in the graph, gradient stays `None`) |
| Q-values | `[0.2851, 0.1814, -0.2915]` → `[0.4008, 0.1232, -0.3288]` |

The same measurement at `beta = 0.05` gives an encoder drift of `7.284e-02`.
**The drift is dominated by stale momentum, not by the objective's gradient** —
which is exactly the 79/21 split the sweep shows.

A related hypothesis was tested and refuted: the representation loss does *not*
leak into the Q head. `zero_grad()` sets `grad = None` and the head is absent
from the representation graph, so Adam skips it. The Q values move only through
the shared encoder, which is the intended channel.

## Decision

The representation objective's *content* accounts for roughly a fifth of the
measured "representation hurts" effect. The remaining four fifths is an
optimizer artifact: every representation step displaces the shared encoder with
leftover Q-learning momentum, whatever the objective predicts.

This retroactively reframes the beta-neighbourhood work. v2.24, v2.29, v2.37,
and `ZG`/`ZH`/`ZE` in v2.45 varied `beta`, which does not change the step count,
so all of those conditions carried the same roughly constant schedule penalty.
The 0.02-0.06 differences those documents ranked were noise on top of a
constant artifact, which is consistent with the v2.46 finding that
`ZE`/`ZG`/`ZH` are not separable from the control.

The conclusion is not "representation-first learning does not work". It is that
this implementation has never tested the hypothesis: the mechanism that was
supposed to carry the effect has been dominated by an optimizer side effect
throughout.

## Next

Give the representation head its own optimizer, or at minimum exclude the Q
network's parameters from the representation step, and re-run v2.47's condition
set. Until that lands, no result in the representation family separates the
hypothesis from the artifact.

Two other items gate any positive claim, from
`docs/experiments/verification-audit-20260823.md`:

- Every condition here is at or below the random-policy floor of 0.283 on
  greedy holdout. The floor is now measured automatically by every sweep.
- `epsilon` is constant at 0.2 and never annealed, so the greedy policy is
  never exercised during training. That is the likeliest direct cause of the
  below-floor greedy policies.
