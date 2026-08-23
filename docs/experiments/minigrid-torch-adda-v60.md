# PyTorch AD/DA v2.49 Separate Representation Optimizer: the Fix Validates, the Setting Fails

Date: 2026-08-23 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/70

## Motivation

v2.47 attributed 79% of the representation effect to the shared Adam: a
representation step moved the shared encoder using state accumulated from
Q-learning gradients, so the objective could not be separated from that
displacement. `agent.separate_representation_optimizer` puts the representation
loss on its own Adam, with the encoder deliberately still in both — the
objective is supposed to shape perception, and removing it would delete the
mechanism under test.

This run repeats the v2.48 condition set unchanged apart from that flag.

## Source State

- Config: `configs/experiments/minigrid-torch-adda-v52.json` (= v51 with the
  flag on): `ZK`, `ZN`, `ZE`, 3200 eval episodes, `holdout_episodes: 120`
- Seeds `5301-5308`, split `rtx4090` / `rtx5060ti`, **both on torch
  `2.11.0+cu128`**, `device=cpu`. The v2.48 build-split caveat does not apply.
- Artifacts: `~/ext4/baby-model-v249/.tmp/v249-separate-opt/` and
  `~/work/baby-model-cuda-56f0c3d/.tmp/v249-separate-opt/`

The first `rtx4090` attempt died 18 minutes in with a pygame segfault inside
minigrid's `process_vis` (exit 134) while 91 GB of memory was free. That host's
`$HOME/work` is a symlink to `/mnt/c/Users/ogosh/work`, so the clone was running
over WSL's drvfs bridge; `rtx5060ti`, on real ext4, was unaffected. Rebuilding
under `$HOME/ext4/` cleared it. Recorded in `docs/current-state.md`.

## Result

Greedy holdout, 120 held-out episodes per seed, random floor **0.279**.

| condition | repr updates | holdout | sd | 95% CI | vs floor |
| --- | ---: | ---: | ---: | :---: | ---: |
| `ZK` (no representation) | 0 | 0.615 | 0.217 | [0.466, 0.743] | +0.335 |
| `ZN` (null control, beta 0) | 72,200 | 0.631 | 0.221 | [0.478, 0.760] | +0.352 |
| `ZE` (beta 0.05) | 170,089 | **0.032** | 0.073 | [0.000, 0.084] | **-0.247** |

| pair | Δ | 95% CI | p | W/L/T |
| --- | ---: | :---: | ---: | :---: |
| `ZN` - `ZK` | **+0.017** | [-0.205, +0.243] | **0.875** | 3/5/0 |
| `ZE` - `ZK` | **-0.582** | [-0.698, -0.446] | **0.0078** | **0/8/0** |
| `ZE` - `ZN` | -0.599 | [-0.721, -0.458] | 0.0078 | 0/8/0 |

Per-seed holdout for `ZE`: `[0.0, 0.0, 0.0, 0.0, 0.208, 0.0, 0.0, 0.05]`.

Training-window success repeats the pattern: `ZK` 0.856, `ZN` 0.863,
`ZE` 0.256; `ZN` - `ZK` p 1.000, `ZE` - `ZK` -0.600 p 0.0078.

## The fix is validated by its own control

`ZN` runs the same objective, the same call sites, and the same number of
optimizer steps with a zero-information gradient. Under the shared optimizer in
v2.48 it cost `-0.298` against the control. Under the separate optimizer it
costs `+0.017` (p 0.875).

**That entire v2.48 penalty was the shared-Adam artifact**, and it is now gone.
This was the pre-registered reading — "if the fix works, `ZN` should match `ZK`"
— and it is the strongest available evidence that the mechanism change does what
it was designed to do rather than merely changing the numbers.

`0.0078` is the exact p-floor at n = 8, and `ZE` lost all eight seeds.

## A refuted prediction, and what it changes

Before the run the expectation was that a separate optimizer would *inflate* the
representation objective's effective step, because Adam's second moment would no
longer be dominated by the larger Q gradients. Measured directly, after 200
interleaved Q and representation updates, the encoder displacement from one
further representation step is:

| optimizer | displacement |
| --- | ---: |
| shared | 2.19e-05 |
| separate | 2.68e-06 |

**0.12x — the separate optimizer makes the representation step about eight times
smaller, not larger.** Under the shared optimizer that step was carried mostly by
stale Q momentum, which is large; under the separate one it carries only the
objective's own gradient, scaled by `beta`.

So the reading of the result is not "the objective is harmful at any strength".
It is:

- Under the shared optimizer, `beta = 0.05` produced a certain effective dose,
  most of it artifact.
- Under the separate optimizer, the same `beta` produces a per-step dose eight
  times smaller — but applied 170,089 times, and the outcome is far worse.
- `beta = 0.05` was hand-picked under the broken mechanism. It has no claim to
  being the right value under the fixed one.

The result therefore condemns **this setting**, not the objective class. That is
a narrower claim than the numbers first suggest, and it is the one the evidence
supports.

One further confound is worth naming: the representation-update count is
**endogenous**. `ZE` performs 170,089 updates against `ZN`'s 72,200, because
`ZE` fails, its episodes run to truncation, and longer episodes mean more
updates. Failure increases the dose, which increases failure.

## A scoping correction on determinism

v2.46 established that `rtx4090`, `rtx5060ti`, and CPU produce bit-identical
results for the same seeds. That was measured at 84 episodes. At 3200 episodes
it no longer holds: comparing `ZK` — a condition that never enters the
representation code at all — between v2.48 and v2.49 on the same seeds, five of
eight values match exactly and three differ (seed 5303: 0.892 vs 0.392).

`torch.get_num_threads()` differs across the hosts (12 on `rtx4090`, 8 on
`rtx5060ti`, 6 on the Mac), which changes CPU reduction order. That is a
well-founded explanation, not a measured one.

**The conclusions are unaffected**, because every comparison here is paired
within a host: each host ran all conditions on its own seeds. This is precisely
what the paired design protects against, and it is why cross-host merging is
safe for the differences while cross-run comparison of absolute levels is not.

Scope the v2.46 claim to the short horizon.

## Next

v2.50 (`configs/experiments/minigrid-torch-adda-v53.json`, running): sweep
`beta` at 0.005 / 0.05 / 0.5 against `ZK` under the separate optimizer, eight
seeds. Until that reports, the honest statement is that the AD/DA hypothesis has
had exactly one fair test, at one hand-me-down hyperparameter, and failed it.
