# PyTorch AD/DA v2.48 Adequate Budget: the Control Learns, the Representation Condition Does Not

Date: 2026-08-23 JST
Issue: https://github.com/Nicolas0315/baby-model/issues/70

## Motivation

The v2.46 audit found every condition at or below the random-policy floor, and
the budget ladder (audit section H) found the cause: at the historic 48 eval
episodes the control's greedy policy is `+0.006` above the floor. BabyAI's own
baselines need 15,900-17,400 RL episodes on the same difficulty tier.

Every prior condition comparison in this repository therefore happened before
learning began. This run repeats the decisive comparison at a budget where the
control demonstrably learns, with enough seeds to be able to reach p < 0.05, and
with the `ZN` null control included so any surviving penalty can be split
between the optimizer schedule and the objective's content.

## Source State

- Config: `configs/experiments/minigrid-torch-adda-v51.json` — `ZK`, `ZN`, `ZE`
  at **3200** eval episodes (vs 48 historically), `holdout_episodes: 120`
- Seeds: `5301-5308` (eight; the exact paired p-floor at n = 8 is 0.008)
- Split across two hosts: `CA-20032518` with torch `2.12.1` for seeds 5301-5304
  and `RTX4090` with torch `2.11.0+cu128` for seeds 5305-5308, both `device=cpu`
- Artifacts: local `v248-local/20260822T235419Z/` and
  `~/work/baby-model-cuda-56f0c3d/.tmp/v248-adequate-budget/20260823T001609Z/`

Splitting across hosts is sound for the device (v2.46 established CPU and CUDA
are bit-identical) but the two halves ran on **different torch builds**, which
was not established as equivalent. The per-half check below is why the merge is
reported anyway.

## Result

Greedy holdout, 120 held-out episodes per seed. Random-policy floor measured on
the same episode seeds: **0.279**.

| condition | repr updates | holdout | sd | 95% CI | vs floor |
| --- | ---: | ---: | ---: | :---: | ---: |
| `ZK` (no representation) | 0 | **0.668** | 0.211 | [0.519, 0.787] | **+0.389** |
| `ZN` (null control, beta 0) | 116,517 | 0.370 | 0.350 | [0.150, 0.605] | +0.091 |
| `ZE` (beta 0.05) | 147,637 | 0.198 | 0.293 | [0.036, 0.427] | **-0.081** |

Paired sign-flip tests:

| pair | Δ | 95% CI | p | W/L/T |
| --- | ---: | :---: | ---: | :---: |
| `ZE` - `ZK` | **-0.470** | **[-0.680, -0.246]** | **0.0156** | **1/7/0** |
| `ZN` - `ZK` | -0.298 | [-0.562, +0.015] | 0.109 | 2/6/0 |
| `ZE` - `ZN` | -0.172 | [-0.433, +0.121] | 0.273 | 2/6/0 |

Training-window success shows the same ordering: `ZK` 0.844, `ZN` 0.531,
`ZE` 0.369, with `ZE` - `ZK` at -0.475, CI [-0.744, -0.169], p 0.047.

### Per-half consistency check

Because the halves used different torch builds, each condition's mean is
reported per half:

| condition | local (torch 2.12.1, seeds 5301-04) | remote (torch 2.11.0, seeds 5305-08) |
| --- | ---: | ---: |
| `ZK` | 0.696 | 0.640 |
| `ZN` | 0.321 | 0.419 |
| `ZE` | 0.194 | 0.202 |
| random floor (no torch involved) | 0.263 | 0.296 |

The halves agree to within 0.10 on every condition, and the random floor — which
involves no torch at all — differs by 0.033 from seeds alone. The build
difference does not produce an offset comparable to the effects being measured.

## Decision

**Two claims are established.**

1. **The harness works and the budget was the problem.** `ZK` reaches 0.668
   against a floor of 0.279, `+0.389`, with a CI that clears the floor. The same
   agent, objective, and curriculum at 48 eval episodes scored `+0.006`. Nothing
   had to change except the episode count.

2. **The representation objective prevents learning.** `ZE` - `ZK` is `-0.470`
   with CI `[-0.680, -0.246]` and p `0.0156`, losing seven of eight seeds. This
   is the **first result in this repository to reach p < 0.05**, and its sign is
   against the project's hypothesis. `ZE` sits *below* the random floor
   (`-0.081`) at a budget where the control clearly learns, so the objective is
   not merely unhelpful — it is worse than not learning a policy at all.

**One claim is not established.** The split between optimizer schedule and
objective content is a point estimate only: schedule 63% (`ZK` to `ZN`) and
objective 37% (`ZN` to `ZE`), against 79/21 at the 84-episode budget in v2.47.
But `ZN` - `ZK` (p 0.109) and `ZE` - `ZN` (p 0.273) are individually
non-significant at n = 8, so only the total is measured, not its decomposition.
`ZN`'s sd of 0.350 is the obstacle: its per-seed values span 0.0 to 0.867.

## Next

The mechanism fix is now the only thing standing between the project and a real
test of its hypothesis: give the representation head its own optimizer, so the
representation loss cannot advance the shared encoder's Adam state, and re-run
this exact condition set. Three outcomes and their readings:

- `ZE` recovers to `ZK`: the penalty was entirely the shared optimizer, and the
  hypothesis has never been tested.
- `ZE` stays below `ZK`: the objective itself is harmful in this form, and the
  next question is whether `state_plus_mission_target` is the wrong target or
  whether auxiliary prediction at this scale is simply a tax.
- `ZE` exceeds `ZK`: the hypothesis gets its first positive evidence, at which
  point it needs replication on a fresh seed set before anything else.

Two prerequisites carried forward from the audit: `epsilon` is still constant at
0.2, so the greedy policy is never exercised during training; and there is still
no ceiling from a standard baseline, so 0.668 has a floor to compare against but
no upper reference.
