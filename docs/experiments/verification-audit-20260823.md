# Verification Audit

Date: 2026-08-23 JST
Scope: the PyTorch AD/DA lane at source commit `56f0c3d`, audited against the
v2.46 run artifacts rather than against the protocol documents.

Every "measured" row below was measured in this session. Rows marked "open" are
gaps with no measurement yet.

## A. Baselines

| # | Item | Status |
| --- | --- | --- |
| A1 | Random-policy floor | **measured — and it changes the reading of every result** |
| A2 | Optimal / scripted-policy ceiling | open |
| A3 | Untrained-network greedy floor | open |
| A4 | Representation head present with beta = 0 | **was not expressible in the config; now added, running as v2.47** |

### A1. The random-policy floor

Uniform-random action selection on `BabyAI-GoToObj-v0`, measured on the exact
episode seeds each metric uses:

| episode seeds | n per seed | random success |
| --- | ---: | ---: |
| the reported training window (stage 2, episodes 28-47) | 20 | 0.269 (sd 0.053) |
| the greedy holdout range (`+99_000`) | 60 | 0.283 (sd 0.052) |

The two ranges are equally hard, so there is no episode-difficulty confound
between the training metric and the holdout.

Placed against the v2.46 eight-seed run:

| condition | repr updates | training window (eps 0.2) | greedy holdout | vs random 0.283 |
| --- | ---: | ---: | ---: | --- |
| `ZK` (no representation) | 0 | 0.431 | 0.254 | below |
| `ZI` (AD-stop) | 235 | 0.481 | 0.192 | below |
| `ZH` (beta 0.0375) | 4098 | 0.431 | 0.158 | below |
| `ZG` (beta 0.025) | 4286 | 0.319 | 0.125 | below |
| `ZE` (beta 0.05) | 4228 | 0.256 | 0.079 | below |

**Every condition's trained greedy policy is at or below the random-policy
floor.** The behaviour policy (epsilon = 0.2) clears the floor for `ZK`, `ZI`,
and `ZH`; the greedy policy does not, for any condition. That pattern — the
epsilon-greedy policy beating random while its own argmax loses to random — is
the signature of a Q-function whose argmax is worse than chance, with the 20%
random component doing the work.

This is the single most consequential missing item in the audit. Fifty-six
experiment documents compare conditions to each other; none establishes that
any condition beats random on a held-out greedy evaluation.

### A4. There is no beta = 0 control

`parse_minigrid_torch_config` rejected `representation_beta <= 0` whenever an
objective was set, so the control that isolates the objective's *information*
from its *optimizer side effects* could not be written. Why that control is
needed is in section E1.

Added as an explicit opt-in `representation_null_control` (the default guard
still rejects an accidental `beta: 0`), and
`configs/experiments/minigrid-torch-adda-v50.json` runs it as `ZN`.

## B. Metric validity

| # | Item | Status |
| --- | --- | --- |
| B1 | Decision metric is measured under exploration on training episodes | fixed — greedy holdout added |
| B2 | `epsilon` is constant 0.2, never annealed | **open — the greedy policy is never exercised during training** |
| B3 | Window is 20 episodes, binomial SE about 0.11 | open — needs a wider eval or more eval episodes |
| B4 | No multiple-comparison control | **open** |
| B5 | Winner-carry-forward selection bias | **open** |

B4/B5 compound: each version promotes the condition that won the previous
version, so the surviving candidate is selected on the same noise it is later
re-tested against. Across 56 documents with roughly 4-6 conditions each, several
hundred pairwise comparisons were made with no alpha adjustment and no
pre-registered stopping rule. The v2.46 outcome is the direct consequence — see
section F.

## C. Logging and evidence durability

| # | Item | Status |
| --- | --- | --- |
| C1 | Per-episode metrics are discarded; no artifact contains a learning curve | **open** |
| C2 | `framework` records torch version and device but not GPU model, driver, or host | **open** |
| C3 | The config body is not embedded in `metrics.json` (only `hypothesis`) | **open** |
| C4 | The source commit is not embedded in the artifact | **open** |
| C5 | GPU evidence lives only in remote `.tmp/`, which is disposable | **open — already caused a loss** |

C1 means "did it converge?" is unanswerable from any stored artifact. The only
trend information is `success_rate_all` versus `success_rate_last_window`, two
points per stage.

C2 and C4 mean every worker, GPU, driver, and commit claim in
`docs/experiments/` was transcribed by hand and cannot be checked against the
artifact it describes.

C5 already bit: the `rtx4090` CUDA venv used for v2.43-v2.45 was found this
session as a broken symlink into a sibling clone that had been deleted, so the
environment behind three gates no longer existed. It was rebuildable only
because `scripts/setup_minigrid_env.sh` is parameterised.

## D. Environment and protocol

| # | Item | Status |
| --- | --- | --- |
| D1 | `max_steps` above the env's own truncation is inert | **measured, 27 configs affected** |
| D2 | The AD-only phase runs entirely in an environment with no mission target | **measured** |
| D3 | `ZI` performs 5.6% of `ZE`'s representation learning | **measured** |
| D4 | Feature-hash collision rate | **measured, 6.6% of tokens per observation** |
| D5 | The GPU has no behavioural effect, and neither does CPU vs CUDA | **measured** |
| D6 | One RNG stream is shared by exploration and replay sampling | open |

### D1. The horizon knob is partly inert

MiniGrid environments truncate on their own `max_steps`; the config value only
applies when it is *smaller*. Measured truncation with a constant action:

| env | env truncation | config value | effective |
| --- | ---: | ---: | --- |
| `BabyAI-GoToObj-v0` | 64 | 80 | **64 — config ignored** |
| `BabyAI-GoToRedBall-v0` | 64 | 80 | **64 — config ignored** |
| `MiniGrid-Empty-5x5-v0` | 100 | 30 | 30 |

27 configs request `max_steps: 80` on a 64-step env. This did not invalidate the
"long horizon" experiments — `_long` doubled `episodes` (12/24/48 vs 6/12/24),
not `max_steps` — but it means the step-horizon axis has never actually been
varied on the evaluation task, and any future attempt to raise it above 64 will
silently do nothing.

### D2 / D3. The AD-first phase does not contain the thing it is supposed to learn

`decoder_delay_episodes` counts *global* episodes across the curriculum. In v48
it is 8, and stage 1 (`empty_warmup`, `MiniGrid-Empty-5x5-v0`) is 12 episodes.
So the entire AD-only phase — 8 of 84 episodes, 9.5% — happens inside an empty
5x5 room with no objects and no mission.

The representation objective is `state_plus_mission_target`. During the only
phase in which `ZI` trains it, there is no mission target in the observation.
Measured representation updates over the eight-seed run:

| condition | representation updates | where |
| --- | ---: | --- |
| `ZE` | 4228 | all three stages |
| `ZG` | 4286 | all three stages |
| `ZH` | 4098 | all three stages |
| `ZI` | 235 | `empty_warmup` only |
| `ZK` | 0 | — |

`ZI`, the condition that has been the leading candidate since v2.42, is
therefore the no-representation control plus 8 forced-random warmup episodes and
a prediction head trained only on target-free observations. Its holdout ordering
(second best, just under `ZK`) is consistent with that reading.

### D5. The GPU is not an independent axis

`rtx4090` and `rtx5060ti` ran the identical config and seed set. Every
behavioural quantity — per-seed, per-condition success, return, and holdout —
is **bit-identical** across the two GPUs. The only differences are auxiliary
representation-loss values at the 1e-10 level, which never flip an `argmax`.

The mechanism is in the code, not in luck: weight initialisation runs on CPU
under `torch.manual_seed` before `.to(device)`, there is no dropout, and no CUDA
RNG is ever drawn. Action selection and replay sampling both come from one
`random.Random(seed)`.

The same eight seeds were then run with `--device cpu` on `rtx4090`, in the
same venv and the same clone. **CPU and CUDA are also bit-identical** on every
behavioural quantity, on the whole `aggregate` block, and on the whole
`statistics` block. The only differing leaves across the entire artifact are
timestamps, the `device` string, and `mean_representation_loss_last_window`.

Runtime for the identical eight-seed sweep:

| device | total | per seed |
| --- | ---: | ---: |
| CUDA `rtx4090` | 8.2 min | 70.0 s |
| CUDA `rtx5060ti` | 10.7 min | 91.3 s |
| CPU `rtx4090` | 9.4 min | 80.8 s |

An RTX 4090 buys **1.15x** over the same host's CPU, because the loop does
single-sample forward passes with a fresh host-to-device transfer of a
1024-vector on every step (E6).

Consequence: the "CUDA replication" gates in v2.34, v2.38, and v2.43, and the
"another CUDA worker" plan in issue #70, carry no independent evidence at all —
not merely no *worker* information, but no information beyond the CPU run. The
whole GPU lane, from v0.8 onward, has consumed fleet GPU time for a 15% wall
clock gain and zero additional evidence. Either retire the lane, or change the
loop to batched or vectorised environments where a GPU can actually contribute.

### D4. Feature hashing

`linear_features` hashes tokens into `feature_dim` buckets. Measured on 200
`BabyAI-GoToObj-v0` observations at `feature_dim: 1024`, `encoder_mode: raw`:

- 154.0 tokens emitted per observation
- 143.9 distinct buckets used
- 10.1 tokens lost to collision, **6.6% per observation**

Colliding cell facts are summed into one input unit, so the state
representation is aliased at that rate. `feature_dim: 4096` would reduce it to
roughly 1.7%. No experiment has varied `feature_dim`.

## E. Implementation level

| # | Item | Status |
| --- | --- | --- |
| E1 | The shared Adam advances the encoder's step count twice as fast in representation conditions | **measured — confound** |
| E2 | Hypothesis: representation loss leaks a momentum update into the Q head | **tested and refuted** |
| E3 | No determinism flags (`use_deterministic_algorithms`, `CUBLAS_WORKSPACE_CONFIG`) | open |
| E4 | TF32 / reduced-precision matmul settings never pinned or recorded | open |
| E5 | `MSELoss` broadcasts silently on a shape mismatch | open |
| E6 | Per-step host-to-device transfer of a dense 1024-vector | open (performance) |
| E7 | `list(self.replay)` copied on every update | open (performance) |

### E1. The confound that motivates v2.47

`update_representation` and `update` both call `step()` on the **same** Adam
instance, and the encoder is shared by both losses. Measured with 50 Q updates
and 50 interleaved representation updates:

| condition | Q updates | Adam steps on encoder | Adam steps on Q head |
| --- | ---: | ---: | ---: |
| no representation | 49 | 49 | 49 |
| representation on | 49 | **99** | 49 |

Adam's per-parameter state is not scale-free in the step count: `beta` scales
the gradient, not the number of steps, so the encoder's bias correction and
second-moment estimate follow a different schedule in every representation
condition. **`beta → 0` therefore does not recover the control.** There is no
continuous path from `ZE` to `ZK`, which means the beta-neighbourhood sweeps
(v2.24, v2.29, v2.37, and `ZG`/`ZH`/`ZE` in v2.45) were varying a parameter that
cannot reach the baseline, while also carrying a doubled optimiser schedule on
the shared encoder.

`ZN` in v2.47 isolates this: the same objective, the same call sites, the same
number of optimiser steps, `beta = 0` so the gradient carries no information.

- If `ZN` matches `ZE`, the objective contributes nothing and the entire
  representation family has been measuring an optimiser schedule.
- If `ZN` matches `ZK`, the schedule is harmless and the objective itself is
  what hurts.
- Anything in between splits the effect.

### E2. Refuted hypothesis, recorded

The prior expectation was that a representation `step()` would apply a ghost
momentum update to the Q head, since Adam updates a parameter with zero
gradient. Measured directly: after priming Adam with 5 real Q updates and then
running 20 representation-only updates, the Q head moved by exactly `0.0` and
its Adam step counter stayed at 5. `zero_grad()` sets `grad = None` and Adam
skips parameters whose gradient is `None`. The Q values do change, but only
through the shared encoder, which is the intended channel.

### E3 / E4. Kernel-level items worth pinning

The bitwise GPU agreement in D5 is currently an accident of the workload, not a
guarantee. Nothing in the repository pins it:

- `torch.use_deterministic_algorithms(True)` is never set, and
  `CUBLAS_WORKSPACE_CONFIG` is never exported. cuBLAS may select a
  split-K reduction whose order varies with workspace availability, so the same
  binary on the same GPU is not contractually reproducible.
- `torch.backends.cuda.matmul.allow_tf32` and
  `torch.backends.cudnn.allow_tf32` are neither set nor recorded. Their
  defaults have changed across torch releases, and the artifact stores only
  `torch.__version__`. A future torch upgrade can change results with no
  recorded cause.
- Only `nn.Linear` and elementwise ops are used, so the high-risk
  nondeterministic kernels (scatter-add, `index_put_` with duplicate indices,
  atomics in embedding backward) are not in the graph. That is why the current
  agreement holds, and it is also why it will stop holding the moment the model
  gains a convolution or an embedding.

The cheap fix is to set the determinism flags once at suite start and record
their resolved values, plus the GPU name and driver, into `framework`. That
turns D5 from an observation into an invariant and closes C2 at the same time.

## F. What the audit implies for the research record

The v2.46 run held the worker, the config, the torch build, and the protocol
fixed and changed only the seed set. The result:

| metric | v2.45 (seeds 4301-4305) | v2.46 (seeds 5301-5308) |
| --- | --- | --- |
| `ZI` Δ success vs control | +0.230, CI [+0.140, +0.340], p 0.062, 5/0/0 | +0.050, CI [-0.044, +0.125], p 0.375, 5/2/1 |
| control `ZK` own level | 0.300 | 0.431 |

Re-running the original seeds `4301-4305` on the rebuilt environment reproduced
v2.45 exactly (0.360 / 0.290 / 0.320 / 0.530 / 0.300), so the environment
rebuild, the driver change from 610.62 to 610.74, and the code changes in this
branch account for none of the difference. **The v2.45 effect was the seed set,
and specifically the control drawing a weak one — the control moved by +0.131
while `ZI` moved by -0.049.**

Combined with A1, the honest current state of the research claim is:

- No condition has been shown to beat a random policy on held-out greedy
  evaluation.
- The leading candidate `ZI` is the condition that does the least
  representation learning, and what little it does happens where the
  representation target does not exist.
- The ordering among the remaining representation conditions is not separable
  from the shared-optimiser schedule until v2.47 reports.

## Priority

1. **A1 as a permanent gate.** Add the random floor as a condition in every
   config so no sweep can report a winner without it. Without this the lane can
   keep producing rankings among below-chance policies.
2. **E1 via v2.47** (running), then give the representation head its own
   optimiser if `ZN` shows the schedule matters.
3. **B2 and B3.** Anneal epsilon so the greedy policy is exercised during
   training, and widen the evaluation beyond 20 episodes.
4. **C1, C2, C4.** Persist per-episode rows, the resolved GPU/driver/host, and
   the source commit. All three are small and all three would have caught
   earlier problems.
5. **D2.** Either extend the AD-only phase into a stage that contains a mission
   target, or stop describing the current protocol as perception-first.
