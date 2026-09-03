# v2.54 Pre-Registration

Written **2026-08-24, before any v2.54 result has been read.** The Mac half is
running and the `rtx5060ti` half is running (PID 199783, verified by two
timestamped probes 61s apart showing +6,126 CPU ticks). No metrics file from
either half has been opened.

Fixed here in response to the audit finding that v2.53 was analysed four times
with no pre-registration.

## Primary

- **Primary comparison**: `ZE lr=1e-4` − `ZK` (no representation).
- **Primary endpoint**: greedy holdout success rate, 120 held-out episodes per
  seed, on `BabyAI-GoToLocal-v0`.
- **Fixed sample size**: **22 seeds** (5301-5322), 11 on the Mac and 11 on
  `rtx5060ti`.
- **Test**: two-sided paired sign-flip randomization on per-seed differences,
  exact at n=22. Significance threshold **alpha = 0.05**, uncorrected, because
  there is exactly one primary comparison and exactly one look.
- **No interim analysis.** The Mac half alone must not be read, aggregated, or
  used to decide anything. If one half fails, the run is re-executed, not
  partially reported.

## Secondary, reported but not confirmatory

- `ZE lr=1e-5` − `ZK` and `ZE lr=1e-3` − `ZK`.
- Steps-to-goal on the holdout.
- Training-window success.
- IQM.
- Performance profile.

These carry no alpha. Any of them moving does not license a claim.

## Per-host effect, mandatory

Seeds and hosts correspond exactly (5301-5311 Mac, 5312-5322 `rtx5060ti`), so a
host effect is fully confounded with a seed-band effect. The paired effect will
be reported **per host as well as pooled**, and if the two hosts disagree in
direction the pooled result is reported as inconclusive regardless of its
p-value.

## Performance profile: not confirmatory as currently implemented

The audit is right that the v2.53 write-up overclaimed. A profile compared on a
handful of hand-picked thresholds cannot establish dominance. Until the profile
is computed at **every unique observed value** and carries a bootstrap CI, it is
descriptive only, and the wording is "empirically above at the thresholds
examined", never "stochastically dominates".

## Gates that must pass before any v2.54 result counts as evidence about the
## hypothesis

These are external-validity gates, not statistics. A significant primary result
with any of these failing means the effect is not established as being about
representation learning.

1. **Mission counterfactual.** The same trained policy, evaluated on the same
   holdout seeds under three conditions: the real mission, a mission shuffled in
   from a different episode, and a blank mission. Success must hold up under the
   real mission and drop under the other two. If it does not drop, the policy is
   not using the mission and `GoToLocal` is being solved some other way.
2. **Feature-hash collisions.** At `feature_dim: 1024` the additive hash puts
   mission tokens and image features in the same index 44.0% of the time (440 of
   1000 initial `GoToLocal` observations; 39.2% for colour and object words
   specifically). 1024 must be compared against 4096 on identical seeds. Until
   then, a difference between conditions cannot be separated from differing
   tolerance to hash collisions.
3. **Provenance.** Runs must record the full commit SHA, the tarball SHA-256,
   and start/end receipts. `BABY_MODEL_SOURCE_COMMIT=v254` is a label, not a
   proof, and the v2.54 runs already in flight carry that weaker label — which is
   itself a limitation of v2.54 that will be stated in its write-up.

## Known limitation of this pre-registration

It is being written while the runs are already in flight, not before they were
launched. That is weaker than a true pre-registration and is disclosed as such.
What it does guarantee is that the analysis plan was fixed before any result was
read.
