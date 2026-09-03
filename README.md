# baby-model

Research framework for the Baby AD/DA asymmetry hypothesis.

The working hypothesis is:

> A model that grows perception first, delays action decoding, and uses
> prediction improvement as intrinsic reward should form more stable concepts
> and transfer better than an end-to-end agent trained from the first step.

This repository starts with a small, dependency-free RL smoke environment so
the research loop can run on any fleet node. The framework is intentionally
small: it gives us a verified loop, run artifacts, tmux entrypoints, and a path
to replace the toy environment with MiniGrid, BabyAI, Habitat, or robot data.

## Current v0

- `A_end_to_end`: raw-observation Q-learning baseline.
- `B_encoder_first`: coarse perceptual representation with a decoder delay.
- `C_baby_surprise`: coarse representation, decoder delay, and raw intrinsic
  surprise reward.
- `D_baby_progress`: coarse representation, decoder delay, and prediction
  improvement reward.

The v0 environment is not a scientific claim. It is a harness to make the
research pipeline testable before we spend GPU time.

## Quick Start

```sh
./scripts/verify.sh
python3 -m baby_model.cli run --config configs/experiments/v0-smoke.json --output-dir runs
./scripts/launch_tmux_local.sh
./scripts/launch_tmux_sweep.sh
```

Optional MiniGrid/BabyAI probe:

```sh
./scripts/verify_minigrid.sh
```

This requires the optional `minigrid` dependency; setup details are in
`docs/experiments/minigrid-protocol.md`.

Optional PyTorch DQN smoke:

```sh
MINIGRID_TORCH_CONFIG=configs/experiments/minigrid-torch-unlock-smoke.json \
./scripts/verify_minigrid.sh
```

This additionally requires optional `torch`; setup details are in
`docs/experiments/minigrid-torch-lane.md`.

## Reading a Sweep

Every PyTorch sweep summary carries a `## Seed-Level Statistics` section with
per-condition dispersion and a paired sign-flip test against the control
condition. Re-analyze an existing artifact without a GPU:

```sh
python3 -m baby_model.stats path/to/metrics.json
```

Every sweep also writes `episodes.jsonl`, one line per episode tagged with seed,
condition, and stage. Learning curves, convergence verdicts, and the first bin
that clears the random floor:

```sh
python3 -m baby_model.curves path/to/run_dir --bins 12 --svg curves.svg
```

The verdict distinguishes `improved`, `flat`, `declined`, and `collapsed`,
because the final window alone cannot tell a condition that gained and lost it
from one that never moved.

The exact p-value floor of that test is `2 / 2**n`, so a five-seed gate cannot
reach p < 0.05. Use at least six seeds, preferably eight, for any gate meant to
claim significance.

Set `holdout_episodes` in a torch config to add a greedy (epsilon = 0),
no-learning evaluation on held-out episode seeds after training. Training-window
metrics mix policy quality with exploration noise; the holdout does not.

## Fleet Loop

Start read-only:

```sh
./scripts/fleet_inventory.sh
```

After the repository is pushed and checked out on worker nodes, use the
commands in `docs/fleet/worker-plan.md` to run local loops under tmux on each
host.

## Tracking

- Research hypothesis: `docs/research/hypothesis.md`
- Source notes: `docs/research/sources.md`
- Experiment protocol: `docs/experiments/v0-protocol.md`
- v0.2 sweep result: `docs/experiments/v02-sweep.md`
- v0.3 sweep result: `docs/experiments/v03-sweep.md`
- MiniGrid/BabyAI migration: `docs/experiments/minigrid-protocol.md`
- BabyAI Unlock hard task: `docs/experiments/minigrid-babyai-unlock.md`
- MiniGrid curriculum to BabyAI Unlock:
  `docs/experiments/minigrid-curriculum-unlock.md`
- MiniGrid linear function approximation:
  `docs/experiments/minigrid-linear-unlock.md`
- MiniGrid linear multi-seed sweep:
  `docs/experiments/minigrid-linear-sweep.md`
- MiniGrid neural encoder: `docs/experiments/minigrid-neural-unlock.md`
- MiniGrid PyTorch DQN lane: `docs/experiments/minigrid-torch-lane.md`
- v2.46 cross-axis result: `docs/experiments/minigrid-torch-adda-v57.md`
- v2.47 null control: `docs/experiments/minigrid-torch-adda-v58.md`
- v2.48 adequate budget: `docs/experiments/minigrid-torch-adda-v59.md`
- v2.49 separate representation optimizer:
  `docs/experiments/minigrid-torch-adda-v60.md`
- v2.50-v2.52 beta is a no-op, the dial is the learning rate:
  `docs/experiments/minigrid-torch-adda-v61.md`
- v2.53 the hypothesis passes at 22 seeds:
  `docs/experiments/minigrid-torch-adda-v62.md`
- Seed statistics and greedy holdout:
  `docs/experiments/seed-statistics-and-holdout.md`
- Verification audit: `docs/experiments/verification-audit-20260823.md`
- Prior art, applicable repos, and learning-order evidence:
  `docs/research/prior-art-and-learning-order.md`
- Voyager (Minecraft x LLM) system analysis:
  `docs/research/voyager-system-analysis.md`
- Pipeline sequence and artifact model:
  `docs/architecture/experiment-pipeline.md`
- Verification of the verification, dependency and adversarial review:
  `docs/architecture/verification-of-verification.md`
- Progress: `docs/progress/STATUS.md`
- Runs: `runs/<timestamp>/`

- [Documentation](docs/)

- [Issue intake](.github/ISSUE_TEMPLATE/)

- [Automation](.github/workflows/)

- [Repository hygiene](.gitignore)
