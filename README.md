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
- Progress: `docs/progress/STATUS.md`
- Runs: `runs/<timestamp>/`
