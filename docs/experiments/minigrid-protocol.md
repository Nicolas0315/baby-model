# MiniGrid/BabyAI Migration Protocol

## Purpose

Move from the dependency-free BabyGrid smoke harness to a maintained sparse
reward benchmark while keeping the default verifier lightweight.

## Source Check

Retrieved: 2026-06-29 JST

- Farama MiniGrid docs via Context7 library `/farama-foundation/minigrid`.
- Install command from docs: `python3 -m pip install minigrid`.
- Environment creation uses `gymnasium.make(...)` after importing `minigrid`.
- Observations are dictionaries with `image`, `direction`, and `mission` fields.
- Step API follows Gymnasium:
  `obs, reward, terminated, truncated, info = env.step(action)`.

## Dependency Boundary

Default `./scripts/verify.sh` stays standard-library-only. MiniGrid/BabyAI checks
live in a separate optional verifier:

```sh
python3 -m venv .venv-minigrid
. .venv-minigrid/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install minigrid
./scripts/verify_minigrid.sh
```

## Probe Environments

The optional probe starts with:

- `MiniGrid-Empty-8x8-v0`
- `MiniGrid-DoorKey-8x8-v0`
- `BabyAI-GoToRedBall-v0`

The probe records action-space size, observation schema, random-policy return,
success rate, and episode length. It is an API and dependency check, not a
scientific result.

## Metrics Schema

Future trained runs should preserve this top-level shape:

- `created_at`
- `hypothesis`
- `env_id`
- `condition`
- `seed`
- `episodes`
- `success_rate_all`
- `success_rate_last_window`
- `mean_return_last_window`
- `mean_steps_success`
- `mean_intrinsic_return_last_window`
- `observation_schema`

## Promotion Gate

Before GPU or long fleet jobs:

- `./scripts/verify.sh` still passes without optional dependencies.
- `./scripts/verify_minigrid.sh` passes in a local optional venv.
- A MiniGrid config compares `A_end_to_end`, `B_encoder_first`, and at least one
  intrinsic variant on the same sparse-reward environment.
- A local tmux run writes summaries under `runs/`.
- Fleet execution uses `git archive | ssh | tmux` and keeps host-level evidence
  outside the repo.
