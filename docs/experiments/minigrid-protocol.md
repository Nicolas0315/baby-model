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

Run the default probe and smoke plus one extra trained config:

```sh
MINIGRID_EXTRA_CONFIG=configs/experiments/minigrid-babyai-unlock.json \
MINIGRID_EXTRA_SEED=201 \
./scripts/verify_minigrid.sh
```

Run the default probe and smoke plus a curriculum config:

```sh
MINIGRID_CURRICULUM_CONFIG=configs/experiments/minigrid-curriculum-unlock.json \
MINIGRID_CURRICULUM_SEED=301 \
./scripts/verify_minigrid.sh
```

Run the default probe and smoke plus a linear function-approximation config:

```sh
MINIGRID_LINEAR_CONFIG=configs/experiments/minigrid-linear-unlock.json \
MINIGRID_LINEAR_SEED=401 \
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

## Local Verification

Local optional venv created on 2026-06-29:

- `minigrid==3.1.0`
- `gymnasium==1.3.0`
- `numpy==2.5.0`
- `pygame-ce==2.5.7`

Command:

```sh
. .venv-minigrid/bin/activate
./scripts/verify_minigrid.sh
```

Probe result:

| env | action_n | obs_keys | success_rate | mean_return | mean_steps |
| --- | ---: | --- | ---: | ---: | ---: |
| `MiniGrid-Empty-8x8-v0` | 7 | `direction,image,mission` | 0.000 | 0.000 | 80.00 |
| `MiniGrid-DoorKey-8x8-v0` | 7 | `direction,image,mission` | 0.000 | 0.000 | 80.00 |
| `BabyAI-GoToRedBall-v0` | 7 | `direction,image,mission` | 0.000 | 0.000 | 64.00 |

Trained smoke result on `MiniGrid-Empty-5x5-v0`, 40 episodes:

| condition | success_all | success_last | return_last | mean_steps_success | intrinsic_last |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A_end_to_end` | 0.900 | 1.000 | 0.937 | 13.00 | 0.000 |
| `B_encoder_first` | 0.325 | 0.450 | 0.220 | 48.85 | 0.000 |
| `E_progress_anneal` | 0.125 | 0.150 | 0.089 | 43.40 | 0.000 |

Harder trained result on `BabyAI-Unlock-v0`, 60 episodes:

| condition | success_all | success_last | return_last | mean_steps_success | intrinsic_last |
| --- | ---: | ---: | ---: | ---: | ---: |
| `A_end_to_end` | 0.033 | 0.050 | 0.040 | 107.50 | 0.000 |
| `B_encoder_first` | 0.050 | 0.000 | 0.000 | 26.00 | 0.000 |
| `E_progress_anneal` | 0.000 | 0.000 | 0.000 |  | 0.177 |

The hard-task local result is recorded in
`docs/experiments/minigrid-babyai-unlock.md`.

Curriculum result on `BabyAI-Unlock-v0` final stage, 60 final-stage episodes:

| condition | final_success_all | final_success_last | final_return_last | mean_steps_success |
| --- | ---: | ---: | ---: | ---: |
| `A_hard_only` | 0.017 | 0.050 | 0.039 | 143.00 |
| `B_curriculum_encoder_first` | 0.017 | 0.000 | 0.000 | 45.00 |
| `E_curriculum_progress` | 0.000 | 0.000 | 0.000 |  |

The curriculum result is recorded in
`docs/experiments/minigrid-curriculum-unlock.md`.

Linear function-approximation result on `BabyAI-Unlock-v0`, 60 episodes:

| condition | success_all | success_last | return_last | mean_steps_success |
| --- | ---: | ---: | ---: | ---: |
| `A_linear_hard_only` | 0.017 | 0.000 | 0.000 | 45.00 |
| `B_linear_encoder_first` | 0.050 | 0.050 | 0.044 | 47.33 |
| `E_linear_progress` | 0.000 | 0.000 | 0.000 |  |

The linear pilot is recorded in `docs/experiments/minigrid-linear-unlock.md`.

## Fleet Verification

The optional verifier and trained smoke were replicated on all four configured
worker classes on 2026-06-29 via `git archive | ssh | tmux` at commit
`d03c46ece84335023bb566ee691b281c24263dab`. Every worker produced the same
trained smoke table as local verification. Exact host-level evidence is kept
outside this repository in local docs.

The harder `BabyAI-Unlock-v0` extra config was replicated on all four
configured worker classes on 2026-06-29 at commit
`b89b50faa7fa50d805d5247372a0c5c5697a3e56`. Every worker produced the same
extra table as local verification.

The curriculum config was replicated on all four configured worker classes on
2026-06-29 at commit `976591913b649e50b2455e0dbf44b39b8a4e1c9e`. Every worker
produced the same final-stage table as local verification.

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
- Harder sparse-reward results are recorded separately from the easy smoke run,
  so the optional verifier can stay lightweight by default.
