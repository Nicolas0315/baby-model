# MiniGrid Curriculum To BabyAI Unlock

Date: 2026-06-29 JST

## Purpose

This run tests the next branch after issue #7. The hard `BabyAI-Unlock-v0`
result showed that tabular Baby-AD/DA did not hold on a sparse BabyAI task.
This experiment asks whether easy-environment warmup helps when the final hard
stage keeps the same 60-episode budget.

## Config

- Config: `configs/experiments/minigrid-curriculum-unlock.json`
- Seed: `301`
- Output: `.tmp/verify-minigrid/curriculum/20260629T012814Z/`
- Final evaluation stage: `BabyAI-Unlock-v0`, 60 episodes, 160 max steps

Stages:

| stage | env | episodes | max_steps |
| --- | --- | ---: | ---: |
| `empty_warmup` | `MiniGrid-Empty-5x5-v0` | 40 | 80 |
| `goto_warmup` | `BabyAI-GoToRedBall-v0` | 40 | 80 |
| `unlock_eval` | `BabyAI-Unlock-v0` | 60 | 160 |

Command:

```sh
. .venv-minigrid/bin/activate
MINIGRID_CURRICULUM_CONFIG=configs/experiments/minigrid-curriculum-unlock.json \
MINIGRID_CURRICULUM_SEED=301 \
./scripts/verify_minigrid.sh
```

## Result

| condition | active_stages | final_success_all | final_success_last | final_return_last | mean_steps_success |
| --- | --- | ---: | ---: | ---: | ---: |
| `A_hard_only` | `unlock_eval` | 0.017 | 0.050 | 0.039 | 143.00 |
| `B_curriculum_encoder_first` | `empty_warmup,goto_warmup,unlock_eval` | 0.017 | 0.000 | 0.000 | 45.00 |
| `E_curriculum_progress` | `empty_warmup,goto_warmup,unlock_eval` | 0.000 | 0.000 | 0.000 |  |

Winner by final-stage last-window success: `A_hard_only`.

## Interpretation

This is another negative result for the current tabular implementation.
Curriculum did not preserve final-stage success on `BabyAI-Unlock-v0`, even
though `B_curriculum_encoder_first` found one early hard-stage success.

The useful conclusion is that the next branch should add function
approximation or a stronger representation model. More hand-tuning of the same
tabular Q-learning plus progress reward is unlikely to resolve the harder
sparse task.

## Fleet Replication

The curriculum config was replicated on all four configured worker classes at
commit `976591913b649e50b2455e0dbf44b39b8a4e1c9e`. Every worker produced the
same final-stage table, with `A_hard_only` as the final last-window winner.
Exact host-level evidence is kept outside this repository in local docs.
