# v0 Experiment Protocol

## Purpose

Build a verified research loop before using heavy libraries or GPUs.

## Conditions

- `A_end_to_end`: raw flattened observation, no decoder delay, no intrinsic
  reward.
- `B_encoder_first`: coarse perceptual representation, 20-episode random
  action phase, no intrinsic reward.
- `C_baby_curiosity`: same delayed phase, plus latent transition surprise.

## Metrics

- `success_rate_all`
- `success_rate_last_window`
- `mean_steps_success`
- `mean_external_return_last_window`
- `mean_intrinsic_return_last_window`
- `mean_unique_features_last_window`

## Commands

```sh
./scripts/verify.sh
python3 -m baby_model.cli run --config configs/experiments/v0-smoke.json --output-dir runs --seed 7
./scripts/launch_tmux_local.sh
```

## Promotion Gate

Move to MiniGrid/BabyAI only after:

- `scripts/verify.sh` passes on the current Mac.
- At least one tmux local loop writes run summaries.
- Fleet read-only inventory proves which hosts can run Python, tmux, and GPU
  commands.
- A GitHub issue exists for the next experiment with protocol and success
  criteria.

