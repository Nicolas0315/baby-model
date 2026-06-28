# v0 Experiment Protocol

## Purpose

Build a verified research loop before using heavy libraries or GPUs.

## Conditions

- `A_end_to_end`: raw flattened observation, no decoder delay, no intrinsic
  reward.
- `B_encoder_first`: coarse perceptual representation, 20-episode random
  action phase, no intrinsic reward.
- `C_baby_surprise`: same delayed phase, plus raw latent transition surprise.
- `D_baby_progress`: same delayed phase, plus transition prediction
  improvement.

## Metrics

- `success_rate_all`
- `success_rate_last_window`
- `mean_steps_success`
- `mean_external_return_last_window`
- `mean_intrinsic_return_last_window`
- `mean_unique_features_last_window`

## v0.1 Interpretation

Raw surprise can over-reward novelty in tiny environments. Prediction
improvement is intended to reward transitions whose model estimate actually
changes, so the intrinsic signal should fade as the transition becomes familiar.

Initial local tmux loop results from 2026-06-28:

- `D_baby_progress` improved clearly over `C_baby_surprise`.
- `B_encoder_first` still won all three local-loop seeds by last-window success.
- Current interpretation: in BabyGrid, the coarse encoder and delayed decoder
  dominate; the intrinsic reward needs either tuning or a richer environment to
  show value.

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
