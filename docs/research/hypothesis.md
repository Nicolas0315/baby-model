# Baby AD/DA Asymmetry Hypothesis

## Claim

Infant-like learning is not just label-free learning. The interesting
asymmetry is that perception has a long high-plasticity phase while action,
speech, and other output channels are low bandwidth.

In this repo:

- AD means sensor-to-latent conversion: perception, attention, and encoding.
- DA means latent-to-world conversion: movement, speech, action, and policy.

The core hypothesis is:

> Delaying DA while AD learns stable latent regularities improves later sample
> efficiency and transfer compared with training perception and action end to
> end from the first step.

## Design Translation

- Broad early input: start with raw multimodal observations.
- Weak priors: bias attention toward novelty, synchrony, motion, faces, voices,
  and caregiver-like response signals when available.
- Delayed decoder: keep policy/output low-capacity or random during an initial
  observation phase.
- Predictive world model: learn latent next-state predictions.
- Intrinsic reward: use prediction improvement, not raw error alone, to avoid
  chasing noise.
- Compression: after enough exposure, collapse unused distinctions and keep
  environment-relevant latent axes.

## First Test

The v0 BabyGrid test compares:

- `A_end_to_end`: raw-observation Q-learning.
- `B_encoder_first`: coarse perceptual encoder plus delayed policy learning.
- `C_baby_surprise`: delayed policy learning plus raw latent transition
  surprise.
- `D_baby_progress`: delayed policy learning plus transition prediction
  improvement.

Supportive evidence would be higher last-window success, fewer successful
steps, or better transfer after a small number of episodes.

Negative evidence is useful: if intrinsic variants do not improve over
baselines in toy environments, the design needs either better intrinsic reward,
better representation learning, or a task with richer hidden structure.
