# Prior Art, Applicable Repositories, and What Is Known About Learning Order

Date: 2026-08-23 JST
Scope: literature and code relevant to (a) baby-model's own hypothesis and
(b) the broader question of whether the *order* of learning can be optimized,
including in LLM pretraining.

Every figure below is quoted from the cited source. Where a source could not be
read directly, that is stated.

## 1. The number that reframes the whole project

The BabyAI platform exists specifically to measure sample efficiency, and it
publishes what its baselines need. On `GoToRedBallGrey` — the same easiest tier
as `GoToObj`, which baby-model uses — reinforcement learning required
**15,900-17,400 episodes**, while imitation learning needed 8,400-12,400
demonstrations (under 2,000 when distilling an RL expert).

baby-model's v48/v49/v50 protocol gives each condition **48 episodes** on
`BabyAI-GoToObj-v0`, after 12 + 24 warmup episodes on easier levels: **84
episodes total.**

That is roughly **two orders of magnitude below** the published requirement for
the same difficulty tier. It is a single sufficient explanation for everything
the 2026-08-23 audit measured:

- every condition's greedy policy sits at or below the random-policy floor,
- per-seed dispersion swamps every between-condition difference,
- the leading candidate flipped when the seed set changed.

At 84 episodes the agents are not weakly trained. They are pre-training. All 56
experiment documents compared conditions inside the noise band that precedes
learning.

Sources: [BabyAI (ICLR 2019)](https://arxiv.org/abs/1810.08272),
[alphaXiv overview](https://www.alphaxiv.org/overview/1810.08272v4),
[BabyAI 1.1](https://arxiv.org/abs/2007.12770).

## 2. baby-model's hypothesis has twenty years of prior art

"Use prediction improvement as intrinsic reward, and let a developmental
schedule emerge" is **Intelligent Adaptive Curiosity** (IAC) and the
learning-progress family that followed it. The mechanism pushes an agent toward
situations that maximize its *learning progress*, and the reported result is
that developmental trajectories — curricula — **self-organize** rather than
being hand-specified.

The important difference from baby-model: in that line of work, learning
progress is a *measured, online quantity that drives behaviour*. In baby-model
it is a hand-set constant (`representation_beta`) applied to a fixed,
hand-written stage list. baby-model asserts the schedule; the prior art derives
it.

- [Intrinsic Motivation Systems for Autonomous Mental Development (IAC)](https://www.cs.swarthmore.edu/~meeden/DevelopmentalRobotics/iac07.pdf)
- [Oudeyer, Active Learning and Artificial Curiosity in Robots](https://www.pyoudeyer.com/active-learning-and-artificial-curiosity-in-robots/)
- [CURIOUS: Intrinsically Motivated Modular Multi-Goal RL](https://arxiv.org/pdf/1810.06284)
- [GRIMGEP: Learning Progress for Robust Goal Sampling](https://arxiv.org/pdf/2008.04388)
- Reference implementation: the Explauto Python library (per Oudeyer's page
  above; the repository itself was not opened for this note).

## 3. Applicable repositories

| Repository | What it gives baby-model |
| --- | --- |
| [flowersteam/TeachMyAgent](https://github.com/flowersteam/TeachMyAgent) | A peer-reviewed **benchmark for Automatic Curriculum Learning** (ICML 2021, Romac / Portelas / Hofmann / Oudeyer). This is the correct venue for the question "does an AD-first schedule get discovered when a teacher is allowed to choose?", instead of hand-setting `decoder_delay_episodes`. |
| [facebookresearch/dcd](https://github.com/facebookresearch/dcd) | PAIRED, Robust PLR, ACCEL — regret-based **Unsupervised Environment Design**. The algorithmic form of "development is scheduled": the curriculum is generated, not written. Papers: NeurIPS 2021, ICML 2022, CoLLAs 2023. |
| [facebookresearch/minimax](https://github.com/facebookresearch/minimax) | JAX autocurricula baselines. Relevant mainly because it is fast enough to make the 15,900-episode budget affordable. |
| [lcswillems/rl-starter-files](https://github.com/lcswillems/rl-starter-files) | The reference PPO/A2C baseline for MiniGrid/BabyAI. The fastest way to establish "what a working agent scores on `GoToObj`", which baby-model has never had. |
| [Farama-Foundation/Minigrid](https://github.com/Farama-Foundation/Minigrid) | Already a dependency; the BabyAI levels and their own truncation limits live here (see audit item D1). |

The highest-value one is `rl-starter-files`: it supplies the missing upper
bound. The audit has a measured random floor (0.283) and no ceiling, so no
result can currently be placed on a scale.

## 4. Learning order in LLMs

The intuition — that *what you learn first* shapes how the substrate gets used,
and that this differs between individuals — is a real and actively studied
question. The evidence is more specific, and more negative, than the intuition
suggests.

### 4.1 Community-scale peer-reviewed evidence: mostly negative

The **BabyLM Challenge** is exactly this hypothesis, run as a shared task:
pretrain on a developmentally plausible corpus at a fixed budget of 100M words
or less. Curriculum learning was one of the most popular strategies.

- First challenge (CoNLL 2023): curriculum-learning attempts, which accounted
  for a large number of submissions, were **largely unsuccessful**, with only
  modest improvements in some cases. Winners came from architecture
  (LTG-BERT), shorter input sequences, and distillation.
- Second challenge (CoNLL 2024, 31 submissions from 17 countries): innovations
  in **architecture, training objective, and dataset construction** were what
  worked; GPT-BERT (hybrid causal-masked) won the Strict and Strict-Small
  tracks. The organizers' own recommendation was that participants interested
  in curriculum learning should **"think beyond data order"**. The one
  curriculum submission singled out as going further changed *what was masked*
  over training rather than the order of the data.

Sources: [Findings of the BabyLM Challenge](https://aclanthology.org/2023.conll-babylm.1/),
[Findings of the Second BabyLM Challenge](https://arxiv.org/abs/2412.05149),
[CLIMB: Curriculum Learning for Infant-inspired Model Building](https://arxiv.org/pdf/2311.08886).

### 4.2 At scale, ordering does buy something — as a warmup

The largest systematic study available trained **over 200 models on up to 100B
tokens** across vanilla curriculum, pacing-based sampling, and interleaved
curricula, with six difficulty metrics:

> "curriculum learning consistently accelerates convergence in early and
> mid-training phases, reducing training steps by 18-45% to reach baseline
> performance. When applied as a warmup strategy before standard random
> sampling, curriculum learning yields sustained improvements up to 3.5%."

Most effective difficulty signals: compression ratio, lexical diversity (MTLD),
readability (Flesch Reading Ease).

Source: [Beyond Random Sampling: Efficient Language Model Pretraining via Curriculum Learning](https://arxiv.org/abs/2506.11300).

Note the shape of the result: ordering helps *reach a given level sooner*, and
helps a little as a *warmup*. It is not reported as a different end state.

### 4.3 Direct answer to the "individual differences in developmental order"
question

A learning-dynamics study pretrained models from **14M to 1B parameters for
300B tokens** under three linguistically motivated curricula — Age-of-
Acquisition, word frequency, and Verb Variation — against random ordering, and
fitted hidden Markov models to the training trajectories jointly across
orderings:

> "training follows a shared sequence of latent phases, while curricula mainly
> change time spent in each phase"

and

> "orderings do not create new macro-phases under this five-state model.
> Instead, the orderings differ in phase occupancy."

Three further findings matter here:

1. **Direction matters.** A reverse-order control "loses much of the accuracy
   advantage of the ascending curriculum", so order is not irrelevant.
2. **The effect is a small-model effect.** Stability differences are present at
   14M-160M and "at larger scales, these stability differences are smaller",
   with measured differences at 1B smaller than at 14M-160M.
3. **The most child-like curriculum was the null one.** Age-of-Acquisition
   ordering — literally sequencing by when children acquire words — showed "no
   consistent cross-probe BLiMP advantages or regressions relative to Random."

Source: [Curriculum Learning for LLM Pretraining: An Analysis of Learning Dynamics](https://arxiv.org/html/2601.21698v2).

So the honest reading of the intuition: the **macro-phase order appears to be a
property of the objective and architecture, not of the data order**. What the
curriculum moves is how long the model dwells in each phase. Individual
variation, in this evidence, is variation in *pacing*, not in *route*.

### 4.4 The bottleneck: the optimizer eats the intervention

The most useful bottleneck paper identifies why curriculum results have been so
weak:

> "This work identifies a critical factor constraining these methods: the
> incompatibility between the ascending data quality order and the decaying
> learning rate (LR) schedule. We find that while curriculum-based training
> substantially outperforms random shuffling when using a constant LR, its
> advantage diminishes under standard LR decay schedules."

Fixes: a moderate LR decay, or replacing decay with checkpoint averaging.
Together they give **+1.64%** over random shuffling on a standard benchmark
suite, validated on **1.5B-parameter models over 30B tokens**. The paper's own
framing is a call to **co-design data curricula with optimization methods**.

Source: [How Learning Rate Decay Wastes Your Best Data in Curriculum-Based LLM Pretraining](https://arxiv.org/abs/2511.18903).

**This is the same failure class baby-model just measured.** In v2.47, 79% of
the apparent "representation objective" effect was the shared Adam schedule
rather than the objective. In LLM pretraining, the curriculum's advantage is
erased by the LR decay schedule. In both cases a data/curriculum intervention
was evaluated without controlling the optimizer, and the optimizer is what got
measured. The generalisable lesson is that **an ordering intervention needs an
optimizer-matched control**, which is exactly what `ZN` is.

### 4.5 Official positions from vendors

There is an asymmetry worth stating plainly:

- **Frontier labs do not publish data order or mixture curricula.** OpenAI
  system cards and Anthropic model cards document evaluations and safety
  testing, not pretraining data sequencing. There is no official vendor
  position to cite on whether learning order matters.
- **Open-weight model reports do document staged curricula**, and are the only
  first-party evidence available. OLMo 2 states it introduces a specialized
  mix "via late-stage curriculum training (i.e. specialized data during the
  annealing phase of pretraining)" and reports that it "significantly improves
  model capabilities across many downstream task benchmarks."

Source: [2 OLMo 2 Furious](https://arxiv.org/abs/2501.00656).

Practically, the industry-standard shape is *staged mixtures* — a long
general-data stage followed by a shorter high-quality "mid-training" or
annealing stage — rather than a per-example difficulty ordering. That is closer
to section 4.2's "curriculum as warmup" result than to a developmental
curriculum. Detailed stage FLOP splits appear in several open-model reports;
the specific figures were not verified for this note and are deliberately not
quoted.

## 5. What this implies for baby-model

1. **The episode budget is the binding constraint, not the hypothesis.** Fix
   section 1 before interpreting any condition comparison. Everything else in
   the audit is downstream of it.
2. **Do not hand-write the schedule.** The hypothesis belongs in an ACL/UED
   harness (TeachMyAgent, dcd) where the schedule is a searched object and the
   question becomes "is AD-first discovered?".
3. **Learning progress should be measured online and drive behaviour**, as in
   IAC, rather than being a fixed `representation_beta`.
4. **Every ordering claim needs an optimizer-matched control.** `ZN` is that
   control for the representation objective; the LLM literature reached the
   same conclusion independently.
5. **Get a ceiling.** `rl-starter-files` on `GoToObj` supplies the upper bound
   the project has never had. Without it, "0.254" has no scale.
6. **Temper the expected effect size.** The strongest scaled evidence for
   ordering is 18-45% faster convergence and single-digit-percent end-state
   gains, shrinking with model size, with the most developmentally faithful
   ordering being the null result. A protocol whose per-seed noise is ±0.15
   cannot see effects of that magnitude.
