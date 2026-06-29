from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from random import Random

from baby_model.envs import ACTIONS


Feature = tuple[int, ...]


@dataclass
class EpisodeMetrics:
    success: bool
    steps: int
    external_return: float
    intrinsic_return: float
    unique_features: int


class QAgent:
    def __init__(
        self,
        actions: int = len(ACTIONS),
        alpha: float = 0.4,
        gamma: float = 0.92,
        epsilon: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.actions = actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = Random(seed)
        self.q: dict[tuple[Feature, int], float] = defaultdict(float)

    def choose(
        self,
        feature: Feature,
        force_random: bool = False,
        action_bonus: dict[int, float] | None = None,
        bonus_weight: float = 1.0,
    ) -> int:
        if force_random or self.rng.random() < self.epsilon:
            return self.rng.randrange(self.actions)
        values = [
            (
                self.q[(feature, action)]
                + bonus_weight * (0.0 if action_bonus is None else action_bonus.get(action, 0.0)),
                action,
            )
            for action in range(self.actions)
        ]
        best_value = max(value for value, _ in values)
        best_actions = [action for value, action in values if value == best_value]
        return self.rng.choice(best_actions)

    def action_values(self, feature: Feature) -> dict[int, float]:
        return {action: self.q[(feature, action)] for action in range(self.actions)}

    def update(
        self,
        feature: Feature,
        action: int,
        reward: float,
        next_feature: Feature,
        done: bool,
    ) -> None:
        next_best = 0.0 if done else max(self.q[(next_feature, a)] for a in range(self.actions))
        key = (feature, action)
        target = reward + self.gamma * next_best
        self.q[key] += self.alpha * (target - self.q[key])


class TransitionSurprise:
    """Counts latent transitions and returns high values for novel outcomes."""

    def __init__(self) -> None:
        self.counts: dict[tuple[Feature, int, Feature], int] = defaultdict(int)
        self.context_counts: dict[tuple[Feature, int], int] = defaultdict(int)

    def surprise(self, feature: Feature, action: int, next_feature: Feature) -> float:
        return 1.0 - self.probability(feature, action, next_feature)

    def probability(self, feature: Feature, action: int, next_feature: Feature) -> float:
        context = (feature, action)
        total = self.context_counts[context]
        seen = self.counts[(feature, action, next_feature)]
        if total == 0:
            return 0.0
        return seen / total

    def learning_progress(self, feature: Feature, action: int, next_feature: Feature) -> float:
        before = self.probability(feature, action, next_feature)
        context = (feature, action)
        total = self.context_counts[context]
        seen = self.counts[(feature, action, next_feature)]
        after = (seen + 1) / (total + 1)
        return max(0.0, after - before)

    def update(self, feature: Feature, action: int, next_feature: Feature) -> None:
        self.context_counts[(feature, action)] += 1
        self.counts[(feature, action, next_feature)] += 1
