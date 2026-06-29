from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from random import Random
from statistics import mean
from typing import Any

from baby_model.agents import EpisodeMetrics, TransitionSurprise
from baby_model.experiment import Condition
from baby_model.minigrid_experiment import _env_call, _intrinsic_signal
from baby_model.minigrid_probe import observation_schema


SparseFeatures = dict[int, float]


@dataclass(frozen=True)
class LinearAgentConfig:
    feature_dim: int
    alpha: float
    gamma: float
    epsilon: float


@dataclass(frozen=True)
class MiniGridLinearConfig:
    env_id: str
    max_steps: int
    quiet_env_output: bool
    agent: LinearAgentConfig
    conditions: tuple[Condition, ...]


class LinearQAgent:
    def __init__(
        self,
        actions: int,
        feature_dim: int,
        alpha: float = 0.05,
        gamma: float = 0.92,
        epsilon: float = 0.2,
        seed: int = 0,
    ) -> None:
        self.actions = actions
        self.feature_dim = feature_dim
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = Random(seed)
        self.weights: list[dict[int, float]] = [defaultdict(float) for _ in range(actions)]

    def q_value(self, features: SparseFeatures, action: int) -> float:
        weights = self.weights[action]
        return sum(weights[index] * value for index, value in features.items())

    def action_values(self, features: SparseFeatures) -> dict[int, float]:
        return {action: self.q_value(features, action) for action in range(self.actions)}

    def choose(
        self,
        features: SparseFeatures,
        force_random: bool = False,
        action_bonus: dict[int, float] | None = None,
        bonus_weight: float = 1.0,
    ) -> int:
        if force_random or self.rng.random() < self.epsilon:
            return self.rng.randrange(self.actions)
        values = [
            (
                self.q_value(features, action)
                + bonus_weight * (0.0 if action_bonus is None else action_bonus.get(action, 0.0)),
                action,
            )
            for action in range(self.actions)
        ]
        best_value = max(value for value, _ in values)
        best_actions = [action for value, action in values if value == best_value]
        return self.rng.choice(best_actions)

    def update(self, features: SparseFeatures, action: int, reward: float, next_features: SparseFeatures, done: bool) -> None:
        current = self.q_value(features, action)
        next_best = 0.0 if done else max(self.q_value(next_features, next_action) for next_action in range(self.actions))
        delta = reward + self.gamma * next_best - current
        norm = max(1.0, sum(abs(value) for value in features.values()))
        weights = self.weights[action]
        for index, value in features.items():
            weights[index] += self.alpha * delta * (value / norm)

    def nonzero_weights(self) -> int:
        return sum(1 for weights in self.weights for value in weights.values() if value != 0.0)


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-minigrid-linear")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/minigrid-linear-unlock.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minigrid-linear"))
    parser.add_argument("--seed", type=int, default=401)
    args = parser.parse_args()

    try:
        report = run_minigrid_linear_suite(json.loads(args.config.read_text(encoding="utf-8")), seed=args.seed)
    except ImportError as exc:
        print(f"missing optional dependency: {exc}")
        print("install with: python3 -m pip install minigrid")
        return 2

    run_dir = write_minigrid_linear_run(report, args.output_dir)
    print(f"minigrid_linear_run_dir={run_dir}")
    print(f"winner_last_window={report['winner_last_window']}")
    return 0


def run_minigrid_linear_suite(config: dict[str, Any], seed: int = 401) -> dict[str, Any]:
    parsed = parse_minigrid_linear_config(config=config, seed=seed)
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
    except ImportError as exc:
        raise ImportError("gymnasium/minigrid") from exc

    results = [
        run_minigrid_linear_condition(
            gym=gym,
            env_id=parsed.env_id,
            condition=condition,
            max_steps=parsed.max_steps,
            agent_config=parsed.agent,
            quiet_env_output=parsed.quiet_env_output,
        )
        for condition in parsed.conditions
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": str(config.get("hypothesis", "Baby-AD/DA MiniGrid linear function approximation")),
        "env_id": parsed.env_id,
        "max_steps": parsed.max_steps,
        "agent": {
            "type": "linear_q",
            "feature_dim": parsed.agent.feature_dim,
            "alpha": parsed.agent.alpha,
            "gamma": parsed.agent.gamma,
            "epsilon": parsed.agent.epsilon,
        },
        "results": results,
        "winner_last_window": max(results, key=lambda row: row["success_rate_last_window"])["name"],
    }


def parse_minigrid_linear_config(config: dict[str, Any], seed: int = 401) -> MiniGridLinearConfig:
    env_cfg = config.get("environment", {})
    if not isinstance(env_cfg, dict):
        raise ValueError("environment must be an object")
    env_id = str(env_cfg.get("id", "BabyAI-Unlock-v0"))
    max_steps = int(env_cfg.get("max_steps", 160))
    if max_steps < 1:
        raise ValueError("environment.max_steps must be positive")
    quiet_env_output = bool(env_cfg.get("quiet_env_output", True))

    agent_cfg = config.get("agent", {})
    if not isinstance(agent_cfg, dict):
        raise ValueError("agent must be an object")
    feature_dim = int(agent_cfg.get("feature_dim", 4096))
    alpha = float(agent_cfg.get("alpha", 0.06))
    gamma = float(agent_cfg.get("gamma", 0.92))
    epsilon = float(agent_cfg.get("epsilon", 0.2))
    if feature_dim < 128:
        raise ValueError("agent.feature_dim must be at least 128")
    if alpha <= 0.0:
        raise ValueError("agent.alpha must be positive")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("agent.gamma out of range")
    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError("agent.epsilon out of range")

    condition_cfgs = config.get("conditions", [])
    if not isinstance(condition_cfgs, list) or not condition_cfgs:
        raise ValueError("conditions must be a non-empty list")
    names: set[str] = set()
    conditions: list[Condition] = []
    for i, item in enumerate(condition_cfgs):
        if not isinstance(item, dict):
            raise ValueError("each condition must be an object")
        name = str(item.get("name", ""))
        if not name:
            raise ValueError("condition.name is required")
        if name in names:
            raise ValueError(f"duplicate condition.name: {name}")
        names.add(name)
        encoder_mode = str(item.get("encoder_mode", "raw"))
        if encoder_mode not in {"raw", "coarse"}:
            raise ValueError(f"invalid encoder_mode for {name}")
        intrinsic_mode = str(item.get("intrinsic_mode", "none"))
        if intrinsic_mode not in {"none", "surprise", "progress"}:
            raise ValueError(f"invalid intrinsic_mode for {name}")
        intrinsic_target = str(item.get("intrinsic_target", "reward"))
        if intrinsic_target not in {"reward", "auxiliary"}:
            raise ValueError(f"invalid intrinsic_target for {name}")
        episodes = int(item.get("episodes", 0))
        delay = int(item.get("decoder_delay_episodes", 0))
        beta = float(item.get("intrinsic_beta", 0.0))
        if episodes < 1:
            raise ValueError(f"episodes must be positive for {name}")
        if delay < 0 or delay > episodes:
            raise ValueError(f"decoder_delay_episodes out of range for {name}")
        if beta < 0:
            raise ValueError(f"intrinsic_beta must be non-negative for {name}")
        conditions.append(
            Condition(
                name=name,
                encoder_mode=encoder_mode,
                episodes=episodes,
                decoder_delay_episodes=delay,
                intrinsic_beta=beta,
                intrinsic_mode=intrinsic_mode,
                seed=seed + i,
                intrinsic_target=intrinsic_target,
            )
        )
    return MiniGridLinearConfig(
        env_id=env_id,
        max_steps=max_steps,
        quiet_env_output=quiet_env_output,
        agent=LinearAgentConfig(feature_dim=feature_dim, alpha=alpha, gamma=gamma, epsilon=epsilon),
        conditions=tuple(conditions),
    )


def run_minigrid_linear_condition(
    gym: Any,
    env_id: str,
    condition: Condition,
    max_steps: int,
    agent_config: LinearAgentConfig,
    quiet_env_output: bool = True,
) -> dict[str, Any]:
    env = gym.make(env_id)
    try:
        actions = int(env.action_space.n)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(condition.seed)
        agent = LinearQAgent(
            actions=actions,
            feature_dim=agent_config.feature_dim,
            alpha=agent_config.alpha,
            gamma=agent_config.gamma,
            epsilon=agent_config.epsilon,
            seed=condition.seed,
        )
        auxiliary_agent = LinearQAgent(
            actions=actions,
            feature_dim=agent_config.feature_dim,
            alpha=agent_config.alpha,
            gamma=agent_config.gamma,
            epsilon=0.0,
            seed=condition.seed + 100_003,
        )
        transition = TransitionSurprise()
        episodes: list[EpisodeMetrics] = []
        first_schema: dict[str, Any] | None = None

        for episode in range(condition.episodes):
            observation, _info = _env_call(
                env.reset,
                quiet=quiet_env_output,
                seed=condition.seed * 1000 + episode,
            )
            if first_schema is None:
                first_schema = observation_schema(observation)
            features = linear_features(observation, condition.encoder_mode, agent_config.feature_dim)
            feature_key = feature_signature(features)
            visited = {feature_key}
            external_return = 0.0
            intrinsic_return = 0.0
            success = False
            steps = 0

            for _ in range(max_steps):
                force_random = episode < condition.decoder_delay_episodes
                action_bonus = None
                if condition.intrinsic_target == "auxiliary" and not force_random:
                    action_bonus = auxiliary_agent.action_values(features)
                action = agent.choose(features, force_random=force_random, action_bonus=action_bonus)
                next_observation, reward, terminated, truncated, _info = _env_call(
                    env.step,
                    action,
                    quiet=quiet_env_output,
                )
                next_features = linear_features(next_observation, condition.encoder_mode, agent_config.feature_dim)
                next_feature_key = feature_signature(next_features)
                intrinsic_signal = _intrinsic_signal(condition.intrinsic_mode, transition, feature_key, action, next_feature_key)
                intrinsic = condition.intrinsic_beta * intrinsic_signal
                total_reward = float(reward) if condition.intrinsic_target == "auxiliary" else float(reward) + intrinsic
                done = bool(terminated or truncated)

                if not force_random:
                    agent.update(features, action, total_reward, next_features, done)
                    if condition.intrinsic_target == "auxiliary":
                        auxiliary_agent.update(features, action, intrinsic, next_features, done)
                transition.update(feature_key, action, next_feature_key)

                external_return += float(reward)
                intrinsic_return += intrinsic
                steps += 1
                visited.add(next_feature_key)
                features = next_features
                feature_key = next_feature_key
                if float(reward) > 0.0:
                    success = True
                if done:
                    break

            episodes.append(
                EpisodeMetrics(
                    success=success,
                    steps=steps,
                    external_return=external_return,
                    intrinsic_return=intrinsic_return,
                    unique_features=len(visited),
                )
            )

        last_window = episodes[-20:] if len(episodes) >= 20 else episodes
        successful_steps = [item.steps for item in episodes if item.success]
        return {
            "name": condition.name,
            "env_id": env_id,
            "agent_type": "linear_q",
            "feature_dim": agent_config.feature_dim,
            "encoder_mode": condition.encoder_mode,
            "episodes": condition.episodes,
            "decoder_delay_episodes": condition.decoder_delay_episodes,
            "intrinsic_beta": condition.intrinsic_beta,
            "intrinsic_mode": condition.intrinsic_mode,
            "intrinsic_target": condition.intrinsic_target,
            "seed": condition.seed,
            "observation_schema": first_schema or {},
            "success_rate_all": mean(1.0 if item.success else 0.0 for item in episodes),
            "success_rate_last_window": mean(1.0 if item.success else 0.0 for item in last_window),
            "mean_steps_success": mean(successful_steps) if successful_steps else None,
            "mean_return_last_window": mean(item.external_return for item in last_window),
            "mean_intrinsic_return_last_window": mean(item.intrinsic_return for item in last_window),
            "mean_unique_features_last_window": mean(item.unique_features for item in last_window),
            "nonzero_weights": agent.nonzero_weights(),
        }
    finally:
        env.close()


def linear_features(observation: Any, mode: str, feature_dim: int) -> SparseFeatures:
    if mode not in {"raw", "coarse"}:
        raise ValueError(f"unknown feature mode: {mode}")
    values: defaultdict[int, float] = defaultdict(float)

    def add(token: str, value: float = 1.0) -> None:
        values[_feature_index(token, feature_dim)] += value

    add("bias")
    add(f"direction:{int(observation['direction'])}")
    mission = observation.get("mission", "")
    if isinstance(mission, str):
        for token in _mission_tokens(mission):
            add(f"mission:{token}")

    image = observation["image"].tolist()
    if mode == "raw":
        for y, row in enumerate(image):
            for x, cell in enumerate(row):
                add(f"cell:{y}:{x}:object:{int(cell[0])}")
                add(f"cell:{y}:{x}:color:{int(cell[1])}")
                add(f"cell:{y}:{x}:state:{int(cell[2])}")
    else:
        counts: defaultdict[int, int] = defaultdict(int)
        for row in image:
            for cell in row:
                counts[int(cell[0])] += 1
        center = image[3][3]
        forward = image[2][3]
        add(f"center:object:{int(center[0])}")
        add(f"center:color:{int(center[1])}")
        add(f"center:state:{int(center[2])}")
        add(f"forward:object:{int(forward[0])}")
        add(f"forward:color:{int(forward[1])}")
        add(f"forward:state:{int(forward[2])}")
        for object_id, count in counts.items():
            add(f"count:object:{object_id}", min(count, 10) / 10.0)

    return dict(values)


def feature_signature(features: SparseFeatures) -> tuple[int, ...]:
    return tuple(sorted(index for index, value in features.items() if value != 0.0))


def _mission_tokens(mission: str) -> list[str]:
    tokens: list[str] = []
    current: list[str] = []
    for char in mission.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.append("".join(current))
            current = []
    if current:
        tokens.append("".join(current))
    return tokens[:12]


def _feature_index(token: str, feature_dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % feature_dim


def write_minigrid_linear_run(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(linear_summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def linear_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# baby-model MiniGrid linear function approximation summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- env_id: `{report['env_id']}`",
        f"- winner_last_window: `{report['winner_last_window']}`",
        "",
        "| condition | success_all | success_last | return_last | mean_steps_success | nonzero_weights |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        mean_steps = row["mean_steps_success"]
        lines.append(
            "| {name} | {all:.3f} | {last:.3f} | {ret:.3f} | {steps} | {weights} |".format(
                name=row["name"],
                all=row["success_rate_all"],
                last=row["success_rate_last_window"],
                ret=row["mean_return_last_window"],
                steps="" if mean_steps is None else f"{mean_steps:.2f}",
                weights=row["nonzero_weights"],
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
