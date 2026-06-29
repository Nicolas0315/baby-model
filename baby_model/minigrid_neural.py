from __future__ import annotations

import argparse
import json
import math
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
from baby_model.minigrid_linear import SparseFeatures, feature_signature, linear_features
from baby_model.minigrid_probe import observation_schema


@dataclass(frozen=True)
class NeuralAgentConfig:
    feature_dim: int
    hidden_dim: int
    alpha_output: float
    alpha_hidden: float
    gamma: float
    epsilon: float
    init_scale: float


@dataclass(frozen=True)
class MiniGridNeuralConfig:
    env_id: str
    max_steps: int
    quiet_env_output: bool
    agent: NeuralAgentConfig
    conditions: tuple[Condition, ...]


class NeuralQAgent:
    def __init__(
        self,
        actions: int,
        feature_dim: int,
        hidden_dim: int,
        alpha_output: float = 0.05,
        alpha_hidden: float = 0.005,
        gamma: float = 0.92,
        epsilon: float = 0.2,
        init_scale: float = 0.04,
        seed: int = 0,
    ) -> None:
        self.actions = actions
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.alpha_output = alpha_output
        self.alpha_hidden = alpha_hidden
        self.gamma = gamma
        self.epsilon = epsilon
        self.rng = Random(seed)
        self.input_weights: list[dict[int, float]] = [defaultdict(float) for _ in range(hidden_dim)]
        self.hidden_bias = [0.0 for _ in range(hidden_dim)]
        self.output_weights = [[self.rng.uniform(-init_scale, init_scale) for _ in range(hidden_dim)] for _ in range(actions)]
        self.output_bias = [0.0 for _ in range(actions)]

    def hidden(self, features: SparseFeatures) -> list[float]:
        activations: list[float] = []
        for hidden_index in range(self.hidden_dim):
            total = self.hidden_bias[hidden_index]
            weights = self.input_weights[hidden_index]
            for feature_index, value in features.items():
                total += weights[feature_index] * value
            activations.append(math.tanh(total))
        return activations

    def q_values_from_hidden(self, hidden: list[float]) -> list[float]:
        return [
            self.output_bias[action] + sum(weight * value for weight, value in zip(self.output_weights[action], hidden))
            for action in range(self.actions)
        ]

    def action_values(self, features: SparseFeatures) -> dict[int, float]:
        values = self.q_values_from_hidden(self.hidden(features))
        return {action: values[action] for action in range(self.actions)}

    def choose(
        self,
        features: SparseFeatures,
        force_random: bool = False,
        action_bonus: dict[int, float] | None = None,
        bonus_weight: float = 1.0,
    ) -> int:
        if force_random or self.rng.random() < self.epsilon:
            return self.rng.randrange(self.actions)
        hidden = self.hidden(features)
        base_values = self.q_values_from_hidden(hidden)
        values = [
            base_values[action] + bonus_weight * (0.0 if action_bonus is None else action_bonus.get(action, 0.0))
            for action in range(self.actions)
        ]
        best_value = max(values)
        best_actions = [action for action, value in enumerate(values) if value == best_value]
        return self.rng.choice(best_actions)

    def update(self, features: SparseFeatures, action: int, reward: float, next_features: SparseFeatures, done: bool) -> None:
        hidden = self.hidden(features)
        values = self.q_values_from_hidden(hidden)
        next_best = 0.0 if done else max(self.q_values_from_hidden(self.hidden(next_features)))
        delta = reward + self.gamma * next_best - values[action]
        old_output = list(self.output_weights[action])

        self.output_bias[action] += self.alpha_output * delta
        for hidden_index, hidden_value in enumerate(hidden):
            self.output_weights[action][hidden_index] += self.alpha_output * delta * hidden_value

        norm = max(1.0, sum(abs(value) for value in features.values()))
        for hidden_index, hidden_value in enumerate(hidden):
            hidden_grad = delta * old_output[hidden_index] * (1.0 - hidden_value * hidden_value)
            self.hidden_bias[hidden_index] += self.alpha_hidden * hidden_grad
            weights = self.input_weights[hidden_index]
            for feature_index, value in features.items():
                weights[feature_index] += self.alpha_hidden * hidden_grad * (value / norm)

    def nonzero_parameters(self) -> int:
        input_count = sum(1 for weights in self.input_weights for value in weights.values() if value != 0.0)
        output_count = sum(1 for row in self.output_weights for value in row if value != 0.0)
        bias_count = sum(1 for value in self.hidden_bias + self.output_bias if value != 0.0)
        return input_count + output_count + bias_count


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-minigrid-neural")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/minigrid-neural-unlock.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minigrid-neural"))
    parser.add_argument("--seed", type=int, default=501)
    args = parser.parse_args()

    try:
        report = run_minigrid_neural_suite(json.loads(args.config.read_text(encoding="utf-8")), seed=args.seed)
    except ImportError as exc:
        print(f"missing optional dependency: {exc}")
        print("install with: python3 -m pip install minigrid")
        return 2

    run_dir = write_minigrid_neural_run(report, args.output_dir)
    print(f"minigrid_neural_run_dir={run_dir}")
    print(f"winner_last_window={report['winner_last_window']}")
    return 0


def run_minigrid_neural_suite(config: dict[str, Any], seed: int = 501) -> dict[str, Any]:
    parsed = parse_minigrid_neural_config(config=config, seed=seed)
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
    except ImportError as exc:
        raise ImportError("gymnasium/minigrid") from exc

    results = [
        run_minigrid_neural_condition(
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
        "hypothesis": str(config.get("hypothesis", "Baby-AD/DA MiniGrid neural encoder")),
        "env_id": parsed.env_id,
        "max_steps": parsed.max_steps,
        "agent": {
            "type": "one_hidden_layer_q",
            "feature_dim": parsed.agent.feature_dim,
            "hidden_dim": parsed.agent.hidden_dim,
            "alpha_output": parsed.agent.alpha_output,
            "alpha_hidden": parsed.agent.alpha_hidden,
            "gamma": parsed.agent.gamma,
            "epsilon": parsed.agent.epsilon,
            "init_scale": parsed.agent.init_scale,
        },
        "results": results,
        "winner_last_window": max(results, key=lambda row: row["success_rate_last_window"])["name"],
    }


def parse_minigrid_neural_config(config: dict[str, Any], seed: int = 501) -> MiniGridNeuralConfig:
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
    feature_dim = int(agent_cfg.get("feature_dim", 1024))
    hidden_dim = int(agent_cfg.get("hidden_dim", 24))
    alpha_output = float(agent_cfg.get("alpha_output", 0.05))
    alpha_hidden = float(agent_cfg.get("alpha_hidden", 0.005))
    gamma = float(agent_cfg.get("gamma", 0.92))
    epsilon = float(agent_cfg.get("epsilon", 0.2))
    init_scale = float(agent_cfg.get("init_scale", 0.04))
    if feature_dim < 128:
        raise ValueError("agent.feature_dim must be at least 128")
    if hidden_dim < 2 or hidden_dim > 256:
        raise ValueError("agent.hidden_dim out of range")
    if alpha_output <= 0.0 or alpha_hidden <= 0.0:
        raise ValueError("agent learning rates must be positive")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("agent.gamma out of range")
    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError("agent.epsilon out of range")
    if init_scale <= 0.0:
        raise ValueError("agent.init_scale must be positive")

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
    return MiniGridNeuralConfig(
        env_id=env_id,
        max_steps=max_steps,
        quiet_env_output=quiet_env_output,
        agent=NeuralAgentConfig(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            alpha_output=alpha_output,
            alpha_hidden=alpha_hidden,
            gamma=gamma,
            epsilon=epsilon,
            init_scale=init_scale,
        ),
        conditions=tuple(conditions),
    )


def run_minigrid_neural_condition(
    gym: Any,
    env_id: str,
    condition: Condition,
    max_steps: int,
    agent_config: NeuralAgentConfig,
    quiet_env_output: bool = True,
) -> dict[str, Any]:
    env = gym.make(env_id)
    try:
        actions = int(env.action_space.n)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(condition.seed)
        agent = NeuralQAgent(actions=actions, seed=condition.seed, **agent_config.__dict__)
        auxiliary_agent = NeuralQAgent(actions=actions, seed=condition.seed + 100_003, epsilon=0.0, **{
            key: value for key, value in agent_config.__dict__.items() if key != "epsilon"
        })
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
            "agent_type": "one_hidden_layer_q",
            "feature_dim": agent_config.feature_dim,
            "hidden_dim": agent_config.hidden_dim,
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
            "nonzero_parameters": agent.nonzero_parameters(),
        }
    finally:
        env.close()


def write_minigrid_neural_run(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(neural_summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def neural_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# baby-model MiniGrid neural encoder summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- env_id: `{report['env_id']}`",
        f"- winner_last_window: `{report['winner_last_window']}`",
        "",
        "| condition | success_all | success_last | return_last | mean_steps_success | nonzero_parameters |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        mean_steps = row["mean_steps_success"]
        lines.append(
            "| {name} | {all:.3f} | {last:.3f} | {ret:.3f} | {steps} | {params} |".format(
                name=row["name"],
                all=row["success_rate_all"],
                last=row["success_rate_last_window"],
                ret=row["mean_return_last_window"],
                steps="" if mean_steps is None else f"{mean_steps:.2f}",
                params=row["nonzero_parameters"],
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
