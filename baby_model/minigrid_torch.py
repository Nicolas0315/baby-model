from __future__ import annotations

import argparse
import json
from collections import deque
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


class TorchDeviceUnavailable(RuntimeError):
    pass


TASK_SIGNAL_DIM = 16
CONTROLLABILITY_DIM = 1
ACTION_LEFT = 0
ACTION_RIGHT = 1
ACTION_FORWARD = 2
ACTION_PICKUP = 3
ACTION_TOGGLE = 5
OBJECT_WALL = 2
OBJECT_DOOR = 4
OBJECT_KEY = 5
OBJECT_BALL = 6
OBJECT_GOAL = 8
VIEW_FORWARD_X = 3
VIEW_FORWARD_Y = 5


@dataclass(frozen=True)
class TorchCurriculumStage:
    name: str
    env_id: str
    max_steps: int
    episodes: int


@dataclass(frozen=True)
class TorchAgentConfig:
    feature_dim: int
    hidden_dim: int
    learning_rate: float
    gamma: float
    epsilon: float
    batch_size: int
    replay_capacity: int
    target_sync_updates: int
    device: str


@dataclass(frozen=True)
class MiniGridTorchConfig:
    env_id: str
    max_steps: int
    quiet_env_output: bool
    agent: TorchAgentConfig
    conditions: tuple[Condition, ...]
    stages: tuple[TorchCurriculumStage, ...] = ()
    active_stages_by_condition: tuple[tuple[str, tuple[str, ...]], ...] = ()


class TorchDQNAgent:
    def __init__(
        self,
        torch: Any,
        actions: int,
        config: TorchAgentConfig,
        device: Any,
        seed: int,
        epsilon: float | None = None,
        representation_objective: str = "none",
        representation_beta: float = 0.0,
    ) -> None:
        self.torch = torch
        self.actions = actions
        self.config = config
        self.device = device
        self.epsilon = config.epsilon if epsilon is None else epsilon
        self.representation_objective = representation_objective
        self.representation_beta = representation_beta
        self.rng = Random(seed)
        self.model = build_q_network(torch, config.feature_dim, config.hidden_dim, actions).to(device)
        self.target = build_q_network(torch, config.feature_dim, config.hidden_dim, actions).to(device)
        self.target.load_state_dict(self.model.state_dict())
        self.target.eval()
        self.representation_predictor = None
        parameters = list(self.model.parameters())
        if representation_objective in {"next_feature", "next_task_signal", "controllability"}:
            target_dim = _representation_target_dim(representation_objective, config.feature_dim)
            self.representation_predictor = build_next_feature_predictor(
                torch,
                hidden_dim=config.hidden_dim,
                actions=actions,
                target_dim=target_dim,
            ).to(device)
            parameters.extend(self.representation_predictor.parameters())
        elif representation_objective == "action_prior":
            self.representation_predictor = build_action_prior_predictor(
                torch,
                hidden_dim=config.hidden_dim,
                actions=actions,
            ).to(device)
            parameters.extend(self.representation_predictor.parameters())
        self.optimizer = torch.optim.Adam(parameters, lr=config.learning_rate)
        self.loss_fn = torch.nn.MSELoss()
        self.classification_loss_fn = torch.nn.CrossEntropyLoss()
        self.replay: deque[tuple[SparseFeatures, int, float, SparseFeatures, bool]] = deque(maxlen=config.replay_capacity)
        self.updates = 0

    def action_values(self, features: SparseFeatures) -> dict[int, float]:
        values = self._q_values(features)
        return {action: values[action] for action in range(self.actions)}

    def action_prior_values(self, features: SparseFeatures) -> dict[int, float]:
        if self.representation_objective != "action_prior" or self.representation_predictor is None:
            return {}
        with self.torch.no_grad():
            state = self._feature_tensor(features).unsqueeze(0)
            hidden = self.model.encode(state)
            logits = self.representation_predictor(hidden)
            probabilities = self.torch.softmax(logits, dim=1).detach().cpu().tolist()[0]
        return {action: float(probabilities[action]) for action in range(self.actions)}

    def choose(
        self,
        features: SparseFeatures,
        force_random: bool = False,
        action_bonus: dict[int, float] | None = None,
        bonus_weight: float = 1.0,
    ) -> int:
        if force_random or self.rng.random() < self.epsilon:
            return self.rng.randrange(self.actions)
        base_values = self._q_values(features)
        values = [
            base_values[action] + bonus_weight * (0.0 if action_bonus is None else action_bonus.get(action, 0.0))
            for action in range(self.actions)
        ]
        best_value = max(values)
        best_actions = [action for action, value in enumerate(values) if value == best_value]
        return self.rng.choice(best_actions)

    def update(self, features: SparseFeatures, action: int, reward: float, next_features: SparseFeatures, done: bool) -> None:
        self.replay.append((features, action, reward, next_features, done))
        if len(self.replay) < self.config.batch_size:
            return

        batch = self.rng.sample(list(self.replay), self.config.batch_size)
        states = self._batch_tensor([item[0] for item in batch])
        actions = self.torch.tensor([item[1] for item in batch], dtype=self.torch.long, device=self.device)
        rewards = self.torch.tensor([item[2] for item in batch], dtype=self.torch.float32, device=self.device)
        next_states = self._batch_tensor([item[3] for item in batch])
        dones = self.torch.tensor([1.0 if item[4] else 0.0 for item in batch], dtype=self.torch.float32, device=self.device)

        q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with self.torch.no_grad():
            next_q_values = self.target(next_states).max(dim=1).values
            targets = rewards + (1.0 - dones) * self.config.gamma * next_q_values

        loss = self.loss_fn(q_values, targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        if self.updates % self.config.target_sync_updates == 0:
            self.target.load_state_dict(self.model.state_dict())

    def update_representation(
        self,
        features: SparseFeatures,
        action: int,
        target_vector: list[float] | int,
    ) -> float | None:
        if self.representation_objective == "none":
            return None
        if self.representation_objective not in {"next_feature", "next_task_signal", "action_prior", "controllability"} or self.representation_predictor is None:
            raise ValueError(f"unsupported representation objective: {self.representation_objective}")

        state = self._feature_tensor(features).unsqueeze(0)
        hidden = self.model.encode(state)
        if self.representation_objective == "action_prior":
            target_action = int(target_vector)
            if target_action < 0 or target_action >= self.actions:
                raise ValueError(f"action_prior target out of range: {target_action}")
            target = self.torch.tensor([target_action], dtype=self.torch.long, device=self.device)
            prediction = self.representation_predictor(hidden)
            loss = self.classification_loss_fn(prediction, target) * self.representation_beta
        else:
            target = self.torch.tensor([target_vector], dtype=self.torch.float32, device=self.device)
            action_one_hot = self.torch.zeros((1, self.actions), dtype=self.torch.float32, device=self.device)
            action_one_hot[0, action] = 1.0
            prediction = self.representation_predictor(self.torch.cat([hidden, action_one_hot], dim=1))
            loss = self.loss_fn(prediction, target) * self.representation_beta
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return float(loss.detach().cpu().item())

    def parameter_count(self) -> int:
        parameters = list(self.model.parameters())
        if self.representation_predictor is not None:
            parameters.extend(self.representation_predictor.parameters())
        return sum(int(parameter.numel()) for parameter in parameters)

    def representation_parameter_count(self) -> int:
        if self.representation_predictor is None:
            return 0
        return sum(int(parameter.numel()) for parameter in self.representation_predictor.parameters())

    def _q_values(self, features: SparseFeatures) -> list[float]:
        with self.torch.no_grad():
            tensor = self._feature_tensor(features).unsqueeze(0)
            return list(self.model(tensor).detach().cpu().tolist()[0])

    def _feature_tensor(self, features: SparseFeatures) -> Any:
        return self.torch.tensor(dense_feature_vector(features, self.config.feature_dim), dtype=self.torch.float32, device=self.device)

    def _batch_tensor(self, batch_features: list[SparseFeatures]) -> Any:
        return self.torch.tensor(
            [dense_feature_vector(features, self.config.feature_dim) for features in batch_features],
            dtype=self.torch.float32,
            device=self.device,
        )


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-minigrid-torch")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/minigrid-torch-unlock-smoke.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minigrid-torch"))
    parser.add_argument("--seed", type=int, default=601)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.device is not None:
            config.setdefault("agent", {})["device"] = args.device
        report = run_minigrid_torch_suite(config, seed=args.seed)
    except ImportError as exc:
        print(f"missing optional dependency: {exc}")
        print("install with: python3 -m pip install minigrid torch")
        return 2
    except TorchDeviceUnavailable as exc:
        print(f"torch device unavailable: {exc}")
        return 2

    run_dir = write_minigrid_torch_run(report, args.output_dir)
    print(f"minigrid_torch_run_dir={run_dir}")
    print(f"torch_device={report['framework']['device']}")
    print(f"winner_last_window={report['winner_last_window']}")
    return 0


def run_minigrid_torch_suite(config: dict[str, Any], seed: int = 601) -> dict[str, Any]:
    parsed = parse_minigrid_torch_config(config=config, seed=seed)
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
        import torch
    except ImportError as exc:
        raise ImportError("gymnasium/minigrid/torch") from exc

    torch.manual_seed(seed)
    device = select_torch_device(torch, parsed.agent.device)
    active_stages_by_condition = dict(parsed.active_stages_by_condition)
    if parsed.stages:
        results = [
            run_minigrid_torch_curriculum_condition(
                gym=gym,
                torch=torch,
                stages=parsed.stages,
                active_stages=active_stages_by_condition[condition.name],
                condition=condition,
                agent_config=parsed.agent,
                device=device,
                quiet_env_output=parsed.quiet_env_output,
            )
            for condition in parsed.conditions
        ]
    else:
        results = [
            run_minigrid_torch_condition(
                gym=gym,
                torch=torch,
                env_id=parsed.env_id,
                condition=condition,
                max_steps=parsed.max_steps,
                agent_config=parsed.agent,
                device=device,
                quiet_env_output=parsed.quiet_env_output,
            )
            for condition in parsed.conditions
        ]
    report = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": str(config.get("hypothesis", "Baby-AD/DA MiniGrid PyTorch DQN")),
        "env_id": parsed.env_id,
        "max_steps": parsed.max_steps,
        "framework": {
            "name": "pytorch",
            "version": str(getattr(torch, "__version__", "unknown")),
            "device": str(device),
            "cuda_available": bool(torch.cuda.is_available()),
            "mps_available": torch_mps_available(torch),
        },
        "agent": {
            "type": "torch_dqn",
            "feature_dim": parsed.agent.feature_dim,
            "hidden_dim": parsed.agent.hidden_dim,
            "learning_rate": parsed.agent.learning_rate,
            "gamma": parsed.agent.gamma,
            "epsilon": parsed.agent.epsilon,
            "batch_size": parsed.agent.batch_size,
            "replay_capacity": parsed.agent.replay_capacity,
            "target_sync_updates": parsed.agent.target_sync_updates,
        },
        "results": results,
        "winner_last_window": max(results, key=lambda row: row["success_rate_last_window"])["name"],
    }
    if parsed.stages:
        report["stages"] = [
            {
                "name": stage.name,
                "env_id": stage.env_id,
                "max_steps": stage.max_steps,
                "episodes": stage.episodes,
            }
            for stage in parsed.stages
        ]
    return report


def _combined_action_bonus(
    agent: TorchDQNAgent,
    auxiliary_agent: TorchDQNAgent,
    features: SparseFeatures,
    condition: Condition,
    force_random: bool,
) -> dict[int, float] | None:
    if force_random:
        return None
    bonus: dict[int, float] = {}
    if condition.intrinsic_target == "auxiliary":
        for action, value in auxiliary_agent.action_values(features).items():
            bonus[action] = bonus.get(action, 0.0) + value
    if condition.action_prior_weight > 0.0:
        for action, value in agent.action_prior_values(features).items():
            bonus[action] = bonus.get(action, 0.0) + condition.action_prior_weight * value
    return bonus or None


def parse_minigrid_torch_config(config: dict[str, Any], seed: int = 601) -> MiniGridTorchConfig:
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
    hidden_dim = int(agent_cfg.get("hidden_dim", 64))
    learning_rate = float(agent_cfg.get("learning_rate", 0.001))
    gamma = float(agent_cfg.get("gamma", 0.92))
    epsilon = float(agent_cfg.get("epsilon", 0.2))
    batch_size = int(agent_cfg.get("batch_size", 16))
    replay_capacity = int(agent_cfg.get("replay_capacity", 512))
    target_sync_updates = int(agent_cfg.get("target_sync_updates", 25))
    device = str(agent_cfg.get("device", "auto"))
    if feature_dim < 16:
        raise ValueError("agent.feature_dim must be at least 16")
    if hidden_dim < 2 or hidden_dim > 2048:
        raise ValueError("agent.hidden_dim out of range")
    if learning_rate <= 0.0:
        raise ValueError("agent.learning_rate must be positive")
    if gamma < 0.0 or gamma > 1.0:
        raise ValueError("agent.gamma out of range")
    if epsilon < 0.0 or epsilon > 1.0:
        raise ValueError("agent.epsilon out of range")
    if batch_size < 1:
        raise ValueError("agent.batch_size must be positive")
    if replay_capacity < batch_size:
        raise ValueError("agent.replay_capacity must be at least batch_size")
    if target_sync_updates < 1:
        raise ValueError("agent.target_sync_updates must be positive")
    if not (device in {"auto", "cpu", "cuda", "mps"} or (device.startswith("cuda:") and device[5:].isdigit())):
        raise ValueError("agent.device must be auto, cpu, cuda, cuda:N, or mps")

    stages = parse_torch_curriculum_stages(config)
    stage_names = {stage.name for stage in stages}
    all_stage_names = tuple(stage.name for stage in stages)

    condition_cfgs = config.get("conditions", [])
    if not isinstance(condition_cfgs, list) or not condition_cfgs:
        raise ValueError("conditions must be a non-empty list")
    names: set[str] = set()
    conditions: list[Condition] = []
    active_stages_by_condition: list[tuple[str, tuple[str, ...]]] = []
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
        representation_objective = str(item.get("representation_objective", "none"))
        if representation_objective not in {"none", "next_feature", "next_task_signal", "action_prior", "controllability"}:
            raise ValueError(f"invalid representation_objective for {name}")
        representation_beta = float(item.get("representation_beta", 0.0))
        action_prior_weight = float(item.get("action_prior_weight", 0.0))
        active_stages = tuple(str(stage_name) for stage_name in item.get("active_stages", all_stage_names))
        if stages:
            if not active_stages:
                raise ValueError(f"active_stages must be non-empty for {name}")
            unknown_stages = sorted(set(active_stages) - stage_names)
            if unknown_stages:
                raise ValueError(f"unknown active_stages for {name}: {','.join(unknown_stages)}")
            episodes = sum(stage.episodes for stage in stages if stage.name in active_stages)
        else:
            if "active_stages" in item:
                raise ValueError(f"active_stages requires top-level stages for {name}")
            episodes = int(item.get("episodes", 0))
        delay = int(item.get("decoder_delay_episodes", 0))
        beta = float(item.get("intrinsic_beta", 0.0))
        if episodes < 1:
            raise ValueError(f"episodes must be positive for {name}")
        if delay < 0 or delay > episodes:
            raise ValueError(f"decoder_delay_episodes out of range for {name}")
        if beta < 0:
            raise ValueError(f"intrinsic_beta must be non-negative for {name}")
        if representation_beta < 0:
            raise ValueError(f"representation_beta must be non-negative for {name}")
        if representation_objective != "none" and representation_beta <= 0.0:
            raise ValueError(f"representation_beta must be positive for {name}")
        if action_prior_weight < 0.0:
            raise ValueError(f"action_prior_weight must be non-negative for {name}")
        if action_prior_weight > 0.0 and representation_objective != "action_prior":
            raise ValueError(f"action_prior_weight requires action_prior for {name}")
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
                representation_objective=representation_objective,
                representation_beta=representation_beta,
                action_prior_weight=action_prior_weight,
            )
        )
        active_stages_by_condition.append((name, active_stages))
    return MiniGridTorchConfig(
        env_id=env_id,
        max_steps=max_steps,
        quiet_env_output=quiet_env_output,
        agent=TorchAgentConfig(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            learning_rate=learning_rate,
            gamma=gamma,
            epsilon=epsilon,
            batch_size=batch_size,
            replay_capacity=replay_capacity,
            target_sync_updates=target_sync_updates,
            device=device,
        ),
        conditions=tuple(conditions),
        stages=stages,
        active_stages_by_condition=tuple(active_stages_by_condition),
    )


def parse_torch_curriculum_stages(config: dict[str, Any]) -> tuple[TorchCurriculumStage, ...]:
    stage_cfgs = config.get("stages", [])
    if stage_cfgs in (None, []):
        return ()
    if not isinstance(stage_cfgs, list):
        raise ValueError("stages must be a list")
    stages: list[TorchCurriculumStage] = []
    names: set[str] = set()
    for item in stage_cfgs:
        if not isinstance(item, dict):
            raise ValueError("each stage must be an object")
        name = str(item.get("name", ""))
        if not name:
            raise ValueError("stage.name is required")
        if name in names:
            raise ValueError(f"duplicate stage.name: {name}")
        names.add(name)
        env_id = str(item.get("env_id", ""))
        max_steps = int(item.get("max_steps", 0))
        episodes = int(item.get("episodes", 0))
        if not env_id:
            raise ValueError(f"stage.env_id is required for {name}")
        if max_steps < 1:
            raise ValueError(f"stage.max_steps must be positive for {name}")
        if episodes < 1:
            raise ValueError(f"stage.episodes must be positive for {name}")
        stages.append(TorchCurriculumStage(name=name, env_id=env_id, max_steps=max_steps, episodes=episodes))
    return tuple(stages)


def run_minigrid_torch_condition(
    gym: Any,
    torch: Any,
    env_id: str,
    condition: Condition,
    max_steps: int,
    agent_config: TorchAgentConfig,
    device: Any,
    quiet_env_output: bool = True,
) -> dict[str, Any]:
    env = gym.make(env_id)
    try:
        actions = int(env.action_space.n)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(condition.seed)
        torch.manual_seed(condition.seed)
        agent = TorchDQNAgent(
            torch=torch,
            actions=actions,
            config=agent_config,
            device=device,
            seed=condition.seed,
            representation_objective=condition.representation_objective,
            representation_beta=condition.representation_beta,
        )
        auxiliary_agent = TorchDQNAgent(
            torch=torch,
            actions=actions,
            config=agent_config,
            device=device,
            seed=condition.seed + 100_003,
            epsilon=0.0,
        )
        transition = TransitionSurprise()
        episodes: list[EpisodeMetrics] = []
        representation_losses: list[float] = []
        representation_update_counts: list[int] = []
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
            representation_loss = 0.0
            representation_updates = 0
            success = False
            steps = 0

            for _ in range(max_steps):
                force_random = episode < condition.decoder_delay_episodes
                action_bonus = _combined_action_bonus(
                    agent=agent,
                    auxiliary_agent=auxiliary_agent,
                    features=features,
                    condition=condition,
                    force_random=force_random,
                )
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

                representation_target = representation_target_for_objective(
                    condition=condition,
                    features=features,
                    observation=observation,
                    next_observation=next_observation,
                    next_features=next_features,
                    feature_dim=agent_config.feature_dim,
                    actions=actions,
                )
                loss = agent.update_representation(features, action, representation_target)
                if loss is not None:
                    representation_loss += loss
                    representation_updates += 1
                if not force_random:
                    agent.update(features, action, total_reward, next_features, done)
                    if condition.intrinsic_target == "auxiliary":
                        auxiliary_agent.update(features, action, intrinsic, next_features, done)
                transition.update(feature_key, action, next_feature_key)

                external_return += float(reward)
                intrinsic_return += intrinsic
                steps += 1
                visited.add(next_feature_key)
                observation = next_observation
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
            representation_losses.append(representation_loss / max(1, representation_updates))
            representation_update_counts.append(representation_updates)

        last_window = episodes[-20:] if len(episodes) >= 20 else episodes
        representation_last_window = (
            representation_losses[-20:] if len(representation_losses) >= 20 else representation_losses
        )
        successful_steps = [item.steps for item in episodes if item.success]
        return {
            "name": condition.name,
            "env_id": env_id,
            "agent_type": "torch_dqn",
            "feature_dim": agent_config.feature_dim,
            "hidden_dim": agent_config.hidden_dim,
            "encoder_mode": condition.encoder_mode,
            "episodes": condition.episodes,
            "decoder_delay_episodes": condition.decoder_delay_episodes,
            "intrinsic_beta": condition.intrinsic_beta,
            "intrinsic_mode": condition.intrinsic_mode,
            "intrinsic_target": condition.intrinsic_target,
            "representation_objective": condition.representation_objective,
            "representation_beta": condition.representation_beta,
            "action_prior_weight": condition.action_prior_weight,
            "seed": condition.seed,
            "observation_schema": first_schema or {},
            "success_rate_all": mean(1.0 if item.success else 0.0 for item in episodes),
            "success_rate_last_window": mean(1.0 if item.success else 0.0 for item in last_window),
            "mean_steps_success": mean(successful_steps) if successful_steps else None,
            "mean_return_last_window": mean(item.external_return for item in last_window),
            "mean_intrinsic_return_last_window": mean(item.intrinsic_return for item in last_window),
            "mean_unique_features_last_window": mean(item.unique_features for item in last_window),
            "mean_representation_loss_last_window": mean(representation_last_window),
            "representation_updates": sum(representation_update_counts),
            "representation_parameter_count": agent.representation_parameter_count(),
            "parameter_count": agent.parameter_count(),
            "updates": agent.updates,
        }
    finally:
        env.close()


def run_minigrid_torch_curriculum_condition(
    gym: Any,
    torch: Any,
    stages: tuple[TorchCurriculumStage, ...],
    active_stages: tuple[str, ...],
    condition: Condition,
    agent_config: TorchAgentConfig,
    device: Any,
    quiet_env_output: bool = True,
) -> dict[str, Any]:
    active_stage_names = set(active_stages)
    agent: TorchDQNAgent | None = None
    auxiliary_agent: TorchDQNAgent | None = None
    transition = TransitionSurprise()
    stage_results: list[dict[str, Any]] = []
    global_episode = 0

    for stage_index, stage in enumerate(stages):
        if stage.name not in active_stage_names:
            continue
        stage_result, agent, auxiliary_agent, global_episode = _run_minigrid_torch_stage(
            gym=gym,
            torch=torch,
            stage=stage,
            condition=condition,
            agent_config=agent_config,
            device=device,
            stage_index=stage_index,
            global_episode_start=global_episode,
            agent=agent,
            auxiliary_agent=auxiliary_agent,
            transition=transition,
            quiet_env_output=quiet_env_output,
        )
        stage_results.append(stage_result)

    if not stage_results or agent is None:
        raise ValueError(f"no active curriculum stages for {condition.name}")
    final_stage = stage_results[-1]
    return {
        "name": condition.name,
        "env_id": final_stage["env_id"],
        "agent_type": "torch_dqn",
        "feature_dim": agent_config.feature_dim,
        "hidden_dim": agent_config.hidden_dim,
        "encoder_mode": condition.encoder_mode,
        "episodes": condition.episodes,
        "decoder_delay_episodes": condition.decoder_delay_episodes,
        "intrinsic_beta": condition.intrinsic_beta,
        "intrinsic_mode": condition.intrinsic_mode,
        "intrinsic_target": condition.intrinsic_target,
        "representation_objective": condition.representation_objective,
        "representation_beta": condition.representation_beta,
        "action_prior_weight": condition.action_prior_weight,
        "seed": condition.seed,
        "active_stages": list(active_stages),
        "stage_results": stage_results,
        "final_stage": final_stage,
        "success_rate_all": final_stage["success_rate_all"],
        "success_rate_last_window": final_stage["success_rate_last_window"],
        "mean_steps_success": final_stage["mean_steps_success"],
        "mean_return_last_window": final_stage["mean_return_last_window"],
        "mean_intrinsic_return_last_window": final_stage["mean_intrinsic_return_last_window"],
        "mean_unique_features_last_window": final_stage["mean_unique_features_last_window"],
        "mean_representation_loss_last_window": final_stage["mean_representation_loss_last_window"],
        "representation_updates": sum(stage["representation_updates"] for stage in stage_results),
        "representation_parameter_count": agent.representation_parameter_count(),
        "parameter_count": agent.parameter_count(),
        "updates": agent.updates,
    }


def _run_minigrid_torch_stage(
    gym: Any,
    torch: Any,
    stage: TorchCurriculumStage,
    condition: Condition,
    agent_config: TorchAgentConfig,
    device: Any,
    stage_index: int,
    global_episode_start: int,
    agent: TorchDQNAgent | None,
    auxiliary_agent: TorchDQNAgent | None,
    transition: TransitionSurprise,
    quiet_env_output: bool,
) -> tuple[dict[str, Any], TorchDQNAgent, TorchDQNAgent, int]:
    env = gym.make(stage.env_id)
    try:
        actions = int(env.action_space.n)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(condition.seed + stage_index)
        torch.manual_seed(condition.seed + stage_index)
        if agent is None:
            agent = TorchDQNAgent(
                torch=torch,
                actions=actions,
                config=agent_config,
                device=device,
                seed=condition.seed,
                representation_objective=condition.representation_objective,
                representation_beta=condition.representation_beta,
            )
        if auxiliary_agent is None:
            auxiliary_agent = TorchDQNAgent(
                torch=torch,
                actions=actions,
                config=agent_config,
                device=device,
                seed=condition.seed + 100_003,
                epsilon=0.0,
            )
        if agent.actions != actions or auxiliary_agent.actions != actions:
            raise ValueError(f"action-space mismatch in stage {stage.name}")

        episodes: list[EpisodeMetrics] = []
        representation_losses: list[float] = []
        representation_update_counts: list[int] = []
        first_schema: dict[str, Any] | None = None

        for stage_episode in range(stage.episodes):
            global_episode = global_episode_start + stage_episode
            observation, _info = _env_call(
                env.reset,
                quiet=quiet_env_output,
                seed=condition.seed * 100_000 + stage_index * 1000 + stage_episode,
            )
            if first_schema is None:
                first_schema = observation_schema(observation)
            features = linear_features(observation, condition.encoder_mode, agent_config.feature_dim)
            feature_key = feature_signature(features)
            visited = {feature_key}
            external_return = 0.0
            intrinsic_return = 0.0
            representation_loss = 0.0
            representation_updates = 0
            success = False
            steps = 0

            for _ in range(stage.max_steps):
                force_random = global_episode < condition.decoder_delay_episodes
                action_bonus = _combined_action_bonus(
                    agent=agent,
                    auxiliary_agent=auxiliary_agent,
                    features=features,
                    condition=condition,
                    force_random=force_random,
                )
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

                representation_target = representation_target_for_objective(
                    condition=condition,
                    features=features,
                    observation=observation,
                    next_observation=next_observation,
                    next_features=next_features,
                    feature_dim=agent_config.feature_dim,
                    actions=actions,
                )
                loss = agent.update_representation(features, action, representation_target)
                if loss is not None:
                    representation_loss += loss
                    representation_updates += 1
                if not force_random:
                    agent.update(features, action, total_reward, next_features, done)
                    if condition.intrinsic_target == "auxiliary":
                        auxiliary_agent.update(features, action, intrinsic, next_features, done)
                transition.update(feature_key, action, next_feature_key)

                external_return += float(reward)
                intrinsic_return += intrinsic
                steps += 1
                visited.add(next_feature_key)
                observation = next_observation
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
            representation_losses.append(representation_loss / max(1, representation_updates))
            representation_update_counts.append(representation_updates)

        return (
            _torch_stage_summary(
                stage=stage,
                condition=condition,
                episodes=episodes,
                first_schema=first_schema,
                representation_losses=representation_losses,
                representation_update_counts=representation_update_counts,
                agent=agent,
            ),
            agent,
            auxiliary_agent,
            global_episode_start + stage.episodes,
        )
    finally:
        env.close()


def _torch_stage_summary(
    stage: TorchCurriculumStage,
    condition: Condition,
    episodes: list[EpisodeMetrics],
    first_schema: dict[str, Any] | None,
    representation_losses: list[float],
    representation_update_counts: list[int],
    agent: TorchDQNAgent,
) -> dict[str, Any]:
    last_window = episodes[-20:] if len(episodes) >= 20 else episodes
    representation_last_window = (
        representation_losses[-20:] if len(representation_losses) >= 20 else representation_losses
    )
    successful_steps = [item.steps for item in episodes if item.success]
    return {
        "stage": stage.name,
        "env_id": stage.env_id,
        "max_steps": stage.max_steps,
        "episodes": stage.episodes,
        "condition_seed": condition.seed,
        "observation_schema": first_schema or {},
        "success_rate_all": mean(1.0 if item.success else 0.0 for item in episodes),
        "success_rate_last_window": mean(1.0 if item.success else 0.0 for item in last_window),
        "mean_steps_success": mean(successful_steps) if successful_steps else None,
        "mean_return_last_window": mean(item.external_return for item in last_window),
        "mean_intrinsic_return_last_window": mean(item.intrinsic_return for item in last_window),
        "mean_unique_features_last_window": mean(item.unique_features for item in last_window),
        "mean_representation_loss_last_window": mean(representation_last_window),
        "representation_updates": sum(representation_update_counts),
        "updates": agent.updates,
    }


def build_q_network(torch: Any, feature_dim: int, hidden_dim: int, actions: int) -> Any:
    class QNetwork(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = torch.nn.Sequential(
                torch.nn.Linear(feature_dim, hidden_dim),
                torch.nn.ReLU(),
            )
            self.head = torch.nn.Linear(hidden_dim, actions)

        def encode(self, x: Any) -> Any:
            return self.encoder(x)

        def forward(self, x: Any) -> Any:
            return self.head(self.encode(x))

    return QNetwork()


def build_next_feature_predictor(torch: Any, hidden_dim: int, actions: int, target_dim: int) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(hidden_dim + actions, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, target_dim),
    )


def build_action_prior_predictor(torch: Any, hidden_dim: int, actions: int) -> Any:
    return torch.nn.Sequential(
        torch.nn.Linear(hidden_dim, hidden_dim),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden_dim, actions),
    )


def _representation_target_dim(representation_objective: str, feature_dim: int) -> int:
    if representation_objective == "next_feature":
        return feature_dim
    if representation_objective == "next_task_signal":
        return TASK_SIGNAL_DIM
    if representation_objective == "controllability":
        return CONTROLLABILITY_DIM
    raise ValueError(f"unsupported representation objective: {representation_objective}")


def representation_target_for_objective(
    condition: Condition,
    features: SparseFeatures,
    observation: Any,
    next_observation: Any,
    next_features: SparseFeatures,
    feature_dim: int,
    actions: int,
) -> list[float] | int:
    if condition.representation_objective == "next_feature":
        return dense_feature_vector(next_features, feature_dim)
    if condition.representation_objective == "next_task_signal":
        return task_signal_vector(next_observation)
    if condition.representation_objective == "controllability":
        return controllability_target(features, next_features)
    if condition.representation_objective == "action_prior":
        return action_prior_label(observation, actions)
    return []


def controllability_target(features: SparseFeatures, next_features: SparseFeatures) -> list[float]:
    changed = feature_signature(features) != feature_signature(next_features)
    return [1.0 if changed else 0.0]


def action_prior_label(observation: Any, actions: int) -> int:
    if actions < 1:
        raise ValueError("actions must be positive")
    if not isinstance(observation, dict):
        return _available_action(ACTION_FORWARD, actions)

    image = observation.get("image")
    if hasattr(image, "tolist"):
        image = image.tolist()
    front = _image_cell(image, VIEW_FORWARD_X, VIEW_FORWARD_Y)
    mission = str(observation.get("mission", "")).lower()

    if front is not None and len(front) >= 3:
        obj_type = int(front[0])
        state = int(front[2])
        if obj_type == OBJECT_KEY and ("key" in mission or "unlock" in mission):
            return _available_action(ACTION_PICKUP, actions)
        if obj_type == OBJECT_DOOR:
            if state == 0:
                return _available_action(ACTION_FORWARD, actions)
            return _available_action(ACTION_TOGGLE, actions)
        if obj_type in {OBJECT_BALL, OBJECT_GOAL}:
            return _available_action(ACTION_FORWARD, actions)
        if obj_type == OBJECT_WALL:
            return _turn_action(observation, actions)

    return _available_action(ACTION_FORWARD, actions)


def task_signal_vector(observation: Any) -> list[float]:
    if not isinstance(observation, dict):
        return [0.0 for _ in range(TASK_SIGNAL_DIM)]
    image = observation.get("image")
    if hasattr(image, "tolist"):
        image = image.tolist()
    cells = _flat_image_cells(image)
    cell_count = max(1, len(cells))
    type_counts: dict[int, int] = {}
    door_state_counts = {0: 0, 1: 0, 2: 0}
    for cell in cells:
        if len(cell) < 3:
            continue
        obj_type = int(cell[0])
        state = int(cell[2])
        type_counts[obj_type] = type_counts.get(obj_type, 0) + 1
        if obj_type == 4 and state in door_state_counts:
            door_state_counts[state] += 1

    mission = str(observation.get("mission", "")).lower()
    direction = float(observation.get("direction", 0))
    norm = float(cell_count)
    return [
        min(1.0, max(0.0, direction / 3.0)),
        1.0 if "unlock" in mission else 0.0,
        1.0 if "open" in mission else 0.0,
        1.0 if "key" in mission else 0.0,
        1.0 if "door" in mission else 0.0,
        type_counts.get(4, 0) / norm,
        type_counts.get(5, 0) / norm,
        type_counts.get(8, 0) / norm,
        type_counts.get(6, 0) / norm,
        type_counts.get(7, 0) / norm,
        type_counts.get(2, 0) / norm,
        door_state_counts[0] / norm,
        door_state_counts[1] / norm,
        door_state_counts[2] / norm,
        1.0 if type_counts.get(5, 0) == 0 and "key" in mission else 0.0,
        1.0 if type_counts.get(4, 0) == 0 and "door" in mission else 0.0,
    ]


def _flat_image_cells(image: Any) -> list[list[int]]:
    if not isinstance(image, list):
        return []
    cells: list[list[int]] = []
    for row in image:
        if not isinstance(row, list):
            continue
        for cell in row:
            if isinstance(cell, list):
                cells.append(cell)
    return cells


def _image_cell(image: Any, x: int, y: int) -> list[int] | None:
    if not isinstance(image, list):
        return None
    if x < 0 or x >= len(image):
        return None
    column = image[x]
    if not isinstance(column, list) or y < 0 or y >= len(column):
        return None
    cell = column[y]
    return cell if isinstance(cell, list) else None


def _available_action(preferred: int, actions: int) -> int:
    if 0 <= preferred < actions:
        return preferred
    if ACTION_FORWARD < actions:
        return ACTION_FORWARD
    return actions - 1


def _turn_action(observation: dict[str, Any], actions: int) -> int:
    direction = int(observation.get("direction", 0))
    preferred = ACTION_LEFT if direction % 2 == 0 else ACTION_RIGHT
    return _available_action(preferred, actions)


def dense_feature_vector(features: SparseFeatures, feature_dim: int) -> list[float]:
    vector = [0.0 for _ in range(feature_dim)]
    for index, value in features.items():
        if index < 0 or index >= feature_dim:
            raise ValueError(f"feature index out of range: {index}")
        vector[index] = float(value)
    return vector


def select_torch_device(torch: Any, preference: str = "auto") -> Any:
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch_mps_available(torch):
            return torch.device("mps")
        return torch.device("cpu")
    if preference in {"cuda"} or preference.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise TorchDeviceUnavailable(f"{preference} requested but torch.cuda.is_available() is false")
        return torch.device(preference)
    if preference == "mps":
        if not torch_mps_available(torch):
            raise TorchDeviceUnavailable("mps requested but torch.backends.mps.is_available() is false")
        return torch.device("mps")
    if preference == "cpu":
        return torch.device("cpu")
    raise ValueError(f"unsupported torch device preference: {preference}")


def torch_mps_available(torch: Any) -> bool:
    backends = getattr(torch, "backends", None)
    mps = getattr(backends, "mps", None)
    checker = getattr(mps, "is_available", None)
    return bool(checker()) if callable(checker) else False


def write_minigrid_torch_run(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(torch_summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def torch_summary_markdown(report: dict[str, Any]) -> str:
    framework = report["framework"]
    is_curriculum = "stages" in report
    lines = [
        "# baby-model MiniGrid PyTorch DQN summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- env_id: `{report['env_id']}`",
        f"- torch_version: `{framework['version']}`",
        f"- device: `{framework['device']}`",
        f"- winner_last_window: `{report['winner_last_window']}`",
        "",
    ]
    if is_curriculum:
        stage_text = ",".join(f"{stage['name']}:{stage['env_id']}:{stage['episodes']}" for stage in report["stages"])
        lines.extend(
            [
                f"- stages: `{stage_text}`",
                "",
                "| condition | active_stages | final_stage | prior | success_all | success_last | return_last | mean_steps_success | rep_loss | rep_updates | updates | parameters |",
                "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    else:
        lines.extend(
            [
                "| condition | prior | success_all | success_last | return_last | mean_steps_success | rep_loss | rep_updates | updates | parameters |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
    for row in report["results"]:
        mean_steps = row["mean_steps_success"]
        rep_loss = row.get("mean_representation_loss_last_window", 0.0)
        if is_curriculum:
            final_stage = row["final_stage"]
            lines.append(
                "| {name} | {stages} | {stage} | {prior:.3f} | {all:.3f} | {last:.3f} | {ret:.3f} | {steps} | {rep_loss:.4f} | {rep_updates} | {updates} | {params} |".format(
                    name=row["name"],
                    stages=",".join(row["active_stages"]),
                    stage=final_stage["stage"],
                    prior=row.get("action_prior_weight", 0.0),
                    all=row["success_rate_all"],
                    last=row["success_rate_last_window"],
                    ret=row["mean_return_last_window"],
                    steps="" if mean_steps is None else f"{mean_steps:.2f}",
                    rep_loss=rep_loss,
                    rep_updates=row.get("representation_updates", 0),
                    updates=row["updates"],
                    params=row["parameter_count"],
                )
            )
        else:
            lines.append(
                "| {name} | {prior:.3f} | {all:.3f} | {last:.3f} | {ret:.3f} | {steps} | {rep_loss:.4f} | {rep_updates} | {updates} | {params} |".format(
                    name=row["name"],
                    prior=row.get("action_prior_weight", 0.0),
                    all=row["success_rate_all"],
                    last=row["success_rate_last_window"],
                    ret=row["mean_return_last_window"],
                    steps="" if mean_steps is None else f"{mean_steps:.2f}",
                    rep_loss=rep_loss,
                    rep_updates=row.get("representation_updates", 0),
                    updates=row["updates"],
                    params=row["parameter_count"],
                )
            )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
