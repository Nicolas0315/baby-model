from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from baby_model.agents import EpisodeMetrics, QAgent, TransitionSurprise
from baby_model.envs import BabyGrid, FeatureEncoder


@dataclass(frozen=True)
class Condition:
    name: str
    encoder_mode: str
    episodes: int
    decoder_delay_episodes: int
    intrinsic_beta: float
    intrinsic_mode: str
    seed: int
    intrinsic_beta_end: float | None = None
    intrinsic_anneal_episodes: int = 0
    intrinsic_schedule: str = "constant"
    intrinsic_gate: str = "none"
    intrinsic_target: str = "reward"
    representation_objective: str = "none"
    representation_beta: float = 0.0
    representation_state_beta: float = 0.0
    representation_target_visibility_beta: float = 0.0
    representation_state_beta_end: float | None = None
    representation_target_visibility_beta_end: float | None = None
    representation_anneal_episodes: int = 0
    representation_schedule: str = "constant"
    action_prior_weight: float = 0.0
    freeze_encoder_after_delay: bool = False
    stop_representation_after_delay: bool = False


def default_conditions(seed: int = 7) -> list[Condition]:
    return [
        Condition("A_end_to_end", "raw", 80, 0, 0.0, "none", seed),
        Condition("B_encoder_first", "coarse", 80, 20, 0.0, "none", seed + 1),
        Condition("C_baby_surprise", "coarse", 80, 20, 0.05, "surprise", seed + 2),
        Condition("D_baby_progress", "coarse", 80, 20, 0.2, "progress", seed + 3),
    ]


def run_condition(
    condition: Condition,
    size: int = 7,
    max_steps: int = 60,
    obstacle_count: int = 0,
    toy_count: int = 3,
) -> dict[str, Any]:
    env = BabyGrid(
        size=size,
        max_steps=max_steps,
        seed=condition.seed,
        obstacle_count=obstacle_count,
        toy_count=toy_count,
    )
    encoder = FeatureEncoder(size=size, mode=condition.encoder_mode)
    agent = QAgent(seed=condition.seed)
    auxiliary_agent = QAgent(seed=condition.seed + 100_003, epsilon=0.0)
    transition = TransitionSurprise()
    episodes: list[EpisodeMetrics] = []

    for episode in range(condition.episodes):
        observation = env.reset(seed=condition.seed * 1000 + episode)
        feature = encoder.encode(observation)
        visited = {feature}
        external_return = 0.0
        intrinsic_return = 0.0
        success = False

        for _ in range(max_steps):
            force_random = episode < condition.decoder_delay_episodes
            action_bonus = None
            if condition.intrinsic_target == "auxiliary" and not force_random:
                action_bonus = auxiliary_agent.action_values(feature)
            action = agent.choose(feature, force_random=force_random, action_bonus=action_bonus)
            result = env.step(action)
            next_feature = encoder.encode(result.observation)
            intrinsic_signal = _intrinsic_signal(condition.intrinsic_mode, transition, feature, action, next_feature)
            intrinsic = _intrinsic_reward(condition, episode, result.reward, intrinsic_signal)
            total_reward = _q_target_reward(condition, result.reward, intrinsic)

            if not force_random:
                agent.update(feature, action, total_reward, next_feature, result.done)
                if condition.intrinsic_target == "auxiliary":
                    auxiliary_agent.update(feature, action, intrinsic, next_feature, result.done)
            transition.update(feature, action, next_feature)

            external_return += result.reward
            intrinsic_return += intrinsic
            visited.add(next_feature)
            feature = next_feature
            if result.reward >= 1.0:
                success = True
            if result.done:
                break

        episodes.append(
            EpisodeMetrics(
                success=success,
                steps=int(env.steps),
                external_return=external_return,
                intrinsic_return=intrinsic_return,
                unique_features=len(visited),
            )
        )

    last_window = episodes[-20:] if len(episodes) >= 20 else episodes
    successful_steps = [item.steps for item in episodes if item.success]
    return {
        "name": condition.name,
        "encoder_mode": condition.encoder_mode,
        "episodes": condition.episodes,
        "decoder_delay_episodes": condition.decoder_delay_episodes,
        "intrinsic_beta": condition.intrinsic_beta,
        "intrinsic_beta_end": condition.intrinsic_beta_end,
        "intrinsic_anneal_episodes": condition.intrinsic_anneal_episodes,
        "intrinsic_schedule": condition.intrinsic_schedule,
        "intrinsic_gate": condition.intrinsic_gate,
        "intrinsic_target": condition.intrinsic_target,
        "intrinsic_mode": condition.intrinsic_mode,
        "seed": condition.seed,
        "success_rate_all": mean(1.0 if item.success else 0.0 for item in episodes),
        "success_rate_last_window": mean(1.0 if item.success else 0.0 for item in last_window),
        "mean_steps_success": mean(successful_steps) if successful_steps else None,
        "mean_external_return_last_window": mean(item.external_return for item in last_window),
        "mean_intrinsic_return_last_window": mean(item.intrinsic_return for item in last_window),
        "mean_unique_features_last_window": mean(item.unique_features for item in last_window),
    }


def run_suite(config: dict[str, Any] | None = None, seed: int = 7) -> dict[str, Any]:
    config = config or {}
    validate_config(config)
    env_cfg = config.get("environment", {})
    condition_cfgs = config.get("conditions")
    if condition_cfgs:
        conditions = [
            Condition(
                name=str(item["name"]),
                encoder_mode=str(item["encoder_mode"]),
                episodes=int(item["episodes"]),
                decoder_delay_episodes=int(item["decoder_delay_episodes"]),
                intrinsic_beta=float(item["intrinsic_beta"]),
                intrinsic_mode=str(item.get("intrinsic_mode", "none")),
                seed=seed + i,
                intrinsic_beta_end=(
                    float(item["intrinsic_beta_end"]) if "intrinsic_beta_end" in item else None
                ),
                intrinsic_anneal_episodes=int(item.get("intrinsic_anneal_episodes", 0)),
                intrinsic_schedule=str(item.get("intrinsic_schedule", "constant")),
                intrinsic_gate=str(item.get("intrinsic_gate", "none")),
                intrinsic_target=str(item.get("intrinsic_target", "reward")),
            )
            for i, item in enumerate(condition_cfgs)
        ]
    else:
        conditions = default_conditions(seed=seed)
    results = [
        run_condition(
            condition,
            size=int(env_cfg.get("size", 7)),
            max_steps=int(env_cfg.get("max_steps", 60)),
            obstacle_count=int(env_cfg.get("obstacle_count", 0)),
            toy_count=int(env_cfg.get("toy_count", 3)),
        )
        for condition in conditions
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Baby-AD/DA asymmetry v0",
        "results": results,
        "winner_last_window": max(results, key=lambda row: row["success_rate_last_window"])["name"],
    }


def write_run(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    metrics_path = run_dir / "metrics.json"
    summary_path = run_dir / "summary.md"
    metrics_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    summary_path.write_text(_summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def validate_config(config: dict[str, Any]) -> None:
    env_cfg = config.get("environment", {})
    if not isinstance(env_cfg, dict):
        raise ValueError("environment must be an object")
    size = int(env_cfg.get("size", 7))
    max_steps = int(env_cfg.get("max_steps", 60))
    obstacle_count = int(env_cfg.get("obstacle_count", 0))
    toy_count = int(env_cfg.get("toy_count", 3))
    if size < 4:
        raise ValueError("environment.size must be at least 4")
    if max_steps < 1:
        raise ValueError("environment.max_steps must be positive")
    if obstacle_count < 0 or obstacle_count > (size * size - 3):
        raise ValueError("environment.obstacle_count out of range")
    if toy_count < 0 or toy_count > (size * size - obstacle_count - 2):
        raise ValueError("environment.toy_count out of range")

    condition_cfgs = config.get("conditions", [])
    if condition_cfgs is None:
        return
    if not isinstance(condition_cfgs, list):
        raise ValueError("conditions must be a list")
    names: set[str] = set()
    for item in condition_cfgs:
        if not isinstance(item, dict):
            raise ValueError("each condition must be an object")
        name = str(item.get("name", ""))
        if not name:
            raise ValueError("condition.name is required")
        if name in names:
            raise ValueError(f"duplicate condition.name: {name}")
        names.add(name)
        if item.get("encoder_mode") not in {"raw", "coarse"}:
            raise ValueError(f"invalid encoder_mode for {name}")
        intrinsic_mode = str(item.get("intrinsic_mode", "none"))
        if intrinsic_mode not in {"none", "surprise", "progress"}:
            raise ValueError(f"invalid intrinsic_mode for {name}")
        intrinsic_schedule = str(item.get("intrinsic_schedule", "constant"))
        if intrinsic_schedule not in {"constant", "linear_anneal"}:
            raise ValueError(f"invalid intrinsic_schedule for {name}")
        intrinsic_gate = str(item.get("intrinsic_gate", "none"))
        if intrinsic_gate not in {"none", "external_flat"}:
            raise ValueError(f"invalid intrinsic_gate for {name}")
        intrinsic_target = str(item.get("intrinsic_target", "reward"))
        if intrinsic_target not in {"reward", "auxiliary"}:
            raise ValueError(f"invalid intrinsic_target for {name}")
        episodes = int(item.get("episodes", 0))
        delay = int(item.get("decoder_delay_episodes", 0))
        beta = float(item.get("intrinsic_beta", 0.0))
        beta_end = float(item.get("intrinsic_beta_end", beta))
        anneal_episodes = int(item.get("intrinsic_anneal_episodes", 0))
        if episodes < 1:
            raise ValueError(f"episodes must be positive for {name}")
        if delay < 0 or delay > episodes:
            raise ValueError(f"decoder_delay_episodes out of range for {name}")
        if beta < 0 or beta_end < 0:
            raise ValueError(f"intrinsic_beta must be non-negative for {name}")
        if anneal_episodes < 0 or anneal_episodes > episodes:
            raise ValueError(f"intrinsic_anneal_episodes out of range for {name}")


def _intrinsic_signal(
    mode: str,
    transition: TransitionSurprise,
    feature: tuple[int, ...],
    action: int,
    next_feature: tuple[int, ...],
) -> float:
    if mode == "none":
        return 0.0
    if mode == "surprise":
        return transition.surprise(feature, action, next_feature)
    if mode == "progress":
        return transition.learning_progress(feature, action, next_feature)
    raise ValueError(f"unknown intrinsic mode: {mode}")


def _intrinsic_reward(condition: Condition, episode: int, external_reward: float, intrinsic_signal: float) -> float:
    intrinsic_beta = _effective_intrinsic_beta(condition, episode)
    if condition.intrinsic_gate == "external_flat" and external_reward > 0.0:
        intrinsic_beta = 0.0
    return intrinsic_beta * intrinsic_signal


def _q_target_reward(condition: Condition, external_reward: float, intrinsic_reward: float) -> float:
    if condition.intrinsic_target == "auxiliary":
        return external_reward
    return external_reward + intrinsic_reward


def _effective_intrinsic_beta(condition: Condition, episode: int) -> float:
    if condition.intrinsic_schedule == "constant":
        return condition.intrinsic_beta
    if condition.intrinsic_schedule == "linear_anneal":
        beta_end = condition.intrinsic_beta if condition.intrinsic_beta_end is None else condition.intrinsic_beta_end
        horizon = condition.intrinsic_anneal_episodes
        if horizon <= 0:
            horizon = max(1, condition.episodes - condition.decoder_delay_episodes)
        progress_episode = max(0, episode - condition.decoder_delay_episodes)
        fraction = min(1.0, progress_episode / horizon)
        return condition.intrinsic_beta + (beta_end - condition.intrinsic_beta) * fraction
    raise ValueError(f"unknown intrinsic_schedule: {condition.intrinsic_schedule}")


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# baby-model run summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- winner_last_window: `{report['winner_last_window']}`",
        "",
        "| condition | success_all | success_last | mean_steps_success | intrinsic_last |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        mean_steps = row["mean_steps_success"]
        mean_steps_text = "" if mean_steps is None else f"{mean_steps:.2f}"
        lines.append(
            "| {name} | {all:.3f} | {last:.3f} | {steps} | {intrinsic:.3f} |".format(
                name=row["name"],
                all=row["success_rate_all"],
                last=row["success_rate_last_window"],
                steps=mean_steps_text,
                intrinsic=row["mean_intrinsic_return_last_window"],
            )
        )
    lines.append("")
    return "\n".join(lines)
