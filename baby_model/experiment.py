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


def default_conditions(seed: int = 7) -> list[Condition]:
    return [
        Condition("A_end_to_end", "raw", 80, 0, 0.0, "none", seed),
        Condition("B_encoder_first", "coarse", 80, 20, 0.0, "none", seed + 1),
        Condition("C_baby_surprise", "coarse", 80, 20, 0.05, "surprise", seed + 2),
        Condition("D_baby_progress", "coarse", 80, 20, 0.2, "progress", seed + 3),
    ]


def run_condition(condition: Condition, size: int = 7, max_steps: int = 60) -> dict[str, Any]:
    env = BabyGrid(size=size, max_steps=max_steps, seed=condition.seed)
    encoder = FeatureEncoder(size=size, mode=condition.encoder_mode)
    agent = QAgent(seed=condition.seed)
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
            action = agent.choose(feature, force_random=force_random)
            result = env.step(action)
            next_feature = encoder.encode(result.observation)
            intrinsic_signal = _intrinsic_signal(condition.intrinsic_mode, transition, feature, action, next_feature)
            intrinsic = condition.intrinsic_beta * intrinsic_signal
            total_reward = result.reward + intrinsic

            if not force_random:
                agent.update(feature, action, total_reward, next_feature, result.done)
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
    if size < 4:
        raise ValueError("environment.size must be at least 4")
    if max_steps < 1:
        raise ValueError("environment.max_steps must be positive")

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
        episodes = int(item.get("episodes", 0))
        delay = int(item.get("decoder_delay_episodes", 0))
        beta = float(item.get("intrinsic_beta", 0.0))
        if episodes < 1:
            raise ValueError(f"episodes must be positive for {name}")
        if delay < 0 or delay > episodes:
            raise ValueError(f"decoder_delay_episodes out of range for {name}")
        if beta < 0:
            raise ValueError(f"intrinsic_beta must be non-negative for {name}")


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
