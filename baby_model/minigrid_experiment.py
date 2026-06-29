from __future__ import annotations

import argparse
import contextlib
import io
import json
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from baby_model.agents import EpisodeMetrics, QAgent, TransitionSurprise
from baby_model.experiment import Condition
from baby_model.minigrid_probe import observation_schema


@dataclass(frozen=True)
class MiniGridConfig:
    env_id: str
    max_steps: int
    quiet_env_output: bool
    conditions: tuple[Condition, ...]


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-minigrid-experiment")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/minigrid-smoke.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minigrid"))
    parser.add_argument("--seed", type=int, default=101)
    args = parser.parse_args()

    try:
        report = run_minigrid_suite(json.loads(args.config.read_text(encoding="utf-8")), seed=args.seed)
    except ImportError as exc:
        print(f"missing optional dependency: {exc}")
        print("install with: python3 -m pip install minigrid")
        return 2

    run_dir = write_minigrid_run(report, args.output_dir)
    print(f"minigrid_run_dir={run_dir}")
    print(f"winner_last_window={report['winner_last_window']}")
    return 0


def run_minigrid_suite(config: dict[str, Any], seed: int = 101) -> dict[str, Any]:
    parsed = parse_minigrid_config(config=config, seed=seed)
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
    except ImportError as exc:
        raise ImportError("gymnasium/minigrid") from exc

    results = [
        run_minigrid_condition(
            gym=gym,
            env_id=parsed.env_id,
            condition=condition,
            max_steps=parsed.max_steps,
            quiet_env_output=parsed.quiet_env_output,
        )
        for condition in parsed.conditions
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": str(config.get("hypothesis", "Baby-AD/DA MiniGrid smoke")),
        "env_id": parsed.env_id,
        "max_steps": parsed.max_steps,
        "results": results,
        "winner_last_window": max(results, key=lambda row: row["success_rate_last_window"])["name"],
    }


def parse_minigrid_config(config: dict[str, Any], seed: int = 101) -> MiniGridConfig:
    env_cfg = config.get("environment", {})
    if not isinstance(env_cfg, dict):
        raise ValueError("environment must be an object")
    env_id = str(env_cfg.get("id", "MiniGrid-Empty-8x8-v0"))
    max_steps = int(env_cfg.get("max_steps", 80))
    if max_steps < 1:
        raise ValueError("environment.max_steps must be positive")
    quiet_env_output = bool(env_cfg.get("quiet_env_output", True))

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
        encoder_mode = str(item.get("encoder_mode", "coarse"))
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
    return MiniGridConfig(
        env_id=env_id,
        max_steps=max_steps,
        quiet_env_output=quiet_env_output,
        conditions=tuple(conditions),
    )


def run_minigrid_condition(
    gym: Any,
    env_id: str,
    condition: Condition,
    max_steps: int,
    quiet_env_output: bool = True,
) -> dict[str, Any]:
    return _run_minigrid_condition(
        gym=gym,
        env_id=env_id,
        condition=condition,
        max_steps=max_steps,
        quiet_env_output=quiet_env_output,
    )


def _run_minigrid_condition(
    gym: Any,
    env_id: str,
    condition: Condition,
    max_steps: int,
    quiet_env_output: bool,
) -> dict[str, Any]:
    env = gym.make(env_id)
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(condition.seed)
        agent = QAgent(actions=int(env.action_space.n), seed=condition.seed)
        auxiliary_agent = QAgent(actions=int(env.action_space.n), seed=condition.seed + 100_003, epsilon=0.0)
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
            feature = encode_observation(observation, condition.encoder_mode)
            visited = {feature}
            external_return = 0.0
            intrinsic_return = 0.0
            success = False
            steps = 0

            for _ in range(max_steps):
                force_random = episode < condition.decoder_delay_episodes
                action_bonus = None
                if condition.intrinsic_target == "auxiliary" and not force_random:
                    action_bonus = auxiliary_agent.action_values(feature)
                action = agent.choose(feature, force_random=force_random, action_bonus=action_bonus)
                next_observation, reward, terminated, truncated, _info = _env_call(
                    env.step,
                    action,
                    quiet=quiet_env_output,
                )
                next_feature = encode_observation(next_observation, condition.encoder_mode)
                intrinsic_signal = _intrinsic_signal(condition.intrinsic_mode, transition, feature, action, next_feature)
                intrinsic = condition.intrinsic_beta * intrinsic_signal
                total_reward = float(reward) if condition.intrinsic_target == "auxiliary" else float(reward) + intrinsic
                done = bool(terminated or truncated)

                if not force_random:
                    agent.update(feature, action, total_reward, next_feature, done)
                    if condition.intrinsic_target == "auxiliary":
                        auxiliary_agent.update(feature, action, intrinsic, next_feature, done)
                transition.update(feature, action, next_feature)

                external_return += float(reward)
                intrinsic_return += intrinsic
                steps += 1
                visited.add(next_feature)
                feature = next_feature
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
        }
    finally:
        env.close()


def _env_call(function: Any, *args: Any, quiet: bool, **kwargs: Any) -> Any:
    if not quiet:
        return function(*args, **kwargs)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        return function(*args, **kwargs)


def write_minigrid_run(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# baby-model MiniGrid run summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- env_id: `{report['env_id']}`",
        f"- winner_last_window: `{report['winner_last_window']}`",
        "",
        "| condition | success_all | success_last | return_last | mean_steps_success | intrinsic_last |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        mean_steps = row["mean_steps_success"]
        lines.append(
            "| {name} | {all:.3f} | {last:.3f} | {ret:.3f} | {steps} | {intrinsic:.3f} |".format(
                name=row["name"],
                all=row["success_rate_all"],
                last=row["success_rate_last_window"],
                ret=row["mean_return_last_window"],
                steps="" if mean_steps is None else f"{mean_steps:.2f}",
                intrinsic=row["mean_intrinsic_return_last_window"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def encode_observation(observation: Any, mode: str) -> tuple[int, ...]:
    if mode not in {"raw", "coarse"}:
        raise ValueError(f"unknown encoder mode: {mode}")
    image = observation["image"]
    direction = int(observation["direction"])
    flat = tuple(int(item) for row in image.tolist() for cell in row for item in cell)
    if mode == "raw":
        return (direction, *flat)

    counts: dict[int, int] = defaultdict(int)
    for row in image.tolist():
        for cell in row:
            counts[int(cell[0])] += 1
    center = image.tolist()[3][3]
    forward = image.tolist()[2][3]
    return (
        direction,
        int(center[0]),
        int(center[1]),
        int(center[2]),
        int(forward[0]),
        int(forward[1]),
        int(forward[2]),
        min(counts.get(2, 0), 8),
        min(counts.get(8, 0), 8),
        min(counts.get(10, 0), 8),
    )


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


if __name__ == "__main__":
    raise SystemExit(main())
