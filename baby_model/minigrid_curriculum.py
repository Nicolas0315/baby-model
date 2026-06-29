from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from baby_model.agents import EpisodeMetrics, QAgent, TransitionSurprise
from baby_model.experiment import Condition
from baby_model.minigrid_experiment import _env_call, _intrinsic_signal, encode_observation
from baby_model.minigrid_probe import observation_schema


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    env_id: str
    max_steps: int
    episodes: int


@dataclass(frozen=True)
class CurriculumCondition:
    condition: Condition
    active_stages: tuple[str, ...]


@dataclass(frozen=True)
class MiniGridCurriculumConfig:
    stages: tuple[CurriculumStage, ...]
    quiet_env_output: bool
    conditions: tuple[CurriculumCondition, ...]


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-minigrid-curriculum")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/minigrid-curriculum-unlock.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minigrid-curriculum"))
    parser.add_argument("--seed", type=int, default=301)
    args = parser.parse_args()

    try:
        report = run_minigrid_curriculum_suite(json.loads(args.config.read_text(encoding="utf-8")), seed=args.seed)
    except ImportError as exc:
        print(f"missing optional dependency: {exc}")
        print("install with: python3 -m pip install minigrid")
        return 2

    run_dir = write_minigrid_curriculum_run(report, args.output_dir)
    print(f"minigrid_curriculum_run_dir={run_dir}")
    print(f"winner_final_last_window={report['winner_final_last_window']}")
    return 0


def run_minigrid_curriculum_suite(config: dict[str, Any], seed: int = 301) -> dict[str, Any]:
    parsed = parse_minigrid_curriculum_config(config=config, seed=seed)
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
    except ImportError as exc:
        raise ImportError("gymnasium/minigrid") from exc

    results = [
        run_minigrid_curriculum_condition(
            gym=gym,
            stages=parsed.stages,
            condition=condition,
            quiet_env_output=parsed.quiet_env_output,
        )
        for condition in parsed.conditions
    ]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": str(config.get("hypothesis", "Baby-AD/DA MiniGrid curriculum")),
        "stages": [
            {
                "name": stage.name,
                "env_id": stage.env_id,
                "max_steps": stage.max_steps,
                "episodes": stage.episodes,
            }
            for stage in parsed.stages
        ],
        "results": results,
        "winner_final_last_window": max(results, key=lambda row: row["final_stage"]["success_rate_last_window"])[
            "name"
        ],
    }


def parse_minigrid_curriculum_config(config: dict[str, Any], seed: int = 301) -> MiniGridCurriculumConfig:
    quiet_env_output = bool(config.get("quiet_env_output", True))

    stage_cfgs = config.get("stages", [])
    if not isinstance(stage_cfgs, list) or not stage_cfgs:
        raise ValueError("stages must be a non-empty list")
    stages: list[CurriculumStage] = []
    stage_names: set[str] = set()
    for item in stage_cfgs:
        if not isinstance(item, dict):
            raise ValueError("each stage must be an object")
        name = str(item.get("name", ""))
        if not name:
            raise ValueError("stage.name is required")
        if name in stage_names:
            raise ValueError(f"duplicate stage.name: {name}")
        stage_names.add(name)
        env_id = str(item.get("env_id", ""))
        max_steps = int(item.get("max_steps", 0))
        episodes = int(item.get("episodes", 0))
        if not env_id:
            raise ValueError(f"stage.env_id is required for {name}")
        if max_steps < 1:
            raise ValueError(f"stage.max_steps must be positive for {name}")
        if episodes < 1:
            raise ValueError(f"stage.episodes must be positive for {name}")
        stages.append(CurriculumStage(name=name, env_id=env_id, max_steps=max_steps, episodes=episodes))

    condition_cfgs = config.get("conditions", [])
    if not isinstance(condition_cfgs, list) or not condition_cfgs:
        raise ValueError("conditions must be a non-empty list")
    names: set[str] = set()
    conditions: list[CurriculumCondition] = []
    all_stage_names = tuple(stage.name for stage in stages)
    for i, item in enumerate(condition_cfgs):
        if not isinstance(item, dict):
            raise ValueError("each condition must be an object")
        name = str(item.get("name", ""))
        if not name:
            raise ValueError("condition.name is required")
        if name in names:
            raise ValueError(f"duplicate condition.name: {name}")
        names.add(name)
        active_stages = tuple(str(stage_name) for stage_name in item.get("active_stages", all_stage_names))
        if not active_stages:
            raise ValueError(f"active_stages must be non-empty for {name}")
        unknown_stages = sorted(set(active_stages) - stage_names)
        if unknown_stages:
            raise ValueError(f"unknown active_stages for {name}: {','.join(unknown_stages)}")
        encoder_mode = str(item.get("encoder_mode", "coarse"))
        if encoder_mode not in {"raw", "coarse"}:
            raise ValueError(f"invalid encoder_mode for {name}")
        intrinsic_mode = str(item.get("intrinsic_mode", "none"))
        if intrinsic_mode not in {"none", "surprise", "progress"}:
            raise ValueError(f"invalid intrinsic_mode for {name}")
        intrinsic_target = str(item.get("intrinsic_target", "reward"))
        if intrinsic_target not in {"reward", "auxiliary"}:
            raise ValueError(f"invalid intrinsic_target for {name}")
        episodes = sum(stage.episodes for stage in stages if stage.name in active_stages)
        delay = int(item.get("decoder_delay_episodes", 0))
        beta = float(item.get("intrinsic_beta", 0.0))
        if delay < 0 or delay > episodes:
            raise ValueError(f"decoder_delay_episodes out of range for {name}")
        if beta < 0:
            raise ValueError(f"intrinsic_beta must be non-negative for {name}")
        conditions.append(
            CurriculumCondition(
                condition=Condition(
                    name=name,
                    encoder_mode=encoder_mode,
                    episodes=episodes,
                    decoder_delay_episodes=delay,
                    intrinsic_beta=beta,
                    intrinsic_mode=intrinsic_mode,
                    seed=seed + i,
                    intrinsic_target=intrinsic_target,
                ),
                active_stages=active_stages,
            )
        )
    return MiniGridCurriculumConfig(
        stages=tuple(stages),
        quiet_env_output=quiet_env_output,
        conditions=tuple(conditions),
    )


def run_minigrid_curriculum_condition(
    gym: Any,
    stages: tuple[CurriculumStage, ...],
    condition: CurriculumCondition,
    quiet_env_output: bool = True,
) -> dict[str, Any]:
    active_stage_names = set(condition.active_stages)
    agent: QAgent | None = None
    auxiliary_agent: QAgent | None = None
    transition = TransitionSurprise()
    stage_results: list[dict[str, Any]] = []
    global_episode = 0

    for stage_index, stage in enumerate(stages):
        if stage.name not in active_stage_names:
            continue
        stage_result, agent, auxiliary_agent, global_episode = _run_curriculum_stage(
            gym=gym,
            stage=stage,
            condition=condition.condition,
            stage_index=stage_index,
            global_episode_start=global_episode,
            agent=agent,
            auxiliary_agent=auxiliary_agent,
            transition=transition,
            quiet_env_output=quiet_env_output,
        )
        stage_results.append(stage_result)

    final_stage = stage_results[-1]
    return {
        "name": condition.condition.name,
        "encoder_mode": condition.condition.encoder_mode,
        "episodes": condition.condition.episodes,
        "decoder_delay_episodes": condition.condition.decoder_delay_episodes,
        "intrinsic_beta": condition.condition.intrinsic_beta,
        "intrinsic_mode": condition.condition.intrinsic_mode,
        "intrinsic_target": condition.condition.intrinsic_target,
        "seed": condition.condition.seed,
        "active_stages": list(condition.active_stages),
        "stage_results": stage_results,
        "final_stage": final_stage,
    }


def _run_curriculum_stage(
    gym: Any,
    stage: CurriculumStage,
    condition: Condition,
    stage_index: int,
    global_episode_start: int,
    agent: QAgent | None,
    auxiliary_agent: QAgent | None,
    transition: TransitionSurprise,
    quiet_env_output: bool,
) -> tuple[dict[str, Any], QAgent, QAgent, int]:
    env = gym.make(stage.env_id)
    try:
        actions = int(env.action_space.n)
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(condition.seed + stage_index)
        if agent is None:
            agent = QAgent(actions=actions, seed=condition.seed)
        if auxiliary_agent is None:
            auxiliary_agent = QAgent(actions=actions, seed=condition.seed + 100_003, epsilon=0.0)
        if agent.actions != actions or auxiliary_agent.actions != actions:
            raise ValueError(f"action-space mismatch in stage {stage.name}")

        episodes: list[EpisodeMetrics] = []
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
            feature = encode_observation(observation, condition.encoder_mode)
            visited = {feature}
            external_return = 0.0
            intrinsic_return = 0.0
            success = False
            steps = 0

            for _ in range(stage.max_steps):
                force_random = global_episode < condition.decoder_delay_episodes
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

        return _stage_summary(stage, condition, episodes, first_schema), agent, auxiliary_agent, global_episode_start + stage.episodes
    finally:
        env.close()


def _stage_summary(
    stage: CurriculumStage,
    condition: Condition,
    episodes: list[EpisodeMetrics],
    first_schema: dict[str, Any] | None,
) -> dict[str, Any]:
    last_window = episodes[-20:] if len(episodes) >= 20 else episodes
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
    }


def write_minigrid_curriculum_run(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(curriculum_summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def curriculum_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# baby-model MiniGrid curriculum summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- winner_final_last_window: `{report['winner_final_last_window']}`",
        "",
        "| condition | active_stages | final_stage | success_all | success_last | return_last | mean_steps_success |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        final_stage = row["final_stage"]
        mean_steps = final_stage["mean_steps_success"]
        lines.append(
            "| {name} | {stages} | {stage} | {all:.3f} | {last:.3f} | {ret:.3f} | {steps} |".format(
                name=row["name"],
                stages=",".join(row["active_stages"]),
                stage=final_stage["stage"],
                all=final_stage["success_rate_all"],
                last=final_stage["success_rate_last_window"],
                ret=final_stage["mean_return_last_window"],
                steps="" if mean_steps is None else f"{mean_steps:.2f}",
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
