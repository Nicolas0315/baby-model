from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any


DEFAULT_ENVS = (
    "MiniGrid-Empty-8x8-v0",
    "MiniGrid-DoorKey-8x8-v0",
    "BabyAI-GoToRedBall-v0",
)


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-minigrid-probe")
    parser.add_argument("--output-dir", type=Path, default=Path(".tmp/verify-minigrid"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--env", action="append", dest="envs", default=[])
    args = parser.parse_args()

    try:
        report = run_probe(
            env_ids=tuple(args.envs) if args.envs else DEFAULT_ENVS,
            episodes=args.episodes,
            max_steps=args.max_steps,
            seed=args.seed,
        )
    except ImportError as exc:
        print(f"missing optional dependency: {exc}")
        print("install with: python3 -m pip install minigrid")
        return 2

    run_dir = write_probe(report, args.output_dir)
    print(f"minigrid_probe_dir={run_dir}")
    print(f"envs={len(report['results'])}")
    return 0


def run_probe(env_ids: tuple[str, ...], episodes: int, max_steps: int, seed: int) -> dict[str, Any]:
    if episodes < 1:
        raise ValueError("episodes must be positive")
    if max_steps < 1:
        raise ValueError("max_steps must be positive")

    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
    except ImportError as exc:
        raise ImportError("gymnasium/minigrid") from exc

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": "Baby-AD/DA MiniGrid optional dependency probe",
        "episodes": episodes,
        "max_steps": max_steps,
        "seed": seed,
        "results": [_probe_env(gym, env_id, episodes, max_steps, seed) for env_id in env_ids],
    }


def write_probe(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "probe.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# baby-model MiniGrid probe summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- episodes: `{report['episodes']}`",
        f"- max_steps: `{report['max_steps']}`",
        f"- seed: `{report['seed']}`",
        "",
        "| env | action_n | obs_keys | success_rate | mean_return | mean_steps |",
        "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in report["results"]:
        lines.append(
            "| {env} | {action_n} | {obs_keys} | {success:.3f} | {ret:.3f} | {steps:.2f} |".format(
                env=row["env_id"],
                action_n=row["action_n"],
                obs_keys=",".join(row["observation_schema"].get("keys", [])),
                success=row["success_rate"],
                ret=row["mean_return"],
                steps=row["mean_steps"],
            )
        )
    lines.append("")
    return "\n".join(lines)


def _probe_env(gym: Any, env_id: str, episodes: int, max_steps: int, seed: int) -> dict[str, Any]:
    env = gym.make(env_id)
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(seed)
        returns: list[float] = []
        steps_list: list[int] = []
        successes = 0
        first_observation: Any | None = None
        for episode in range(episodes):
            observation, _info = env.reset(seed=seed + episode)
            if first_observation is None:
                first_observation = observation
            total_return = 0.0
            steps = 0
            for _ in range(max_steps):
                observation, reward, terminated, truncated, _info = env.step(env.action_space.sample())
                total_return += float(reward)
                steps += 1
                if float(reward) > 0.0:
                    successes += 1
                if terminated or truncated:
                    break
            returns.append(total_return)
            steps_list.append(steps)
        return {
            "env_id": env_id,
            "action_n": int(env.action_space.n),
            "observation_schema": observation_schema(first_observation),
            "success_rate": successes / episodes,
            "mean_return": mean(returns),
            "mean_steps": mean(steps_list),
        }
    finally:
        env.close()


def observation_schema(observation: Any) -> dict[str, Any]:
    if isinstance(observation, dict):
        return {
            "type": "dict",
            "keys": sorted(str(key) for key in observation.keys()),
            "fields": {str(key): _value_schema(value) for key, value in observation.items()},
        }
    return {"type": type(observation).__name__, "value": _value_schema(observation)}


def _value_schema(value: Any) -> dict[str, Any]:
    schema: dict[str, Any] = {"type": type(value).__name__}
    shape = getattr(value, "shape", None)
    if shape is not None:
        schema["shape"] = [int(item) for item in shape]
    dtype = getattr(value, "dtype", None)
    if dtype is not None:
        schema["dtype"] = str(dtype)
    return schema


if __name__ == "__main__":
    raise SystemExit(main())
