from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from baby_model.minigrid_torch import run_minigrid_torch_suite
from baby_model.sweep import parse_seeds


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-minigrid-torch-sweep")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/minigrid-torch-unlock-smoke.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minigrid-torch-sweeps"))
    parser.add_argument("--seeds", default="601,602,603")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    try:
        config = json.loads(args.config.read_text(encoding="utf-8"))
        if args.device is not None:
            config.setdefault("agent", {})["device"] = args.device
        report = run_minigrid_torch_sweep(config, seeds=parse_seeds(args.seeds))
    except ImportError as exc:
        print(f"missing optional dependency: {exc}")
        print("install with: python3 -m pip install minigrid torch")
        return 2

    run_dir = write_minigrid_torch_sweep(report, args.output_dir)
    print(f"minigrid_torch_sweep_dir={run_dir}")
    print(f"winner_by_mean_success_last_window={report['winner_by_mean_success_last_window']}")
    return 0


def run_minigrid_torch_sweep(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    runs = [run_minigrid_torch_suite(config, seed=seed) for seed in seeds]
    aggregate = aggregate_torch_reports(runs=runs, seeds=seeds)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": str(config.get("hypothesis", "Baby-AD/DA MiniGrid PyTorch DQN sweep")),
        "seeds": seeds,
        "runs": runs,
        "frameworks": [run["framework"] for run in runs],
        "aggregate": aggregate,
        "winner_by_mean_success_last_window": max(
            aggregate,
            key=lambda row: (row["mean_success_rate_last_window"], row["win_count"]),
        )["name"],
    }


def aggregate_torch_reports(runs: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
    if len(runs) != len(seeds):
        raise ValueError("runs and seeds length mismatch")
    by_name: dict[str, list[dict[str, Any]]] = defaultdict(list)
    wins: defaultdict[str, int] = defaultdict(int)
    for run in runs:
        wins[str(run["winner_last_window"])] += 1
        for row in run["results"]:
            by_name[str(row["name"])].append(row)

    aggregate: list[dict[str, Any]] = []
    for name in sorted(by_name):
        rows = by_name[name]
        success_last = [float(row["success_rate_last_window"]) for row in rows]
        success_all = [float(row["success_rate_all"]) for row in rows]
        returns = [float(row["mean_return_last_window"]) for row in rows]
        updates = [int(row["updates"]) for row in rows]
        parameters = [int(row["parameter_count"]) for row in rows]
        aggregate.append(
            {
                "name": name,
                "seeds": seeds,
                "condition_seeds": [int(row["seed"]) for row in rows],
                "win_count": wins[name],
                "mean_success_rate_all": mean(success_all),
                "mean_success_rate_last_window": mean(success_last),
                "median_success_rate_last_window": median(success_last),
                "mean_return_last_window": mean(returns),
                "mean_updates": mean(updates),
                "mean_parameter_count": mean(parameters),
            }
        )
    return aggregate


def write_minigrid_torch_sweep(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(torch_sweep_summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def torch_sweep_summary_markdown(report: dict[str, Any]) -> str:
    devices = sorted({str(framework.get("device", "unknown")) for framework in report["frameworks"]})
    versions = sorted({str(framework.get("version", "unknown")) for framework in report["frameworks"]})
    lines = [
        "# baby-model MiniGrid PyTorch sweep summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- seeds: `{','.join(str(seed) for seed in report['seeds'])}`",
        f"- torch_versions: `{','.join(versions)}`",
        f"- devices: `{','.join(devices)}`",
        f"- winner_by_mean_success_last_window: `{report['winner_by_mean_success_last_window']}`",
        "",
        "| condition | wins | mean_success_all | mean_success_last | median_success_last | mean_return_last | mean_updates | parameters |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["aggregate"]:
        lines.append(
            "| {name} | {wins} | {all:.3f} | {last:.3f} | {median:.3f} | {ret:.3f} | {updates:.1f} | {params:.0f} |".format(
                name=row["name"],
                wins=row["win_count"],
                all=row["mean_success_rate_all"],
                last=row["mean_success_rate_last_window"],
                median=row["median_success_rate_last_window"],
                ret=row["mean_return_last_window"],
                updates=row["mean_updates"],
                params=row["mean_parameter_count"],
            )
        )
    lines.append("")
    lines.extend(["## Per-Seed Winners", ""])
    for seed, run in zip(report["seeds"], report["runs"], strict=True):
        framework = run["framework"]
        lines.append(
            f"- `{seed}`: `{run['winner_last_window']}` on `{framework['device']}` with torch `{framework['version']}`"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
