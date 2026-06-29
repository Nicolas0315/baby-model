from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

from baby_model.minigrid_linear import run_minigrid_linear_suite
from baby_model.sweep import parse_seeds


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-minigrid-linear-sweep")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/minigrid-linear-unlock.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minigrid-linear-sweeps"))
    parser.add_argument("--seeds", default="401,402,403")
    args = parser.parse_args()

    try:
        report = run_minigrid_linear_sweep(
            json.loads(args.config.read_text(encoding="utf-8")),
            seeds=parse_seeds(args.seeds),
        )
    except ImportError as exc:
        print(f"missing optional dependency: {exc}")
        print("install with: python3 -m pip install minigrid")
        return 2

    run_dir = write_minigrid_linear_sweep(report, args.output_dir)
    print(f"minigrid_linear_sweep_dir={run_dir}")
    print(f"winner_by_mean_success_last_window={report['winner_by_mean_success_last_window']}")
    return 0


def run_minigrid_linear_sweep(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    runs = [run_minigrid_linear_suite(config, seed=seed) for seed in seeds]
    aggregate = aggregate_linear_reports(runs=runs, seeds=seeds)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": str(config.get("hypothesis", "Baby-AD/DA MiniGrid linear function approximation sweep")),
        "seeds": seeds,
        "runs": runs,
        "aggregate": aggregate,
        "winner_by_mean_success_last_window": max(
            aggregate,
            key=lambda row: (row["mean_success_rate_last_window"], row["win_count"]),
        )["name"],
    }


def aggregate_linear_reports(runs: list[dict[str, Any]], seeds: list[int]) -> list[dict[str, Any]]:
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
        weights = [int(row["nonzero_weights"]) for row in rows]
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
                "mean_nonzero_weights": mean(weights),
            }
        )
    return aggregate


def write_minigrid_linear_sweep(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(linear_sweep_summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def linear_sweep_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# baby-model MiniGrid linear sweep summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- seeds: `{','.join(str(seed) for seed in report['seeds'])}`",
        f"- winner_by_mean_success_last_window: `{report['winner_by_mean_success_last_window']}`",
        "",
        "| condition | wins | mean_success_all | mean_success_last | median_success_last | mean_return_last | mean_nonzero_weights |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["aggregate"]:
        lines.append(
            "| {name} | {wins} | {all:.3f} | {last:.3f} | {median:.3f} | {ret:.3f} | {weights:.1f} |".format(
                name=row["name"],
                wins=row["win_count"],
                all=row["mean_success_rate_all"],
                last=row["mean_success_rate_last_window"],
                median=row["median_success_rate_last_window"],
                ret=row["mean_return_last_window"],
                weights=row["mean_nonzero_weights"],
            )
        )
    lines.append("")
    lines.extend(["## Per-Seed Winners", ""])
    for seed, run in zip(report["seeds"], report["runs"], strict=True):
        lines.append(f"- `{seed}`: `{run['winner_last_window']}`")
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
