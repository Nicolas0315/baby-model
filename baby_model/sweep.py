from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from baby_model.experiment import run_suite, validate_config


def parse_seeds(seed_text: str) -> list[int]:
    seeds = [int(item.strip()) for item in seed_text.split(",") if item.strip()]
    if not seeds:
        raise ValueError("at least one seed is required")
    return seeds


def run_sweep(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    validate_config(config)
    reports = [{"seed": seed, "report": run_suite(config=config, seed=seed)} for seed in seeds]
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in reports:
        sweep_seed = item["seed"]
        for row in item["report"]["results"]:
            grouped[row["name"]].append(
                {
                    **row,
                    "condition_seed": row["seed"],
                    "sweep_seed": sweep_seed,
                }
            )

    aggregate = []
    for name, rows in sorted(grouped.items()):
        aggregate.append(
            {
                "name": name,
                "seeds": [row["sweep_seed"] for row in rows],
                "condition_seeds": [row["condition_seed"] for row in rows],
                "wins": sum(1 for item in reports if item["report"]["winner_last_window"] == name),
                "mean_success_rate_last_window": mean(row["success_rate_last_window"] for row in rows),
                "mean_success_rate_all": mean(row["success_rate_all"] for row in rows),
                "mean_external_return_last_window": mean(row["mean_external_return_last_window"] for row in rows),
                "mean_intrinsic_return_last_window": mean(row["mean_intrinsic_return_last_window"] for row in rows),
                "mean_steps_success": _mean_optional(row["mean_steps_success"] for row in rows),
            }
        )

    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": str(config.get("hypothesis", "Baby-AD/DA experiment sweep")),
        "seeds": seeds,
        "aggregate": aggregate,
        "reports": reports,
        "winner_by_mean_success_last_window": max(
            aggregate, key=lambda row: row["mean_success_rate_last_window"]
        )["name"],
    }


def write_sweep(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "sweep.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(_summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def _mean_optional(values: Any) -> float | None:
    present = [value for value in values if value is not None]
    if not present:
        return None
    return mean(present)


def _summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# baby-model sweep summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- seeds: `{','.join(str(seed) for seed in report['seeds'])}`",
        f"- winner_by_mean_success_last_window: `{report['winner_by_mean_success_last_window']}`",
        "",
        "| condition | wins | mean_success_last | mean_success_all | mean_steps_success | intrinsic_last |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in report["aggregate"]:
        mean_steps = row["mean_steps_success"]
        lines.append(
            "| {name} | {wins} | {last:.3f} | {all:.3f} | {steps} | {intrinsic:.3f} |".format(
                name=row["name"],
                wins=row["wins"],
                last=row["mean_success_rate_last_window"],
                all=row["mean_success_rate_all"],
                steps="" if mean_steps is None else f"{mean_steps:.2f}",
                intrinsic=row["mean_intrinsic_return_last_window"],
            )
        )
    lines.append("")
    return "\n".join(lines)
