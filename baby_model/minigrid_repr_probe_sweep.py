from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from baby_model.minigrid_repr_probe import run_minigrid_representation_probe
from baby_model.sweep import parse_seeds


DEFAULT_CONFIG_PATH = Path("configs/experiments/minigrid-repr-probe-v35.json")


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-minigrid-repr-probe-sweep")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minigrid-repr-probe-sweeps"))
    parser.add_argument("--seeds", default="2901,2902,2903")
    args = parser.parse_args()

    try:
        report = run_minigrid_representation_probe_sweep(
            json.loads(args.config.read_text(encoding="utf-8")),
            seeds=parse_seeds(args.seeds),
        )
    except ImportError as exc:
        print(f"missing optional dependency: {exc}")
        print("install with: python3 -m pip install minigrid")
        return 2

    run_dir = write_representation_probe_sweep(report, args.output_dir)
    print(f"minigrid_repr_probe_sweep_dir={run_dir}")
    print(f"decision_met={str(report['decision']['met']).lower()}")
    print(f"mean_transition_lift_delta={report['aggregate']['mean_transition_lift_delta']:.6f}")
    return 0


def run_minigrid_representation_probe_sweep(config: dict[str, Any], seeds: list[int]) -> dict[str, Any]:
    if not seeds:
        raise ValueError("seeds must be non-empty")
    runs = [run_minigrid_representation_probe(config, seed=seed) for seed in seeds]
    aggregate = aggregate_representation_probe_runs(runs=runs, seeds=seeds)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": str(config.get("hypothesis", "Baby-AD/DA MiniGrid representation probe sweep")),
        "seeds": seeds,
        "runs": runs,
        "aggregate": aggregate,
        "decision": evaluate_representation_probe_sweep_decision(aggregate),
    }


def aggregate_representation_probe_runs(runs: list[dict[str, Any]], seeds: list[int]) -> dict[str, Any]:
    if len(runs) != len(seeds):
        raise ValueError("runs and seeds length mismatch")
    if not runs:
        raise ValueError("runs must be non-empty")

    per_seed: list[dict[str, Any]] = []
    for seed, run in zip(seeds, runs, strict=True):
        if "seed" in run and int(run["seed"]) != seed:
            raise ValueError(f"run seed mismatch: expected {seed}, got {run['seed']}")
        decision = run["decision"]
        transition_label = str(decision["rule"]["transition_label"])
        comparisons = decision["comparisons"]
        transition = comparisons[transition_label]
        mission_object = comparisons["mission_object"]
        mission_color = comparisons["mission_color"]
        per_seed.append(
            {
                "seed": seed,
                "decision_met": bool(decision["met"]),
                "transition_label": transition_label,
                "transition_lift_delta": float(transition["lift_delta"]),
                "transition_baseline_lift": float(transition["baseline_lift"]),
                "transition_candidate_lift": float(transition["candidate_lift"]),
                "mission_object_accuracy_delta": float(mission_object["accuracy_delta"]),
                "mission_color_accuracy_delta": float(mission_color["accuracy_delta"]),
                "transition_test_examples": int(transition["candidate_test_examples"]),
            }
        )

    transition_deltas = [row["transition_lift_delta"] for row in per_seed]
    object_deltas = [row["mission_object_accuracy_delta"] for row in per_seed]
    color_deltas = [row["mission_color_accuracy_delta"] for row in per_seed]
    test_examples = [row["transition_test_examples"] for row in per_seed]
    nonnegative_count = sum(1 for value in transition_deltas if value >= 0.0)
    return {
        "transition_label": per_seed[0]["transition_label"],
        "seed_count": len(per_seed),
        "per_seed": per_seed,
        "mean_transition_lift_delta": mean(transition_deltas),
        "nonnegative_transition_lift_delta_count": nonnegative_count,
        "min_mission_object_accuracy_delta": min(object_deltas),
        "min_mission_color_accuracy_delta": min(color_deltas),
        "mean_mission_object_accuracy_delta": mean(object_deltas),
        "mean_mission_color_accuracy_delta": mean(color_deltas),
        "min_transition_test_examples": min(test_examples),
    }


def evaluate_representation_probe_sweep_decision(aggregate: dict[str, Any]) -> dict[str, Any]:
    mean_transition_passed = aggregate["mean_transition_lift_delta"] >= 0.010
    mission_object_passed = aggregate["min_mission_object_accuracy_delta"] >= -0.050
    mission_color_passed = aggregate["min_mission_color_accuracy_delta"] >= -0.050
    seed_count = int(aggregate["seed_count"])
    nonnegative_required = (seed_count // 2) + 1
    nonnegative_passed = aggregate["nonnegative_transition_lift_delta_count"] >= nonnegative_required
    examples_passed = aggregate["min_transition_test_examples"] >= 10
    met = bool(
        mean_transition_passed
        and mission_object_passed
        and mission_color_passed
        and nonnegative_passed
        and examples_passed
    )
    return {
        "met": met,
        "rule": {
            "mean_transition_lift_delta_min": 0.010,
            "min_mission_accuracy_delta": -0.050,
            "nonnegative_transition_lift_delta_min_count": nonnegative_required,
            "min_transition_test_examples": 10,
        },
        "checks": {
            "mean_transition_passed": mean_transition_passed,
            "mission_object_passed": mission_object_passed,
            "mission_color_passed": mission_color_passed,
            "nonnegative_passed": nonnegative_passed,
            "examples_passed": examples_passed,
        },
    }


def write_representation_probe_sweep(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(representation_probe_sweep_summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def representation_probe_sweep_summary_markdown(report: dict[str, Any]) -> str:
    aggregate = report["aggregate"]
    decision = report["decision"]
    lines = [
        "# baby-model MiniGrid representation probe sweep summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- seeds: `{','.join(str(seed) for seed in report['seeds'])}`",
        f"- transition_label: `{aggregate['transition_label']}`",
        f"- decision_met: `{str(decision['met']).lower()}`",
        f"- mean_transition_lift_delta: `{aggregate['mean_transition_lift_delta']:.6f}`",
        f"- nonnegative_transition_lift_delta_count: `{aggregate['nonnegative_transition_lift_delta_count']}`",
        f"- min_mission_object_accuracy_delta: `{aggregate['min_mission_object_accuracy_delta']:.6f}`",
        f"- min_mission_color_accuracy_delta: `{aggregate['min_mission_color_accuracy_delta']:.6f}`",
        f"- min_transition_test_examples: `{aggregate['min_transition_test_examples']}`",
        "",
        "| seed | decision_met | baseline_lift | candidate_lift | lift_delta | object_delta | color_delta | test |",
        "| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in aggregate["per_seed"]:
        lines.append(
            "| {seed} | {met} | {base:.3f} | {candidate:.3f} | {delta:.3f} | {object_delta:.3f} | {color_delta:.3f} | {test} |".format(
                seed=row["seed"],
                met=str(row["decision_met"]).lower(),
                base=row["transition_baseline_lift"],
                candidate=row["transition_candidate_lift"],
                delta=row["transition_lift_delta"],
                object_delta=row["mission_object_accuracy_delta"],
                color_delta=row["mission_color_accuracy_delta"],
                test=row["transition_test_examples"],
            )
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
