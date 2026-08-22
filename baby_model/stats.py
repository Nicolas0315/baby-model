"""Seed-level uncertainty statistics for sweep reports.

Sweep aggregates previously reported only point estimates (mean, median, win
count). With five seeds and a 20-episode evaluation window, differences of
0.05-0.10 in success rate are inside the noise band, so a point-estimate
winner is not evidence. This module adds the two things needed to read a sweep
honestly:

- per-condition dispersion (sd, sem, bootstrap CI),
- a paired comparison against a baseline condition, using the fact that every
  condition is run on the same seed set.

Standard library only, deterministic (fixed bootstrap seed), and usable
offline on an existing ``metrics.json`` artifact:

    python3 -m baby_model.stats path/to/metrics.json
"""

from __future__ import annotations

import argparse
import json
import random
from itertools import product
from pathlib import Path
from statistics import mean, median, stdev
from typing import Any

BOOTSTRAP_SEED = 20260823
BOOTSTRAP_RESAMPLES = 10000
EXACT_SIGNFLIP_MAX_N = 20


def describe(values: list[float]) -> dict[str, float | int]:
    """Point estimate plus dispersion for one condition's per-seed values."""
    if not values:
        raise ValueError("values must be non-empty")
    n = len(values)
    sd = stdev(values) if n > 1 else 0.0
    lo, hi = bootstrap_ci(values)
    return {
        "n": n,
        "mean": mean(values),
        "median": median(values),
        "sd": sd,
        "sem": sd / (n**0.5) if n > 1 else 0.0,
        "ci95_low": lo,
        "ci95_high": hi,
    }


def bootstrap_ci(
    values: list[float],
    resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
) -> tuple[float, float]:
    """Percentile bootstrap CI for the mean. Degenerate for n == 1."""
    if not values:
        raise ValueError("values must be non-empty")
    if len(values) == 1:
        return values[0], values[0]
    rng = random.Random(seed)
    n = len(values)
    means = sorted(mean(rng.choices(values, k=n)) for _ in range(resamples))
    return _percentile(means, 0.025), _percentile(means, 0.975)


def signflip_p_value(diffs: list[float], seed: int = BOOTSTRAP_SEED) -> float:
    """Two-sided paired randomization test on per-seed differences.

    Conditions share the seed set, so the null "condition label carries no
    information" is exactly a random sign assignment per seed. For n <= 20 all
    2**n assignments are enumerated, giving an exact p-value; above that the
    null is sampled.
    """
    if not diffs:
        raise ValueError("diffs must be non-empty")
    observed = abs(mean(diffs))
    n = len(diffs)
    if n <= EXACT_SIGNFLIP_MAX_N:
        total = 0
        extreme = 0
        for signs in product((1.0, -1.0), repeat=n):
            total += 1
            if abs(mean(s * d for s, d in zip(signs, diffs, strict=True))) >= observed - 1e-12:
                extreme += 1
        return extreme / total
    rng = random.Random(seed)
    total = BOOTSTRAP_RESAMPLES
    extreme = sum(
        1
        for _ in range(total)
        if abs(mean(rng.choice((1.0, -1.0)) * d for d in diffs)) >= observed - 1e-12
    )
    # +1 smoothing so a sampled p-value is never reported as exactly zero.
    return (extreme + 1) / (total + 1)


def paired_comparison(treatment: list[float], baseline: list[float]) -> dict[str, float | int]:
    """Paired seed-level comparison of one condition against a baseline."""
    if len(treatment) != len(baseline):
        raise ValueError("treatment and baseline must be seed-aligned")
    diffs = [t - b for t, b in zip(treatment, baseline, strict=True)]
    lo, hi = bootstrap_ci(diffs)
    return {
        "n": len(diffs),
        "mean_diff": mean(diffs),
        "ci95_low": lo,
        "ci95_high": hi,
        "p_two_sided": signflip_p_value(diffs),
        "wins": sum(1 for d in diffs if d > 0.0),
        "losses": sum(1 for d in diffs if d < 0.0),
        "ties": sum(1 for d in diffs if d == 0.0),
    }


def per_seed_metric(report: dict[str, Any], metric: str) -> dict[str, list[float]]:
    """Extract seed-aligned per-condition values for one metric from a sweep report."""
    runs = report.get("runs")
    if not runs:
        raise ValueError("report has no runs")
    series: dict[str, list[float]] = {}
    for run in runs:
        for row in run["results"]:
            name = str(row["name"])
            if metric not in row:
                raise KeyError(f"metric {metric!r} missing for condition {name}")
            series.setdefault(name, []).append(float(row[metric]))
    lengths = {len(values) for values in series.values()}
    if len(lengths) != 1:
        raise ValueError("conditions are not seed-aligned across runs")
    return series


def baseline_condition_name(report: dict[str, Any]) -> str:
    """Baseline is the first condition declared in the config, i.e. the control."""
    return str(report["runs"][0]["results"][0]["name"])


def analyze_report(
    report: dict[str, Any],
    metrics: tuple[str, ...] = ("success_rate_last_window", "mean_return_last_window"),
    baseline: str | None = None,
) -> dict[str, Any]:
    base = baseline or baseline_condition_name(report)
    out: dict[str, Any] = {
        "seeds": list(report.get("seeds", [])),
        "baseline": base,
        "metrics": {},
    }
    for metric in metrics:
        series = per_seed_metric(report, metric)
        if base not in series:
            raise KeyError(f"baseline condition {base!r} not in report")
        out["metrics"][metric] = {
            name: {
                "per_seed": values,
                "summary": describe(values),
                "vs_baseline": None if name == base else paired_comparison(values, series[base]),
            }
            for name, values in sorted(series.items())
        }
    return out


def statistics_markdown(analysis: dict[str, Any]) -> str:
    lines: list[str] = ["## Seed-Level Statistics", ""]
    lines.append(f"- baseline: `{analysis['baseline']}`")
    lines.append(
        "- paired test: two-sided sign-flip randomization on per-seed differences "
        "(exact for n <= 20); CI: percentile bootstrap"
    )
    lines.append("")
    for metric, rows in analysis["metrics"].items():
        lines.append(f"### `{metric}`")
        lines.append("")
        lines.append("| condition | n | mean | sd | 95% CI | Δ vs baseline | Δ 95% CI | p | W/L/T |")
        lines.append("| --- | ---: | ---: | ---: | :---: | ---: | :---: | ---: | :---: |")
        for name, row in rows.items():
            s = row["summary"]
            cmp_ = row["vs_baseline"]
            if cmp_ is None:
                delta, delta_ci, pval, wlt = "—", "—", "—", "—"
            else:
                delta = f"{cmp_['mean_diff']:+.3f}"
                delta_ci = f"[{cmp_['ci95_low']:+.3f}, {cmp_['ci95_high']:+.3f}]"
                pval = f"{cmp_['p_two_sided']:.3f}"
                wlt = f"{cmp_['wins']}/{cmp_['losses']}/{cmp_['ties']}"
            lines.append(
                f"| `{name}` | {s['n']} | {s['mean']:.3f} | {s['sd']:.3f} | "
                f"[{s['ci95_low']:.3f}, {s['ci95_high']:.3f}] | {delta} | {delta_ci} | {pval} | {wlt} |"
            )
        lines.append("")
    return "\n".join(lines)


def _percentile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("sorted_values must be non-empty")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    low = int(position)
    high = min(low + 1, len(sorted_values) - 1)
    weight = position - low
    return sorted_values[low] * (1.0 - weight) + sorted_values[high] * weight


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-stats")
    parser.add_argument("metrics", type=Path, help="path to a sweep metrics.json artifact")
    parser.add_argument("--baseline", default=None, help="baseline condition name (default: first declared)")
    parser.add_argument(
        "--metrics-keys",
        default="success_rate_last_window,mean_return_last_window",
        help="comma-separated per-condition metric keys",
    )
    parser.add_argument("--json", action="store_true", help="emit JSON instead of markdown")
    args = parser.parse_args()

    report = json.loads(args.metrics.read_text(encoding="utf-8"))
    keys = tuple(key.strip() for key in args.metrics_keys.split(",") if key.strip())
    analysis = analyze_report(report, metrics=keys, baseline=args.baseline)
    if args.json:
        print(json.dumps(analysis, indent=2, sort_keys=True))
    else:
        print(statistics_markdown(analysis))
    return 0


def demo() -> None:
    """Self-check: a real effect is detected, pure noise is not."""
    baseline = [0.30, 0.35, 0.25, 0.30, 0.35]
    strong = [0.55, 0.60, 0.50, 0.55, 0.60]
    cmp_strong = paired_comparison(strong, baseline)
    assert cmp_strong["mean_diff"] > 0.2, cmp_strong
    assert cmp_strong["p_two_sided"] <= 0.0625, cmp_strong  # 2/32, the exact floor at n=5
    assert cmp_strong["ci95_low"] > 0.0, cmp_strong

    noisy = [0.35, 0.25, 0.30, 0.40, 0.20]
    cmp_noise = paired_comparison(noisy, baseline)
    assert cmp_noise["p_two_sided"] > 0.3, cmp_noise
    assert cmp_noise["ci95_low"] < 0.0 < cmp_noise["ci95_high"], cmp_noise

    # Identical series: no difference can be claimed.
    cmp_same = paired_comparison(baseline, baseline)
    assert cmp_same["mean_diff"] == 0.0
    assert cmp_same["p_two_sided"] == 1.0

    described = describe(baseline)
    assert described["n"] == 5
    assert described["ci95_low"] <= described["mean"] <= described["ci95_high"]
    print("stats demo ok")


if __name__ == "__main__":
    raise SystemExit(main())
