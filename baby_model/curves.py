"""Learning-curve analysis over an `episodes.jsonl` log.

Before this existed, the only trend information an artifact carried was
`success_rate_all` versus `success_rate_last_window` — two points per stage — so
"did it converge?", "when did it start learning?", and "did it collapse?" were
unanswerable after the fact. The 2026-08-23 audit listed that as item C1.

Standard library only, so it runs in the default lane:

    python3 -m baby_model.curves path/to/run_dir
    python3 -m baby_model.curves path/to/episodes.jsonl --bins 16 --svg curves.svg
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

# Trend verdicts. The distinctions matter because the final window alone cannot
# tell them apart: a condition that gained and lost it (collapsed), one that only
# ever got worse (declined), and one that never moved (flat) can all land on the
# same final number.
IMPROVED = "improved"
FLAT = "flat"
COLLAPSED = "collapsed"
DECLINED = "declined"
NOISE_BAND = 0.05


def load_episodes(path: Path) -> list[dict[str, Any]]:
    """Read an episodes.jsonl, or find one inside a run directory."""
    if path.is_dir():
        candidate = path / "episodes.jsonl"
        if not candidate.exists():
            candidate = path / "latest" / "episodes.jsonl"
        path = candidate
    if not path.exists():
        raise FileNotFoundError(f"no episode log at {path}")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def binned_curve(rows: list[dict[str, Any]], bins: int, key: str = "success") -> list[float]:
    """Mean of `key` over `bins` equal slices of the episode sequence.

    Binned **per seed** and then averaged bin-wise. Binning the concatenated
    sequence instead makes every bin span parts of different seeds, so the result
    is a seed-ordering artefact rather than a learning curve. That produced a
    visibly wrong verdict on an 8-seed log before this was fixed.
    """
    if bins < 1:
        raise ValueError("bins must be positive")
    if not rows:
        return []
    by_seed: dict[int, list[float]] = defaultdict(list)
    for row in sorted(rows, key=lambda r: (int(r["seed"]), int(r["episode"]))):
        by_seed[int(row["seed"])].append(float(row[key]))
    per_seed_curves: list[list[float]] = []
    for values in by_seed.values():
        edges = [round(index * len(values) / bins) for index in range(bins + 1)]
        per_seed_curves.append(
            [mean(values[a:b]) if b > a else 0.0 for a, b in zip(edges[:-1], edges[1:], strict=True)]
        )
    return [mean(curve[i] for curve in per_seed_curves) for i in range(bins)]


def classify(curve: list[float], noise_band: float = NOISE_BAND) -> str:
    """Improved, flat, or collapsed, judged against the run's own peak.

    A drop from the peak that exceeds the noise band is reported as a collapse
    even when the end still sits above the start, because a condition that peaks
    and falls back is not the same result as one that plateaus.
    """
    if len(curve) < 2:
        return FLAT
    # Compare quarter means, not single bins. A bin of ~67 episodes has a
    # binomial SE near 0.05 at p=0.8, which is the whole noise band, so peak-bin
    # comparisons flag any end-of-run wobble as a collapse. This was a real false
    # positive on the first live run.
    size = max(1, len(curve) // 4)
    quarters = [mean(curve[i : i + size]) for i in range(0, len(curve) - size + 1, size)] or [mean(curve)]
    first_quarter, last_quarter = quarters[0], quarters[-1]
    peak_quarter = max(quarters)
    if peak_quarter - last_quarter > noise_band and peak_quarter > first_quarter + noise_band:
        return COLLAPSED
    if last_quarter - first_quarter > noise_band:
        return IMPROVED
    if first_quarter - last_quarter > noise_band:
        # Got worse without ever gaining, which is not the same result as
        # gaining and losing it, and is not "flat" either.
        return DECLINED
    return FLAT


def first_crossing(curve: list[float], threshold: float) -> int | None:
    """Index of the first bin at or above `threshold`, or None."""
    for index, value in enumerate(curve):
        if value >= threshold:
            return index
    return None


def analyze(rows: list[dict[str, Any]], bins: int = 12, floor: float | None = None) -> dict[str, Any]:
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)

    stages: list[str] = []
    for row in rows:
        if row["stage"] not in stages:
            stages.append(str(row["stage"]))

    out: dict[str, Any] = {"bins": bins, "floor": floor, "stages": stages, "conditions": {}}
    for name in sorted(by_condition):
        condition_rows = by_condition[name]
        # The evaluation stage is the last one declared; per-stage curves keep a
        # warmup improvement from being read as an evaluation improvement.
        eval_rows = [row for row in condition_rows if row["stage"] == stages[-1]] if stages else []
        curve = binned_curve(eval_rows or condition_rows, bins)
        entry: dict[str, Any] = {
            "episodes": len(condition_rows),
            "eval_episodes": len(eval_rows),
            "curve": curve,
            "verdict": classify(curve),
            "peak": max(curve) if curve else 0.0,
            "final_bin": curve[-1] if curve else 0.0,
            "mean_steps": mean(float(row["steps"]) for row in (eval_rows or condition_rows)),
            "per_stage": {
                stage: mean(float(row["success"]) for row in condition_rows if row["stage"] == stage)
                for stage in stages
                if any(row["stage"] == stage for row in condition_rows)
            },
        }
        if floor is not None:
            entry["bins_above_floor"] = sum(1 for value in curve if value > floor)
            entry["first_bin_above_floor"] = first_crossing(curve, floor)
        out["conditions"][name] = entry
    return out


def sparkline(curve: list[float], width: int = 24) -> str:
    blocks = " ▁▂▃▄▅▆▇█"
    if not curve:
        return ""
    return "".join(blocks[min(len(blocks) - 1, int(round(value * (len(blocks) - 1))))] for value in curve[:width])


def curves_markdown(analysis: dict[str, Any]) -> str:
    floor = analysis.get("floor")
    lines = ["# Learning Curves", ""]
    lines.append(f"- bins: `{analysis['bins']}` over the evaluation stage")
    lines.append(f"- stages: {', '.join(f'`{s}`' for s in analysis['stages'])}")
    if floor is not None:
        lines.append(f"- random-policy floor: **{floor:.3f}**")
    lines.append("")
    header = "| condition | verdict | curve | peak | final | "
    header += "bins>floor | first>floor | " if floor is not None else ""
    header += "eval eps | mean steps |"
    lines.append(header)
    divider = "| --- | --- | --- | ---: | ---: | " + ("---: | ---: | " if floor is not None else "") + "---: | ---: |"
    lines.append(divider)
    for name, entry in analysis["conditions"].items():
        row = (
            f"| `{name}` | **{entry['verdict']}** | `{sparkline(entry['curve'])}` | "
            f"{entry['peak']:.3f} | {entry['final_bin']:.3f} | "
        )
        if floor is not None:
            first = entry["first_bin_above_floor"]
            row += f"{entry['bins_above_floor']}/{analysis['bins']} | {'-' if first is None else first} | "
        row += f"{entry['eval_episodes']} | {entry['mean_steps']:.1f} |"
        lines.append(row)
    lines.append("")
    lines.append("## Per-Stage Success")
    lines.append("")
    lines.append("| condition | " + " | ".join(f"`{s}`" for s in analysis["stages"]) + " |")
    lines.append("| --- |" + " ---: |" * len(analysis["stages"]))
    for name, entry in analysis["conditions"].items():
        cells = " | ".join(f"{entry['per_stage'].get(s, float('nan')):.3f}" for s in analysis["stages"])
        lines.append(f"| `{name}` | {cells} |")
    lines.append("")
    return "\n".join(lines)


def curves_svg(analysis: dict[str, Any], width: int = 720, height: int = 260) -> str:
    """Inline SVG, no plotting dependency, so the default lane keeps working."""
    palette = ["#2b7bba", "#d1495b", "#4c956c", "#e09f3e", "#7d5ba6", "#118ab2"]
    pad = 36
    plot_w, plot_h = width - 2 * pad, height - 2 * pad
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" width="{width}" height="{height}">',
        f'<rect width="{width}" height="{height}" fill="#ffffff"/>',
        f'<line x1="{pad}" y1="{pad}" x2="{pad}" y2="{pad+plot_h}" stroke="#333"/>',
        f'<line x1="{pad}" y1="{pad+plot_h}" x2="{pad+plot_w}" y2="{pad+plot_h}" stroke="#333"/>',
        f'<text x="4" y="{pad+6}" font-size="11" fill="#333">1.0</text>',
        f'<text x="4" y="{pad+plot_h}" font-size="11" fill="#333">0.0</text>',
        f'<text x="{pad}" y="{height-8}" font-size="11" fill="#333">episode bin (evaluation stage)</text>',
    ]
    floor = analysis.get("floor")
    if floor is not None:
        y = pad + plot_h * (1.0 - floor)
        parts.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{pad+plot_w}" y2="{y:.1f}" '
            f'stroke="#999" stroke-dasharray="5,4"/>'
            f'<text x="{pad+plot_w-96}" y="{y-5:.1f}" font-size="11" fill="#666">random {floor:.3f}</text>'
        )
    for index, (name, entry) in enumerate(analysis["conditions"].items()):
        curve = entry["curve"]
        if len(curve) < 2:
            continue
        colour = palette[index % len(palette)]
        step = plot_w / (len(curve) - 1)
        points = " ".join(
            f"{pad + i*step:.1f},{pad + plot_h*(1.0-v):.1f}" for i, v in enumerate(curve)
        )
        parts.append(f'<polyline points="{points}" fill="none" stroke="{colour}" stroke-width="2"/>')
        parts.append(
            f'<text x="{pad+8}" y="{pad+14+index*15}" font-size="12" fill="{colour}">{name}</text>'
        )
    parts.append("</svg>")
    return "\n".join(parts)


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-curves")
    parser.add_argument("path", type=Path, help="run directory or episodes.jsonl")
    parser.add_argument("--bins", type=int, default=12)
    parser.add_argument("--floor", type=float, default=None, help="random-policy floor to draw")
    parser.add_argument("--svg", type=Path, default=None)
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    floor = args.floor
    if floor is None and args.path.is_dir():
        for candidate in (args.path / "metrics.json", args.path / "latest" / "metrics.json"):
            if candidate.exists():
                block = json.loads(candidate.read_text(encoding="utf-8")).get("random_policy_floor")
                if block:
                    floor = float(block["mean_success"])
                break

    analysis = analyze(load_episodes(args.path), bins=args.bins, floor=floor)
    if args.svg:
        args.svg.write_text(curves_svg(analysis), encoding="utf-8")
        print(f"svg={args.svg}")
    print(json.dumps(analysis, indent=2, sort_keys=True) if args.json else curves_markdown(analysis))
    return 0


def demo() -> None:
    """Self-check: the three verdicts are told apart, and a collapse is not read as flat."""
    def rows(values: list[float]) -> list[dict[str, Any]]:
        return [
            {"seed": 1, "episode": i, "success": v, "steps": 10, "stage": "eval", "condition": "c"}
            for i, v in enumerate(values)
        ]

    rising = [0.0] * 20 + [1.0] * 20
    assert classify(binned_curve(rows(rising), 8)) == IMPROVED
    flat = [0.3] * 40
    assert classify(binned_curve(rows(flat), 8)) == FLAT
    declining = [0.8] * 20 + [0.2] * 20
    assert classify(binned_curve(rows(declining), 8)) == DECLINED
    # Peaks in the middle and falls back: must not read as flat just because the
    # endpoints match.
    collapsing = [0.1] * 10 + [0.9] * 20 + [0.1] * 10
    assert classify(binned_curve(rows(collapsing), 8)) == COLLAPSED
    # A single noisy end bin must NOT read as a collapse; quarter means absorb it.
    wobbly = binned_curve(rows([0.2] * 10 + [0.9] * 30), 12)
    wobbly[-2] = wobbly[-2] - 0.08
    assert classify(wobbly) == IMPROVED, wobbly

    # Two seeds with opposite trends average to flat. Binning the concatenation
    # would instead read as one long ramp followed by one long fall.
    seed_a = [{"seed": 1, "episode": i, "success": v, "steps": 10, "stage": "eval", "condition": "c"}
              for i, v in enumerate([0.0] * 20 + [1.0] * 20)]
    seed_b = [{"seed": 2, "episode": i, "success": v, "steps": 10, "stage": "eval", "condition": "c"}
              for i, v in enumerate([1.0] * 20 + [0.0] * 20)]
    averaged = binned_curve(seed_a + seed_b, 4)
    assert all(abs(v - 0.5) < 1e-9 for v in averaged), averaged
    assert classify(averaged) == FLAT, averaged

    curve = binned_curve(rows(rising), 4)
    assert len(curve) == 4 and curve[0] == 0.0 and curve[-1] == 1.0, curve
    assert first_crossing(curve, 0.5) == 2, curve
    assert first_crossing(curve, 2.0) is None
    assert sparkline([0.0, 1.0]) == " █", repr(sparkline([0.0, 1.0]))

    analysis = analyze(rows(rising), bins=4, floor=0.2)
    entry = analysis["conditions"]["c"]
    assert entry["verdict"] == IMPROVED
    assert entry["bins_above_floor"] == 2, entry
    assert "Learning Curves" in curves_markdown(analysis)
    assert curves_svg(analysis).startswith("<svg")

    try:
        binned_curve(rows(flat), 0)
    except ValueError:
        pass
    else:
        raise AssertionError("bins=0 accepted")
    print("curves demo ok")


if __name__ == "__main__":
    raise SystemExit(main())
