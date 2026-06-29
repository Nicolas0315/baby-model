from __future__ import annotations

import argparse
import json
from pathlib import Path

from baby_model.experiment import run_suite, validate_config, write_run
from baby_model.sweep import parse_seeds, run_sweep, write_sweep


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run an experiment suite")
    run_parser.add_argument("--config", type=Path, default=None)
    run_parser.add_argument("--output-dir", type=Path, default=Path("runs"))
    run_parser.add_argument("--seed", type=int, default=7)

    sweep_parser = subparsers.add_parser("sweep", help="run a multi-seed experiment sweep")
    sweep_parser.add_argument("--config", type=Path, required=True)
    sweep_parser.add_argument("--output-dir", type=Path, default=Path("runs/sweeps"))
    sweep_parser.add_argument("--seeds", default="101,102,103")

    verify_parser = subparsers.add_parser("verify-config", help="parse a config file")
    verify_parser.add_argument("config", type=Path)

    args = parser.parse_args()
    if args.command == "verify-config":
        validate_config(_load_config(args.config))
        print(f"ok config={args.config}")
        return 0
    if args.command == "run":
        config = _load_config(args.config) if args.config else {}
        validate_config(config)
        report = run_suite(config=config, seed=args.seed)
        run_dir = write_run(report, args.output_dir)
        print(f"run_dir={run_dir}")
        print(f"winner_last_window={report['winner_last_window']}")
        return 0
    if args.command == "sweep":
        config = _load_config(args.config)
        seeds = parse_seeds(args.seeds)
        report = run_sweep(config=config, seeds=seeds)
        run_dir = write_sweep(report, args.output_dir)
        print(f"sweep_dir={run_dir}")
        print(f"winner_by_mean_success_last_window={report['winner_by_mean_success_last_window']}")
        return 0
    raise AssertionError(args.command)


def _load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    raise SystemExit(main())
