from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True, order=True)
class DriverVersion:
    major: int
    minor: int = 0
    patch: int = 0

    @classmethod
    def parse(cls, value: str) -> "DriverVersion":
        parts = value.strip().split(".")
        if not parts or not parts[0].isdigit():
            raise ValueError(f"invalid driver version: {value}")
        numbers = [int(part) if part.isdigit() else 0 for part in parts[:3]]
        while len(numbers) < 3:
            numbers.append(0)
        return cls(numbers[0], numbers[1], numbers[2])

    def __str__(self) -> str:
        return f"{self.major}.{self.minor}.{self.patch}"


@dataclass(frozen=True)
class WheelPolicy:
    tag: str
    index_url: str
    min_driver: DriverVersion | None
    cuda_major: int | None
    notes: str


WHEEL_POLICIES: dict[str, WheelPolicy] = {
    "cpu": WheelPolicy(
        tag="cpu",
        index_url="https://download.pytorch.org/whl/cpu",
        min_driver=None,
        cuda_major=None,
        notes="CPU-only fallback; no CUDA driver requirement.",
    ),
    "cu126": WheelPolicy(
        tag="cu126",
        index_url="https://download.pytorch.org/whl/cu126",
        min_driver=DriverVersion(560, 0, 0),
        cuda_major=12,
        notes="CUDA 12.6 wheel; useful for older CUDA-capable workers.",
    ),
    "cu130": WheelPolicy(
        tag="cu130",
        index_url="https://download.pytorch.org/whl/cu130",
        min_driver=DriverVersion(580, 0, 0),
        cuda_major=13,
        notes="CUDA 13.0 wheel; requires a CUDA 13-capable driver branch.",
    ),
    "cu132": WheelPolicy(
        tag="cu132",
        index_url="https://download.pytorch.org/whl/cu132",
        min_driver=DriverVersion(580, 0, 0),
        cuda_major=13,
        notes="CUDA 13.2 wheel; preferred for newer CUDA 13-only workers.",
    ),
}


@dataclass(frozen=True)
class WorkerPolicy:
    worker_class: str
    driver_version: DriverVersion | None
    cuda_umd_major: int | None
    requires_cuda13_wheel: bool
    primary_wheel: str
    fallback_wheel: str = "cpu"
    fallback_device: str = "cpu"


def parse_cuda_major(value: str | int | None) -> int | None:
    if value is None:
        return None
    if isinstance(value, int):
        return value
    head = value.strip().split(".", 1)[0]
    if not head.isdigit():
        return None
    return int(head)


def parse_worker_policy(data: dict[str, Any]) -> WorkerPolicy:
    driver = data.get("driver_version")
    return WorkerPolicy(
        worker_class=str(data["worker_class"]),
        driver_version=DriverVersion.parse(str(driver)) if driver else None,
        cuda_umd_major=parse_cuda_major(data.get("cuda_umd")),
        requires_cuda13_wheel=bool(data.get("requires_cuda13_wheel", False)),
        primary_wheel=str(data["primary_wheel"]),
        fallback_wheel=str(data.get("fallback_wheel", "cpu")),
        fallback_device=str(data.get("fallback_device", "cpu")),
    )


def evaluate_worker(policy: WorkerPolicy) -> dict[str, str]:
    wheel = WHEEL_POLICIES[policy.primary_wheel]
    fallback = WHEEL_POLICIES[policy.fallback_wheel]
    reasons: list[str] = []

    if policy.driver_version is None:
        reasons.append("no CUDA driver observed")
    elif wheel.min_driver and policy.driver_version < wheel.min_driver:
        reasons.append(f"driver {policy.driver_version} below {wheel.min_driver} for {wheel.tag}")

    if wheel.cuda_major is not None and policy.cuda_umd_major is not None and policy.cuda_umd_major < wheel.cuda_major:
        reasons.append(f"CUDA UMD {policy.cuda_umd_major} below CUDA {wheel.cuda_major} for {wheel.tag}")

    if policy.requires_cuda13_wheel and wheel.cuda_major != 13:
        reasons.append("worker class requires a CUDA 13 wheel")

    if reasons:
        return {
            "worker_class": policy.worker_class,
            "status": "fallback_required",
            "primary_wheel": wheel.tag,
            "primary_index_url": wheel.index_url,
            "fallback_wheel": fallback.tag,
            "fallback_index_url": fallback.index_url,
            "fallback_device": policy.fallback_device,
            "reason": "; ".join(reasons),
        }

    return {
        "worker_class": policy.worker_class,
        "status": "gpu_candidate",
        "primary_wheel": wheel.tag,
        "primary_index_url": wheel.index_url,
        "fallback_wheel": fallback.tag,
        "fallback_index_url": fallback.index_url,
        "fallback_device": policy.fallback_device,
        "reason": "driver and wheel family are compatible; run bounded CUDA smoke",
    }


def render_policy_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# GPU Wheel Policy Report",
        "",
        f"- created_at: `{report.get('created_at', '')}`",
        f"- source_commit: `{report.get('source_commit', '')}`",
        "",
        "| worker_class | status | primary_wheel | fallback_wheel | reason |",
        "| --- | --- | --- | --- | --- |",
    ]
    for row in report["results"]:
        lines.append(
            "| {worker_class} | {status} | {primary_wheel} | {fallback_wheel} | {reason} |".format(**row)
        )
    lines.append("")
    return "\n".join(lines)


def build_report(config: dict[str, Any]) -> dict[str, Any]:
    workers = [parse_worker_policy(row) for row in config["workers"]]
    return {
        "created_at": config.get("created_at", ""),
        "source_commit": config.get("source_commit", ""),
        "results": [evaluate_worker(worker) for worker in workers],
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Evaluate anonymous GPU worker wheel policy.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="")
    args = parser.parse_args(argv)

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    report = build_report(config)
    rendered = render_policy_markdown(report)
    if args.output:
        Path(args.output).write_text(rendered, encoding="utf-8")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
