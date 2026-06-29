from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from baby_model.minigrid_experiment import _env_call
from baby_model.minigrid_linear import SparseFeatures, feature_signature, linear_features
from baby_model.minigrid_probe import observation_schema
from baby_model.minigrid_torch import affordance_progress_vector, controllability_target


DEFAULT_LABELS = ("mission_object", "mission_color", "changed")
DEFAULT_FEATURE_SETS = ("raw_current", "affordance_current")
OBJECT_WORDS = ("ball", "box", "key", "door", "goal")
COLOR_WORDS = ("red", "green", "blue", "purple", "yellow", "grey", "gray")


@dataclass(frozen=True)
class ProbeEnvConfig:
    name: str
    env_id: str
    episodes: int
    max_steps: int


@dataclass(frozen=True)
class ProbeDecisionConfig:
    labels: tuple[str, ...]
    min_accuracy: float
    min_lift: float
    min_test_examples: int


@dataclass(frozen=True)
class MiniGridRepresentationProbeConfig:
    envs: tuple[ProbeEnvConfig, ...]
    feature_dim: int
    encoder_mode: str
    policy: str
    test_every: int
    signature_buckets: int
    feature_sets: tuple[str, ...]
    quiet_env_output: bool
    decision: ProbeDecisionConfig


def main() -> int:
    parser = argparse.ArgumentParser(prog="baby-model-minigrid-repr-probe")
    parser.add_argument("--config", type=Path, default=Path("configs/experiments/minigrid-repr-probe-v28.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("runs/minigrid-repr-probe"))
    parser.add_argument("--seed", type=int, default=2301)
    args = parser.parse_args()

    try:
        report = run_minigrid_representation_probe(json.loads(args.config.read_text(encoding="utf-8")), seed=args.seed)
    except ImportError as exc:
        print(f"missing optional dependency: {exc}")
        print("install with: python3 -m pip install minigrid")
        return 2

    run_dir = write_representation_probe(report, args.output_dir)
    print(f"minigrid_repr_probe_dir={run_dir}")
    print(f"decision_met={str(report['decision']['met']).lower()}")
    print(f"best_feature_set={report['decision']['best_feature_set']}")
    return 0


def run_minigrid_representation_probe(config: dict[str, Any], seed: int = 2301) -> dict[str, Any]:
    parsed = parse_minigrid_representation_probe_config(config)
    try:
        import gymnasium as gym
        import minigrid  # noqa: F401
    except ImportError as exc:
        raise ImportError("gymnasium/minigrid") from exc

    transitions = collect_probe_transitions(gym=gym, config=parsed, seed=seed)
    feature_reports = [
        evaluate_feature_set(transitions, feature_set=feature_set, config=parsed)
        for feature_set in parsed.feature_sets
    ]
    decision = evaluate_probe_decision(feature_reports, parsed.decision)
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "hypothesis": str(config.get("hypothesis", "Baby-AD/DA non-DQN representation probe")),
        "seed": seed,
        "transition_count": len(transitions),
        "config": {
            "feature_dim": parsed.feature_dim,
            "encoder_mode": parsed.encoder_mode,
            "policy": parsed.policy,
            "test_every": parsed.test_every,
            "signature_buckets": parsed.signature_buckets,
            "feature_sets": list(parsed.feature_sets),
            "decision": {
                "labels": list(parsed.decision.labels),
                "min_accuracy": parsed.decision.min_accuracy,
                "min_lift": parsed.decision.min_lift,
                "min_test_examples": parsed.decision.min_test_examples,
            },
        },
        "envs": [
            {"name": env.name, "env_id": env.env_id, "episodes": env.episodes, "max_steps": env.max_steps}
            for env in parsed.envs
        ],
        "observation_schema": transitions[0]["observation_schema"] if transitions else {},
        "feature_reports": feature_reports,
        "decision": decision,
    }


def parse_minigrid_representation_probe_config(config: dict[str, Any]) -> MiniGridRepresentationProbeConfig:
    dataset = config.get("dataset", {})
    if not isinstance(dataset, dict):
        raise ValueError("dataset must be an object")
    env_items = dataset.get("envs", [])
    if not isinstance(env_items, list) or not env_items:
        raise ValueError("dataset.envs must be a non-empty list")
    envs: list[ProbeEnvConfig] = []
    names: set[str] = set()
    for item in env_items:
        if not isinstance(item, dict):
            raise ValueError("each dataset.envs item must be an object")
        name = str(item.get("name", ""))
        env_id = str(item.get("env_id", ""))
        episodes = int(item.get("episodes", 0))
        max_steps = int(item.get("max_steps", 0))
        if not name:
            raise ValueError("dataset.envs.name is required")
        if name in names:
            raise ValueError(f"duplicate dataset.envs.name: {name}")
        if not env_id:
            raise ValueError(f"dataset.envs.env_id is required for {name}")
        if episodes < 1:
            raise ValueError(f"dataset.envs.episodes must be positive for {name}")
        if max_steps < 1:
            raise ValueError(f"dataset.envs.max_steps must be positive for {name}")
        names.add(name)
        envs.append(ProbeEnvConfig(name=name, env_id=env_id, episodes=episodes, max_steps=max_steps))

    feature_cfg = config.get("features", {})
    if not isinstance(feature_cfg, dict):
        raise ValueError("features must be an object")
    feature_dim = int(feature_cfg.get("feature_dim", 1024))
    encoder_mode = str(feature_cfg.get("encoder_mode", "raw"))
    feature_sets = tuple(str(item) for item in feature_cfg.get("feature_sets", DEFAULT_FEATURE_SETS))
    if feature_dim < 128:
        raise ValueError("features.feature_dim must be at least 128")
    if encoder_mode not in {"raw", "coarse"}:
        raise ValueError("features.encoder_mode must be raw or coarse")
    if not feature_sets:
        raise ValueError("features.feature_sets must be non-empty")
    for feature_set in feature_sets:
        if feature_set not in {"raw_current", "affordance_current"}:
            raise ValueError(f"unsupported feature set: {feature_set}")

    policy = str(dataset.get("policy", "random"))
    test_every = int(dataset.get("test_every", 5))
    signature_buckets = int(dataset.get("signature_buckets", 16))
    quiet_env_output = bool(dataset.get("quiet_env_output", True))
    if policy != "random":
        raise ValueError("dataset.policy must be random")
    if test_every < 2:
        raise ValueError("dataset.test_every must be at least 2")
    if signature_buckets < 2:
        raise ValueError("dataset.signature_buckets must be at least 2")

    decision_cfg = config.get("decision", {})
    if not isinstance(decision_cfg, dict):
        raise ValueError("decision must be an object")
    labels = tuple(str(item) for item in decision_cfg.get("labels", DEFAULT_LABELS))
    min_accuracy = float(decision_cfg.get("min_accuracy", 0.60))
    min_lift = float(decision_cfg.get("min_lift", 0.05))
    min_test_examples = int(decision_cfg.get("min_test_examples", 10))
    if not labels:
        raise ValueError("decision.labels must be non-empty")
    for label in labels:
        if label not in {"mission_object", "mission_color", "changed", "next_signature_bucket"}:
            raise ValueError(f"unsupported decision label: {label}")
    if min_accuracy < 0.0 or min_accuracy > 1.0:
        raise ValueError("decision.min_accuracy out of range")
    if min_lift < 0.0 or min_lift > 1.0:
        raise ValueError("decision.min_lift out of range")
    if min_test_examples < 1:
        raise ValueError("decision.min_test_examples must be positive")

    return MiniGridRepresentationProbeConfig(
        envs=tuple(envs),
        feature_dim=feature_dim,
        encoder_mode=encoder_mode,
        policy=policy,
        test_every=test_every,
        signature_buckets=signature_buckets,
        feature_sets=feature_sets,
        quiet_env_output=quiet_env_output,
        decision=ProbeDecisionConfig(
            labels=labels,
            min_accuracy=min_accuracy,
            min_lift=min_lift,
            min_test_examples=min_test_examples,
        ),
    )


def collect_probe_transitions(gym: Any, config: MiniGridRepresentationProbeConfig, seed: int) -> list[dict[str, Any]]:
    transitions: list[dict[str, Any]] = []
    for env_index, env_config in enumerate(config.envs):
        env = gym.make(env_config.env_id)
        try:
            if hasattr(env.action_space, "seed"):
                env.action_space.seed(seed + env_index)
            for episode in range(env_config.episodes):
                observation, _info = _env_call(
                    env.reset,
                    quiet=config.quiet_env_output,
                    seed=seed * 100_000 + env_index * 1000 + episode,
                )
                first_schema = observation_schema(observation)
                for step in range(env_config.max_steps):
                    action = int(env.action_space.sample())
                    next_observation, reward, terminated, truncated, _info = _env_call(
                        env.step,
                        action,
                        quiet=config.quiet_env_output,
                    )
                    features = linear_features(observation, config.encoder_mode, config.feature_dim)
                    next_features = linear_features(next_observation, config.encoder_mode, config.feature_dim)
                    labels = transition_probe_labels(
                        observation=observation,
                        next_observation=next_observation,
                        features=features,
                        next_features=next_features,
                        signature_buckets=config.signature_buckets,
                    )
                    transitions.append(
                        {
                            "env_name": env_config.name,
                            "env_id": env_config.env_id,
                            "episode": episode,
                            "step": step,
                            "action": action,
                            "reward": float(reward),
                            "done": bool(terminated or truncated),
                            "features": features,
                            "next_features": next_features,
                            "affordance_features": vector_to_sparse_features(affordance_progress_vector(observation)),
                            "labels": labels,
                            "observation_schema": first_schema,
                        }
                    )
                    observation = next_observation
                    if terminated or truncated:
                        break
        finally:
            env.close()
    return transitions


def transition_probe_labels(
    observation: Any,
    next_observation: Any,
    features: SparseFeatures,
    next_features: SparseFeatures,
    signature_buckets: int,
) -> dict[str, str]:
    mission = str(observation.get("mission", "") if isinstance(observation, dict) else "")
    changed = controllability_target(features, next_features)[0] > 0.0
    return {
        "mission_object": _first_matching_token(mission, OBJECT_WORDS),
        "mission_color": _first_matching_token(mission, COLOR_WORDS),
        "changed": "changed" if changed else "same",
        "next_signature_bucket": f"bucket:{_signature_bucket(next_features, signature_buckets)}",
    }


def vector_to_sparse_features(vector: list[float]) -> SparseFeatures:
    return {index: float(value) for index, value in enumerate(vector) if value != 0.0}


def evaluate_feature_set(
    transitions: list[dict[str, Any]],
    feature_set: str,
    config: MiniGridRepresentationProbeConfig,
) -> dict[str, Any]:
    examples = [transition_features(transition, feature_set) for transition in transitions]
    labels_by_name: dict[str, list[str]] = {
        label_name: [str(transition["labels"][label_name]) for transition in transitions]
        for label_name in ("mission_object", "mission_color", "changed", "next_signature_bucket")
    }
    label_reports = {
        label_name: centroid_probe_metrics(
            examples=examples,
            labels=labels,
            test_every=config.test_every,
        )
        for label_name, labels in labels_by_name.items()
    }
    return {
        "feature_set": feature_set,
        "labels": label_reports,
    }


def transition_features(transition: dict[str, Any], feature_set: str) -> SparseFeatures:
    if feature_set == "raw_current":
        return dict(transition["features"])
    if feature_set == "affordance_current":
        return dict(transition["affordance_features"])
    raise ValueError(f"unsupported feature set: {feature_set}")


def centroid_probe_metrics(examples: list[SparseFeatures], labels: list[str], test_every: int) -> dict[str, Any]:
    if len(examples) != len(labels):
        raise ValueError("examples and labels length mismatch")
    train_examples: list[SparseFeatures] = []
    train_labels: list[str] = []
    test_examples: list[SparseFeatures] = []
    test_labels: list[str] = []
    for index, (example, label) in enumerate(zip(examples, labels, strict=True)):
        if (index + 1) % test_every == 0:
            test_examples.append(example)
            test_labels.append(label)
        else:
            train_examples.append(example)
            train_labels.append(label)
    if not test_examples and train_examples:
        test_examples.append(train_examples.pop())
        test_labels.append(train_labels.pop())
    if not train_examples or not test_examples:
        return {
            "accuracy": 0.0,
            "majority_baseline": 0.0,
            "lift": 0.0,
            "train_examples": len(train_examples),
            "test_examples": len(test_examples),
            "classes": sorted(set(labels)),
        }

    centroids = _centroids(train_examples, train_labels)
    predictions = [_predict_centroid(example, centroids) for example in test_examples]
    correct = sum(1 for expected, actual in zip(test_labels, predictions, strict=True) if expected == actual)
    accuracy = correct / len(test_labels)
    majority_label, majority_count = Counter(test_labels).most_common(1)[0]
    majority_baseline = majority_count / len(test_labels)
    return {
        "accuracy": accuracy,
        "majority_baseline": majority_baseline,
        "lift": accuracy - majority_baseline,
        "train_examples": len(train_examples),
        "test_examples": len(test_examples),
        "classes": sorted(set(labels)),
        "majority_label": majority_label,
    }


def evaluate_probe_decision(feature_reports: list[dict[str, Any]], decision: ProbeDecisionConfig) -> dict[str, Any]:
    candidates: list[dict[str, Any]] = []
    for report in feature_reports:
        label_metrics = [report["labels"][label] for label in decision.labels]
        passed = all(
            metric["test_examples"] >= decision.min_test_examples
            and metric["accuracy"] >= decision.min_accuracy
            and metric["lift"] >= decision.min_lift
            for metric in label_metrics
        )
        candidates.append(
            {
                "feature_set": report["feature_set"],
                "passed": passed,
                "mean_accuracy": mean(metric["accuracy"] for metric in label_metrics),
                "mean_lift": mean(metric["lift"] for metric in label_metrics),
            }
        )
    best = max(candidates, key=lambda item: (item["passed"], item["mean_accuracy"], item["mean_lift"]))
    return {
        "met": bool(best["passed"]),
        "best_feature_set": best["feature_set"],
        "candidates": candidates,
        "rule": {
            "labels": list(decision.labels),
            "min_accuracy": decision.min_accuracy,
            "min_lift": decision.min_lift,
            "min_test_examples": decision.min_test_examples,
        },
    }


def write_representation_probe(report: dict[str, Any], output_dir: Path) -> Path:
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    (run_dir / "metrics.json").write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / "summary.md").write_text(representation_probe_summary_markdown(report), encoding="utf-8")
    latest_path = output_dir / "latest"
    if latest_path.exists() or latest_path.is_symlink():
        latest_path.unlink()
    latest_path.symlink_to(run_dir.name)
    return run_dir


def representation_probe_summary_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# baby-model MiniGrid representation probe summary",
        "",
        f"- created_at: `{report['created_at']}`",
        f"- hypothesis: `{report['hypothesis']}`",
        f"- seed: `{report['seed']}`",
        f"- transitions: `{report['transition_count']}`",
        f"- decision_met: `{str(report['decision']['met']).lower()}`",
        f"- best_feature_set: `{report['decision']['best_feature_set']}`",
        "",
        "| feature_set | label | accuracy | majority | lift | train | test | classes |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for feature_report in report["feature_reports"]:
        feature_set = feature_report["feature_set"]
        for label_name, metrics in feature_report["labels"].items():
            lines.append(
                "| {feature_set} | {label} | {accuracy:.3f} | {majority:.3f} | {lift:.3f} | {train} | {test} | {classes} |".format(
                    feature_set=feature_set,
                    label=label_name,
                    accuracy=metrics["accuracy"],
                    majority=metrics["majority_baseline"],
                    lift=metrics["lift"],
                    train=metrics["train_examples"],
                    test=metrics["test_examples"],
                    classes=",".join(metrics["classes"]),
                )
            )
    lines.append("")
    return "\n".join(lines)


def _centroids(examples: list[SparseFeatures], labels: list[str]) -> dict[str, SparseFeatures]:
    sums: dict[str, defaultdict[int, float]] = {}
    counts: Counter[str] = Counter()
    for example, label in zip(examples, labels, strict=True):
        if label not in sums:
            sums[label] = defaultdict(float)
        counts[label] += 1
        for index, value in example.items():
            sums[label][index] += value
    return {
        label: {index: value / counts[label] for index, value in values.items()}
        for label, values in sums.items()
    }


def _predict_centroid(example: SparseFeatures, centroids: dict[str, SparseFeatures]) -> str:
    if not centroids:
        return ""
    example_norm = _sparse_norm(example)
    scored = []
    for label, centroid in centroids.items():
        denom = max(1e-12, example_norm * _sparse_norm(centroid))
        scored.append((_sparse_dot(example, centroid) / denom, label))
    best_score = max(score for score, _label in scored)
    return sorted(label for score, label in scored if score == best_score)[0]


def _sparse_dot(left: SparseFeatures, right: SparseFeatures) -> float:
    if len(left) > len(right):
        left, right = right, left
    return sum(value * right.get(index, 0.0) for index, value in left.items())


def _sparse_norm(features: SparseFeatures) -> float:
    return sum(value * value for value in features.values()) ** 0.5


def _first_matching_token(text: str, candidates: tuple[str, ...]) -> str:
    tokens = _tokens(text)
    for candidate in candidates:
        if candidate in tokens:
            return "grey" if candidate == "gray" else candidate
    return "unknown"


def _tokens(text: str) -> set[str]:
    current: list[str] = []
    tokens: set[str] = set()
    for char in text.lower():
        if char.isalnum():
            current.append(char)
        elif current:
            tokens.add("".join(current))
            current = []
    if current:
        tokens.add("".join(current))
    return tokens


def _signature_bucket(features: SparseFeatures, buckets: int) -> int:
    payload = ",".join(str(index) for index in feature_signature(features))
    digest = hashlib.blake2b(payload.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % buckets


if __name__ == "__main__":
    raise SystemExit(main())
