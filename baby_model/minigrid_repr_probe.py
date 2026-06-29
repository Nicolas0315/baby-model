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
SUPPORTED_LABELS = ("mission_object", "mission_color", "changed", "next_signature_bucket")
OBJECT_WORDS = ("ball", "box", "key", "door", "goal")
COLOR_WORDS = ("red", "green", "blue", "purple", "yellow", "grey", "gray")
MINIGRID_OBJECT_TO_IDX = {"door": 4, "key": 5, "ball": 6, "box": 7, "goal": 8}
MINIGRID_COLOR_TO_IDX = {"red": 0, "green": 1, "blue": 2, "purple": 3, "yellow": 4, "grey": 5, "gray": 5}
SCRIPTED_ACTION_LEFT = 0
SCRIPTED_ACTION_RIGHT = 1
SCRIPTED_ACTION_FORWARD = 2


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
    mode: str
    baseline_feature_set: str
    reference_feature_set: str
    candidate_feature_set: str
    transition_label: str
    changed_min_lift_delta: float
    transition_min_lift_delta: float
    external_transition_lift_baseline: float
    max_mission_accuracy_drop: float


@dataclass(frozen=True)
class PredictiveEncoderConfig:
    name: str
    target_label: str
    epochs: int
    learning_rate: float
    include_action: bool
    include_raw_passthrough: bool
    score_scale: float
    prediction_weight: float


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
    predictive_encoder: PredictiveEncoderConfig
    predictive_encoders: tuple[PredictiveEncoderConfig, ...]


@dataclass(frozen=True)
class PredictiveLinearEncoder:
    classes: tuple[str, ...]
    weights: dict[str, SparseFeatures]
    config: PredictiveEncoderConfig
    feature_dim: int

    def embed(self, transition: dict[str, Any]) -> SparseFeatures:
        embedding: SparseFeatures = {}
        if self.config.include_raw_passthrough:
            embedding.update(dict(transition["features"]))
        if not self.classes:
            return embedding

        features = predictive_input_features(
            transition=transition,
            feature_dim=self.feature_dim,
            include_action=self.config.include_action,
        )
        scores = [_sparse_dot(features, self.weights[label]) for label in self.classes]
        max_abs_score = max(1.0, max(abs(score) for score in scores))
        offset = self.feature_dim
        for class_index, score in enumerate(scores):
            if score != 0.0:
                embedding[offset + class_index] = (score / max_abs_score) * self.config.score_scale
        prediction = _predict_linear_label(features, self.classes, self.weights)
        if prediction:
            embedding[offset + len(self.classes) + self.classes.index(prediction)] = self.config.prediction_weight
        return embedding


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
    predictive_encoders: dict[str, PredictiveLinearEncoder] = {}
    training_reports: dict[str, dict[str, Any]] = {}
    for encoder_config in parsed.predictive_encoders:
        if encoder_config.name in parsed.feature_sets:
            encoder, report = train_predictive_encoder(transitions, parsed, encoder_config)
            predictive_encoders[encoder_config.name] = encoder
            training_reports[encoder_config.name] = report
    feature_reports = [
        evaluate_feature_set(
            transitions,
            feature_set=feature_set,
            config=parsed,
            predictive_encoders=predictive_encoders,
        )
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
            "training": {
                "predictive_encoders": [
                    {
                        "name": encoder_config.name,
                        "target_label": encoder_config.target_label,
                        "epochs": encoder_config.epochs,
                        "learning_rate": encoder_config.learning_rate,
                        "include_action": encoder_config.include_action,
                        "include_raw_passthrough": encoder_config.include_raw_passthrough,
                        "score_scale": encoder_config.score_scale,
                        "prediction_weight": encoder_config.prediction_weight,
                    }
                    for encoder_config in parsed.predictive_encoders
                ]
            },
            "decision": {
                "mode": parsed.decision.mode,
                "labels": list(parsed.decision.labels),
                "min_accuracy": parsed.decision.min_accuracy,
                "min_lift": parsed.decision.min_lift,
                "min_test_examples": parsed.decision.min_test_examples,
                "baseline_feature_set": parsed.decision.baseline_feature_set,
                "reference_feature_set": parsed.decision.reference_feature_set,
                "candidate_feature_set": parsed.decision.candidate_feature_set,
                "transition_label": parsed.decision.transition_label,
                "changed_min_lift_delta": parsed.decision.changed_min_lift_delta,
                "transition_min_lift_delta": parsed.decision.transition_min_lift_delta,
                "external_transition_lift_baseline": parsed.decision.external_transition_lift_baseline,
                "max_mission_accuracy_drop": parsed.decision.max_mission_accuracy_drop,
            },
        },
        "envs": [
            {"name": env.name, "env_id": env.env_id, "episodes": env.episodes, "max_steps": env.max_steps}
            for env in parsed.envs
        ],
        "observation_schema": transitions[0]["observation_schema"] if transitions else {},
        "training_report": training_reports.get("predictive_encoder"),
        "training_reports": training_reports,
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
    predictive_names = predictive_encoder_names(config)
    supported_feature_sets = {"raw_current", "affordance_current"} | predictive_names
    for feature_set in feature_sets:
        if feature_set not in supported_feature_sets:
            raise ValueError(f"unsupported feature set: {feature_set}")

    policy = str(dataset.get("policy", "random"))
    test_every = int(dataset.get("test_every", 5))
    signature_buckets = int(dataset.get("signature_buckets", 16))
    quiet_env_output = bool(dataset.get("quiet_env_output", True))
    if policy not in {"random", "scripted_object"}:
        raise ValueError("dataset.policy must be random or scripted_object")
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
        if label not in SUPPORTED_LABELS:
            raise ValueError(f"unsupported decision label: {label}")
    if min_accuracy < 0.0 or min_accuracy > 1.0:
        raise ValueError("decision.min_accuracy out of range")
    if min_lift < 0.0 or min_lift > 1.0:
        raise ValueError("decision.min_lift out of range")
    if min_test_examples < 1:
        raise ValueError("decision.min_test_examples must be positive")
    mode = str(decision_cfg.get("mode", "absolute_all_labels"))
    if mode not in {"absolute_all_labels", "relative_to_baseline", "relative_to_reference", "external_transition_baseline"}:
        raise ValueError(
            "decision.mode must be absolute_all_labels, relative_to_baseline, relative_to_reference, or external_transition_baseline"
        )
    baseline_feature_set = str(decision_cfg.get("baseline_feature_set", "raw_current"))
    reference_feature_set = str(decision_cfg.get("reference_feature_set", "predictive_encoder"))
    candidate_feature_set = str(decision_cfg.get("candidate_feature_set", "predictive_encoder"))
    transition_label = str(decision_cfg.get("transition_label", "changed"))
    changed_min_lift_delta = float(decision_cfg.get("changed_min_lift_delta", 0.05))
    transition_min_lift_delta = float(decision_cfg.get("transition_min_lift_delta", 0.01))
    external_transition_lift_baseline = float(decision_cfg.get("external_transition_lift_baseline", 0.0))
    max_mission_accuracy_drop = float(decision_cfg.get("max_mission_accuracy_drop", 0.05))
    if transition_label not in SUPPORTED_LABELS:
        raise ValueError(f"unsupported decision transition_label: {transition_label}")
    if changed_min_lift_delta < 0.0 or changed_min_lift_delta > 1.0:
        raise ValueError("decision.changed_min_lift_delta out of range")
    if transition_min_lift_delta < 0.0 or transition_min_lift_delta > 1.0:
        raise ValueError("decision.transition_min_lift_delta out of range")
    if external_transition_lift_baseline < -1.0 or external_transition_lift_baseline > 1.0:
        raise ValueError("decision.external_transition_lift_baseline out of range")
    if max_mission_accuracy_drop < 0.0 or max_mission_accuracy_drop > 1.0:
        raise ValueError("decision.max_mission_accuracy_drop out of range")
    if mode == "relative_to_baseline":
        for feature_set in (baseline_feature_set, candidate_feature_set):
            if feature_set not in feature_sets:
                raise ValueError(f"decision feature set is not enabled: {feature_set}")
    if mode == "relative_to_reference":
        for feature_set in (baseline_feature_set, reference_feature_set, candidate_feature_set):
            if feature_set not in feature_sets:
                raise ValueError(f"decision feature set is not enabled: {feature_set}")
    if mode == "external_transition_baseline":
        for feature_set in (baseline_feature_set, candidate_feature_set):
            if feature_set not in feature_sets:
                raise ValueError(f"decision feature set is not enabled: {feature_set}")

    predictive_encoders = parse_predictive_encoder_configs(config)
    predictive_encoder = predictive_encoders[0]

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
            mode=mode,
            baseline_feature_set=baseline_feature_set,
            reference_feature_set=reference_feature_set,
            candidate_feature_set=candidate_feature_set,
            transition_label=transition_label,
            changed_min_lift_delta=changed_min_lift_delta,
            transition_min_lift_delta=transition_min_lift_delta,
            external_transition_lift_baseline=external_transition_lift_baseline,
            max_mission_accuracy_drop=max_mission_accuracy_drop,
        ),
        predictive_encoder=predictive_encoder,
        predictive_encoders=predictive_encoders,
    )


def predictive_encoder_names(config: dict[str, Any]) -> set[str]:
    return {encoder.name for encoder in parse_predictive_encoder_configs(config)}


def parse_predictive_encoder_configs(config: dict[str, Any]) -> tuple[PredictiveEncoderConfig, ...]:
    training_cfg = config.get("training", {})
    if training_cfg is None:
        training_cfg = {}
    if not isinstance(training_cfg, dict):
        raise ValueError("training must be an object")
    encoder_items = training_cfg.get("predictive_encoders")
    if encoder_items is None:
        encoder_items = [training_cfg.get("predictive_encoder", training_cfg)]
    if not isinstance(encoder_items, list) or not encoder_items:
        raise ValueError("training.predictive_encoders must be a non-empty list")

    encoders: list[PredictiveEncoderConfig] = []
    names: set[str] = set()
    for item in encoder_items:
        if item is None:
            item = {}
        if not isinstance(item, dict):
            raise ValueError("each training.predictive_encoders item must be an object")
        name = str(item.get("name", "predictive_encoder"))
        target_label = str(item.get("target_label", "changed"))
        epochs = int(item.get("epochs", 4))
        learning_rate = float(item.get("learning_rate", 0.10))
        include_action = bool(item.get("include_action", True))
        include_raw_passthrough = bool(item.get("include_raw_passthrough", True))
        score_scale = float(item.get("score_scale", 1.0))
        prediction_weight = float(item.get("prediction_weight", 1.0))
        if not name:
            raise ValueError("training.predictive_encoders.name is required")
        if name in names:
            raise ValueError(f"duplicate predictive encoder name: {name}")
        if name in {"raw_current", "affordance_current"}:
            raise ValueError(f"predictive encoder name conflicts with built-in feature set: {name}")
        if target_label not in SUPPORTED_LABELS:
            raise ValueError(f"unsupported predictive encoder target_label: {target_label}")
        if epochs < 1:
            raise ValueError("training.predictive_encoders.epochs must be positive")
        if learning_rate <= 0.0:
            raise ValueError("training.predictive_encoders.learning_rate must be positive")
        if score_scale < 0.0:
            raise ValueError("training.predictive_encoders.score_scale must be non-negative")
        if prediction_weight < 0.0:
            raise ValueError("training.predictive_encoders.prediction_weight must be non-negative")
        names.add(name)
        encoders.append(
            PredictiveEncoderConfig(
                name=name,
                target_label=target_label,
                epochs=epochs,
                learning_rate=learning_rate,
                include_action=include_action,
                include_raw_passthrough=include_raw_passthrough,
                score_scale=score_scale,
                prediction_weight=prediction_weight,
            )
        )
    return tuple(encoders)


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
                    action = choose_probe_action(
                        observation=observation,
                        policy=config.policy,
                        fallback_action_space=env.action_space,
                        episode=episode,
                        step=step,
                        seed=seed + env_index,
                    )
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


def choose_probe_action(
    observation: Any,
    policy: str,
    fallback_action_space: Any,
    episode: int,
    step: int,
    seed: int,
) -> int:
    if policy == "random":
        return int(fallback_action_space.sample())
    if policy == "scripted_object":
        return scripted_object_action(observation=observation, episode=episode, step=step, seed=seed)
    raise ValueError(f"unsupported probe policy: {policy}")


def scripted_object_action(observation: Any, episode: int, step: int, seed: int) -> int:
    if not isinstance(observation, dict):
        return deterministic_explore_action(episode=episode, step=step, seed=seed)
    image = observation.get("image")
    target = target_from_mission(str(observation.get("mission", "")))
    target_cell = nearest_visible_target_cell(image=image, target=target)
    if target_cell is None:
        return deterministic_explore_action(episode=episode, step=step, seed=seed)
    x, y, width, height = target_cell
    center_x = width // 2
    agent_y = height - 1
    if x < center_x:
        return SCRIPTED_ACTION_LEFT
    if x > center_x:
        return SCRIPTED_ACTION_RIGHT
    if y < agent_y:
        return SCRIPTED_ACTION_FORWARD
    return deterministic_explore_action(episode=episode, step=step, seed=seed)


def deterministic_explore_action(episode: int, step: int, seed: int) -> int:
    pattern = (SCRIPTED_ACTION_FORWARD, SCRIPTED_ACTION_RIGHT, SCRIPTED_ACTION_FORWARD, SCRIPTED_ACTION_LEFT)
    return pattern[(episode * 17 + step + seed) % len(pattern)]


def target_from_mission(mission: str) -> dict[str, str]:
    return {
        "object": _first_matching_token(mission, OBJECT_WORDS),
        "color": _first_matching_token(mission, COLOR_WORDS),
    }


def nearest_visible_target_cell(image: Any, target: dict[str, str]) -> tuple[int, int, int, int] | None:
    columns = image_columns(image)
    if not columns:
        return None
    width = len(columns)
    height = len(columns[0]) if columns[0] else 0
    if height == 0:
        return None
    target_object = target.get("object", "unknown")
    target_color = target.get("color", "unknown")
    target_object_idx = MINIGRID_OBJECT_TO_IDX.get(target_object)
    target_color_idx = MINIGRID_COLOR_TO_IDX.get(target_color)
    center_x = width // 2
    agent_y = height - 1
    candidates: list[tuple[int, int, int, int]] = []
    for x, column in enumerate(columns):
        for y, cell in enumerate(column):
            object_idx, color_idx = cell_object_color(cell)
            if object_idx is None:
                continue
            object_match = target_object_idx is not None and object_idx == target_object_idx
            color_match = target_color_idx is not None and color_idx == target_color_idx
            if target_object_idx is not None and target_color_idx is not None:
                if not (object_match and color_match):
                    continue
                priority = 0
            elif target_object_idx is not None:
                if not object_match:
                    continue
                priority = 1
            elif target_color_idx is not None:
                if not color_match:
                    continue
                priority = 2
            elif object_idx not in set(MINIGRID_OBJECT_TO_IDX.values()):
                continue
            else:
                priority = 3
            distance = abs(x - center_x) + abs(y - agent_y)
            candidates.append((priority, distance, x, y))
    if not candidates:
        return None
    _priority, _distance, x, y = min(candidates)
    return x, y, width, height


def image_columns(image: Any) -> list[list[Any]]:
    if image is None:
        return []
    if hasattr(image, "tolist"):
        image = image.tolist()
    if not isinstance(image, list):
        return []
    if not image:
        return []
    first = image[0]
    if isinstance(first, list) and first and isinstance(first[0], list):
        return image
    return []


def cell_object_color(cell: Any) -> tuple[int | None, int | None]:
    if hasattr(cell, "tolist"):
        cell = cell.tolist()
    if not isinstance(cell, (list, tuple)) or len(cell) < 2:
        return None, None
    return int(cell[0]), int(cell[1])


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
    predictive_encoders: dict[str, PredictiveLinearEncoder] | None = None,
) -> dict[str, Any]:
    examples = [
        transition_features(
            transition,
            feature_set=feature_set,
            predictive_encoders=predictive_encoders,
        )
        for transition in transitions
    ]
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


def transition_features(
    transition: dict[str, Any],
    feature_set: str,
    predictive_encoders: dict[str, PredictiveLinearEncoder] | None = None,
) -> SparseFeatures:
    if feature_set == "raw_current":
        return dict(transition["features"])
    if feature_set == "affordance_current":
        return dict(transition["affordance_features"])
    if predictive_encoders is not None and feature_set in predictive_encoders:
        return predictive_encoders[feature_set].embed(transition)
    if feature_set.startswith("predictive_") or feature_set == "predictive_encoder":
        if predictive_encoders is None:
            raise ValueError("predictive_encoder feature set requires a trained encoder")
        raise ValueError(f"missing trained encoder for feature set: {feature_set}")
    raise ValueError(f"unsupported feature set: {feature_set}")


def train_predictive_encoder(
    transitions: list[dict[str, Any]],
    config: MiniGridRepresentationProbeConfig,
    encoder_config: PredictiveEncoderConfig | None = None,
) -> tuple[PredictiveLinearEncoder, dict[str, Any]]:
    encoder_config = encoder_config or config.predictive_encoder
    target_label = encoder_config.target_label
    train_transitions: list[dict[str, Any]] = []
    test_transitions: list[dict[str, Any]] = []
    for index, transition in enumerate(transitions):
        if (index + 1) % config.test_every == 0:
            test_transitions.append(transition)
        else:
            train_transitions.append(transition)
    if not test_transitions and train_transitions:
        test_transitions.append(train_transitions.pop())

    classes = tuple(sorted({str(transition["labels"][target_label]) for transition in train_transitions}))
    weights: dict[str, SparseFeatures] = {label: {} for label in classes}
    epoch_mistakes: list[int] = []
    for _epoch in range(encoder_config.epochs):
        mistakes = 0
        for transition in train_transitions:
            features = predictive_input_features(
                transition=transition,
                feature_dim=config.feature_dim,
                include_action=encoder_config.include_action,
            )
            expected = str(transition["labels"][target_label])
            predicted = _predict_linear_label(features, classes, weights)
            if predicted != expected:
                mistakes += 1
                _add_scaled_features(weights[expected], features, encoder_config.learning_rate)
                if predicted:
                    _add_scaled_features(weights[predicted], features, -encoder_config.learning_rate)
        epoch_mistakes.append(mistakes)

    encoder = PredictiveLinearEncoder(
        classes=classes,
        weights=weights,
        config=encoder_config,
        feature_dim=config.feature_dim,
    )
    report = {
        "name": encoder_config.name,
        "target_label": target_label,
        "classes": list(classes),
        "epochs": encoder_config.epochs,
        "learning_rate": encoder_config.learning_rate,
        "include_action": encoder_config.include_action,
        "include_raw_passthrough": encoder_config.include_raw_passthrough,
        "score_scale": encoder_config.score_scale,
        "prediction_weight": encoder_config.prediction_weight,
        "train_examples": len(train_transitions),
        "test_examples": len(test_transitions),
        "epoch_mistakes": epoch_mistakes,
        "train_accuracy": predictive_encoder_accuracy(train_transitions, encoder),
        "test_accuracy": predictive_encoder_accuracy(test_transitions, encoder),
        "test_majority_baseline": majority_baseline(
            [str(transition["labels"][target_label]) for transition in test_transitions]
        ),
    }
    report["test_lift"] = report["test_accuracy"] - report["test_majority_baseline"]
    return encoder, report


def predictive_input_features(
    transition: dict[str, Any],
    feature_dim: int,
    include_action: bool,
) -> SparseFeatures:
    features = dict(transition["features"])
    if include_action:
        action_index = feature_dim + int(transition["action"])
        features[action_index] = features.get(action_index, 0.0) + 1.0
    return features


def predictive_encoder_accuracy(
    transitions: list[dict[str, Any]],
    encoder: PredictiveLinearEncoder,
) -> float:
    if not transitions:
        return 0.0
    correct = 0
    for transition in transitions:
        features = predictive_input_features(
            transition=transition,
            feature_dim=encoder.feature_dim,
            include_action=encoder.config.include_action,
        )
        prediction = _predict_linear_label(features, encoder.classes, encoder.weights)
        if prediction == str(transition["labels"][encoder.config.target_label]):
            correct += 1
    return correct / len(transitions)


def majority_baseline(labels: list[str]) -> float:
    if not labels:
        return 0.0
    return Counter(labels).most_common(1)[0][1] / len(labels)


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
    if decision.mode == "relative_to_baseline":
        return evaluate_relative_probe_decision(feature_reports, decision)
    if decision.mode == "relative_to_reference":
        return evaluate_reference_probe_decision(feature_reports, decision)
    if decision.mode == "external_transition_baseline":
        return evaluate_external_transition_decision(feature_reports, decision)
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
            "mode": decision.mode,
            "labels": list(decision.labels),
            "min_accuracy": decision.min_accuracy,
            "min_lift": decision.min_lift,
            "min_test_examples": decision.min_test_examples,
        },
    }


def evaluate_relative_probe_decision(
    feature_reports: list[dict[str, Any]],
    decision: ProbeDecisionConfig,
) -> dict[str, Any]:
    reports_by_feature_set = {report["feature_set"]: report for report in feature_reports}
    baseline = reports_by_feature_set[decision.baseline_feature_set]
    candidate = reports_by_feature_set[decision.candidate_feature_set]
    required_labels = ("changed", "mission_object", "mission_color")
    missing_labels = [
        label
        for label in required_labels
        if label not in baseline["labels"] or label not in candidate["labels"]
    ]
    if missing_labels:
        raise ValueError(f"missing relative decision labels: {','.join(missing_labels)}")

    comparisons: dict[str, dict[str, float | int]] = {}
    enough_examples = True
    for label in required_labels:
        base_metrics = baseline["labels"][label]
        candidate_metrics = candidate["labels"][label]
        enough_examples = (
            enough_examples
            and base_metrics["test_examples"] >= decision.min_test_examples
            and candidate_metrics["test_examples"] >= decision.min_test_examples
        )
        comparisons[label] = {
            "baseline_accuracy": base_metrics["accuracy"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "accuracy_delta": candidate_metrics["accuracy"] - base_metrics["accuracy"],
            "baseline_lift": base_metrics["lift"],
            "candidate_lift": candidate_metrics["lift"],
            "lift_delta": candidate_metrics["lift"] - base_metrics["lift"],
            "baseline_test_examples": base_metrics["test_examples"],
            "candidate_test_examples": candidate_metrics["test_examples"],
        }

    changed_passed = comparisons["changed"]["lift_delta"] >= decision.changed_min_lift_delta
    mission_object_passed = comparisons["mission_object"]["accuracy_delta"] >= -decision.max_mission_accuracy_drop
    mission_color_passed = comparisons["mission_color"]["accuracy_delta"] >= -decision.max_mission_accuracy_drop
    passed = bool(enough_examples and changed_passed and mission_object_passed and mission_color_passed)
    candidates = [
        {
            "feature_set": decision.baseline_feature_set,
            "passed": False,
            "mean_accuracy": mean(baseline["labels"][label]["accuracy"] for label in decision.labels),
            "mean_lift": mean(baseline["labels"][label]["lift"] for label in decision.labels),
        },
        {
            "feature_set": decision.candidate_feature_set,
            "passed": passed,
            "mean_accuracy": mean(candidate["labels"][label]["accuracy"] for label in decision.labels),
            "mean_lift": mean(candidate["labels"][label]["lift"] for label in decision.labels),
        },
    ]
    return {
        "met": passed,
        "best_feature_set": decision.candidate_feature_set if passed else decision.baseline_feature_set,
        "candidates": candidates,
        "comparisons": comparisons,
        "rule": {
            "mode": decision.mode,
            "labels": list(decision.labels),
            "min_test_examples": decision.min_test_examples,
            "baseline_feature_set": decision.baseline_feature_set,
            "candidate_feature_set": decision.candidate_feature_set,
            "changed_min_lift_delta": decision.changed_min_lift_delta,
            "max_mission_accuracy_drop": decision.max_mission_accuracy_drop,
        },
    }


def evaluate_reference_probe_decision(
    feature_reports: list[dict[str, Any]],
    decision: ProbeDecisionConfig,
) -> dict[str, Any]:
    reports_by_feature_set = {report["feature_set"]: report for report in feature_reports}
    baseline = reports_by_feature_set[decision.baseline_feature_set]
    reference = reports_by_feature_set[decision.reference_feature_set]
    candidate = reports_by_feature_set[decision.candidate_feature_set]
    required_labels = ("mission_object", "mission_color", decision.transition_label)
    missing_labels = [
        label
        for label in required_labels
        if label not in baseline["labels"] or label not in reference["labels"] or label not in candidate["labels"]
    ]
    if missing_labels:
        raise ValueError(f"missing reference decision labels: {','.join(missing_labels)}")

    comparisons: dict[str, dict[str, float | int]] = {}
    enough_examples = True
    for label in required_labels:
        base_metrics = baseline["labels"][label]
        reference_metrics = reference["labels"][label]
        candidate_metrics = candidate["labels"][label]
        enough_examples = (
            enough_examples
            and base_metrics["test_examples"] >= decision.min_test_examples
            and reference_metrics["test_examples"] >= decision.min_test_examples
            and candidate_metrics["test_examples"] >= decision.min_test_examples
        )
        comparisons[label] = {
            "baseline_accuracy": base_metrics["accuracy"],
            "reference_accuracy": reference_metrics["accuracy"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "accuracy_delta_vs_baseline": candidate_metrics["accuracy"] - base_metrics["accuracy"],
            "accuracy_delta_vs_reference": candidate_metrics["accuracy"] - reference_metrics["accuracy"],
            "baseline_lift": base_metrics["lift"],
            "reference_lift": reference_metrics["lift"],
            "candidate_lift": candidate_metrics["lift"],
            "lift_delta_vs_baseline": candidate_metrics["lift"] - base_metrics["lift"],
            "lift_delta_vs_reference": candidate_metrics["lift"] - reference_metrics["lift"],
            "baseline_test_examples": base_metrics["test_examples"],
            "reference_test_examples": reference_metrics["test_examples"],
            "candidate_test_examples": candidate_metrics["test_examples"],
        }

    transition = comparisons[decision.transition_label]
    transition_passed = (
        transition["lift_delta_vs_baseline"] >= decision.transition_min_lift_delta
        and transition["lift_delta_vs_reference"] >= decision.transition_min_lift_delta
    )
    mission_object_passed = (
        comparisons["mission_object"]["accuracy_delta_vs_baseline"] >= -decision.max_mission_accuracy_drop
    )
    mission_color_passed = (
        comparisons["mission_color"]["accuracy_delta_vs_baseline"] >= -decision.max_mission_accuracy_drop
    )
    passed = bool(enough_examples and transition_passed and mission_object_passed and mission_color_passed)
    candidates = [
        {
            "feature_set": report["feature_set"],
            "passed": bool(report["feature_set"] == decision.candidate_feature_set and passed),
            "mean_accuracy": mean(report["labels"][label]["accuracy"] for label in decision.labels),
            "mean_lift": mean(report["labels"][label]["lift"] for label in decision.labels),
        }
        for report in (baseline, reference, candidate)
    ]
    return {
        "met": passed,
        "best_feature_set": decision.candidate_feature_set if passed else decision.reference_feature_set,
        "candidates": candidates,
        "comparisons": comparisons,
        "rule": {
            "mode": decision.mode,
            "labels": list(decision.labels),
            "min_test_examples": decision.min_test_examples,
            "baseline_feature_set": decision.baseline_feature_set,
            "reference_feature_set": decision.reference_feature_set,
            "candidate_feature_set": decision.candidate_feature_set,
            "transition_label": decision.transition_label,
            "transition_min_lift_delta": decision.transition_min_lift_delta,
            "max_mission_accuracy_drop": decision.max_mission_accuracy_drop,
        },
    }


def evaluate_external_transition_decision(
    feature_reports: list[dict[str, Any]],
    decision: ProbeDecisionConfig,
) -> dict[str, Any]:
    reports_by_feature_set = {report["feature_set"]: report for report in feature_reports}
    baseline = reports_by_feature_set[decision.baseline_feature_set]
    candidate = reports_by_feature_set[decision.candidate_feature_set]
    required_labels = ("mission_object", "mission_color", decision.transition_label)
    missing_labels = [
        label
        for label in required_labels
        if label not in baseline["labels"] or label not in candidate["labels"]
    ]
    if missing_labels:
        raise ValueError(f"missing external decision labels: {','.join(missing_labels)}")

    comparisons: dict[str, dict[str, float | int]] = {}
    enough_examples = True
    for label in required_labels:
        base_metrics = baseline["labels"][label]
        candidate_metrics = candidate["labels"][label]
        enough_examples = (
            enough_examples
            and base_metrics["test_examples"] >= decision.min_test_examples
            and candidate_metrics["test_examples"] >= decision.min_test_examples
        )
        comparisons[label] = {
            "baseline_accuracy": base_metrics["accuracy"],
            "candidate_accuracy": candidate_metrics["accuracy"],
            "accuracy_delta": candidate_metrics["accuracy"] - base_metrics["accuracy"],
            "baseline_lift": base_metrics["lift"],
            "candidate_lift": candidate_metrics["lift"],
            "lift_delta": candidate_metrics["lift"] - base_metrics["lift"],
            "external_lift_baseline": decision.external_transition_lift_baseline,
            "candidate_lift_delta_vs_external": candidate_metrics["lift"] - decision.external_transition_lift_baseline,
            "baseline_test_examples": base_metrics["test_examples"],
            "candidate_test_examples": candidate_metrics["test_examples"],
        }

    transition = comparisons[decision.transition_label]
    transition_passed = transition["candidate_lift_delta_vs_external"] >= decision.transition_min_lift_delta
    mission_object_passed = comparisons["mission_object"]["accuracy_delta"] >= -decision.max_mission_accuracy_drop
    mission_color_passed = comparisons["mission_color"]["accuracy_delta"] >= -decision.max_mission_accuracy_drop
    passed = bool(enough_examples and transition_passed and mission_object_passed and mission_color_passed)
    candidates = [
        {
            "feature_set": decision.baseline_feature_set,
            "passed": False,
            "mean_accuracy": mean(baseline["labels"][label]["accuracy"] for label in decision.labels),
            "mean_lift": mean(baseline["labels"][label]["lift"] for label in decision.labels),
        },
        {
            "feature_set": decision.candidate_feature_set,
            "passed": passed,
            "mean_accuracy": mean(candidate["labels"][label]["accuracy"] for label in decision.labels),
            "mean_lift": mean(candidate["labels"][label]["lift"] for label in decision.labels),
        },
    ]
    return {
        "met": passed,
        "best_feature_set": decision.candidate_feature_set if passed else decision.baseline_feature_set,
        "candidates": candidates,
        "comparisons": comparisons,
        "rule": {
            "mode": decision.mode,
            "labels": list(decision.labels),
            "min_test_examples": decision.min_test_examples,
            "baseline_feature_set": decision.baseline_feature_set,
            "candidate_feature_set": decision.candidate_feature_set,
            "transition_label": decision.transition_label,
            "transition_min_lift_delta": decision.transition_min_lift_delta,
            "external_transition_lift_baseline": decision.external_transition_lift_baseline,
            "max_mission_accuracy_drop": decision.max_mission_accuracy_drop,
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
    training_reports = report.get("training_reports")
    if isinstance(training_reports, dict) and training_reports:
        lines.extend(
            [
                "## Predictive encoders",
                "",
                "| name | target_label | train_accuracy | test_accuracy | test_majority | test_lift | epoch_mistakes |",
                "| --- | --- | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for name in sorted(training_reports):
            training_report = training_reports[name]
            lines.append(
                "| {name} | {target_label} | {train_accuracy:.3f} | {test_accuracy:.3f} | {test_majority:.3f} | {test_lift:.3f} | {epoch_mistakes} |".format(
                    name=name,
                    target_label=training_report["target_label"],
                    train_accuracy=training_report["train_accuracy"],
                    test_accuracy=training_report["test_accuracy"],
                    test_majority=training_report["test_majority_baseline"],
                    test_lift=training_report["test_lift"],
                    epoch_mistakes=",".join(str(item) for item in training_report["epoch_mistakes"]),
                )
            )
        lines.append("")
    else:
        training_report = report.get("training_report")
        if training_report:
            lines.extend(
                [
                    "## Predictive encoder",
                    "",
                    f"- target_label: `{training_report['target_label']}`",
                    f"- train_accuracy: `{training_report['train_accuracy']:.3f}`",
                    f"- test_accuracy: `{training_report['test_accuracy']:.3f}`",
                    f"- test_majority: `{training_report['test_majority_baseline']:.3f}`",
                    f"- test_lift: `{training_report['test_lift']:.3f}`",
                    f"- epoch_mistakes: `{','.join(str(item) for item in training_report['epoch_mistakes'])}`",
                    "",
                ]
            )
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


def _predict_linear_label(
    features: SparseFeatures,
    classes: tuple[str, ...],
    weights: dict[str, SparseFeatures],
) -> str:
    if not classes:
        return ""
    scored = [(_sparse_dot(features, weights[label]), label) for label in classes]
    best_score = max(score for score, _label in scored)
    return sorted(label for score, label in scored if score == best_score)[0]


def _add_scaled_features(target: SparseFeatures, features: SparseFeatures, scale: float) -> None:
    for index, value in features.items():
        updated = target.get(index, 0.0) + value * scale
        if abs(updated) < 1e-12:
            target.pop(index, None)
        else:
            target[index] = updated


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
