from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from baby_model.agents import QAgent
from baby_model.experiment import (
    Condition,
    _effective_intrinsic_beta,
    _intrinsic_reward,
    _q_target_reward,
    run_suite,
    validate_config,
    write_run,
)
from baby_model.gpu_compat import DriverVersion, build_report, evaluate_worker, parse_worker_policy
from baby_model.envs import BabyGrid
from baby_model.minigrid_curriculum import (
    curriculum_summary_markdown,
    parse_minigrid_curriculum_config,
    run_minigrid_curriculum_suite,
)
from baby_model.minigrid_experiment import encode_observation, parse_minigrid_config, run_minigrid_suite
from baby_model.minigrid_linear import (
    LinearQAgent,
    linear_features,
    linear_summary_markdown,
    parse_minigrid_linear_config,
    run_minigrid_linear_suite,
)
from baby_model.minigrid_linear_sweep import aggregate_linear_reports, linear_sweep_summary_markdown
from baby_model.minigrid_neural import (
    NeuralQAgent,
    neural_summary_markdown,
    parse_minigrid_neural_config,
    run_minigrid_neural_suite,
)
from baby_model.minigrid_probe import observation_schema, summary_markdown
from baby_model.minigrid_repr_probe import (
    centroid_probe_metrics,
    evaluate_probe_decision,
    evaluate_feature_set,
    parse_minigrid_representation_probe_config,
    scripted_object_action,
    target_relation_label,
    train_predictive_encoder,
    representation_probe_summary_markdown,
    transition_probe_labels,
    vector_to_sparse_features,
)
from baby_model.minigrid_repr_probe_sweep import (
    DEFAULT_CONFIG_PATH as REPR_PROBE_SWEEP_DEFAULT_CONFIG_PATH,
    aggregate_representation_probe_runs,
    evaluate_representation_probe_sweep_decision,
    representation_probe_sweep_summary_markdown,
)
import baby_model.minigrid_torch as minigrid_torch_module
from baby_model.minigrid_torch import (
    TorchAgentConfig,
    TorchCurriculumStage,
    action_prior_label,
    affordance_progress_vector,
    controllability_target,
    dense_feature_vector,
    mission_preservation_probe,
    mission_target_transition_vector,
    parse_minigrid_torch_config,
    run_minigrid_torch_curriculum_condition,
    select_torch_device,
    state_plus_delta_vector,
    state_plus_mission_target_vector,
    state_plus_target_visibility_vector,
    summarize_mission_preservation_probes,
    subgoal_progress_vector,
    target_visibility_relation,
    target_visibility_transition_vector,
    task_signal_vector,
    torch_summary_markdown,
    transition_group_vector,
)
from baby_model.minigrid_torch_sweep import aggregate_torch_reports, torch_sweep_summary_markdown
from baby_model.sweep import parse_seeds, run_sweep


class ExperimentTest(unittest.TestCase):
    def test_run_suite_returns_all_conditions(self) -> None:
        report = run_suite(seed=3)
        names = [row["name"] for row in report["results"]]
        self.assertEqual(
            names,
            ["A_end_to_end", "B_encoder_first", "C_baby_surprise", "D_baby_progress"],
        )
        self.assertIn(report["winner_last_window"], names)

    def test_qagent_action_bonus_can_change_greedy_choice(self) -> None:
        agent = QAgent(actions=2, epsilon=0.0, seed=1)
        feature = (0,)
        agent.q[(feature, 0)] = 1.0
        self.assertEqual(agent.choose(feature), 0)
        self.assertEqual(agent.choose(feature, action_bonus={1: 2.0}), 1)

    def test_write_run_creates_metrics_summary_and_latest_link(self) -> None:
        report = run_suite(
            config={
                "environment": {"size": 5, "max_steps": 20},
                "conditions": [
                        {
                            "name": "tiny",
                            "encoder_mode": "coarse",
                            "episodes": 4,
                            "decoder_delay_episodes": 1,
                            "intrinsic_beta": 0.01,
                            "intrinsic_mode": "progress",
                        }
                ],
            },
            seed=1,
        )
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = write_run(report, Path(tmp))
            self.assertTrue((run_dir / "metrics.json").exists())
            self.assertTrue((run_dir / "summary.md").exists())
            self.assertTrue((Path(tmp) / "latest").exists())

    def test_validate_config_rejects_invalid_delay(self) -> None:
        with self.assertRaises(ValueError):
            validate_config(
                {
                    "conditions": [
                        {
                            "name": "bad",
                            "encoder_mode": "coarse",
                            "episodes": 3,
                            "decoder_delay_episodes": 4,
                            "intrinsic_beta": 0.0,
                            "intrinsic_mode": "none",
                        }
                    ]
                }
            )

    def test_validate_config_rejects_invalid_intrinsic_mode(self) -> None:
        with self.assertRaises(ValueError):
            validate_config(
                {
                    "conditions": [
                        {
                            "name": "bad",
                            "encoder_mode": "coarse",
                            "episodes": 3,
                            "decoder_delay_episodes": 0,
                            "intrinsic_beta": 0.0,
                            "intrinsic_mode": "noise",
                        }
                    ]
                }
            )

    def test_validate_config_accepts_v03_intrinsic_controls(self) -> None:
        report = run_suite(
            config={
                "environment": {"size": 5, "max_steps": 20, "obstacle_count": 3, "toy_count": 1},
                "conditions": [
                    {
                        "name": "anneal",
                        "encoder_mode": "coarse",
                        "episodes": 4,
                        "decoder_delay_episodes": 1,
                        "intrinsic_beta": 0.05,
                        "intrinsic_beta_end": 0.0,
                        "intrinsic_anneal_episodes": 2,
                        "intrinsic_schedule": "linear_anneal",
                        "intrinsic_gate": "external_flat",
                        "intrinsic_target": "auxiliary",
                        "intrinsic_mode": "progress",
                    }
                ],
            },
            seed=2,
        )
        row = report["results"][0]
        self.assertEqual(row["intrinsic_schedule"], "linear_anneal")
        self.assertEqual(row["intrinsic_gate"], "external_flat")
        self.assertEqual(row["intrinsic_target"], "auxiliary")

    def test_validate_config_rejects_invalid_intrinsic_target(self) -> None:
        with self.assertRaises(ValueError):
            validate_config(
                {
                    "conditions": [
                        {
                            "name": "bad-target",
                            "encoder_mode": "coarse",
                            "episodes": 4,
                            "decoder_delay_episodes": 1,
                            "intrinsic_beta": 0.05,
                            "intrinsic_target": "actor_loss",
                            "intrinsic_mode": "progress",
                        }
                    ]
                }
            )

    def test_intrinsic_control_math(self) -> None:
        anneal = Condition(
            name="anneal",
            encoder_mode="coarse",
            episodes=10,
            decoder_delay_episodes=2,
            intrinsic_beta=0.1,
            intrinsic_mode="progress",
            seed=1,
            intrinsic_beta_end=0.0,
            intrinsic_anneal_episodes=4,
            intrinsic_schedule="linear_anneal",
        )
        self.assertAlmostEqual(_effective_intrinsic_beta(anneal, 2), 0.1)
        self.assertAlmostEqual(_effective_intrinsic_beta(anneal, 4), 0.05)
        self.assertAlmostEqual(_effective_intrinsic_beta(anneal, 6), 0.0)

        gated = Condition(
            name="gate",
            encoder_mode="coarse",
            episodes=4,
            decoder_delay_episodes=1,
            intrinsic_beta=0.1,
            intrinsic_mode="progress",
            seed=1,
            intrinsic_gate="external_flat",
        )
        self.assertEqual(_intrinsic_reward(gated, 2, 1.0, 3.0), 0.0)
        self.assertAlmostEqual(_intrinsic_reward(gated, 2, -0.01, 3.0), 0.3)

        auxiliary = Condition(
            name="aux",
            encoder_mode="coarse",
            episodes=4,
            decoder_delay_episodes=1,
            intrinsic_beta=0.1,
            intrinsic_mode="progress",
            seed=1,
            intrinsic_target="auxiliary",
        )
        self.assertEqual(_q_target_reward(auxiliary, 1.0, 0.3), 1.0)
        self.assertEqual(_q_target_reward(gated, 1.0, 0.3), 1.3)

    def test_obstacle_grid_has_walls_and_remains_runnable(self) -> None:
        env = BabyGrid(size=6, max_steps=20, seed=5, obstacle_count=6, toy_count=2)
        observation = env.reset(seed=5)
        self.assertEqual(sum(1 for value in observation if value == 1), 6)
        result = env.step(0)
        self.assertEqual(len(result.observation), 36)

    def test_obstacle_grid_rejects_overfull_layout(self) -> None:
        with self.assertRaises(ValueError):
            BabyGrid(size=4, obstacle_count=14, toy_count=1)

    def test_wall_collision_keeps_position_and_reports_blocked(self) -> None:
        env = BabyGrid(size=4, max_steps=10, collision_penalty=-0.25)
        env.agent = (1, 1)
        env.goal = (3, 3)
        env.walls = ((2, 1),)
        env.toys = ()
        result = env.step(1)
        self.assertEqual(env.agent, (1, 1))
        self.assertEqual(result.info["blocked"], 1)
        self.assertEqual(result.reward, -0.25)

    def test_run_sweep_aggregates_conditions(self) -> None:
        config = {
            "environment": {"size": 5, "max_steps": 20, "obstacle_count": 3, "toy_count": 1},
            "conditions": [
                {
                    "name": "base",
                    "encoder_mode": "coarse",
                    "episodes": 4,
                    "decoder_delay_episodes": 1,
                    "intrinsic_beta": 0.0,
                    "intrinsic_mode": "none",
                },
                {
                    "name": "progress",
                    "encoder_mode": "coarse",
                    "episodes": 4,
                    "decoder_delay_episodes": 1,
                    "intrinsic_beta": 0.1,
                    "intrinsic_mode": "progress",
                },
            ],
        }
        report = run_sweep(config, parse_seeds("1,2"))
        self.assertEqual(len(report["aggregate"]), 2)
        self.assertIn(report["winner_by_mean_success_last_window"], {"base", "progress"})
        self.assertEqual(report["seeds"], [1, 2])
        for row in report["aggregate"]:
            self.assertEqual(row["seeds"], [1, 2])
            self.assertIn("condition_seeds", row)

    def test_minigrid_probe_schema_and_summary_are_dependency_free(self) -> None:
        schema = observation_schema({"mission": "go to goal", "direction": 1, "image": [[0]]})
        self.assertEqual(schema["type"], "dict")
        self.assertEqual(schema["keys"], ["direction", "image", "mission"])

        summary = summary_markdown(
            {
                "created_at": "2026-06-29T00:00:00+00:00",
                "hypothesis": "probe",
                "episodes": 1,
                "max_steps": 2,
                "seed": 3,
                "results": [
                    {
                        "env_id": "MiniGrid-Empty-8x8-v0",
                        "action_n": 7,
                        "observation_schema": schema,
                        "success_rate": 0.0,
                        "mean_return": 0.0,
                        "mean_steps": 2.0,
                    }
                ],
            }
        )
        self.assertIn("MiniGrid-Empty-8x8-v0", summary)

    def test_minigrid_config_and_encoder_are_dependency_free(self) -> None:
        parsed = parse_minigrid_config(
            {
                "environment": {"id": "MiniGrid-Empty-8x8-v0", "max_steps": 4},
                "conditions": [
                    {
                        "name": "base",
                        "encoder_mode": "coarse",
                        "episodes": 3,
                        "decoder_delay_episodes": 1,
                        "intrinsic_beta": 0.0,
                        "intrinsic_mode": "none",
                    }
                ],
            },
            seed=7,
        )
        self.assertEqual(parsed.env_id, "MiniGrid-Empty-8x8-v0")
        self.assertIs(parsed.quiet_env_output, True)
        self.assertEqual(parsed.conditions[0].seed, 7)

        class FakeImage:
            def tolist(self) -> list[list[list[int]]]:
                return [[[0, 0, 0] for _ in range(7)] for _ in range(7)]

        feature = encode_observation({"image": FakeImage(), "direction": 2}, "coarse")
        self.assertEqual(feature[:4], (2, 0, 0, 0))

    def test_minigrid_suite_forwards_quiet_env_output(self) -> None:
        class FakeImage:
            def tolist(self) -> list[list[list[int]]]:
                return [[[0, 0, 0] for _ in range(7)] for _ in range(7)]

        observation = {"image": FakeImage(), "direction": 0, "mission": "go"}

        class FakeActionSpace:
            n = 2

            def seed(self, seed: int) -> None:
                self.last_seed = seed

        class FakeEnv:
            action_space = FakeActionSpace()

            def reset(self, seed: int) -> tuple[dict[str, object], dict[str, object]]:
                print("fake-reset")
                return observation, {}

            def step(self, action: int) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
                print("fake-step")
                return observation, 0.0, True, False, {}

            def close(self) -> None:
                pass

        fake_gym = types.ModuleType("gymnasium")
        fake_gym.make = lambda env_id: FakeEnv()  # type: ignore[attr-defined]
        fake_minigrid = types.ModuleType("minigrid")
        previous_gym = sys.modules.get("gymnasium")
        previous_minigrid = sys.modules.get("minigrid")
        sys.modules["gymnasium"] = fake_gym
        sys.modules["minigrid"] = fake_minigrid
        try:
            quiet_config = {
                "environment": {"id": "FakeMiniGrid-v0", "max_steps": 1, "quiet_env_output": True},
                "conditions": [
                    {
                        "name": "base",
                        "encoder_mode": "coarse",
                        "episodes": 1,
                        "decoder_delay_episodes": 0,
                        "intrinsic_beta": 0.0,
                        "intrinsic_mode": "none",
                    }
                ],
            }
            noisy_config = {
                **quiet_config,
                "environment": {"id": "FakeMiniGrid-v0", "max_steps": 1, "quiet_env_output": False},
            }

            quiet_stdout = io.StringIO()
            with contextlib.redirect_stdout(quiet_stdout):
                run_minigrid_suite(quiet_config, seed=11)
            self.assertEqual(quiet_stdout.getvalue(), "")

            noisy_stdout = io.StringIO()
            with contextlib.redirect_stdout(noisy_stdout):
                run_minigrid_suite(noisy_config, seed=11)
            self.assertIn("fake-reset", noisy_stdout.getvalue())
            self.assertIn("fake-step", noisy_stdout.getvalue())
        finally:
            if previous_gym is None:
                sys.modules.pop("gymnasium", None)
            else:
                sys.modules["gymnasium"] = previous_gym
            if previous_minigrid is None:
                sys.modules.pop("minigrid", None)
            else:
                sys.modules["minigrid"] = previous_minigrid

    def test_minigrid_curriculum_config_and_runner_are_dependency_free(self) -> None:
        config = {
            "stages": [
                {"name": "warmup", "env_id": "FakeWarmup-v0", "max_steps": 1, "episodes": 2},
                {"name": "eval", "env_id": "FakeEval-v0", "max_steps": 1, "episodes": 2},
            ],
            "conditions": [
                {
                    "name": "hard_only",
                    "encoder_mode": "raw",
                    "active_stages": ["eval"],
                    "decoder_delay_episodes": 0,
                    "intrinsic_beta": 0.0,
                    "intrinsic_mode": "none",
                },
                {
                    "name": "curriculum",
                    "encoder_mode": "coarse",
                    "active_stages": ["warmup", "eval"],
                    "decoder_delay_episodes": 1,
                    "intrinsic_beta": 0.0,
                    "intrinsic_mode": "none",
                },
            ],
        }
        parsed = parse_minigrid_curriculum_config(config, seed=17)
        self.assertEqual([stage.name for stage in parsed.stages], ["warmup", "eval"])
        self.assertEqual(parsed.conditions[0].condition.episodes, 2)
        self.assertEqual(parsed.conditions[1].condition.episodes, 4)

        class FakeImage:
            def tolist(self) -> list[list[list[int]]]:
                return [[[0, 0, 0] for _ in range(7)] for _ in range(7)]

        observation = {"image": FakeImage(), "direction": 0, "mission": "go"}
        made_envs: list[str] = []

        class FakeActionSpace:
            n = 2

            def seed(self, seed: int) -> None:
                self.last_seed = seed

        class FakeEnv:
            def __init__(self, env_id: str) -> None:
                self.env_id = env_id
                self.action_space = FakeActionSpace()

            def reset(self, seed: int) -> tuple[dict[str, object], dict[str, object]]:
                return observation, {}

            def step(self, action: int) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
                reward = 1.0 if self.env_id == "FakeEval-v0" else 0.0
                return observation, reward, True, False, {}

            def close(self) -> None:
                pass

        fake_gym = types.ModuleType("gymnasium")

        def make_env(env_id: str) -> FakeEnv:
            made_envs.append(env_id)
            return FakeEnv(env_id)

        fake_gym.make = make_env  # type: ignore[attr-defined]
        fake_minigrid = types.ModuleType("minigrid")
        previous_gym = sys.modules.get("gymnasium")
        previous_minigrid = sys.modules.get("minigrid")
        sys.modules["gymnasium"] = fake_gym
        sys.modules["minigrid"] = fake_minigrid
        try:
            report = run_minigrid_curriculum_suite(config, seed=17)
        finally:
            if previous_gym is None:
                sys.modules.pop("gymnasium", None)
            else:
                sys.modules["gymnasium"] = previous_gym
            if previous_minigrid is None:
                sys.modules.pop("minigrid", None)
            else:
                sys.modules["minigrid"] = previous_minigrid

        self.assertEqual(made_envs, ["FakeEval-v0", "FakeWarmup-v0", "FakeEval-v0"])
        self.assertIn(report["winner_final_last_window"], {"hard_only", "curriculum"})
        self.assertEqual(report["results"][1]["final_stage"]["stage"], "eval")
        self.assertIn("MiniGrid curriculum", curriculum_summary_markdown(report))

    def test_minigrid_linear_agent_updates_weights(self) -> None:
        agent = LinearQAgent(actions=2, feature_dim=128, alpha=0.5, gamma=0.0, epsilon=0.0, seed=1)
        features = {3: 1.0, 7: 1.0}
        self.assertEqual(agent.q_value(features, 1), 0.0)
        agent.update(features, action=1, reward=1.0, next_features=features, done=True)
        self.assertGreater(agent.q_value(features, 1), 0.0)

    def test_minigrid_linear_config_features_and_runner_are_dependency_free(self) -> None:
        config = {
            "environment": {"id": "FakeLinear-v0", "max_steps": 1},
            "agent": {"feature_dim": 256, "alpha": 0.1, "gamma": 0.9, "epsilon": 0.0},
            "conditions": [
                {
                    "name": "linear",
                    "encoder_mode": "raw",
                    "episodes": 2,
                    "decoder_delay_episodes": 0,
                    "intrinsic_beta": 0.0,
                    "intrinsic_mode": "none",
                }
            ],
        }
        parsed = parse_minigrid_linear_config(config, seed=23)
        self.assertEqual(parsed.agent.feature_dim, 256)
        self.assertEqual(parsed.conditions[0].seed, 23)

        class FakeImage:
            def tolist(self) -> list[list[list[int]]]:
                return [[[0, 0, 0] for _ in range(7)] for _ in range(7)]

        observation = {"image": FakeImage(), "direction": 1, "mission": "unlock red door"}
        self.assertEqual(linear_features(observation, "raw", 256), linear_features(observation, "raw", 256))
        self.assertLessEqual(max(linear_features(observation, "raw", 256)), 255)

        class FakeActionSpace:
            n = 2

            def seed(self, seed: int) -> None:
                self.last_seed = seed

        class FakeEnv:
            action_space = FakeActionSpace()

            def reset(self, seed: int) -> tuple[dict[str, object], dict[str, object]]:
                return observation, {}

            def step(self, action: int) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
                return observation, 1.0, True, False, {}

            def close(self) -> None:
                pass

        fake_gym = types.ModuleType("gymnasium")
        fake_gym.make = lambda env_id: FakeEnv()  # type: ignore[attr-defined]
        fake_minigrid = types.ModuleType("minigrid")
        previous_gym = sys.modules.get("gymnasium")
        previous_minigrid = sys.modules.get("minigrid")
        sys.modules["gymnasium"] = fake_gym
        sys.modules["minigrid"] = fake_minigrid
        try:
            report = run_minigrid_linear_suite(config, seed=23)
        finally:
            if previous_gym is None:
                sys.modules.pop("gymnasium", None)
            else:
                sys.modules["gymnasium"] = previous_gym
            if previous_minigrid is None:
                sys.modules.pop("minigrid", None)
            else:
                sys.modules["minigrid"] = previous_minigrid

        self.assertEqual(report["winner_last_window"], "linear")
        self.assertGreater(report["results"][0]["nonzero_weights"], 0)
        self.assertIn("linear function approximation", linear_summary_markdown(report))

    def test_minigrid_linear_sweep_aggregate_is_dependency_free(self) -> None:
        runs = [
            {
                "winner_last_window": "B",
                "results": [
                    {
                        "name": "A",
                        "seed": 1,
                        "success_rate_all": 0.0,
                        "success_rate_last_window": 0.0,
                        "mean_return_last_window": 0.0,
                        "nonzero_weights": 10,
                    },
                    {
                        "name": "B",
                        "seed": 2,
                        "success_rate_all": 0.1,
                        "success_rate_last_window": 0.2,
                        "mean_return_last_window": 0.3,
                        "nonzero_weights": 20,
                    },
                ],
            },
            {
                "winner_last_window": "A",
                "results": [
                    {
                        "name": "A",
                        "seed": 3,
                        "success_rate_all": 0.2,
                        "success_rate_last_window": 0.4,
                        "mean_return_last_window": 0.5,
                        "nonzero_weights": 30,
                    },
                    {
                        "name": "B",
                        "seed": 4,
                        "success_rate_all": 0.0,
                        "success_rate_last_window": 0.0,
                        "mean_return_last_window": 0.0,
                        "nonzero_weights": 40,
                    },
                ],
            },
        ]
        aggregate = aggregate_linear_reports(runs, seeds=[101, 102])
        by_name = {row["name"]: row for row in aggregate}
        self.assertEqual(by_name["A"]["win_count"], 1)
        self.assertEqual(by_name["B"]["win_count"], 1)
        self.assertAlmostEqual(by_name["A"]["mean_success_rate_last_window"], 0.2)
        self.assertEqual(by_name["B"]["condition_seeds"], [2, 4])
        summary = linear_sweep_summary_markdown(
            {
                "created_at": "2026-06-29T00:00:00+00:00",
                "hypothesis": "linear sweep",
                "seeds": [101, 102],
                "winner_by_mean_success_last_window": "A",
                "aggregate": aggregate,
                "runs": runs,
            }
        )
        self.assertIn("Per-Seed Winners", summary)

    def test_minigrid_neural_agent_updates_parameters(self) -> None:
        agent = NeuralQAgent(
            actions=2,
            feature_dim=128,
            hidden_dim=4,
            alpha_output=0.2,
            alpha_hidden=0.1,
            gamma=0.0,
            epsilon=0.0,
            seed=3,
        )
        features = {3: 1.0, 7: 1.0}
        before = agent.nonzero_parameters()
        agent.update(features, action=1, reward=1.0, next_features=features, done=True)
        self.assertGreater(agent.nonzero_parameters(), before)

    def test_minigrid_neural_config_and_runner_are_dependency_free(self) -> None:
        config = {
            "environment": {"id": "FakeNeural-v0", "max_steps": 1},
            "agent": {
                "feature_dim": 256,
                "hidden_dim": 4,
                "alpha_output": 0.1,
                "alpha_hidden": 0.01,
                "gamma": 0.9,
                "epsilon": 0.0,
                "init_scale": 0.02,
            },
            "conditions": [
                {
                    "name": "neural",
                    "encoder_mode": "raw",
                    "episodes": 2,
                    "decoder_delay_episodes": 0,
                    "intrinsic_beta": 0.0,
                    "intrinsic_mode": "none",
                }
            ],
        }
        parsed = parse_minigrid_neural_config(config, seed=31)
        self.assertEqual(parsed.agent.hidden_dim, 4)
        self.assertEqual(parsed.conditions[0].seed, 31)

        class FakeImage:
            def tolist(self) -> list[list[list[int]]]:
                return [[[0, 0, 0] for _ in range(7)] for _ in range(7)]

        observation = {"image": FakeImage(), "direction": 1, "mission": "unlock red door"}

        class FakeActionSpace:
            n = 2

            def seed(self, seed: int) -> None:
                self.last_seed = seed

        class FakeEnv:
            action_space = FakeActionSpace()

            def reset(self, seed: int) -> tuple[dict[str, object], dict[str, object]]:
                return observation, {}

            def step(self, action: int) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
                return observation, 1.0, True, False, {}

            def close(self) -> None:
                pass

        fake_gym = types.ModuleType("gymnasium")
        fake_gym.make = lambda env_id: FakeEnv()  # type: ignore[attr-defined]
        fake_minigrid = types.ModuleType("minigrid")
        previous_gym = sys.modules.get("gymnasium")
        previous_minigrid = sys.modules.get("minigrid")
        sys.modules["gymnasium"] = fake_gym
        sys.modules["minigrid"] = fake_minigrid
        try:
            report = run_minigrid_neural_suite(config, seed=31)
        finally:
            if previous_gym is None:
                sys.modules.pop("gymnasium", None)
            else:
                sys.modules["gymnasium"] = previous_gym
            if previous_minigrid is None:
                sys.modules.pop("minigrid", None)
            else:
                sys.modules["minigrid"] = previous_minigrid

        self.assertEqual(report["winner_last_window"], "neural")
        self.assertGreater(report["results"][0]["nonzero_parameters"], 0)
        self.assertIn("neural encoder", neural_summary_markdown(report))

    def test_minigrid_torch_config_helpers_are_dependency_free(self) -> None:
        config = {
            "environment": {"id": "BabyAI-Unlock-v0", "max_steps": 3},
            "agent": {
                "feature_dim": 32,
                "hidden_dim": 8,
                "learning_rate": 0.001,
                "gamma": 0.9,
                "epsilon": 0.0,
                "batch_size": 2,
                "replay_capacity": 8,
                "target_sync_updates": 3,
                "device": "auto",
            },
            "conditions": [
                {
                    "name": "torch",
                    "encoder_mode": "raw",
                    "episodes": 2,
                    "decoder_delay_episodes": 1,
                    "intrinsic_beta": 0.0,
                    "intrinsic_mode": "none",
                }
            ],
        }
        parsed = parse_minigrid_torch_config(config, seed=41)
        self.assertEqual(parsed.agent.feature_dim, 32)
        self.assertEqual(parsed.conditions[0].seed, 41)

        vector = dense_feature_vector({0: 1.0, 3: 0.5}, 5)
        self.assertEqual(vector, [1.0, 0.0, 0.0, 0.5, 0.0])

        class FakeCuda:
            def __init__(self, available: bool) -> None:
                self.available = available

            def is_available(self) -> bool:
                return self.available

        class FakeMps:
            def __init__(self, available: bool) -> None:
                self.available = available

            def is_available(self) -> bool:
                return self.available

        class FakeTorch:
            def __init__(self, cuda_available: bool, mps_available: bool) -> None:
                self.cuda = FakeCuda(cuda_available)
                self.backends = types.SimpleNamespace(mps=FakeMps(mps_available))

            def device(self, name: str) -> str:
                return f"device:{name}"

        self.assertEqual(select_torch_device(FakeTorch(cuda_available=True, mps_available=True), "auto"), "device:cuda")
        self.assertEqual(select_torch_device(FakeTorch(cuda_available=False, mps_available=True), "auto"), "device:mps")
        self.assertEqual(select_torch_device(FakeTorch(cuda_available=False, mps_available=False), "auto"), "device:cpu")

        summary = torch_summary_markdown(
            {
                "created_at": "2026-06-29T00:00:00+00:00",
                "hypothesis": "torch dqn",
                "env_id": "BabyAI-Unlock-v0",
                "framework": {"version": "fake", "device": "cpu"},
                "winner_last_window": "torch",
                "results": [
                    {
                        "name": "torch",
                        "success_rate_all": 0.0,
                        "success_rate_last_window": 0.0,
                        "mean_return_last_window": 0.0,
                        "mean_steps_success": None,
                        "updates": 1,
                        "parameter_count": 10,
                    }
                ],
            }
        )
        self.assertIn("PyTorch DQN", summary)
        self.assertIn("target_visible_last", summary)

    def test_minigrid_torch_v11_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v11.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=601)
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only",
                "F_torch_short_delay",
                "G_torch_aux_progress_short",
                "H_torch_aux_progress_coarse",
            ],
        )
        aux = parsed.conditions[2]
        self.assertEqual(aux.decoder_delay_episodes, 2)
        self.assertEqual(aux.intrinsic_mode, "progress")
        self.assertEqual(aux.intrinsic_target, "auxiliary")

    def test_minigrid_torch_v12_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v12.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=701)
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "I_torch_long_delay",
                "J_torch_long_aux_progress",
                "K_torch_long_coarse_aux",
            ],
        )
        self.assertEqual({condition.episodes for condition in parsed.conditions}, {48})
        self.assertEqual(parsed.agent.batch_size, 16)
        self.assertEqual(parsed.agent.replay_capacity, 1024)
        aux = parsed.conditions[2]
        self.assertEqual(aux.decoder_delay_episodes, 4)
        self.assertEqual(aux.intrinsic_mode, "progress")
        self.assertEqual(aux.intrinsic_target, "auxiliary")

    def test_minigrid_torch_v13_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v13.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=901)
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "L_torch_predictive_delay",
                "M_torch_predictive_aux_progress",
            ],
        )
        predictive = parsed.conditions[1]
        self.assertEqual(predictive.representation_objective, "next_feature")
        self.assertEqual(predictive.representation_beta, 0.10)
        self.assertEqual(predictive.decoder_delay_episodes, 4)
        self.assertEqual(parsed.conditions[2].intrinsic_target, "auxiliary")

    def test_minigrid_torch_task_signal_vector_is_dependency_free(self) -> None:
        observation = {
            "direction": 2,
            "mission": "unlock the door with the key",
            "image": [
                [[4, 1, 2], [5, 0, 0]],
                [[8, 0, 0], [2, 0, 0]],
            ],
        }
        vector = task_signal_vector(observation)
        self.assertEqual(len(vector), 16)
        self.assertAlmostEqual(vector[0], 2 / 3)
        self.assertEqual(vector[1:5], [1.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(vector[5], 0.25)
        self.assertAlmostEqual(vector[6], 0.25)
        self.assertAlmostEqual(vector[7], 0.25)
        self.assertAlmostEqual(vector[13], 0.25)

    def test_minigrid_torch_action_prior_label_is_dependency_free(self) -> None:
        def observation_with_front(front_cell: list[int], mission: str = "unlock the door with the key") -> dict[str, object]:
            image = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
            image[2][3] = [2, 0, 0]
            image[3][5] = front_cell
            return {"direction": 0, "mission": mission, "image": image}

        self.assertEqual(action_prior_label(observation_with_front([5, 0, 0]), actions=7), 3)
        self.assertEqual(action_prior_label(observation_with_front([4, 0, 2]), actions=7), 5)
        self.assertEqual(action_prior_label(observation_with_front([2, 0, 0]), actions=7), 0)
        self.assertEqual(action_prior_label(observation_with_front([8, 0, 0], mission="go to the goal"), actions=7), 2)

    def test_minigrid_torch_controllability_target_is_dependency_free(self) -> None:
        self.assertEqual(controllability_target({1: 1.0, 2: 0.0}, {1: 1.0}), [0.0])
        self.assertEqual(controllability_target({1: 1.0}, {1: 1.0, 3: 1.0}), [1.0])

    def test_minigrid_torch_affordance_progress_vector_is_dependency_free(self) -> None:
        image = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        image[0][0] = [5, 0, 0]
        image[1][0] = [8, 0, 0]
        image[2][0] = [4, 0, 2]
        image[3][5] = [4, 0, 2]
        vector = affordance_progress_vector(
            {"direction": 3, "mission": "unlock the door with the key", "image": image}
        )
        self.assertEqual(len(vector), 16)
        self.assertEqual(vector[0], 1.0)
        self.assertEqual(vector[1:5], [1.0, 0.0, 1.0, 1.0])
        self.assertEqual(vector[5:11], [0.0, 1.0, 0.0, 1.0, 0.0, 0.0])
        self.assertEqual(vector[11:16], [1.0, 1.0, 1.0, 1.0, 0.0])

    def test_minigrid_torch_transition_group_vector_is_dependency_free(self) -> None:
        before = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        after = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        before[3][5] = [4, 0, 2]
        after[3][5] = [4, 0, 0]
        after[0][0] = [5, 0, 0]
        vector = transition_group_vector(
            {"direction": 0, "mission": "unlock the door with the key", "image": before},
            {"direction": 1, "mission": "unlock the door with the key", "image": after},
        )
        self.assertEqual(len(vector), 16)
        self.assertEqual(vector[0], 1.0)
        self.assertEqual(vector[6:9], [0.0, 1.0, 1.0])
        self.assertEqual(vector[11:14], [1.0, 0.0, 1.0])
        self.assertEqual(vector[15], 1.0)

    def test_minigrid_torch_target_visibility_vector_is_dependency_free(self) -> None:
        before = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        after = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        after[3][5] = [6, 0, 0]
        before_observation = {"direction": 0, "mission": "go to the red ball", "image": before}
        after_observation = {"direction": 0, "mission": "go to the red ball", "image": after}
        vector = target_visibility_transition_vector(before_observation, after_observation)
        self.assertEqual(len(vector), 49)
        self.assertEqual(sum(vector), 1.0)
        self.assertEqual(target_visibility_relation(before_observation), "absent")
        self.assertEqual(target_visibility_relation(after_observation), "center_near")
        self.assertEqual(vector[3], 1.0)

        left_far = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        right_near = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        left_far[0][0] = [7, 2, 0]
        right_near[5][6] = [7, 2, 0]
        left_observation = {"direction": 0, "mission": "go to the blue box", "image": left_far}
        right_observation = {"direction": 0, "mission": "go to the blue box", "image": right_near}
        relation_vector = target_visibility_transition_vector(left_observation, right_observation)
        self.assertEqual(target_visibility_relation(left_observation), "left_far")
        self.assertEqual(target_visibility_relation(right_observation), "right_near")
        self.assertEqual(relation_vector[2 * 7 + 5], 1.0)

    def test_minigrid_torch_mission_preservation_probe_is_dependency_free(self) -> None:
        absent = {"direction": 0, "mission": "go to the red ball", "image": [[[0, 0, 0] for _ in range(7)] for _ in range(7)]}
        center_near_image = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        center_near_image[3][5] = [6, 0, 0]
        center_near = {"direction": 0, "mission": "go to the red ball", "image": center_near_image}
        right_far_image = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        right_far_image[6][0] = [7, 2, 0]
        right_far = {"direction": 0, "mission": "go to the blue box", "image": right_far_image}

        absent_probe = mission_preservation_probe(absent)
        center_probe = mission_preservation_probe(center_near)
        far_probe = mission_preservation_probe(right_far)
        summary = summarize_mission_preservation_probes([absent_probe, center_probe, far_probe])

        self.assertEqual(absent_probe["mission_target_relation"], "absent")
        self.assertEqual(center_probe["mission_target_relation"], "center_near")
        self.assertEqual(far_probe["mission_target_relation"], "right_far")
        self.assertEqual(center_probe["mission_target_known"], 1.0)
        self.assertEqual(center_probe["mission_target_visible"], 1.0)
        self.assertEqual(center_probe["mission_target_center"], 1.0)
        self.assertEqual(center_probe["mission_target_near"], 1.0)
        self.assertAlmostEqual(summary["mission_target_visible_rate_all"], 2 / 3)
        self.assertAlmostEqual(summary["mission_target_center_rate_last_window"], 1 / 3)
        self.assertAlmostEqual(summary["mission_target_near_rate_last_window"], 1 / 3)

    def test_minigrid_torch_subgoal_progress_vector_is_dependency_free(self) -> None:
        before = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        after = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        before[0][0] = [5, 0, 0]
        before[1][0] = [4, 0, 2]
        after[1][0] = [4, 0, 0]
        after[2][0] = [8, 0, 0]
        after[3][5] = [8, 0, 0]
        vector = subgoal_progress_vector(
            {"direction": 0, "mission": "unlock the door with the key", "image": before},
            {"direction": 1, "mission": "unlock the door with the key", "image": after},
        )
        self.assertEqual(len(vector), 10)
        self.assertEqual(vector, [1.0, 0.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0])

        key_only_before = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        key_only_before[0][0] = [5, 0, 0]
        approach = subgoal_progress_vector(
            {"direction": 0, "mission": "unlock the door with the key", "image": key_only_before},
            {
                "direction": 0,
                "mission": "unlock the door with the key",
                "image": [[[0, 0, 0] for _ in range(7)] for _ in range(7)],
            },
        )
        self.assertEqual(approach, [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])

    def test_minigrid_torch_state_plus_delta_vector_is_dependency_free(self) -> None:
        before = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        after = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        before[0][0] = [5, 0, 0]
        before[1][0] = [4, 0, 2]
        after[1][0] = [4, 0, 0]
        after[2][0] = [8, 0, 0]
        before_observation = {"direction": 0, "mission": "unlock the door with the key", "image": before}
        after_observation = {"direction": 1, "mission": "unlock the door with the key", "image": after}
        vector = state_plus_delta_vector(before_observation, after_observation)
        current = affordance_progress_vector(before_observation)
        next_state = affordance_progress_vector(after_observation)
        transition = transition_group_vector(before_observation, after_observation)
        subgoal = subgoal_progress_vector(before_observation, after_observation)

        self.assertEqual(len(vector), 58)
        self.assertEqual(vector[:16], current)
        self.assertEqual(vector[16:32], next_state)
        self.assertEqual(vector[32:48], transition)
        self.assertEqual(vector[48:], subgoal)

    def test_minigrid_torch_state_plus_target_visibility_vector_is_dependency_free(self) -> None:
        before = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        after = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        before[0][0] = [5, 0, 0]
        after[3][5] = [6, 0, 0]
        before_observation = {"direction": 0, "mission": "go to the red ball", "image": before}
        after_observation = {"direction": 1, "mission": "go to the red ball", "image": after}
        state_delta = state_plus_delta_vector(before_observation, after_observation)
        target_visibility = target_visibility_transition_vector(before_observation, after_observation)
        vector = state_plus_target_visibility_vector(before_observation, after_observation)

        self.assertEqual(len(vector), 107)
        self.assertEqual(vector[:58], state_delta)
        self.assertEqual(vector[58:], target_visibility)
        self.assertEqual(sum(vector[58:]), 1.0)

    def test_minigrid_torch_mission_target_transition_vector_is_dependency_free(self) -> None:
        absent = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        visible = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        visible[3][5] = [6, 0, 0]
        absent_observation = {"direction": 0, "mission": "go to the red ball", "image": absent}
        visible_observation = {"direction": 0, "mission": "go to the red ball", "image": visible}
        unknown_observation = {"direction": 0, "mission": "go to the thing", "image": visible}

        absent_to_absent = mission_target_transition_vector(absent_observation, absent_observation)
        absent_to_visible = mission_target_transition_vector(absent_observation, visible_observation)
        unknown_to_visible = mission_target_transition_vector(unknown_observation, visible_observation)
        combined = state_plus_mission_target_vector(absent_observation, visible_observation)

        self.assertEqual(len(absent_to_absent), 49)
        self.assertEqual(sum(absent_to_absent), 0.0)
        self.assertEqual(sum(unknown_to_visible), 0.0)
        self.assertEqual(sum(absent_to_visible), 1.0)
        self.assertEqual(absent_to_visible[3], 1.0)
        self.assertEqual(len(combined), 107)
        self.assertEqual(combined[:58], state_plus_delta_vector(absent_observation, visible_observation))
        self.assertEqual(combined[58:], absent_to_visible)

    def test_minigrid_torch_v14_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v14.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=1001)
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "N_torch_task_signal_delay",
                "O_torch_task_signal_aux_progress",
            ],
        )
        predictive = parsed.conditions[1]
        self.assertEqual(predictive.representation_objective, "next_task_signal")
        self.assertEqual(predictive.representation_beta, 0.3)
        self.assertEqual(parsed.conditions[2].intrinsic_target, "auxiliary")

    def test_minigrid_torch_v15_curriculum_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v15.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=1101)
        self.assertEqual([stage.name for stage in parsed.stages], ["empty_warmup", "goto_warmup", "unlock_eval"])
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "P_torch_curriculum_task_signal_delay",
                "Q_torch_curriculum_task_signal_aux_progress",
            ],
        )
        active = dict(parsed.active_stages_by_condition)
        self.assertEqual(active["A_torch_hard_only_long"], ("unlock_eval",))
        self.assertEqual(
            active["P_torch_curriculum_task_signal_delay"],
            ("empty_warmup", "goto_warmup", "unlock_eval"),
        )
        self.assertEqual(parsed.conditions[0].episodes, 48)
        self.assertEqual(parsed.conditions[1].episodes, 84)
        self.assertEqual(parsed.conditions[1].representation_objective, "next_task_signal")
        self.assertEqual(parsed.conditions[2].intrinsic_target, "auxiliary")

    def test_minigrid_torch_v16_action_prior_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v16.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=1201)
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "R_torch_action_prior_delay",
                "S_torch_action_prior_policy_mix",
            ],
        )
        self.assertEqual(parsed.conditions[1].representation_objective, "action_prior")
        self.assertEqual(parsed.conditions[1].representation_beta, 0.2)
        self.assertEqual(parsed.conditions[1].action_prior_weight, 0.0)
        self.assertEqual(parsed.conditions[2].representation_objective, "action_prior")
        self.assertEqual(parsed.conditions[2].action_prior_weight, 0.25)

    def test_minigrid_torch_v17_controllability_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v17.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=1301)
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "T_torch_controllability_delay",
                "U_torch_controllability_aux_progress",
            ],
        )
        self.assertEqual(parsed.conditions[1].representation_objective, "controllability")
        self.assertEqual(parsed.conditions[1].representation_beta, 0.3)
        self.assertEqual(parsed.conditions[2].intrinsic_target, "auxiliary")

    def test_minigrid_torch_v18_dense_ladder_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v18.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=1401)
        self.assertEqual(
            [stage.name for stage in parsed.stages],
            [
                "empty_warmup",
                "goto_warmup",
                "goto_door_warmup",
                "open_door_warmup",
                "doorkey_warmup",
                "unlock_local_warmup",
                "unlock_eval",
            ],
        )
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "T_torch_controllability_delay",
                "V_torch_dense_ladder_controllability_delay",
                "W_torch_dense_ladder_controllability_aux_progress",
            ],
        )
        active = dict(parsed.active_stages_by_condition)
        self.assertEqual(active["A_torch_hard_only_long"], ("unlock_eval",))
        self.assertEqual(active["T_torch_controllability_delay"], ("empty_warmup", "goto_warmup", "unlock_eval"))
        self.assertIn("doorkey_warmup", active["V_torch_dense_ladder_controllability_delay"])
        self.assertEqual(parsed.conditions[2].episodes, 144)
        self.assertEqual(parsed.conditions[2].representation_objective, "controllability")
        self.assertEqual(parsed.conditions[3].intrinsic_target, "auxiliary")

    def test_minigrid_torch_v19_affordance_progress_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v19.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=1501)
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "T_torch_controllability_delay",
                "X_torch_affordance_progress_delay",
                "Y_torch_affordance_progress_aux_progress",
            ],
        )
        self.assertEqual(parsed.conditions[2].representation_objective, "affordance_progress")
        self.assertEqual(parsed.conditions[2].representation_beta, 0.3)
        self.assertEqual(parsed.conditions[3].intrinsic_target, "auxiliary")

    def test_minigrid_torch_v20_transition_group_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v20.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=1601)
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "T_torch_controllability_delay",
                "Z_torch_transition_group_delay",
                "ZA_torch_transition_group_aux_progress",
            ],
        )
        self.assertEqual(parsed.conditions[2].representation_objective, "transition_group")
        self.assertEqual(parsed.conditions[2].representation_beta, 0.3)
        self.assertEqual(parsed.conditions[3].intrinsic_target, "auxiliary")

    def test_minigrid_torch_v21_subgoal_progress_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v21.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=1701)
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "T_torch_controllability_delay",
                "ZB_torch_subgoal_progress_delay",
                "ZC_torch_subgoal_progress_aux_progress",
            ],
        )
        self.assertEqual(parsed.conditions[2].representation_objective, "subgoal_progress")
        self.assertEqual(parsed.conditions[2].representation_beta, 0.3)
        self.assertEqual(parsed.conditions[3].intrinsic_target, "auxiliary")

    def test_minigrid_torch_v22_state_plus_delta_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v22.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=1801)
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "T_torch_controllability_delay",
                "ZD_torch_state_plus_delta_delay",
                "ZE_torch_state_plus_delta_aux_progress",
            ],
        )
        self.assertEqual(parsed.conditions[2].representation_objective, "state_plus_delta")
        self.assertEqual(parsed.conditions[2].representation_beta, 0.3)
        self.assertEqual(parsed.conditions[3].intrinsic_target, "auxiliary")

    def test_minigrid_torch_v23_dense_keydoor_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v23.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=1901)
        self.assertEqual(
            [stage.name for stage in parsed.stages],
            [
                "empty_warmup",
                "goto_warmup",
                "goto_door_warmup",
                "open_door_warmup",
                "doorkey_warmup",
                "unlock_local_warmup",
                "unlock_pickup_warmup",
                "unlock_eval",
            ],
        )
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "T_torch_controllability_delay",
                "ZF_torch_dense_keydoor_state_plus_delta_delay",
                "ZG_torch_dense_keydoor_state_plus_delta_aux_progress",
            ],
        )
        active = dict(parsed.active_stages_by_condition)
        self.assertEqual(active["A_torch_hard_only_long"], ("unlock_eval",))
        self.assertEqual(active["T_torch_controllability_delay"], ("empty_warmup", "goto_warmup", "unlock_eval"))
        self.assertIn("unlock_pickup_warmup", active["ZF_torch_dense_keydoor_state_plus_delta_delay"])
        self.assertEqual(parsed.conditions[2].episodes, 164)
        self.assertEqual(parsed.conditions[2].representation_objective, "state_plus_delta")
        self.assertEqual(parsed.conditions[3].intrinsic_target, "auxiliary")

    def test_minigrid_torch_v24_two_phase_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v24.json")
        parsed = parse_minigrid_torch_config(json.loads(config_path.read_text(encoding="utf-8")), seed=2001)
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_hard_only_long",
                "T_torch_controllability_delay",
                "ZF_torch_dense_keydoor_state_plus_delta_delay",
                "ZH_torch_two_phase_state_plus_delta_frozen",
            ],
        )
        active = dict(parsed.active_stages_by_condition)
        self.assertEqual(active["ZH_torch_two_phase_state_plus_delta_frozen"], tuple(stage.name for stage in parsed.stages))
        two_phase = parsed.conditions[3]
        self.assertEqual(two_phase.episodes, 164)
        self.assertEqual(two_phase.decoder_delay_episodes, 116)
        self.assertEqual(two_phase.representation_objective, "state_plus_delta")
        self.assertTrue(two_phase.freeze_encoder_after_delay)
        self.assertTrue(two_phase.stop_representation_after_delay)

    def test_minigrid_torch_v25_gotoobj_task_family_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v25.json")
        parsed = parse_minigrid_torch_config(
            json.loads(config_path.read_text(encoding="utf-8")),
            seed=2101,
        )
        self.assertEqual(parsed.env_id, "BabyAI-GoToObj-v0")
        self.assertEqual(
            [stage.name for stage in parsed.stages],
            ["empty_warmup", "goto_red_ball_warmup", "goto_obj_eval"],
        )
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_gotoobj_hard_only",
                "T_torch_gotoobj_controllability_delay",
                "ZI_torch_gotoobj_state_plus_delta_delay",
                "ZJ_torch_gotoobj_two_phase_state_plus_delta_frozen",
            ],
        )
        active = dict(parsed.active_stages_by_condition)
        self.assertEqual(active["A_torch_gotoobj_hard_only"], ("goto_obj_eval",))
        self.assertEqual(
            active["ZJ_torch_gotoobj_two_phase_state_plus_delta_frozen"],
            ("empty_warmup", "goto_red_ball_warmup", "goto_obj_eval"),
        )
        two_phase = parsed.conditions[3]
        self.assertEqual(two_phase.episodes, 84)
        self.assertEqual(two_phase.decoder_delay_episodes, 36)
        self.assertTrue(two_phase.freeze_encoder_after_delay)
        self.assertTrue(two_phase.stop_representation_after_delay)

    def test_minigrid_torch_v26_gotoobj_matched_repr_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v26.json")
        parsed = parse_minigrid_torch_config(
            json.loads(config_path.read_text(encoding="utf-8")),
            seed=2201,
        )
        self.assertEqual(parsed.env_id, "BabyAI-GoToObj-v0")
        self.assertEqual(
            [stage.name for stage in parsed.stages],
            ["empty_warmup", "goto_red_ball_warmup", "goto_obj_eval"],
        )
        names = [condition.name for condition in parsed.conditions]
        self.assertEqual(
            names,
            [
                "A_torch_gotoobj_hard_only",
                "ZK_torch_gotoobj_curriculum_no_repr_delay",
                "ZL_torch_gotoobj_controllability_matched_delay",
                "ZM_torch_gotoobj_state_plus_delta_matched_delay",
            ],
        )
        active = dict(parsed.active_stages_by_condition)
        self.assertEqual(active["A_torch_gotoobj_hard_only"], ("goto_obj_eval",))
        self.assertEqual(
            active["ZK_torch_gotoobj_curriculum_no_repr_delay"],
            ("empty_warmup", "goto_red_ball_warmup", "goto_obj_eval"),
        )
        for condition in parsed.conditions[1:]:
            self.assertEqual(condition.episodes, 84)
            self.assertEqual(condition.decoder_delay_episodes, 8)
        self.assertEqual(parsed.conditions[1].representation_objective, "none")
        self.assertEqual(parsed.conditions[2].representation_objective, "controllability")
        self.assertEqual(parsed.conditions[3].representation_objective, "state_plus_delta")

    def test_minigrid_torch_v36_semantic_rl_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v36.json")
        parsed = parse_minigrid_torch_config(
            json.loads(config_path.read_text(encoding="utf-8")),
            seed=3001,
        )
        self.assertEqual(parsed.agent.device, "cpu")
        self.assertEqual(parsed.env_id, "BabyAI-GoToObj-v0")
        self.assertEqual(
            [condition.name for condition in parsed.conditions],
            [
                "ZK_torch_gotoobj_curriculum_no_repr_delay",
                "ZM_torch_gotoobj_state_plus_delta_matched_delay",
                "ZN_torch_gotoobj_target_visibility_matched_delay",
            ],
        )
        self.assertEqual(parsed.conditions[0].representation_objective, "none")
        self.assertEqual(parsed.conditions[1].representation_objective, "state_plus_delta")
        self.assertEqual(parsed.conditions[2].representation_objective, "target_visibility_transition")
        for condition in parsed.conditions:
            self.assertEqual(condition.episodes, 42)
            self.assertEqual(condition.decoder_delay_episodes, 4)

    def test_minigrid_torch_v37_combined_semantic_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v37.json")
        parsed = parse_minigrid_torch_config(
            json.loads(config_path.read_text(encoding="utf-8")),
            seed=3101,
        )
        self.assertEqual(parsed.agent.device, "cpu")
        self.assertEqual(parsed.env_id, "BabyAI-GoToObj-v0")
        self.assertEqual(
            [condition.name for condition in parsed.conditions],
            [
                "ZK_torch_gotoobj_curriculum_no_repr_delay",
                "ZM_torch_gotoobj_state_plus_delta_matched_delay",
                "ZN_torch_gotoobj_target_visibility_matched_delay",
                "ZO_torch_gotoobj_state_plus_target_visibility_delay",
            ],
        )
        self.assertEqual(parsed.conditions[0].representation_objective, "none")
        self.assertEqual(parsed.conditions[1].representation_objective, "state_plus_delta")
        self.assertEqual(parsed.conditions[2].representation_objective, "target_visibility_transition")
        self.assertEqual(parsed.conditions[3].representation_objective, "state_plus_target_visibility")
        for condition in parsed.conditions:
            self.assertEqual(condition.episodes, 42)
            self.assertEqual(condition.decoder_delay_episodes, 4)

    def test_minigrid_torch_v38_mission_conditioned_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v38.json")
        parsed = parse_minigrid_torch_config(
            json.loads(config_path.read_text(encoding="utf-8")),
            seed=3201,
        )
        self.assertEqual(parsed.agent.device, "cpu")
        self.assertEqual(parsed.env_id, "BabyAI-GoToObj-v0")
        self.assertEqual(
            [condition.name for condition in parsed.conditions],
            [
                "ZK_torch_gotoobj_curriculum_no_repr_delay",
                "ZM_torch_gotoobj_state_plus_delta_matched_delay",
                "ZN_torch_gotoobj_target_visibility_matched_delay",
                "ZO_torch_gotoobj_state_plus_target_visibility_delay",
                "ZP_torch_gotoobj_mission_target_visibility_delay",
                "ZQ_torch_gotoobj_state_plus_mission_target_delay",
            ],
        )
        self.assertEqual(parsed.conditions[4].representation_objective, "mission_target_transition")
        self.assertEqual(parsed.conditions[5].representation_objective, "state_plus_mission_target")
        for condition in parsed.conditions:
            self.assertEqual(condition.episodes, 42)
            self.assertEqual(condition.decoder_delay_episodes, 4)

    def test_minigrid_torch_v39_combined_beta_sweep_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v39.json")
        parsed = parse_minigrid_torch_config(
            json.loads(config_path.read_text(encoding="utf-8")),
            seed=3401,
        )
        self.assertEqual(parsed.agent.device, "cpu")
        self.assertEqual(parsed.env_id, "BabyAI-GoToObj-v0")
        self.assertEqual(
            [condition.name for condition in parsed.conditions],
            [
                "ZK_torch_gotoobj_curriculum_no_repr_delay",
                "ZM_torch_gotoobj_state_plus_delta_matched_delay",
                "ZN_torch_gotoobj_target_visibility_matched_delay",
                "ZR_torch_gotoobj_state_plus_target_visibility_b010",
                "ZO_torch_gotoobj_state_plus_target_visibility_b030",
                "ZS_torch_gotoobj_state_plus_target_visibility_b050",
            ],
        )
        beta_by_name = {condition.name: condition.representation_beta for condition in parsed.conditions}
        self.assertEqual(beta_by_name["ZR_torch_gotoobj_state_plus_target_visibility_b010"], 0.1)
        self.assertEqual(beta_by_name["ZO_torch_gotoobj_state_plus_target_visibility_b030"], 0.3)
        self.assertEqual(beta_by_name["ZS_torch_gotoobj_state_plus_target_visibility_b050"], 0.5)
        for condition in parsed.conditions[3:]:
            self.assertEqual(condition.representation_objective, "state_plus_target_visibility")
            self.assertEqual(condition.episodes, 42)
            self.assertEqual(condition.decoder_delay_episodes, 4)

    def test_minigrid_torch_v40_tight_beta_sweep_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v40.json")
        parsed = parse_minigrid_torch_config(
            json.loads(config_path.read_text(encoding="utf-8")),
            seed=3501,
        )
        self.assertEqual(parsed.agent.device, "cpu")
        self.assertEqual(parsed.env_id, "BabyAI-GoToObj-v0")
        self.assertEqual(
            [condition.name for condition in parsed.conditions],
            [
                "ZK_torch_gotoobj_curriculum_no_repr_delay",
                "ZT_torch_gotoobj_state_plus_target_visibility_b005",
                "ZU_torch_gotoobj_state_plus_target_visibility_b0075",
                "ZR_torch_gotoobj_state_plus_target_visibility_b010",
                "ZV_torch_gotoobj_state_plus_target_visibility_b0125",
                "ZW_torch_gotoobj_state_plus_target_visibility_b015",
            ],
        )
        beta_by_name = {condition.name: condition.representation_beta for condition in parsed.conditions}
        self.assertEqual(beta_by_name["ZT_torch_gotoobj_state_plus_target_visibility_b005"], 0.05)
        self.assertEqual(beta_by_name["ZU_torch_gotoobj_state_plus_target_visibility_b0075"], 0.075)
        self.assertEqual(beta_by_name["ZR_torch_gotoobj_state_plus_target_visibility_b010"], 0.1)
        self.assertEqual(beta_by_name["ZV_torch_gotoobj_state_plus_target_visibility_b0125"], 0.125)
        self.assertEqual(beta_by_name["ZW_torch_gotoobj_state_plus_target_visibility_b015"], 0.15)
        for condition in parsed.conditions[1:]:
            self.assertEqual(condition.representation_objective, "state_plus_target_visibility")
            self.assertEqual(condition.episodes, 42)
            self.assertEqual(condition.decoder_delay_episodes, 4)

    def test_minigrid_torch_v41_longer_horizon_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v41.json")
        parsed = parse_minigrid_torch_config(
            json.loads(config_path.read_text(encoding="utf-8")),
            seed=3601,
        )
        self.assertEqual(parsed.agent.device, "cpu")
        self.assertEqual(parsed.env_id, "BabyAI-GoToObj-v0")
        self.assertEqual(
            [(stage.name, stage.episodes) for stage in parsed.stages],
            [("empty_warmup", 12), ("goto_red_ball_warmup", 24), ("goto_obj_eval", 48)],
        )
        self.assertEqual(
            [condition.name for condition in parsed.conditions],
            [
                "ZK_torch_gotoobj_curriculum_no_repr_delay_long",
                "ZT_torch_gotoobj_state_plus_target_visibility_b005_long",
                "ZU_torch_gotoobj_state_plus_target_visibility_b0075_long",
                "ZR_torch_gotoobj_state_plus_target_visibility_b010_long",
            ],
        )
        beta_by_name = {condition.name: condition.representation_beta for condition in parsed.conditions}
        self.assertEqual(beta_by_name["ZT_torch_gotoobj_state_plus_target_visibility_b005_long"], 0.05)
        self.assertEqual(beta_by_name["ZU_torch_gotoobj_state_plus_target_visibility_b0075_long"], 0.075)
        self.assertEqual(beta_by_name["ZR_torch_gotoobj_state_plus_target_visibility_b010_long"], 0.1)
        for condition in parsed.conditions:
            self.assertEqual(condition.episodes, 84)
            self.assertEqual(condition.decoder_delay_episodes, 8)
        for condition in parsed.conditions[1:]:
            self.assertEqual(condition.representation_objective, "state_plus_target_visibility")

    def test_minigrid_torch_v42_two_head_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-torch-adda-v42.json")
        parsed = parse_minigrid_torch_config(
            json.loads(config_path.read_text(encoding="utf-8")),
            seed=3701,
        )
        self.assertEqual(parsed.agent.device, "cpu")
        self.assertEqual(parsed.env_id, "BabyAI-GoToObj-v0")
        self.assertEqual(
            [condition.name for condition in parsed.conditions],
            [
                "ZK_torch_gotoobj_curriculum_no_repr_delay",
                "ZU_torch_gotoobj_state_plus_target_visibility_b0075",
                "ZX_torch_gotoobj_two_head_state050_visibility025",
                "ZY_torch_gotoobj_two_head_state0375_visibility0375",
            ],
        )
        combined = parsed.conditions[1]
        self.assertEqual(combined.representation_objective, "state_plus_target_visibility")
        self.assertEqual(combined.representation_beta, 0.075)
        two_head = parsed.conditions[2]
        self.assertEqual(two_head.representation_objective, "state_delta_and_target_visibility")
        self.assertEqual(two_head.representation_beta, 0.0)
        self.assertEqual(two_head.representation_state_beta, 0.05)
        self.assertEqual(two_head.representation_target_visibility_beta, 0.025)
        balanced = parsed.conditions[3]
        self.assertEqual(balanced.representation_state_beta, 0.0375)
        self.assertEqual(balanced.representation_target_visibility_beta, 0.0375)
        for condition in parsed.conditions:
            self.assertEqual(condition.episodes, 42)
            self.assertEqual(condition.decoder_delay_episodes, 4)

    def test_minigrid_torch_two_head_target_is_dependency_free(self) -> None:
        before_observation = {
            "image": [[[0, 0, 0] for _ in range(7)] for _ in range(7)],
            "direction": 0,
            "mission": "go to the red ball",
        }
        after_image = [[[0, 0, 0] for _ in range(7)] for _ in range(7)]
        after_image[3][5] = [6, 0, 0]
        after_observation = {
            "image": after_image,
            "direction": 1,
            "mission": "go to the red ball",
        }
        condition = Condition(
            name="two_head",
            encoder_mode="raw",
            episodes=1,
            decoder_delay_episodes=0,
            intrinsic_beta=0.0,
            intrinsic_mode="none",
            seed=1,
            representation_objective="state_delta_and_target_visibility",
            representation_state_beta=0.05,
            representation_target_visibility_beta=0.025,
        )
        target = minigrid_torch_module.representation_target_for_objective(
            condition=condition,
            features={},
            observation=before_observation,
            next_observation=after_observation,
            next_features={},
            feature_dim=128,
            actions=7,
        )
        self.assertEqual(set(target.keys()), {"state_plus_delta", "target_visibility_transition"})
        self.assertEqual(target["state_plus_delta"], state_plus_delta_vector(before_observation, after_observation))
        self.assertEqual(
            target["target_visibility_transition"],
            target_visibility_transition_vector(before_observation, after_observation),
        )
        self.assertEqual(len(target["state_plus_delta"]), 58)
        self.assertEqual(len(target["target_visibility_transition"]), 49)

    def test_minigrid_torch_two_head_config_rejects_ambiguous_betas(self) -> None:
        config = json.loads(Path("configs/experiments/minigrid-torch-adda-v42.json").read_text(encoding="utf-8"))
        two_head = config["conditions"][2]

        scalar_beta_config = json.loads(json.dumps(config))
        scalar_beta_config["conditions"][2] = dict(two_head, representation_beta=0.075)
        with self.assertRaisesRegex(ValueError, "representation_beta must stay zero"):
            parse_minigrid_torch_config(scalar_beta_config, seed=3701)

        missing_head_beta_config = json.loads(json.dumps(config))
        missing_head_beta_config["conditions"][2].pop("representation_target_visibility_beta")
        with self.assertRaisesRegex(ValueError, "two-head representation betas must be positive"):
            parse_minigrid_torch_config(missing_head_beta_config, seed=3701)

        non_two_head_config = json.loads(json.dumps(config))
        non_two_head_config["conditions"][1]["representation_state_beta"] = 0.01
        with self.assertRaisesRegex(ValueError, "two-head representation betas require"):
            parse_minigrid_torch_config(non_two_head_config, seed=3701)

    def test_minigrid_repr_probe_v28_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-repr-probe-v28.json")
        parsed = parse_minigrid_representation_probe_config(json.loads(config_path.read_text(encoding="utf-8")))
        self.assertEqual([env.name for env in parsed.envs], ["goto_red_ball", "goto_obj"])
        self.assertEqual([env.env_id for env in parsed.envs], ["BabyAI-GoToRedBall-v0", "BabyAI-GoToObj-v0"])
        self.assertEqual(parsed.feature_dim, 1024)
        self.assertEqual(parsed.encoder_mode, "raw")
        self.assertEqual(parsed.feature_sets, ("raw_current", "affordance_current"))
        self.assertEqual(parsed.decision.labels, ("mission_object", "mission_color", "changed"))
        self.assertEqual(parsed.decision.min_test_examples, 10)

    def test_minigrid_repr_probe_v29_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-repr-probe-v29.json")
        parsed = parse_minigrid_representation_probe_config(json.loads(config_path.read_text(encoding="utf-8")))
        self.assertEqual(parsed.feature_sets, ("raw_current", "predictive_encoder"))
        self.assertEqual(parsed.predictive_encoder.target_label, "changed")
        self.assertTrue(parsed.predictive_encoder.include_action)
        self.assertTrue(parsed.predictive_encoder.include_raw_passthrough)
        self.assertEqual(parsed.decision.mode, "relative_to_baseline")
        self.assertEqual(parsed.decision.baseline_feature_set, "raw_current")
        self.assertEqual(parsed.decision.candidate_feature_set, "predictive_encoder")
        self.assertEqual(parsed.decision.changed_min_lift_delta, 0.05)
        self.assertEqual(parsed.decision.max_mission_accuracy_drop, 0.05)

    def test_minigrid_repr_probe_labels_and_centroid_metrics_are_dependency_free(self) -> None:
        before = {
            "direction": 0,
            "mission": "go to the yellow key",
            "image": [[[0, 0, 0] for _ in range(7)] for _ in range(7)],
        }
        after = {
            "direction": 1,
            "mission": "go to the yellow key",
            "image": [[[0, 0, 0] for _ in range(7)] for _ in range(7)],
        }
        labels = transition_probe_labels(
            observation=before,
            next_observation=after,
            features={1: 1.0},
            next_features={1: 1.0, 2: 1.0},
            signature_buckets=8,
        )
        self.assertEqual(labels["mission_object"], "key")
        self.assertEqual(labels["mission_color"], "yellow")
        self.assertEqual(labels["changed"], "changed")
        self.assertTrue(labels["next_signature_bucket"].startswith("bucket:"))
        self.assertEqual(labels["target_visibility_transition"], "absent->absent")

        self.assertEqual(vector_to_sparse_features([0.0, 1.0, 0.5]), {1: 1.0, 2: 0.5})
        examples = [{0: 1.0}, {0: 0.9}, {1: 1.0}, {1: 0.9}, {0: 1.0}, {1: 1.0}]
        metrics = centroid_probe_metrics(examples, ["a", "a", "b", "b", "a", "b"], test_every=3)
        self.assertEqual(metrics["test_examples"], 2)
        self.assertGreaterEqual(metrics["accuracy"], metrics["majority_baseline"])

    def test_minigrid_repr_target_visibility_transition_is_dependency_free(self) -> None:
        empty = [0, 0, 0]
        red_ball = [6, 0, 0]
        before_image = [[list(empty) for _y in range(7)] for _x in range(7)]
        after_image = [[list(empty) for _y in range(7)] for _x in range(7)]
        after_image[3][5] = red_ball
        before = {"mission": "go to the red ball", "image": before_image}
        after = {"mission": "go to the red ball", "image": after_image}
        self.assertEqual(target_relation_label(before), "absent")
        self.assertEqual(target_relation_label(after), "center_near")
        labels = transition_probe_labels(
            observation=before,
            next_observation=after,
            features={1: 1.0},
            next_features={1: 1.0, 2: 1.0},
            signature_buckets=8,
        )
        self.assertEqual(labels["target_visibility_transition"], "absent->center_near")

    def test_minigrid_repr_predictive_encoder_is_dependency_free(self) -> None:
        config = parse_minigrid_representation_probe_config(
            {
                "dataset": {
                    "policy": "random",
                    "test_every": 3,
                    "signature_buckets": 4,
                    "envs": [{"name": "fake", "env_id": "Fake-v0", "episodes": 1, "max_steps": 12}],
                },
                "features": {
                    "feature_dim": 128,
                    "encoder_mode": "raw",
                    "feature_sets": ["raw_current", "predictive_encoder"],
                },
                "training": {
                    "predictive_encoder": {
                        "target_label": "changed",
                        "epochs": 3,
                        "learning_rate": 0.2,
                        "include_action": True,
                        "include_raw_passthrough": True,
                    }
                },
                "decision": {
                    "mode": "relative_to_baseline",
                    "labels": ["mission_object", "mission_color", "changed"],
                    "baseline_feature_set": "raw_current",
                    "candidate_feature_set": "predictive_encoder",
                    "min_test_examples": 2,
                },
            }
        )
        transitions = []
        for index in range(12):
            is_changed = index % 2 == 1
            transitions.append(
                {
                    "action": 1 if is_changed else 0,
                    "features": {1 if is_changed else 2: 1.0},
                    "affordance_features": {},
                    "labels": {
                        "mission_object": "ball" if index < 6 else "key",
                        "mission_color": "red" if index < 6 else "blue",
                        "changed": "changed" if is_changed else "same",
                        "next_signature_bucket": f"bucket:{index % 4}",
                        "target_visibility_transition": "absent->center_near" if is_changed else "absent->absent",
                    },
                }
            )
        encoder, training_report = train_predictive_encoder(transitions, config)
        self.assertEqual(training_report["target_label"], "changed")
        self.assertGreaterEqual(training_report["test_accuracy"], training_report["test_majority_baseline"])

        report = evaluate_feature_set(
            transitions,
            feature_set="predictive_encoder",
            config=config,
            predictive_encoders={"predictive_encoder": encoder},
        )
        self.assertIn("changed", report["labels"])
        self.assertGreaterEqual(report["labels"]["changed"]["accuracy"], report["labels"]["changed"]["majority_baseline"])

    def test_minigrid_repr_feature_eval_uses_config_labels_only(self) -> None:
        config = parse_minigrid_representation_probe_config(
            {
                "dataset": {
                    "policy": "random",
                    "test_every": 3,
                    "signature_buckets": 4,
                    "envs": [{"name": "fake", "env_id": "Fake-v0", "episodes": 1, "max_steps": 6}],
                },
                "features": {"feature_dim": 128, "encoder_mode": "raw", "feature_sets": ["raw_current"]},
                "decision": {
                    "mode": "absolute_all_labels",
                    "labels": ["mission_object", "mission_color", "changed"],
                    "min_test_examples": 1,
                },
            }
        )
        transitions = [
            {
                "features": {index: 1.0},
                "affordance_features": {},
                "labels": {
                    "mission_object": "ball",
                    "mission_color": "red",
                    "changed": "changed" if index % 2 else "same",
                    "next_signature_bucket": f"bucket:{index % 4}",
                },
            }
            for index in range(6)
        ]
        report = evaluate_feature_set(transitions, feature_set="raw_current", config=config)
        self.assertEqual(set(report["labels"]), {"mission_object", "mission_color", "changed"})

    def test_minigrid_repr_probe_v30_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-repr-probe-v30.json")
        parsed = parse_minigrid_representation_probe_config(json.loads(config_path.read_text(encoding="utf-8")))
        self.assertEqual(parsed.feature_sets, ("raw_current", "predictive_changed", "predictive_next_signature"))
        self.assertEqual([encoder.name for encoder in parsed.predictive_encoders], ["predictive_changed", "predictive_next_signature"])
        self.assertEqual([encoder.target_label for encoder in parsed.predictive_encoders], ["changed", "next_signature_bucket"])
        self.assertEqual(parsed.decision.mode, "relative_to_reference")
        self.assertEqual(parsed.decision.baseline_feature_set, "raw_current")
        self.assertEqual(parsed.decision.reference_feature_set, "predictive_changed")
        self.assertEqual(parsed.decision.candidate_feature_set, "predictive_next_signature")
        self.assertEqual(parsed.decision.transition_label, "next_signature_bucket")
        self.assertEqual(parsed.decision.transition_min_lift_delta, 0.01)

    def test_minigrid_repr_probe_v31_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-repr-probe-v31.json")
        parsed = parse_minigrid_representation_probe_config(json.loads(config_path.read_text(encoding="utf-8")))
        self.assertEqual(parsed.policy, "scripted_object")
        self.assertEqual(parsed.feature_sets, ("raw_current", "predictive_changed"))
        self.assertEqual(parsed.predictive_encoders[0].name, "predictive_changed")
        self.assertEqual(parsed.decision.mode, "external_transition_baseline")
        self.assertEqual(parsed.decision.transition_label, "changed")
        self.assertEqual(parsed.decision.external_transition_lift_baseline, 0.209)

    def test_minigrid_repr_probe_v32_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-repr-probe-v32.json")
        parsed = parse_minigrid_representation_probe_config(json.loads(config_path.read_text(encoding="utf-8")))
        self.assertEqual(parsed.policy, "scripted_object")
        self.assertEqual(parsed.feature_sets, ("raw_current", "predictive_next_signature"))
        self.assertEqual(parsed.predictive_encoders[0].name, "predictive_next_signature")
        self.assertEqual(parsed.predictive_encoders[0].target_label, "next_signature_bucket")
        self.assertEqual(parsed.decision.mode, "relative_to_baseline")
        self.assertEqual(parsed.decision.labels, ("mission_object", "mission_color", "next_signature_bucket"))
        self.assertEqual(parsed.decision.candidate_feature_set, "predictive_next_signature")
        self.assertEqual(parsed.decision.transition_label, "next_signature_bucket")
        self.assertEqual(parsed.decision.transition_min_lift_delta, 0.01)
        self.assertEqual(parsed.decision.max_mission_accuracy_drop, 0.05)

    def test_minigrid_repr_probe_v33_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-repr-probe-v33.json")
        parsed = parse_minigrid_representation_probe_config(json.loads(config_path.read_text(encoding="utf-8")))
        self.assertEqual(parsed.policy, "scripted_object")
        self.assertEqual(
            parsed.feature_sets,
            ("raw_current", "predictive_next_signature", "predictive_next_signature_pure"),
        )
        self.assertEqual(
            [encoder.name for encoder in parsed.predictive_encoders],
            ["predictive_next_signature", "predictive_next_signature_pure"],
        )
        self.assertEqual(
            [encoder.include_raw_passthrough for encoder in parsed.predictive_encoders],
            [True, False],
        )
        self.assertEqual(parsed.decision.mode, "relative_to_reference")
        self.assertEqual(parsed.decision.labels, ("mission_object", "mission_color", "next_signature_bucket"))
        self.assertEqual(parsed.decision.baseline_feature_set, "raw_current")
        self.assertEqual(parsed.decision.reference_feature_set, "predictive_next_signature")
        self.assertEqual(parsed.decision.candidate_feature_set, "predictive_next_signature_pure")
        self.assertEqual(parsed.decision.transition_label, "next_signature_bucket")
        self.assertEqual(parsed.decision.transition_min_lift_delta, 0.01)
        self.assertEqual(parsed.decision.max_mission_accuracy_drop, 0.05)
        self.assertEqual(parsed.decision.min_test_examples, 10)

    def test_minigrid_repr_probe_v34_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-repr-probe-v34.json")
        parsed = parse_minigrid_representation_probe_config(json.loads(config_path.read_text(encoding="utf-8")))
        self.assertEqual(parsed.policy, "scripted_object")
        self.assertEqual(parsed.feature_sets, ("raw_current", "predictive_target_visibility"))
        self.assertEqual(parsed.predictive_encoders[0].name, "predictive_target_visibility")
        self.assertEqual(parsed.predictive_encoders[0].target_label, "target_visibility_transition")
        self.assertTrue(parsed.predictive_encoders[0].include_raw_passthrough)
        self.assertEqual(parsed.decision.mode, "relative_to_baseline")
        self.assertEqual(parsed.decision.labels, ("mission_object", "mission_color", "target_visibility_transition"))
        self.assertEqual(parsed.decision.baseline_feature_set, "raw_current")
        self.assertEqual(parsed.decision.candidate_feature_set, "predictive_target_visibility")
        self.assertEqual(parsed.decision.transition_label, "target_visibility_transition")
        self.assertEqual(parsed.decision.transition_min_lift_delta, 0.01)
        self.assertEqual(parsed.decision.max_mission_accuracy_drop, 0.05)
        self.assertEqual(parsed.decision.min_test_examples, 10)

    def test_minigrid_repr_probe_v35_config_is_dependency_free(self) -> None:
        config_path = Path("configs/experiments/minigrid-repr-probe-v35.json")
        parsed = parse_minigrid_representation_probe_config(json.loads(config_path.read_text(encoding="utf-8")))
        self.assertEqual(parsed.policy, "scripted_object")
        self.assertEqual(parsed.feature_sets, ("raw_current", "predictive_target_visibility"))
        self.assertEqual(parsed.predictive_encoders[0].target_label, "target_visibility_transition")
        self.assertEqual(parsed.decision.labels, ("mission_object", "mission_color", "target_visibility_transition"))
        self.assertEqual(parsed.decision.transition_label, "target_visibility_transition")

    def test_minigrid_repr_probe_sweep_aggregate_is_dependency_free(self) -> None:
        def run(
            seed: int,
            *,
            lift_delta: float,
            object_delta: float = 0.0,
            color_delta: float = 0.0,
            test_examples: int = 20,
        ) -> dict[str, object]:
            return {
                "decision": {
                    "met": lift_delta >= 0.01,
                    "rule": {"transition_label": "target_visibility_transition"},
                    "comparisons": {
                        "mission_object": {"accuracy_delta": object_delta},
                        "mission_color": {"accuracy_delta": color_delta},
                        "target_visibility_transition": {
                            "lift_delta": lift_delta,
                            "baseline_lift": 0.30,
                            "candidate_lift": 0.30 + lift_delta,
                            "candidate_test_examples": test_examples,
                        },
                    },
                },
                "seed": seed,
            }

        runs = [
            run(1, lift_delta=0.02),
            run(2, lift_delta=0.01),
            run(3, lift_delta=-0.005),
        ]
        aggregate = aggregate_representation_probe_runs(runs, seeds=[1, 2, 3])
        self.assertEqual(aggregate["seed_count"], 3)
        self.assertAlmostEqual(aggregate["mean_transition_lift_delta"], 0.008333333333333333)
        self.assertEqual(aggregate["nonnegative_transition_lift_delta_count"], 2)
        self.assertEqual(aggregate["min_transition_test_examples"], 20)
        decision = evaluate_representation_probe_sweep_decision(aggregate)
        self.assertFalse(decision["met"])

        positive = aggregate_representation_probe_runs(
            [run(1, lift_delta=0.02), run(2, lift_delta=0.015), run(3, lift_delta=0.0)],
            seeds=[1, 2, 3],
        )
        self.assertTrue(evaluate_representation_probe_sweep_decision(positive)["met"])
        low_sample = aggregate_representation_probe_runs(
            [run(1, lift_delta=0.02), run(2, lift_delta=0.015, test_examples=9), run(3, lift_delta=0.0)],
            seeds=[1, 2, 3],
        )
        self.assertFalse(evaluate_representation_probe_sweep_decision(low_sample)["met"])
        with self.assertRaises(ValueError):
            aggregate_representation_probe_runs([run(99, lift_delta=0.02)], seeds=[1])
        summary = representation_probe_sweep_summary_markdown(
            {
                "created_at": "2026-06-29T00:00:00+00:00",
                "hypothesis": "repr sweep",
                "seeds": [1, 2, 3],
                "aggregate": aggregate,
                "decision": decision,
            }
        )
        self.assertIn("mean_transition_lift_delta", summary)
        self.assertEqual(REPR_PROBE_SWEEP_DEFAULT_CONFIG_PATH, Path("configs/experiments/minigrid-repr-probe-v35.json"))

    def test_minigrid_repr_relative_decision_is_dependency_free(self) -> None:
        config = parse_minigrid_representation_probe_config(
            {
                "dataset": {
                    "policy": "random",
                    "test_every": 3,
                    "signature_buckets": 4,
                    "envs": [{"name": "fake", "env_id": "Fake-v0", "episodes": 1, "max_steps": 1}],
                },
                "features": {
                    "feature_dim": 128,
                    "encoder_mode": "raw",
                    "feature_sets": ["raw_current", "predictive_encoder"],
                },
                "decision": {
                    "mode": "relative_to_baseline",
                    "labels": ["mission_object", "mission_color", "changed"],
                    "baseline_feature_set": "raw_current",
                    "candidate_feature_set": "predictive_encoder",
                    "changed_min_lift_delta": 0.05,
                    "max_mission_accuracy_drop": 0.05,
                    "min_test_examples": 10,
                },
            }
        )
        base_labels = {
            "mission_object": {"accuracy": 0.90, "majority_baseline": 0.50, "lift": 0.40, "test_examples": 20},
            "mission_color": {"accuracy": 0.80, "majority_baseline": 0.50, "lift": 0.30, "test_examples": 20},
            "changed": {"accuracy": 0.60, "majority_baseline": 0.58, "lift": 0.02, "test_examples": 20},
        }
        candidate_labels = {
            "mission_object": {"accuracy": 0.86, "majority_baseline": 0.50, "lift": 0.36, "test_examples": 20},
            "mission_color": {"accuracy": 0.78, "majority_baseline": 0.50, "lift": 0.28, "test_examples": 20},
            "changed": {"accuracy": 0.66, "majority_baseline": 0.58, "lift": 0.08, "test_examples": 20},
        }
        result = evaluate_probe_decision(
            [
                {"feature_set": "raw_current", "labels": base_labels},
                {"feature_set": "predictive_encoder", "labels": candidate_labels},
            ],
            config.decision,
        )
        self.assertTrue(result["met"])
        self.assertEqual(result["best_feature_set"], "predictive_encoder")
        self.assertGreaterEqual(result["comparisons"]["changed"]["lift_delta"], 0.05)

    def test_minigrid_repr_relative_decision_rejects_failed_gates(self) -> None:
        config = parse_minigrid_representation_probe_config(
            {
                "dataset": {
                    "policy": "random",
                    "test_every": 3,
                    "signature_buckets": 4,
                    "envs": [{"name": "fake", "env_id": "Fake-v0", "episodes": 1, "max_steps": 1}],
                },
                "features": {
                    "feature_dim": 128,
                    "encoder_mode": "raw",
                    "feature_sets": ["raw_current", "predictive_encoder"],
                },
                "decision": {
                    "mode": "relative_to_baseline",
                    "labels": ["mission_object", "mission_color", "changed"],
                    "baseline_feature_set": "raw_current",
                    "candidate_feature_set": "predictive_encoder",
                    "changed_min_lift_delta": 0.05,
                    "max_mission_accuracy_drop": 0.05,
                    "min_test_examples": 10,
                },
            }
        )

        def result_for(
            *,
            candidate_object_accuracy: float = 0.86,
            candidate_color_accuracy: float = 0.78,
            candidate_changed_lift: float = 0.08,
            candidate_test_examples: int = 20,
        ) -> dict[str, object]:
            base_labels = {
                "mission_object": {"accuracy": 0.90, "majority_baseline": 0.50, "lift": 0.40, "test_examples": 20},
                "mission_color": {"accuracy": 0.80, "majority_baseline": 0.50, "lift": 0.30, "test_examples": 20},
                "changed": {"accuracy": 0.60, "majority_baseline": 0.58, "lift": 0.02, "test_examples": 20},
            }
            candidate_labels = {
                "mission_object": {
                    "accuracy": candidate_object_accuracy,
                    "majority_baseline": 0.50,
                    "lift": candidate_object_accuracy - 0.50,
                    "test_examples": candidate_test_examples,
                },
                "mission_color": {
                    "accuracy": candidate_color_accuracy,
                    "majority_baseline": 0.50,
                    "lift": candidate_color_accuracy - 0.50,
                    "test_examples": candidate_test_examples,
                },
                "changed": {
                    "accuracy": candidate_changed_lift + 0.58,
                    "majority_baseline": 0.58,
                    "lift": candidate_changed_lift,
                    "test_examples": candidate_test_examples,
                },
            }
            return evaluate_probe_decision(
                [
                    {"feature_set": "raw_current", "labels": base_labels},
                    {"feature_set": "predictive_encoder", "labels": candidate_labels},
                ],
                config.decision,
            )

        self.assertFalse(result_for(candidate_changed_lift=0.06)["met"])
        self.assertFalse(result_for(candidate_object_accuracy=0.84)["met"])
        self.assertFalse(result_for(candidate_color_accuracy=0.74)["met"])
        self.assertFalse(result_for(candidate_test_examples=9)["met"])

    def test_minigrid_repr_relative_decision_uses_transition_label(self) -> None:
        config = parse_minigrid_representation_probe_config(
            {
                "dataset": {
                    "policy": "scripted_object",
                    "test_every": 3,
                    "signature_buckets": 4,
                    "envs": [{"name": "fake", "env_id": "Fake-v0", "episodes": 1, "max_steps": 1}],
                },
                "features": {
                    "feature_dim": 128,
                    "encoder_mode": "raw",
                    "feature_sets": ["raw_current", "predictive_next_signature"],
                },
                "training": {
                    "predictive_encoders": [
                        {"name": "predictive_next_signature", "target_label": "next_signature_bucket"}
                    ]
                },
                "decision": {
                    "mode": "relative_to_baseline",
                    "labels": ["mission_object", "mission_color", "next_signature_bucket"],
                    "baseline_feature_set": "raw_current",
                    "candidate_feature_set": "predictive_next_signature",
                    "transition_label": "next_signature_bucket",
                    "transition_min_lift_delta": 0.01,
                    "max_mission_accuracy_drop": 0.05,
                    "min_test_examples": 10,
                },
            }
        )
        base_labels = {
            "mission_object": {"accuracy": 0.90, "majority_baseline": 0.50, "lift": 0.40, "test_examples": 20},
            "mission_color": {"accuracy": 0.80, "majority_baseline": 0.50, "lift": 0.30, "test_examples": 20},
            "next_signature_bucket": {"accuracy": 0.70, "majority_baseline": 0.20, "lift": 0.50, "test_examples": 20},
        }
        candidate_labels = {
            "mission_object": {"accuracy": 0.87, "majority_baseline": 0.50, "lift": 0.37, "test_examples": 20},
            "mission_color": {"accuracy": 0.78, "majority_baseline": 0.50, "lift": 0.28, "test_examples": 20},
            "next_signature_bucket": {"accuracy": 0.72, "majority_baseline": 0.20, "lift": 0.52, "test_examples": 20},
        }
        result = evaluate_probe_decision(
            [
                {"feature_set": "raw_current", "labels": base_labels},
                {"feature_set": "predictive_next_signature", "labels": candidate_labels},
            ],
            config.decision,
        )
        self.assertTrue(result["met"])
        self.assertAlmostEqual(result["comparisons"]["next_signature_bucket"]["lift_delta"], 0.02)

        candidate_labels["next_signature_bucket"] = {
            "accuracy": 0.705,
            "majority_baseline": 0.20,
            "lift": 0.505,
            "test_examples": 20,
        }
        failed = evaluate_probe_decision(
            [
                {"feature_set": "raw_current", "labels": base_labels},
                {"feature_set": "predictive_next_signature", "labels": candidate_labels},
            ],
            config.decision,
        )
        self.assertFalse(failed["met"])

    def test_minigrid_repr_scripted_policy_is_dependency_free(self) -> None:
        empty = [0, 0, 0]
        red_ball = [6, 0, 0]
        image_left = [[list(empty) for _y in range(7)] for _x in range(7)]
        image_left[1][5] = red_ball
        image_center = [[list(empty) for _y in range(7)] for _x in range(7)]
        image_center[3][5] = red_ball
        self.assertEqual(
            scripted_object_action(
                {"mission": "go to the red ball", "image": image_left},
                episode=0,
                step=0,
                seed=0,
            ),
            0,
        )
        self.assertEqual(
            scripted_object_action(
                {"mission": "go to the red ball", "image": image_center},
                episode=0,
                step=0,
                seed=0,
            ),
            2,
        )
        self.assertIn(scripted_object_action({"mission": "go to the blue key", "image": image_left}, 1, 2, 3), {0, 1, 2})

    def test_minigrid_repr_reference_decision_is_dependency_free(self) -> None:
        config = parse_minigrid_representation_probe_config(
            {
                "dataset": {
                    "policy": "random",
                    "test_every": 3,
                    "signature_buckets": 4,
                    "envs": [{"name": "fake", "env_id": "Fake-v0", "episodes": 1, "max_steps": 1}],
                },
                "features": {
                    "feature_dim": 128,
                    "encoder_mode": "raw",
                    "feature_sets": ["raw_current", "predictive_changed", "predictive_next_signature"],
                },
                "training": {
                    "predictive_encoders": [
                        {"name": "predictive_changed", "target_label": "changed"},
                        {"name": "predictive_next_signature", "target_label": "next_signature_bucket"},
                    ]
                },
                "decision": {
                    "mode": "relative_to_reference",
                    "labels": ["mission_object", "mission_color", "next_signature_bucket"],
                    "baseline_feature_set": "raw_current",
                    "reference_feature_set": "predictive_changed",
                    "candidate_feature_set": "predictive_next_signature",
                    "transition_label": "next_signature_bucket",
                    "transition_min_lift_delta": 0.01,
                    "max_mission_accuracy_drop": 0.05,
                    "min_test_examples": 10,
                },
            }
        )
        base_labels = {
            "mission_object": {"accuracy": 0.90, "majority_baseline": 0.50, "lift": 0.40, "test_examples": 20},
            "mission_color": {"accuracy": 0.80, "majority_baseline": 0.50, "lift": 0.30, "test_examples": 20},
            "next_signature_bucket": {"accuracy": 0.30, "majority_baseline": 0.20, "lift": 0.10, "test_examples": 20},
        }
        reference_labels = {
            "mission_object": {"accuracy": 0.89, "majority_baseline": 0.50, "lift": 0.39, "test_examples": 20},
            "mission_color": {"accuracy": 0.79, "majority_baseline": 0.50, "lift": 0.29, "test_examples": 20},
            "next_signature_bucket": {"accuracy": 0.32, "majority_baseline": 0.20, "lift": 0.12, "test_examples": 20},
        }
        candidate_labels = {
            "mission_object": {"accuracy": 0.86, "majority_baseline": 0.50, "lift": 0.36, "test_examples": 20},
            "mission_color": {"accuracy": 0.77, "majority_baseline": 0.50, "lift": 0.27, "test_examples": 20},
            "next_signature_bucket": {"accuracy": 0.34, "majority_baseline": 0.20, "lift": 0.14, "test_examples": 20},
        }
        result = evaluate_probe_decision(
            [
                {"feature_set": "raw_current", "labels": base_labels},
                {"feature_set": "predictive_changed", "labels": reference_labels},
                {"feature_set": "predictive_next_signature", "labels": candidate_labels},
            ],
            config.decision,
        )
        self.assertTrue(result["met"])
        self.assertEqual(result["best_feature_set"], "predictive_next_signature")

        candidate_labels["next_signature_bucket"] = {
            "accuracy": 0.325,
            "majority_baseline": 0.20,
            "lift": 0.125,
            "test_examples": 20,
        }
        failed = evaluate_probe_decision(
            [
                {"feature_set": "raw_current", "labels": base_labels},
                {"feature_set": "predictive_changed", "labels": reference_labels},
                {"feature_set": "predictive_next_signature", "labels": candidate_labels},
            ],
            config.decision,
        )
        self.assertFalse(failed["met"])

    def test_minigrid_repr_external_decision_is_dependency_free(self) -> None:
        config = parse_minigrid_representation_probe_config(
            {
                "dataset": {
                    "policy": "scripted_object",
                    "test_every": 3,
                    "signature_buckets": 4,
                    "envs": [{"name": "fake", "env_id": "Fake-v0", "episodes": 1, "max_steps": 1}],
                },
                "features": {
                    "feature_dim": 128,
                    "encoder_mode": "raw",
                    "feature_sets": ["raw_current", "predictive_changed"],
                },
                "training": {"predictive_encoders": [{"name": "predictive_changed", "target_label": "changed"}]},
                "decision": {
                    "mode": "external_transition_baseline",
                    "labels": ["mission_object", "mission_color", "changed"],
                    "baseline_feature_set": "raw_current",
                    "candidate_feature_set": "predictive_changed",
                    "transition_label": "changed",
                    "transition_min_lift_delta": 0.01,
                    "external_transition_lift_baseline": 0.20,
                    "max_mission_accuracy_drop": 0.05,
                    "min_test_examples": 10,
                },
            }
        )
        base_labels = {
            "mission_object": {"accuracy": 0.80, "majority_baseline": 0.50, "lift": 0.30, "test_examples": 20},
            "mission_color": {"accuracy": 0.70, "majority_baseline": 0.50, "lift": 0.20, "test_examples": 20},
            "changed": {"accuracy": 0.62, "majority_baseline": 0.50, "lift": 0.12, "test_examples": 20},
        }
        candidate_labels = {
            "mission_object": {"accuracy": 0.77, "majority_baseline": 0.50, "lift": 0.27, "test_examples": 20},
            "mission_color": {"accuracy": 0.66, "majority_baseline": 0.50, "lift": 0.16, "test_examples": 20},
            "changed": {"accuracy": 0.72, "majority_baseline": 0.50, "lift": 0.22, "test_examples": 20},
        }
        result = evaluate_probe_decision(
            [
                {"feature_set": "raw_current", "labels": base_labels},
                {"feature_set": "predictive_changed", "labels": candidate_labels},
            ],
            config.decision,
        )
        self.assertTrue(result["met"])

        candidate_labels["mission_color"] = {
            "accuracy": 0.64,
            "majority_baseline": 0.50,
            "lift": 0.14,
            "test_examples": 20,
        }
        failed = evaluate_probe_decision(
            [
                {"feature_set": "raw_current", "labels": base_labels},
                {"feature_set": "predictive_changed", "labels": candidate_labels},
            ],
            config.decision,
        )
        self.assertFalse(failed["met"])

    def test_minigrid_repr_probe_summary_is_dependency_free(self) -> None:
        summary = representation_probe_summary_markdown(
            {
                "created_at": "2026-06-29T00:00:00+00:00",
                "hypothesis": "repr probe",
                "seed": 1,
                "transition_count": 3,
                "decision": {"met": True, "best_feature_set": "raw_current"},
                "feature_reports": [
                    {
                        "feature_set": "raw_current",
                        "labels": {
                            "mission_object": {
                                "accuracy": 1.0,
                                "majority_baseline": 0.5,
                                "lift": 0.5,
                                "train_examples": 2,
                                "test_examples": 1,
                                "classes": ["ball", "key"],
                            }
                        },
                    }
                ],
            }
        )
        self.assertIn("decision_met: `true`", summary)
        self.assertIn("raw_current", summary)

    def test_minigrid_torch_curriculum_runner_is_dependency_free(self) -> None:
        class FakeTorch:
            def manual_seed(self, seed: int) -> None:
                self.last_seed = seed

        class FakeImage:
            def tolist(self) -> list[list[list[int]]]:
                return [[[0, 0, 0] for _ in range(7)] for _ in range(7)]

        observation = {"image": FakeImage(), "direction": 0, "mission": "unlock the door"}
        made_envs: list[str] = []

        class FakeActionSpace:
            n = 2

            def seed(self, seed: int) -> None:
                self.last_seed = seed

        class FakeEnv:
            def __init__(self, env_id: str) -> None:
                self.env_id = env_id
                self.action_space = FakeActionSpace()

            def reset(self, seed: int) -> tuple[dict[str, object], dict[str, object]]:
                return observation, {}

            def step(self, action: int) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
                reward = 1.0 if self.env_id == "FakeEval-v0" else 0.0
                return observation, reward, True, False, {}

            def close(self) -> None:
                pass

        class FakeGym:
            def make(self, env_id: str) -> FakeEnv:
                made_envs.append(env_id)
                return FakeEnv(env_id)

        class FakeAgent:
            instances: list["FakeAgent"] = []

            def __init__(
                self,
                torch: object,
                actions: int,
                config: TorchAgentConfig,
                device: object,
                seed: int,
                epsilon: float | None = None,
                representation_objective: str = "none",
                representation_beta: float = 0.0,
                representation_state_beta: float = 0.0,
                representation_target_visibility_beta: float = 0.0,
            ) -> None:
                self.actions = actions
                self.updates = 0
                self.representation_objective = representation_objective
                self.representation_updates = 0
                self.encoder_frozen = False
                self.freeze_calls = 0
                FakeAgent.instances.append(self)

            def choose(
                self,
                features: dict[int, float],
                force_random: bool = False,
                action_bonus: dict[int, float] | None = None,
                bonus_weight: float = 1.0,
            ) -> int:
                return 0

            def action_values(self, features: dict[int, float]) -> dict[int, float]:
                return {0: 0.0, 1: 0.0}

            def action_prior_values(self, features: dict[int, float]) -> dict[int, float]:
                return {0: 0.0, 1: 1.0}

            def update(
                self,
                features: dict[int, float],
                action: int,
                reward: float,
                next_features: dict[int, float],
                done: bool,
            ) -> None:
                self.updates += 1

            def update_representation(
                self,
                features: dict[int, float],
                action: int,
                target_vector: list[float] | int,
            ) -> float | None:
                if self.representation_objective == "none":
                    return None
                self.representation_updates += 1
                self.last_representation_target = target_vector
                return 0.25

            def freeze_encoder(self) -> None:
                self.encoder_frozen = True
                self.freeze_calls += 1

            def parameter_count(self) -> int:
                return 10

            def representation_parameter_count(self) -> int:
                return 2 if self.representation_objective != "none" else 0

        stages = (
            TorchCurriculumStage("warmup", "FakeWarmup-v0", max_steps=1, episodes=2),
            TorchCurriculumStage("eval", "FakeEval-v0", max_steps=1, episodes=2),
        )
        condition = Condition(
            name="curriculum",
            encoder_mode="raw",
            episodes=4,
            decoder_delay_episodes=1,
            intrinsic_beta=0.0,
            intrinsic_mode="none",
            seed=31,
            representation_objective="action_prior",
            representation_beta=0.3,
            action_prior_weight=0.2,
        )
        agent_config = TorchAgentConfig(
            feature_dim=128,
            hidden_dim=8,
            learning_rate=0.001,
            gamma=0.9,
            epsilon=0.0,
            batch_size=1,
            replay_capacity=4,
            target_sync_updates=2,
            device="cpu",
        )

        with mock.patch.object(minigrid_torch_module, "TorchDQNAgent", FakeAgent):
            report = run_minigrid_torch_curriculum_condition(
                gym=FakeGym(),
                torch=FakeTorch(),
                stages=stages,
                active_stages=("warmup", "eval"),
                condition=condition,
                agent_config=agent_config,
                device="cpu",
            )

        self.assertEqual(made_envs, ["FakeWarmup-v0", "FakeEval-v0"])
        self.assertEqual(report["final_stage"]["stage"], "eval")
        self.assertEqual(report["success_rate_last_window"], 1.0)
        self.assertEqual(report["representation_updates"], 4)
        self.assertEqual(report["updates"], 3)
        self.assertEqual(report["action_prior_weight"], 0.2)
        self.assertIsInstance(FakeAgent.instances[0].last_representation_target, int)
        self.assertIn(
            "warmup",
            torch_summary_markdown(
                {
                    "created_at": "2026-06-29T00:00:00+00:00",
                    "hypothesis": "torch curriculum",
                    "env_id": "BabyAI-Unlock-v0",
                    "framework": {"version": "fake", "device": "cpu"},
                    "winner_last_window": "curriculum",
                    "stages": [
                        {"name": "warmup", "env_id": "FakeWarmup-v0", "episodes": 2},
                        {"name": "eval", "env_id": "FakeEval-v0", "episodes": 2},
                    ],
                    "results": [report],
                }
            ),
        )
        self.assertIn("mission_target_visible_rate_last_window", report["final_stage"])

    def test_minigrid_torch_two_phase_protocol_freezes_and_stops_representation(self) -> None:
        class FakeTorch:
            def manual_seed(self, seed: int) -> None:
                self.last_seed = seed

        class FakeImage:
            def tolist(self) -> list[list[list[int]]]:
                return [[[0, 0, 0] for _ in range(7)] for _ in range(7)]

        observation = {"image": FakeImage(), "direction": 0, "mission": "go"}
        made_envs: list[str] = []

        class FakeActionSpace:
            n = 2

            def seed(self, seed: int) -> None:
                self.last_seed = seed

        class FakeEnv:
            def __init__(self, env_id: str) -> None:
                self.env_id = env_id
                self.action_space = FakeActionSpace()

            def reset(self, seed: int) -> tuple[dict[str, object], dict[str, object]]:
                return observation, {}

            def step(self, action: int) -> tuple[dict[str, object], float, bool, bool, dict[str, object]]:
                return observation, 1.0 if self.env_id == "FakeEval-v0" else 0.0, True, False, {}

            def close(self) -> None:
                pass

        class FakeGym:
            def make(self, env_id: str) -> FakeEnv:
                made_envs.append(env_id)
                return FakeEnv(env_id)

        class FakeAgent:
            instances: list["FakeAgent"] = []

            def __init__(
                self,
                torch: object,
                actions: int,
                config: TorchAgentConfig,
                device: object,
                seed: int,
                epsilon: float | None = None,
                representation_objective: str = "none",
                representation_beta: float = 0.0,
                representation_state_beta: float = 0.0,
                representation_target_visibility_beta: float = 0.0,
            ) -> None:
                self.actions = actions
                self.updates = 0
                self.representation_objective = representation_objective
                self.representation_updates = 0
                self.encoder_frozen = False
                self.freeze_calls = 0
                FakeAgent.instances.append(self)

            def choose(
                self,
                features: dict[int, float],
                force_random: bool = False,
                action_bonus: dict[int, float] | None = None,
                bonus_weight: float = 1.0,
            ) -> int:
                return 0

            def action_values(self, features: dict[int, float]) -> dict[int, float]:
                return {0: 0.0, 1: 0.0}

            def action_prior_values(self, features: dict[int, float]) -> dict[int, float]:
                return {}

            def update(
                self,
                features: dict[int, float],
                action: int,
                reward: float,
                next_features: dict[int, float],
                done: bool,
            ) -> None:
                self.updates += 1

            def update_representation(
                self,
                features: dict[int, float],
                action: int,
                target_vector: list[float] | int,
            ) -> float | None:
                self.representation_updates += 1
                return 0.5

            def freeze_encoder(self) -> None:
                self.encoder_frozen = True
                self.freeze_calls += 1

            def parameter_count(self) -> int:
                return 10

            def representation_parameter_count(self) -> int:
                return 2

        stages = (
            TorchCurriculumStage("ad_warmup", "FakeWarmup-v0", max_steps=1, episodes=2),
            TorchCurriculumStage("da_eval", "FakeEval-v0", max_steps=1, episodes=2),
        )
        condition = Condition(
            name="two_phase",
            encoder_mode="raw",
            episodes=4,
            decoder_delay_episodes=2,
            intrinsic_beta=0.0,
            intrinsic_mode="none",
            seed=41,
            representation_objective="state_plus_delta",
            representation_beta=0.3,
            freeze_encoder_after_delay=True,
            stop_representation_after_delay=True,
        )
        agent_config = TorchAgentConfig(
            feature_dim=128,
            hidden_dim=8,
            learning_rate=0.001,
            gamma=0.9,
            epsilon=0.0,
            batch_size=1,
            replay_capacity=4,
            target_sync_updates=2,
            device="cpu",
        )

        with mock.patch.object(minigrid_torch_module, "TorchDQNAgent", FakeAgent):
            report = run_minigrid_torch_curriculum_condition(
                gym=FakeGym(),
                torch=FakeTorch(),
                stages=stages,
                active_stages=("ad_warmup", "da_eval"),
                condition=condition,
                agent_config=agent_config,
                device="cpu",
            )

        agent = FakeAgent.instances[0]
        self.assertEqual(made_envs, ["FakeWarmup-v0", "FakeEval-v0"])
        self.assertEqual(agent.representation_updates, 2)
        self.assertEqual(agent.updates, 2)
        self.assertEqual(agent.freeze_calls, 1)
        self.assertTrue(report["encoder_frozen"])
        self.assertEqual(report["representation_updates"], 2)
        self.assertEqual(report["final_stage"]["representation_updates"], 0)

    def test_minigrid_torch_rejects_invalid_representation_objective(self) -> None:
        with self.assertRaises(ValueError):
            parse_minigrid_torch_config(
                {
                    "conditions": [
                        {
                            "name": "bad",
                            "encoder_mode": "raw",
                            "episodes": 4,
                            "decoder_delay_episodes": 1,
                            "intrinsic_beta": 0.0,
                            "intrinsic_mode": "none",
                            "representation_objective": "mystery",
                            "representation_beta": 0.1,
                        }
                    ]
                },
                seed=1,
            )

    def test_minigrid_torch_rejects_action_prior_weight_without_action_prior(self) -> None:
        with self.assertRaises(ValueError):
            parse_minigrid_torch_config(
                {
                    "conditions": [
                        {
                            "name": "bad",
                            "encoder_mode": "raw",
                            "episodes": 4,
                            "decoder_delay_episodes": 1,
                            "intrinsic_beta": 0.0,
                            "intrinsic_mode": "none",
                            "representation_objective": "next_task_signal",
                            "representation_beta": 0.1,
                            "action_prior_weight": 0.1,
                        }
                    ]
                },
                seed=1,
            )

    def test_minigrid_torch_sweep_aggregate_is_dependency_free(self) -> None:
        runs = [
            {
                "winner_last_window": "A",
                "framework": {"version": "2.x", "device": "cuda"},
                "results": [
                    {
                        "name": "A",
                        "seed": 601,
                        "success_rate_all": 0.1,
                        "success_rate_last_window": 0.2,
                        "mean_return_last_window": 0.3,
                        "mission_target_visible_rate_last_window": 0.6,
                        "mission_target_center_rate_last_window": 0.4,
                        "mission_target_near_rate_last_window": 0.5,
                        "updates": 10,
                        "parameter_count": 20,
                    },
                    {
                        "name": "B",
                        "seed": 602,
                        "success_rate_all": 0.0,
                        "success_rate_last_window": 0.0,
                        "mean_return_last_window": 0.0,
                        "mission_target_visible_rate_last_window": 0.1,
                        "mission_target_center_rate_last_window": 0.0,
                        "mission_target_near_rate_last_window": 0.1,
                        "updates": 8,
                        "parameter_count": 20,
                    },
                ],
            },
            {
                "winner_last_window": "B",
                "framework": {"version": "2.x", "device": "cuda"},
                "results": [
                    {
                        "name": "A",
                        "seed": 603,
                        "success_rate_all": 0.0,
                        "success_rate_last_window": 0.0,
                        "mean_return_last_window": 0.0,
                        "mission_target_visible_rate_last_window": 0.2,
                        "mission_target_center_rate_last_window": 0.0,
                        "mission_target_near_rate_last_window": 0.1,
                        "updates": 6,
                        "parameter_count": 20,
                    },
                    {
                        "name": "B",
                        "seed": 604,
                        "success_rate_all": 0.2,
                        "success_rate_last_window": 0.4,
                        "mean_return_last_window": 0.5,
                        "mission_target_visible_rate_last_window": 0.7,
                        "mission_target_center_rate_last_window": 0.5,
                        "mission_target_near_rate_last_window": 0.6,
                        "updates": 12,
                        "parameter_count": 20,
                    },
                ],
            },
        ]
        aggregate = aggregate_torch_reports(runs, seeds=[701, 702])
        by_name = {row["name"]: row for row in aggregate}
        self.assertEqual(by_name["A"]["win_count"], 1)
        self.assertEqual(by_name["B"]["win_count"], 1)
        self.assertAlmostEqual(by_name["B"]["mean_success_rate_last_window"], 0.2)
        self.assertAlmostEqual(by_name["A"]["median_return_last_window"], 0.15)
        self.assertAlmostEqual(by_name["B"]["median_return_last_window"], 0.25)
        self.assertAlmostEqual(by_name["A"]["mean_mission_target_visible_rate_last_window"], 0.4)
        self.assertAlmostEqual(by_name["B"]["mean_mission_target_center_rate_last_window"], 0.25)
        self.assertEqual(by_name["A"]["condition_seeds"], [601, 603])
        summary = torch_sweep_summary_markdown(
            {
                "created_at": "2026-06-29T00:00:00+00:00",
                "hypothesis": "torch sweep",
                "seeds": [701, 702],
                "winner_by_mean_success_last_window": "B",
                "frameworks": [run["framework"] for run in runs],
                "aggregate": aggregate,
                "runs": runs,
            }
        )
        self.assertIn("PyTorch sweep", summary)
        self.assertIn("median_return_last", summary)
        self.assertIn("target_visible_last", summary)
        self.assertIn("| A | 1 | 0.050 | 0.100 | 0.100 | 0.150 | 0.150 | 0.400 | 0.200 | 0.300 | 8.0 | 20 |", summary)
        self.assertIn("Per-Seed Winners", summary)

    def test_gpu_compat_policy_is_dependency_free(self) -> None:
        self.assertLess(DriverVersion.parse("576.88"), DriverVersion.parse("580.0"))
        self.assertLess(DriverVersion.parse("560.99"), DriverVersion.parse("580.0"))

        ready = evaluate_worker(
            parse_worker_policy(
                {
                    "worker_class": "ready",
                    "driver_version": "596.21",
                    "cuda_umd": "13.2",
                    "primary_wheel": "cu132",
                }
            )
        )
        self.assertEqual(ready["status"], "gpu_candidate")
        self.assertEqual(ready["primary_index_url"], "https://download.pytorch.org/whl/cu132")

        blocked = evaluate_worker(
            parse_worker_policy(
                {
                    "worker_class": "driver-blocked",
                    "driver_version": "576.88",
                    "cuda_umd": "12.9",
                    "requires_cuda13_wheel": True,
                    "primary_wheel": "cu132",
                }
            )
        )
        self.assertEqual(blocked["status"], "fallback_required")
        self.assertIn("below", blocked["reason"])
        self.assertIn("CUDA UMD 12", blocked["reason"])

        report = build_report(
            {
                "created_at": "2026-06-29 JST",
                "source_commit": "test",
                "workers": [
                    {
                        "worker_class": "legacy-cuda",
                        "driver_version": "596.21",
                        "cuda_umd": "13.2",
                        "primary_wheel": "cu126",
                    }
                ],
            }
        )
        self.assertEqual(report["results"][0]["status"], "gpu_candidate")


if __name__ == "__main__":
    unittest.main()
