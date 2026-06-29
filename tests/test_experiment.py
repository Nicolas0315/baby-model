from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import types
import unittest
from pathlib import Path

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
from baby_model.minigrid_torch import (
    dense_feature_vector,
    parse_minigrid_torch_config,
    select_torch_device,
    torch_summary_markdown,
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
                        "updates": 10,
                        "parameter_count": 20,
                    },
                    {
                        "name": "B",
                        "seed": 602,
                        "success_rate_all": 0.0,
                        "success_rate_last_window": 0.0,
                        "mean_return_last_window": 0.0,
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
                        "updates": 6,
                        "parameter_count": 20,
                    },
                    {
                        "name": "B",
                        "seed": 604,
                        "success_rate_all": 0.2,
                        "success_rate_last_window": 0.4,
                        "mean_return_last_window": 0.5,
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
