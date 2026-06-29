from __future__ import annotations

import tempfile
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
from baby_model.envs import BabyGrid
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


if __name__ == "__main__":
    unittest.main()
