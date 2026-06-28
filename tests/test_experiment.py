from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from baby_model.experiment import run_suite, validate_config, write_run


class ExperimentTest(unittest.TestCase):
    def test_run_suite_returns_all_conditions(self) -> None:
        report = run_suite(seed=3)
        names = [row["name"] for row in report["results"]]
        self.assertEqual(names, ["A_end_to_end", "B_encoder_first", "C_baby_curiosity"])
        self.assertIn(report["winner_last_window"], names)

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
                        }
                    ]
                }
            )


if __name__ == "__main__":
    unittest.main()
