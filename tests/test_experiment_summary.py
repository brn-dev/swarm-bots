import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_logs.experiment_results import (
    EP_REW_EMA_COLUMN,
    EP_SUCCESS_RATE_EMA_COLUMN,
    ExperimentGroup,
    ExperimentRunLog,
)
from plot_logs.experiment_summary import (
    summarize_threshold_timesteps,
    summarize_experiment_groups,
    write_experiment_summary,
)


def make_run(
    name: str,
    reward_values: tuple[float, ...],
    success_values: tuple[float, ...],
) -> ExperimentRunLog:
    return ExperimentRunLog(
        group_name="variant",
        run_name=name,
        path=Path(f"{name}.csv"),
        x_values=np.asarray((1.0, 2.0, 3.0)),
        series={
            EP_REW_EMA_COLUMN: np.asarray(reward_values),
            EP_SUCCESS_RATE_EMA_COLUMN: np.asarray(success_values),
        },
    )


class ExperimentSummaryTests(unittest.TestCase):
    def test_summarizes_per_run_tail_averages_and_missing_metrics(self) -> None:
        group = ExperimentGroup(
            name="variant",
            display_name="Variant",
            runs=[
                make_run("run-1", (1.0, 3.0, np.nan), (np.nan, np.nan, np.nan)),
                make_run("run-2", (4.0, 5.0, 7.0), (0.0, 50.0, 100.0)),
            ],
        )

        summary = summarize_experiment_groups([group], tail_points=2)
        variant = summary["variants"]["variant"]

        self.assertEqual(variant["number_of_runs"], 2)
        self.assertEqual(
            variant[EP_REW_EMA_COLUMN],
            {"number_of_runs": 2, "mean": 4.0, "std": 2.0},
        )
        self.assertEqual(
            variant[EP_SUCCESS_RATE_EMA_COLUMN],
            {"number_of_runs": 1, "mean": 75.0, "std": 0.0},
        )
        self.assertEqual(
            variant["success_rate_threshold_timesteps"],
            {
                "25": {"number_of_runs": 1, "mean": 2.0, "std": 0.0},
                "50": {"number_of_runs": 1, "mean": 2.0, "std": 0.0},
                "75": {"number_of_runs": 1, "mean": 3.0, "std": 0.0},
                "90": {"number_of_runs": 1, "mean": 3.0, "std": 0.0},
                "95": {"number_of_runs": 1, "mean": 3.0, "std": 0.0},
                "98": {"number_of_runs": 1, "mean": 3.0, "std": 0.0},
            },
        )

    def test_summarizes_first_per_run_threshold_timesteps(self) -> None:
        runs = [
            make_run("run-1", (0.0, 0.0, 0.0), (80.0, 90.0, 95.0)),
            make_run("run-2", (0.0, 0.0, 0.0), (90.0, 90.0, 100.0)),
        ]

        summaries = summarize_threshold_timesteps(
            runs,
            EP_SUCCESS_RATE_EMA_COLUMN,
            thresholds=(50.0, 90.0, 98.0),
        )

        self.assertEqual(
            summaries,
            {
                "50": {"number_of_runs": 2, "mean": 1.0, "std": 0.0},
                "90": {"number_of_runs": 2, "mean": 1.5, "std": 0.5},
                "98": {"number_of_runs": 1, "mean": 3.0, "std": 0.0},
            },
        )

    def test_unreached_thresholds_have_no_timestep_statistics(self) -> None:
        runs = [make_run("run-1", (0.0, 0.0, 0.0), (80.0, 89.0, np.nan))]

        summaries = summarize_threshold_timesteps(
            runs,
            EP_SUCCESS_RATE_EMA_COLUMN,
            thresholds=(90.0, 95.0),
        )

        self.assertEqual(
            summaries,
            {
                "90": {"number_of_runs": 0, "mean": None, "std": None},
                "95": {"number_of_runs": 0, "mean": None, "std": None},
            },
        )

    def test_run_length_limit_excludes_later_values(self) -> None:
        group = ExperimentGroup(
            name="variant",
            display_name="Variant",
            runs=[make_run("run-1", (1.0, 2.0, 100.0), (1.0, 2.0, 100.0))],
        )

        summary = summarize_experiment_groups(
            [group],
            tail_points=1,
            run_length_limit=2,
            cut_at_limit=True,
        )

        self.assertEqual(
            summary["variants"]["variant"][EP_REW_EMA_COLUMN]["mean"],
            2.0,
        )
        self.assertEqual(summary["run_length_limit"], 2)
        self.assertEqual(
            summary["variants"]["variant"]["success_rate_threshold_timesteps"],
            {
                threshold: {"number_of_runs": 0, "mean": None, "std": None}
                for threshold in ("25", "50", "75", "90", "95", "98")
            },
        )

    def test_writes_strict_json(self) -> None:
        summary = {
            "variants": {
                "variant": {
                    EP_REW_EMA_COLUMN: {
                        "number_of_runs": 0,
                        "mean": None,
                        "std": None,
                    }
                }
            }
        }
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_path = Path(tmp_dir) / "nested" / "summary.json"
            resolved_path = write_experiment_summary(summary, output_path)

            self.assertEqual(resolved_path, output_path.resolve())
            self.assertIn('"mean": null', output_path.read_text(encoding="utf-8"))

    def test_rejects_empty_tail(self) -> None:
        with self.assertRaisesRegex(ValueError, "tail_points"):
            summarize_experiment_groups([], tail_points=0)


if __name__ == "__main__":
    unittest.main()
