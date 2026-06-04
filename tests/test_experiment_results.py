import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_logs.experiment_results import (
    DEFAULT_X_COLUMN,
    EP_REW_EMA_COLUMN,
    EP_SUCCESS_RATE_EMA_COLUMN,
    load_experiment_groups,
    plot_experiment_results,
)


class ExperimentResultsTests(unittest.TestCase):
    def test_load_experiment_groups_ignores_missing_extra_group_source(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            experiment_run_dir = tmp_path / "experiment"
            experiment_run_dir.mkdir()

            extra_source_dir = tmp_path / "extra-source"
            extra_source_dir.mkdir()
            log_path = extra_source_dir / "log.csv"
            log_path.write_text(
                f"{DEFAULT_X_COLUMN};{EP_REW_EMA_COLUMN}\n1;2.5\n",
                encoding="utf-8",
            )

            groups = load_experiment_groups(
                experiment_run_dir,
                extra_group_sources={
                    "extra": [
                        tmp_path / "missing-source",
                        extra_source_dir,
                    ],
                },
            )

        self.assertEqual(len(groups), 1)
        self.assertEqual(groups[0].name, "extra")
        self.assertEqual(len(groups[0].runs), 1)
        self.assertEqual(groups[0].runs[0].path, log_path)

    def test_plot_experiment_results_writes_success_rate_plots(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            experiment_run_dir = tmp_path / "experiment"
            output_dir = tmp_path / "plots"
            for group_name, run_name, reward_values, success_values in (
                ("alpha", "run-1", (1.0, 2.0), (0.0, 50.0)),
                ("beta", "run-1", (3.0, 4.0), (100.0, 100.0)),
            ):
                run_dir = experiment_run_dir / group_name / run_name
                run_dir.mkdir(parents=True, exist_ok=True)
                run_dir.joinpath("log.csv").write_text(
                    "\n".join(
                        [
                            f"{DEFAULT_X_COLUMN};{EP_REW_EMA_COLUMN};{EP_SUCCESS_RATE_EMA_COLUMN}",
                            f"1;{reward_values[0]};{success_values[0]}",
                            f"2;{reward_values[1]};{success_values[1]}",
                        ]
                    ),
                    encoding="utf-8",
                )

            result = plot_experiment_results(experiment_run_dir, output_dir)
            output_names = {path.name for path in result.output_paths}

            self.assertIn(EP_SUCCESS_RATE_EMA_COLUMN, result.groups[0].runs[0].series)
            self.assertEqual(
                output_names,
                {
                    "ep_rew_ema_individual_runs.png",
                    "ep_rew_ema_grouped.png",
                    "ep_success_rate_ema_individual_runs.png",
                    "ep_success_rate_ema_grouped.png",
                },
            )
            for path in result.output_paths:
                self.assertTrue(path.exists(), path)


if __name__ == "__main__":
    unittest.main()
