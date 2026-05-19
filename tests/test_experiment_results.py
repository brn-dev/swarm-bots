import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_logs.experiment_results import DEFAULT_X_COLUMN, EP_REW_EMA_COLUMN, load_experiment_groups


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


if __name__ == "__main__":
    unittest.main()
