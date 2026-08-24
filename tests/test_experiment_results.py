import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_logs.experiment_results import (
    DEFAULT_X_COLUMN,
    EVALUATION_PLOT_SPECS,
    EVAL_EP_REW_COLUMN,
    EVAL_SUCCESS_RATE_COLUMN,
    EP_REW_EMA_COLUMN,
    EP_SUCCESS_RATE_EMA_COLUMN,
    ExperimentGroup,
    ExperimentPlotSelection,
    LEGEND_FONT_SIZE,
    PLOT_FONT_SIZE,
    load_experiment_groups,
    joined_plot_title,
    plot_experiment_results,
    plot_experiment_selection,
    selected_groups_for_plot_selection,
)


class ExperimentResultsTests(unittest.TestCase):
    def test_joined_plot_title_appends_nonempty_suffixes(self) -> None:
        self.assertEqual(
            joined_plot_title(
                "Episode Reward EMA By Group",
                "PO-Wall (Medium)",
                "MAT-QCX NOP Ablation",
            ),
            "Episode Reward EMA By Group - PO-Wall (Medium) - MAT-QCX NOP Ablation",
        )

    def test_plot_selection_can_override_display_names(self) -> None:
        group = ExperimentGroup(name="baseline", display_name="Baseline", runs=[])
        selection = ExperimentPlotSelection(
            name="ablation",
            group_names=("baseline",),
            display_name_overrides={"baseline": "Baseline + feature"},
        )

        selected_groups = selected_groups_for_plot_selection(selection, [group])

        self.assertEqual(selected_groups[0].display_name, "Baseline + feature")
        self.assertEqual(group.display_name, "Baseline")

    def test_plot_selection_title_includes_scenario_and_selection(self) -> None:
        groups = [
            SimpleNamespace(name="baseline"),
            SimpleNamespace(name="ablation"),
        ]
        selection = ExperimentPlotSelection(
            name="nop_ablation",
            group_names=("baseline", "ablation"),
            title_suffix="NOP Ablation",
        )

        with (
            patch(
                "plot_logs.experiment_results.metric_has_finite_values",
                return_value=True,
            ),
            patch(
                "plot_logs.experiment_results.plot_individual_metric",
                return_value=[],
            ) as plot_individual_metric,
            patch(
                "plot_logs.experiment_results.plot_group_metric",
                return_value=[],
            ) as plot_group_metric,
        ):
            plot_experiment_selection(
                selection=selection,
                groups=groups,
                output_dir=Path("unused-output"),
                x_column=DEFAULT_X_COLUMN,
                dpis=(100,),
                theoretical_maximum=None,
                run_length_limit=100,
                cut_at_limit=True,
                group_line_width=1.0,
                group_line_alpha=1.0,
                colors={},
                title_suffix="Find-Opening",
            )

        for call in (*plot_individual_metric.call_args_list, *plot_group_metric.call_args_list):
            self.assertTrue(
                call.kwargs["title"].endswith(" - Find-Opening - NOP Ablation")
            )
            self.assertEqual(call.kwargs["font_size"], PLOT_FONT_SIZE)
            self.assertEqual(call.kwargs["legend_font_size"], LEGEND_FONT_SIZE)

    def test_plot_selection_can_override_font_sizes(self) -> None:
        groups = [SimpleNamespace(name="baseline"), SimpleNamespace(name="ablation")]
        selection = ExperimentPlotSelection(
            name="compact_panel",
            group_names=("baseline", "ablation"),
            font_size=22,
            legend_font_size=24,
        )

        with (
            patch("plot_logs.experiment_results.metric_has_finite_values", return_value=True),
            patch("plot_logs.experiment_results.plot_individual_metric", return_value=[]) as individual,
            patch("plot_logs.experiment_results.plot_group_metric", return_value=[]) as grouped,
        ):
            plot_experiment_selection(
                selection=selection,
                groups=groups,
                output_dir=Path("unused-output"),
                x_column=DEFAULT_X_COLUMN,
                dpis=(100,),
                theoretical_maximum=None,
                run_length_limit=100,
                cut_at_limit=True,
                group_line_width=1.0,
                group_line_alpha=1.0,
                colors={},
            )

        for call in (*individual.call_args_list, *grouped.call_args_list):
            self.assertEqual(call.kwargs["font_size"], 22)
            self.assertEqual(call.kwargs["legend_font_size"], 24)

    def test_plot_selection_can_omit_selection_title_suffix(self) -> None:
        groups = [SimpleNamespace(name="baseline"), SimpleNamespace(name="ablation")]
        selection = ExperimentPlotSelection(
            name="nop_ablation",
            group_names=("baseline", "ablation"),
            title_suffix="NOP Ablation",
        )

        with (
            patch("plot_logs.experiment_results.metric_has_finite_values", return_value=True),
            patch("plot_logs.experiment_results.plot_individual_metric", return_value=[]) as individual,
            patch("plot_logs.experiment_results.plot_group_metric", return_value=[]) as grouped,
        ):
            plot_experiment_selection(
                selection=selection,
                groups=groups,
                output_dir=Path("unused-output"),
                x_column=DEFAULT_X_COLUMN,
                dpis=(100,),
                theoretical_maximum=None,
                run_length_limit=100,
                cut_at_limit=True,
                group_line_width=1.0,
                group_line_alpha=1.0,
                colors={},
                title_suffix="Find-Opening",
                include_selection_title_suffix=False,
            )

        self.assertEqual(
            {call.kwargs["title"] for call in individual.call_args_list},
            {
                "Episode Reward EMA Per Run - Find-Opening",
                "Episode Success Rate EMA Per Run - Find-Opening",
            },
        )
        self.assertEqual(
            {call.kwargs["title"] for call in grouped.call_args_list},
            {
                "Episode Reward EMA By Group - Find-Opening",
                "Episode Success Rate EMA By Group - Find-Opening",
            },
        )

    def test_main_group_names_excludes_loaded_groups_only_from_main_plots(self) -> None:
        baseline_group = SimpleNamespace(name="baseline")
        ablation_group = SimpleNamespace(name="ablation")

        with (
            patch(
                "plot_logs.experiment_results.load_experiment_groups",
                return_value=[baseline_group, ablation_group],
            ),
            patch(
                "plot_logs.experiment_results.metric_has_finite_values",
                return_value=True,
            ),
            patch(
                "plot_logs.experiment_results.plot_individual_metric",
                return_value=[],
            ) as plot_individual_metric,
            patch(
                "plot_logs.experiment_results.plot_group_metric",
                return_value=[],
            ) as plot_group_metric,
        ):
            result = plot_experiment_results(
                Path("unused"),
                Path("unused-output"),
                main_group_names=("baseline",),
                title_suffix="PO-Wall (Medium)",
            )

        self.assertEqual(result.groups, [baseline_group, ablation_group])
        for call in (*plot_individual_metric.call_args_list, *plot_group_metric.call_args_list):
            self.assertEqual(call.args[0], [baseline_group])
            self.assertTrue(call.kwargs["title"].endswith(" - PO-Wall (Medium)"))

    def test_plot_experiment_results_applies_group_color_overrides(self) -> None:
        groups = [SimpleNamespace(name="baseline"), SimpleNamespace(name="variant")]

        with (
            patch(
                "plot_logs.experiment_results.load_experiment_groups",
                return_value=groups,
            ),
            patch(
                "plot_logs.experiment_results.metric_has_finite_values",
                return_value=True,
            ),
            patch(
                "plot_logs.experiment_results.plot_individual_metric",
                return_value=[],
            ) as individual,
            patch(
                "plot_logs.experiment_results.plot_group_metric",
                return_value=[],
            ) as grouped,
        ):
            plot_experiment_results(
                Path("unused"),
                Path("unused-output"),
                group_color_overrides={"baseline": "#CC79A7"},
            )

        for call in (*individual.call_args_list, *grouped.call_args_list):
            self.assertEqual(
                call.kwargs["colors"]["baseline"],
                (0.8, 0.4745098039215686, 0.6549019607843137, 1.0),
            )
            self.assertEqual(
                call.kwargs["colors"]["variant"],
                (0.9019607843137255, 0.6235294117647059, 0.0, 1.0),
            )

    def test_plot_experiment_results_can_plot_evaluation_logs(self) -> None:
        groups = [SimpleNamespace(name="baseline")]

        with (
            patch(
                "plot_logs.experiment_results.load_experiment_groups",
                return_value=groups,
            ) as load_groups,
            patch(
                "plot_logs.experiment_results.metric_has_finite_values",
                return_value=True,
            ),
            patch(
                "plot_logs.experiment_results.plot_individual_metric",
                return_value=[],
            ) as individual,
            patch(
                "plot_logs.experiment_results.plot_group_metric",
                return_value=[],
            ) as grouped,
        ):
            plot_experiment_results(
                Path("unused"),
                Path("unused-output"),
                log_stem="eval_log",
                plot_specs=EVALUATION_PLOT_SPECS,
            )

        self.assertEqual(load_groups.call_args.kwargs["log_stem"], "eval_log")
        self.assertEqual(
            load_groups.call_args.kwargs["required_metric_columns"],
            (EVAL_EP_REW_COLUMN,),
        )
        self.assertEqual(
            load_groups.call_args.kwargs["optional_metric_columns"],
            (EVAL_SUCCESS_RATE_COLUMN,),
        )
        plotted_columns = {
            call.kwargs["metric"].column
            for call in (*individual.call_args_list, *grouped.call_args_list)
        }
        self.assertEqual(
            plotted_columns,
            {EVAL_EP_REW_COLUMN, EVAL_SUCCESS_RATE_COLUMN},
        )

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
