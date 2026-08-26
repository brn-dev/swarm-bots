import io
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from experiments import plot_all_thesis
from plot_logs.experiment_results import (
    DEFAULT_X_COLUMN,
    EP_REW_EMA_COLUMN,
    EP_SUCCESS_RATE_EMA_COLUMN,
    GROUP_MARKER_SIZE,
    ExperimentGroup,
    ExperimentPlotSelection,
    ExperimentRunLog,
    LEGEND_FONT_SIZE,
    PLOT_FONT_SIZE,
    LineStyle,
    load_experiment_groups,
    joined_plot_title,
    plot_experiment_results,
    plot_experiment_selection,
    selected_groups_for_plot_selection,
    local_average_marker_points,
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

    def test_line_styles_and_markers_reach_curves_and_legends_in_main_and_selection_plots(self) -> None:
        styles: dict[str, LineStyle] = {
            "dotted": ":",
            "dash_dot_dot": (0, (3, 2, 1, 2, 1, 2)),
            "dash_dot": (0, (5, 2, 1, 2)),
            "dashed": (0, (7, 3)),
            "solid": "-",
        }
        names = (*styles, "default")
        groups = [
            ExperimentGroup(
                name=name,
                display_name=name,
                runs=[
                    ExperimentRunLog(
                        group_name=name,
                        run_name=f"run-{index}",
                        path=Path("unused.csv"),
                        x_values=np.linspace(0.0, 200.0, 201),
                        series={
                            EP_REW_EMA_COLUMN: np.linspace(0.0, 1.0, 201),
                            EP_SUCCESS_RATE_EMA_COLUMN: np.linspace(0.0, 50.0, 201),
                        },
                    )
                    for index in range(2)
                ],
            )
            for name in names
        ]
        selection = ExperimentPlotSelection(
            name="ablation",
            group_names=("default", "dash_dot_dot", "solid"),
            display_name_overrides={"solid": "Renamed baseline"},
        )

        with (
            patch("plot_logs.experiment_results.load_experiment_groups", return_value=groups),
            patch("plot_logs.experiment_results.save_figure_variants", return_value=[]) as save,
        ):
            try:
                plot_experiment_results(
                    Path("unused"),
                    Path("unused-output"),
                    main_group_names=tuple(reversed(names)),
                    extra_plot_selections=(selection,),
                    group_linestyle_overrides=styles,
                    group_marker_overrides={"solid": "D"},
                    theoretical_maximum=100.0,
                    run_length_limit=100,
                    cut_at_limit=True,
                )

                self.assertEqual(save.call_count, 8)
                for call in save.call_args_list:
                    figure, output_path = call.args
                    figure.canvas.draw()
                    axis = figure.axes[0]
                    is_selection = "ablation" in output_path.stem
                    plot_names = selection.group_names if is_selection else tuple(reversed(names))
                    runs_per_group = 2 if "individual" in output_path.stem else 1
                    legend_lines = axis.get_legend().get_lines()
                    curve_lines = [line for line in axis.lines if line.get_linestyle() != "None"]
                    marker_layers = [line for line in axis.lines if line.get_linestyle() == "None"]
                    self.assertEqual(len(curve_lines), len(plot_names) * runs_per_group + 1)
                    self.assertEqual(len(marker_layers), runs_per_group)
                    self.assertEqual(len(legend_lines), len(plot_names) + 1)
                    for index, name in enumerate(plot_names):
                        expected = Line2D([], [], linestyle=styles.get(name, "-"))
                        lines = curve_lines[index * runs_per_group:(index + 1) * runs_per_group]
                        for line in (*lines, legend_lines[index]):
                            # get_linestyle() collapses every custom dash sequence to "--".
                            self.assertEqual(line._unscaled_dash_pattern, expected._unscaled_dash_pattern)
                            self.assertEqual(line.get_marker(), "D" if name == "solid" else "None")
                            if name == "solid":
                                self.assertEqual(line.get_markersize(), GROUP_MARKER_SIZE)
                                self.assertEqual(line.get_markerfacecolor(), "white")
                        for line in lines:
                            np.testing.assert_array_equal(line.get_xdata(), np.arange(101))
                            scale = 50.0 if "success_rate" in output_path.stem else 1.0
                            np.testing.assert_allclose(line.get_ydata(), np.arange(101) / 200 * scale)
                            self.assertEqual(line.get_markevery(), [])
                    for layer in marker_layers:
                        marker_x, marker_y = layer.get_data()
                        np.testing.assert_allclose(marker_x, np.arange(1, 13) * 100 / 13)
                        local_mean_steps = np.array([
                            7.5, 15.5, 23.5, 30.5, 38.5, 46.5,
                            53.5, 61.5, 69.5, 76.5, 84.5, 92.5,
                        ])
                        np.testing.assert_allclose(marker_y, local_mean_steps / 200 * scale)
                        self.assertEqual(layer.get_marker(), "D")
                        self.assertGreater(layer.get_zorder(), curve_lines[0].get_zorder())
                        self.assertEqual(layer.get_label(), "_nolegend_")
                    self.assertEqual(axis.lines[-1].get_linestyle(), "--")
                    self.assertEqual(legend_lines[-1].get_linestyle(), "--")
            finally:
                for call in save.call_args_list:
                    plt.close(call.args[0])

    @patch("plot_logs.experiment_results.GROUP_MARKER_COUNT", 7)
    def test_marker_averages_dampen_spikes_without_modifying_the_curve(self) -> None:
        x_values = np.linspace(0, 100, 1001)
        y_values = np.full_like(x_values, 10)
        y_values[x_values == 50] = 31
        original_y = y_values.copy()

        marker_x, marker_y = local_average_marker_points(x_values, y_values)

        np.testing.assert_allclose(marker_x, np.linspace(12.5, 87.5, 7))
        np.testing.assert_allclose(marker_y, [10, 10, 10, 11, 10, 10, 10])
        np.testing.assert_array_equal(y_values, original_y)

    @patch("plot_logs.experiment_results.GROUP_MARKER_COUNT", 7)
    def test_marker_averages_skip_empty_windows_and_nonfinite_samples(self) -> None:
        x_values = np.concatenate(([np.nan], np.linspace(0, 100, 1001), [np.inf]))
        y_values = 2 * x_values + 3
        y_values[(x_values >= 45) & (x_values <= 55)] = np.nan

        marker_x, marker_y = local_average_marker_points(x_values, y_values)

        np.testing.assert_allclose(marker_x, [12.5, 25, 37.5, 62.5, 75, 87.5])
        np.testing.assert_allclose(marker_y, 2 * marker_x + 3)
        for x, y, expected_x, expected_y in (
            ([], [], [], []),
            ([1.0], [np.nan], [], []),
            ([0.0, 100.0], [1.0, 2.0], [], []),
            ([1.0], [2.0], [1.0], [2.0]),
            ([1.0, 1.0], [2.0, 4.0], [1.0], [3.0]),
        ):
            with self.subTest(x=x, y=y):
                actual_x, actual_y = local_average_marker_points(np.array(x), np.array(y))
                np.testing.assert_array_equal(actual_x, expected_x)
                np.testing.assert_array_equal(actual_y, expected_y)

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


class ThesisPlotRunnerTests(unittest.TestCase):
    def test_discovery_includes_main_extended_and_finetuning_plots(self) -> None:
        scripts = plot_all_thesis.discover_plot_scripts()
        names = {path.relative_to(plot_all_thesis.EXPERIMENTS_DIR).as_posix() for path in scripts}

        self.assertTrue({
            "thesis_mjw_find_opening/plot_results.py",
            "thesis_mjw_po_wall_medium/plot_results.py",
            "thesis_mjw_po_wall_medium/plot_250m_results.py",
            "thesis_mjw_po_wall_disconnected_finetune/plot_results.py",
            "thesis_mjw_po_wall_disconnected_finetune/plot_hard_wall_results.py",
            "thesis_parallel_env_ablation_po_wall_medium/plot_results.py",
        }.issubset(names))
        self.assertNotIn("thesis_plot_common.py", names)
        self.assertNotIn("plot_all_thesis.py", names)
        self.assertEqual(scripts, sorted(scripts))

    def test_runner_attempts_every_script_and_reports_failures(self) -> None:
        scripts = plot_all_thesis.discover_plot_scripts()[:3]
        for returncodes, expected_exit in (((0, 0, 0), 0), ((0, 2, 0), 1)):
            with (
                self.subTest(returncodes=returncodes),
                patch.object(plot_all_thesis, "discover_plot_scripts", return_value=scripts),
                patch.object(
                    plot_all_thesis.subprocess, "run",
                    side_effect=[SimpleNamespace(returncode=code) for code in returncodes],
                ) as run,
                patch("sys.stdout", new_callable=io.StringIO) as output,
            ):
                self.assertEqual(plot_all_thesis.main([]), expected_exit)
                self.assertEqual(run.call_count, len(scripts))
                for call, script in zip(run.call_args_list, scripts, strict=True):
                    self.assertEqual(call.args, ([sys.executable, str(script)],))
                    self.assertEqual(call.kwargs, {"cwd": plot_all_thesis.REPO_ROOT, "check": False})
                if expected_exit:
                    self.assertIn(
                        f"FAILED (exit 2): {scripts[1].relative_to(plot_all_thesis.REPO_ROOT).as_posix()}",
                        output.getvalue(),
                    )
                else:
                    self.assertIn("3/3 thesis plot scripts succeeded", output.getvalue())

    def test_dry_run_lists_scripts_without_invoking_them(self) -> None:
        with (
            patch.object(plot_all_thesis.subprocess, "run") as run,
            patch("sys.stdout", new_callable=io.StringIO) as output,
        ):
            self.assertEqual(plot_all_thesis.main(["--dry-run"]), 0)

        run.assert_not_called()
        self.assertEqual(output.getvalue().splitlines(), [
            script.relative_to(plot_all_thesis.REPO_ROOT).as_posix()
            for script in plot_all_thesis.discover_plot_scripts()
        ])

    def test_no_scripts_is_reported_as_failure(self) -> None:
        with (
            patch.object(plot_all_thesis, "discover_plot_scripts", return_value=[]),
            patch.object(plot_all_thesis.subprocess, "run") as run,
            patch("sys.stderr", new_callable=io.StringIO) as output,
        ):
            self.assertEqual(plot_all_thesis.main([]), 1)

        run.assert_not_called()
        self.assertIn("No thesis plot scripts found", output.getvalue())


if __name__ == "__main__":
    unittest.main()
