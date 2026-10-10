from __future__ import annotations

# Optional dependency detection and backend selection must precede plotting imports.
# ruff: noqa: E402

import bz2
import gzip
import lzma
import subprocess
import sys
import textwrap
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import pytest

matplotlib = pytest.importorskip("matplotlib")
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from swarmbots.plotting import (
    group_statistics,
    load_groups,
    load_log,
    log_columns,
    plot_experiment_results,
    plot_groups,
    plot_logs,
)

def write_log(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def test_resumed_duplicates_are_discarded_before_smoothing(tmp_path: Path) -> None:
    source = write_log(tmp_path / "resumed.csv", "timesteps;metric\n0;0\n2;100\n2;4\n4;8\n")
    statistics = group_statistics([load_log(source, columns=["metric"])], "metric", smooth=0.5)
    np.testing.assert_allclose(statistics.x_values, [0, 2, 4])
    np.testing.assert_allclose(statistics.mean, [0, 2, 5])


@pytest.fixture
def training_csv() -> str:
    return "timesteps;ep_rew_ema;ep_success_rate_ema;loss;loss__std\n0;0;;2;0.5\n1;2;25;1;0.25\n2;4;50;0.5;0.1\n"


@pytest.mark.parametrize("suffix", [".csv", ".csv.gz", ".csv.bz2", ".csv.xz", ".csv.zip", ".zip"])
def test_compressed_logs_load_the_same_metrics(tmp_path: Path, training_csv: str, suffix: str) -> None:
    path = tmp_path / f"log{suffix}"
    if suffix.endswith(".zip"):
        with ZipFile(path, "w") as archive:
            archive.writestr("nested/log.csv", training_csv)
            archive.writestr("metadata.json", "{}")
    elif suffix == ".csv":
        path.write_text(training_csv, encoding="utf-8")
    else:
        opener = {".csv.gz": gzip.open, ".csv.bz2": bz2.open, ".csv.xz": lzma.open}[suffix]
        with opener(path, "wt", encoding="utf-8") as handle:
            handle.write(training_csv)
    run = load_log(tmp_path)
    np.testing.assert_array_equal(run.x_values, [0, 1, 2])
    np.testing.assert_allclose(run.series["ep_rew_ema"], [0, 2, 4])
    np.testing.assert_allclose(run.series["ep_success_rate_ema"], [np.nan, 25, 50])
    assert "loss" in log_columns(path)


def test_zip_prefers_log_csv_and_rejects_ambiguous_archives(tmp_path: Path, training_csv: str) -> None:
    path = tmp_path / "logs.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("log.csv", training_csv)
        archive.writestr("eval.csv", "other\n1\n")
    assert load_log(path).series["ep_rew_ema"].tolist() == [0, 2, 4]
    with ZipFile(path, "w") as archive:
        archive.writestr("a.csv", training_csv)
        archive.writestr("b.csv", training_csv)
    with pytest.raises(ValueError, match="one CSV"):
        load_log(path)


def test_missing_optional_success_and_unusable_rows(tmp_path: Path) -> None:
    path = write_log(tmp_path / "log.csv", "timesteps;ep_rew_ema\n;2\nnan;3\ninf;4\n0;\n1;5\n")
    run = load_log(path)
    np.testing.assert_array_equal(run.x_values, [0, 1])
    np.testing.assert_allclose(run.series["ep_rew_ema"], [np.nan, 5])
    with pytest.raises(ValueError, match="Missing columns.*ep_success_rate_ema"):
        load_log(path, columns=["ep_success_rate_ema"])


def test_non_scalar_metrics_report_the_cell(tmp_path: Path) -> None:
    path = write_log(tmp_path / "log.csv", "timesteps;histogram\n1;[1, 2]\n")
    with pytest.raises(ValueError, match="row 2, column 'histogram'"):
        load_log(path, columns=["histogram"])


def test_timestamp_axes_and_mixed_types(tmp_path: Path) -> None:
    path = write_log(tmp_path / "log.csv", "timestamp;loss\n2026-01-01T00:00:00Z;2\n2026-01-02T00:00:00Z;1\n")
    run = load_log(path, columns=["loss"], x_column="timestamp")
    assert run.x_is_datetime
    assert run.x_values[1] - run.x_values[0] == 1
    figure = plot_logs(path, columns=["loss"], x_column="timestamp")
    try:
        assert figure.axes[0].get_xlabel() == "timestamp"
        assert isinstance(figure.axes[0].xaxis.get_major_formatter(), matplotlib.dates.ConciseDateFormatter)
    finally:
        plt.close(figure)
    with pytest.raises(ValueError, match="numeric X"):
        load_log(path, columns=["loss"], x_column="timestamp", max_steps=2)
    write_log(path, "timestamp;loss\n1;2\n2026-01-01;3\n")
    with pytest.raises(ValueError, match="Mixed numeric"):
        load_log(path, columns=["loss"], x_column="timestamp")


def test_group_statistics_align_seeds_without_extrapolating(tmp_path: Path) -> None:
    first = load_log(write_log(tmp_path / "first.csv", "timesteps;ep_rew_ema\n0;0\n2;4\n4;8\n"))
    short = load_log(write_log(tmp_path / "short.csv", "timesteps;ep_rew_ema\n1;4\n2;6\n3;8\n"))
    stats = group_statistics([first, short], "ep_rew_ema")
    np.testing.assert_array_equal(stats.x_values, [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(stats.counts, [1, 2, 2, 2, 1])
    np.testing.assert_allclose(stats.mean, [0, 3, 5, 7, 8])
    np.testing.assert_allclose(stats.std, [0, 1, 1, 1, 0])


def test_single_point_missing_metric_and_resumed_values(tmp_path: Path) -> None:
    resumed = load_log(write_log(tmp_path / "resumed.csv", "timesteps;ep_rew_ema\n2;2\n0;0\n2;4\n4;8\n"))
    single = load_log(write_log(tmp_path / "single.csv", "timesteps;ep_rew_ema\n2;6\n"))
    absent = load_log(write_log(tmp_path / "absent.csv", "timesteps;ep_rew_ema;ep_success_rate_ema\n2;;50\n"))
    stats = group_statistics([resumed, single, absent], "ep_rew_ema")
    np.testing.assert_array_equal(stats.counts, [1, 2, 1])
    np.testing.assert_allclose(stats.mean, [0, 5, 8])
    np.testing.assert_allclose(stats.std, [0, 1, 0])


def test_cutoff_precedes_interpolation(tmp_path: Path) -> None:
    late = write_log(tmp_path / "late.csv", "timesteps;ep_rew_ema\n0;0\n10;100\n")
    short = write_log(tmp_path / "short.csv", "timesteps;ep_rew_ema\n0;0\n5;10\n")
    groups = load_groups({"A": [late, short]}, max_steps=5)
    stats = group_statistics(groups["A"], "ep_rew_ema")
    np.testing.assert_array_equal(stats.x_values, [0, 5])
    np.testing.assert_array_equal(stats.counts, [2, 1])
    np.testing.assert_allclose(stats.mean, [0, 10])


def test_discovery_and_explicit_sources_preserve_groups(tmp_path: Path, training_csv: str) -> None:
    first = write_log(tmp_path / "A" / "seed-1" / "log.csv", training_csv)
    second = write_log(tmp_path / "A" / "seed-2" / "log.csv", training_csv)
    direct = write_log(tmp_path / "B" / "log.csv", training_csv)
    (tmp_path / "unrelated").mkdir()
    discovered = load_groups(tmp_path)
    assert list(discovered) == ["A", "B"]
    assert [len(runs) for runs in discovered.values()] == [2, 1]
    named = load_groups({"Baseline": direct.parent, "Alternative": [first, second.parent, first]})
    assert list(named) == ["Baseline", "Alternative"]
    assert [len(runs) for runs in named.values()] == [1, 2]
    with pytest.raises(FileNotFoundError):
        load_groups({"Typo": tmp_path / "missing"})


def test_scalar_plots_smooth_values_and_keep_logged_std_bands(tmp_path: Path, training_csv: str) -> None:
    path = write_log(tmp_path / "log.csv", training_csv)
    output = tmp_path / "plots" / "loss.svg"
    figure = plot_logs(
        path, columns=["loss"], std_columns={"loss": "loss__std"}, smooth=0.5, labels=["Training"], output_path=output
    )
    try:
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [2, 1.5, 1])
        assert figure.axes[0].lines[0].get_label() == "Training"
        assert len(figure.axes[0].collections) == 1
        assert output.is_file()
    finally:
        plt.close(figure)
    with pytest.raises(ValueError, match="smooth"):
        plot_logs(path, smooth=0)


def test_default_labels_distinguish_files_in_the_same_directory(tmp_path: Path, training_csv: str) -> None:
    first = write_log(tmp_path / "first.csv", training_csv)
    second = write_log(tmp_path / "second.csv", training_csv)
    figure = plot_logs([first, second])
    try:
        labels = [line.get_label() for line in figure.axes[0].lines]
        assert len(set(labels)) == 2
    finally:
        plt.close(figure)


def test_group_plots_report_metric_run_count_and_respect_style(tmp_path: Path) -> None:
    first = write_log(tmp_path / "first.csv", "timesteps;ep_rew_ema;ep_success_rate_ema\n0;0;20\n1;2;50\n")
    second = write_log(tmp_path / "second.csv", "timesteps;ep_rew_ema\n0;2\n1;4\n")
    groups = load_groups({"Method": [first, second]})
    figure = plot_groups(groups, colors={"Method": "red"}, linestyles={"Method": "--"})
    try:
        assert figure.axes[0].lines[0].get_label() == "Method (n=2)"
        assert figure.axes[1].lines[0].get_label() == "Method (n=1)"
        assert figure.axes[0].lines[0].get_color() == "red"
        assert figure.axes[0].lines[0].get_linestyle() == "--"
        np.testing.assert_allclose(figure.axes[0].lines[0].get_ydata(), [1, 3])
    finally:
        plt.close(figure)


def test_experiment_exports_each_metric_and_closes_figures(tmp_path: Path, training_csv: str) -> None:
    path = write_log(tmp_path / "method" / "log.csv", training_csv)
    before = plt.get_fignums()
    result = plot_experiment_results({"Method": path}, tmp_path / "plots", formats=["png", "pdf", "svg"])
    assert len(result.output_paths) == 12
    assert {path.suffix for path in result.output_paths} == {".png", ".pdf", ".svg"}
    assert all(path.is_file() and path.stat().st_size > 100 for path in result.output_paths)
    assert plt.get_fignums() == before


@pytest.mark.parametrize("module", ["plot_logs", "plot_experiment_results"])
def test_tools_run_on_training_logs(tmp_path: Path, training_csv: str, module: str) -> None:
    source = write_log(tmp_path / "method" / "log.csv", training_csv).parent
    if module == "plot_logs":
        args = [str(source), "--output", str(tmp_path / "training.png")]
        expected = [tmp_path / "training.png"]
    else:
        args = ["--group", "Method", str(source), "--output-dir", str(tmp_path / "plots")]
        expected = [
            tmp_path / "plots" / "ep_rew_ema_grouped.png",
            tmp_path / "plots" / "ep_success_rate_ema_individual_runs.png",
        ]
    completed = subprocess.run(
        [sys.executable, "-m", f"swarmbots.tools.{module}", *args], cwd=tmp_path, capture_output=True, text=True
    )
    assert completed.returncode == 0, completed.stderr
    assert all(path.is_file() for path in expected)


def test_numeric_reading_does_not_require_the_plot_extra(tmp_path: Path, training_csv: str) -> None:
    path = write_log(tmp_path / "log.csv", training_csv)
    code = textwrap.dedent("""
        import importlib.abc
        import sys

        class WithoutMatplotlib(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname: str, path: object = None, target: object = None) -> None:
                if fullname == "matplotlib" or fullname.startswith("matplotlib."):
                    raise ModuleNotFoundError("No module named 'matplotlib'", name="matplotlib")

        sys.meta_path.insert(0, WithoutMatplotlib())
        from swarmbots.plotting import load_log
        from swarmbots.plotting.plot_logs import plot_logs as module_function
        from swarmbots.plotting import plot_logs
        assert plot_logs is module_function
        assert load_log(sys.argv[1]).series["ep_rew_ema"].tolist() == [0, 2, 4]
        try:
            plot_logs(sys.argv[1])
        except ModuleNotFoundError as error:
            assert "swarmbots[plot]" in str(error)
        else:
            raise AssertionError("The plotting dependency should be required")
    """)
    completed = subprocess.run([sys.executable, "-c", code, str(path)], cwd=tmp_path, capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr
