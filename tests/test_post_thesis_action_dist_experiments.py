import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from experiments import post_thesis_action_dist_common as action_dist_common
from experiments import post_thesis_action_dist_plot_common as action_dist_plot_common
from experiments.post_thesis_mjw_find_opening_action_dists import (
    plot_results as find_opening_plot,
)
from experiments.post_thesis_mjw_find_opening_action_dists.scripts import (
    common as find_opening_common,
)
from experiments.post_thesis_mjw_po_wall_medium_action_dists import (
    plot_results as po_wall_plot,
)
from experiments.post_thesis_mjw_po_wall_medium_action_dists.scripts import (
    common as po_wall_common,
)
from experiments.thesis_mjw_find_opening import plot_results as thesis_find_opening_plot
from experiments.thesis_mjw_po_wall_medium import plot_results as thesis_po_wall_plot


def test_shared_runner_uses_thesis_tmasac_architecture_and_names_distribution() -> None:
    entrypoint_path = Path(__file__)
    scenario_kwargs = {"continuous_connector_actions": True}

    with patch.object(action_dist_common, "run_tmasac_experiment") as run:
        action_dist_common.run_post_thesis_action_dist_experiment(
            experiment_run_name="post_thesis_test",
            scenario_name="find_opening",
            scenario_kwargs=scenario_kwargs,
            continuous_action_dist="rqs_6",
            entrypoint_path=entrypoint_path,
        )

    run.assert_called_once_with(
        experiment_run_name="post_thesis_test",
        scenario_name="find_opening",
        scenario_kwargs=scenario_kwargs,
        variant="tmasac_baseline",
        entrypoint_path=entrypoint_path,
        continuous_action_dist="rqs_6",
        use_nop=True,
        variant_name="tmasac_baseline_rqs_6",
    )


@pytest.mark.parametrize("suite_common", (po_wall_common, find_opening_common))
def test_each_suite_has_one_launcher_per_action_distribution(
        monkeypatch: pytest.MonkeyPatch,
        suite_common: ModuleType,
) -> None:
    scripts_dir = Path(suite_common.__file__).parent
    launchers = sorted(scripts_dir.glob("run_tmasac_*.py"))
    invocations: list[dict[str, object]] = []

    def fake_run_experiment(**kwargs: object) -> None:
        invocations.append(kwargs)

    launcher_common = ModuleType("common")
    launcher_common.run_experiment = fake_run_experiment
    monkeypatch.setitem(sys.modules, "common", launcher_common)
    for launcher in launchers:
        runpy.run_path(str(launcher), run_name="__main__")

    assert {launcher.name for launcher in launchers} == {
        f"run_tmasac_{variant}.py"
        for variant in action_dist_common.ACTION_DIST_VARIANTS
    }
    assert {
        invocation["continuous_action_dist"]
        for invocation in invocations
    } == set(action_dist_common.ACTION_DIST_VARIANTS)


def test_suites_keep_thesis_scenarios_and_post_thesis_run_names() -> None:
    assert po_wall_common.EXPERIMENT_RUN_NAME == (
        "post_thesis_mjw_po_wall_medium_action_dists"
    )
    assert find_opening_common.EXPERIMENT_RUN_NAME == (
        "post_thesis_mjw_find_opening_action_dists"
    )
    assert po_wall_common.SCENARIO_KWARGS["continuous_connector_actions"] is True
    assert find_opening_common.SCENARIO_KWARGS == {
        "continuous_connector_actions": True,
    }


def test_plots_merge_thesis_baseline_runs_with_their_extra_runs() -> None:
    assert po_wall_plot.BASELINE_SOURCES == (
        thesis_po_wall_plot.EXPERIMENT_RUN_DIR / "tmasac_baseline",
        *thesis_po_wall_plot.EXTRA_GROUP_SOURCES["tmasac_baseline"],
    )
    assert find_opening_plot.BASELINE_SOURCES == (
        thesis_find_opening_plot.EXPERIMENT_RUN_DIR / "tmasac_baseline",
        *thesis_find_opening_plot.EXTRA_GROUP_SOURCES["tmasac_baseline"],
    )


def test_plot_order_contains_only_baseline_and_new_action_distributions() -> None:
    assert action_dist_plot_common.GROUP_ORDER == (
        "tmasac_baseline",
        "tmasac_baseline_bernstein_6",
        "tmasac_baseline_bernstein_8",
        "tmasac_baseline_rqs_4",
        "tmasac_baseline_rqs_6",
    )
