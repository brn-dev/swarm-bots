import runpy
import sys
from pathlib import Path
from types import ModuleType
from unittest.mock import patch

import pytest

from experiments import post_thesis_action_dist_common as action_dist_common
from experiments import post_thesis_action_dist_plot_common as action_dist_plot_common
from experiments import thesis_experiment_common
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


def test_shared_runner_supports_slstm_tmasac() -> None:
    entrypoint_path = Path(__file__)
    scenario_kwargs = {"continuous_connector_actions": True}

    with patch.object(action_dist_common, "run_tmasac_experiment") as run:
        action_dist_common.run_post_thesis_action_dist_experiment(
            experiment_run_name="post_thesis_test",
            scenario_name="find_opening",
            scenario_kwargs=scenario_kwargs,
            continuous_action_dist="bernstein_8",
            entrypoint_path=entrypoint_path,
            tmasac_variant="slstm_two_small_actor_state_critic",
        )

    run.assert_called_once_with(
        experiment_run_name="post_thesis_test",
        scenario_name="find_opening",
        scenario_kwargs=scenario_kwargs,
        variant="slstm_two_small_actor_state_critic",
        entrypoint_path=entrypoint_path,
        continuous_action_dist="bernstein_8",
        use_nop=True,
        variant_name="slstm_two_small_actor_state_critic_bernstein_8",
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


def test_find_opening_has_one_slstm_tmasac_launcher_per_action_distribution(
        monkeypatch: pytest.MonkeyPatch,
) -> None:
    scripts_dir = Path(find_opening_common.__file__).parent
    launchers = sorted(scripts_dir.glob("run_slstm_tmasac_*.py"))
    invocations: list[dict[str, object]] = []

    def fake_run_slstm_tmasac_experiment(**kwargs: object) -> None:
        invocations.append(kwargs)

    launcher_common = ModuleType("common")
    launcher_common.run_slstm_tmasac_experiment = fake_run_slstm_tmasac_experiment
    monkeypatch.setitem(sys.modules, "common", launcher_common)
    for launcher in launchers:
        runpy.run_path(str(launcher), run_name="__main__")

    assert {launcher.name for launcher in launchers} == {
        f"run_slstm_tmasac_{variant}.py"
        for variant in action_dist_common.ACTION_DIST_VARIANTS
    }
    assert {
        invocation["continuous_action_dist"]
        for invocation in invocations
    } == set(action_dist_common.ACTION_DIST_VARIANTS)


def test_find_opening_common_forwards_slstm_tmasac_variant() -> None:
    entrypoint_path = Path(__file__)

    with patch.object(
            find_opening_common,
            "run_post_thesis_action_dist_experiment",
    ) as run:
        find_opening_common.run_slstm_tmasac_experiment(
            continuous_action_dist="rqs_4",
            entrypoint_path=entrypoint_path,
        )

    run.assert_called_once_with(
        experiment_run_name=find_opening_common.EXPERIMENT_RUN_NAME,
        scenario_name="find_opening",
        scenario_kwargs=find_opening_common.SCENARIO_KWARGS,
        continuous_action_dist="rqs_4",
        entrypoint_path=entrypoint_path,
        tmasac_variant="slstm_two_small_actor_state_critic",
    )


@pytest.mark.parametrize("policy_variant", ("mat_ind", "mat_qcx"))
@pytest.mark.parametrize("action_dist", action_dist_common.ACTION_DIST_VARIANTS)
def test_po_wall_mat_launchers_keep_thesis_ppo_setup_with_selected_distribution(
        monkeypatch: pytest.MonkeyPatch,
        policy_variant: str,
        action_dist: str,
) -> None:
    launcher = Path(po_wall_common.__file__).parent / f"run_{policy_variant}_{action_dist}.py"
    monkeypatch.setitem(sys.modules, "common", po_wall_common)

    with patch.object(thesis_experiment_common, "run_mjw_experiment") as run:
        runpy.run_path(str(launcher), run_name="__main__")

    run.assert_called_once()
    config = run.call_args.kwargs
    assert config["policy_variant"] == policy_variant
    assert config["continuous_action_dist"] == action_dist
    assert config["variant_name"] == f"{policy_variant}_{action_dist}"
    assert config["experiment_run_name"] == po_wall_common.EXPERIMENT_RUN_NAME
    assert config["scenario_name"] == "wall"
    assert config["scenario_kwargs"] == po_wall_common.SCENARIO_KWARGS
    assert config["entrypoint_path"] == launcher.resolve()
    assert config["total_timesteps"] == 100_000_000
    assert (config["num_envs"], config["rollout_steps_per_env"]) == (1024, 4)
    assert config["use_nop"] is True
    assert config["mat_use_agent_attention"] is True
    assert config["use_transition_obs"] is False


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
    assert po_wall_plot.BASELINE_SOURCES == {
        algorithm_variant: (
            thesis_po_wall_plot.EXPERIMENT_RUN_DIR / algorithm_variant,
            *thesis_po_wall_plot.EXTRA_GROUP_SOURCES[algorithm_variant],
        )
        for algorithm_variant in ("tmasac_baseline", "mat_ind", "mat_qcx")
    }
    assert find_opening_plot.BASELINE_SOURCES == {
        "tmasac_baseline": (
            thesis_find_opening_plot.EXPERIMENT_RUN_DIR / "tmasac_baseline",
            *thesis_find_opening_plot.EXTRA_GROUP_SOURCES["tmasac_baseline"],
        ),
        "slstm_two_small_actor_state_critic": (
            thesis_find_opening_plot.EXPERIMENT_RUN_DIR
            / "slstm_two_small_actor_state_critic",
            *thesis_find_opening_plot.EXTRA_GROUP_SOURCES[
                "slstm_two_small_actor_state_critic"
            ],
        ),
    }


@pytest.mark.parametrize(
    ("suite_plot", "algorithms"),
    (
        (po_wall_plot, ("tmasac_baseline", "mat_ind", "mat_qcx")),
        (find_opening_plot, ("tmasac_baseline", "slstm_two_small_actor_state_critic")),
    ),
)
def test_reporting_includes_all_algorithms_and_action_distributions(
        suite_plot: ModuleType,
        algorithms: tuple[str, ...],
) -> None:
    expected_group_order = tuple(
        group
        for algorithm in algorithms
        for group in (
            algorithm,
            f"{algorithm}_bernstein_6",
            f"{algorithm}_bernstein_8",
            f"{algorithm}_bernstein_12",
            f"{algorithm}_bernstein_18",
            f"{algorithm}_rqs_4",
            f"{algorithm}_rqs_6",
        )
    )

    with patch.object(action_dist_plot_common, "plot_experiment_results") as plot:
        action_dist_plot_common.plot_post_thesis_action_dist_results(
            experiment_run_dir=suite_plot.EXPERIMENT_RUN_DIR,
            output_dir=suite_plot.OUTPUT_DIR,
            baseline_sources=suite_plot.BASELINE_SOURCES,
            scenario_title="Test scenario",
            algorithm_variants=suite_plot.ALGORITHM_VARIANTS,
        )

    assert plot.call_args.kwargs["group_filter"] == expected_group_order
    assert plot.call_args.kwargs["extra_group_sources"] == suite_plot.BASELINE_SOURCES

    with (
        patch.object(action_dist_plot_common, "load_experiment_groups") as load,
        patch.object(
            action_dist_plot_common,
            "summarize_thesis_groups",
            return_value=Path("summary.json"),
        ),
    ):
        action_dist_plot_common.summarize_post_thesis_action_dist_results(
            experiment_run_dir=suite_plot.EXPERIMENT_RUN_DIR,
            baseline_sources=suite_plot.BASELINE_SOURCES,
            output_path=Path("summary.json"),
            tail_points=10,
            algorithm_variants=suite_plot.ALGORITHM_VARIANTS,
        )

    assert load.call_args.kwargs["group_filter"] == expected_group_order
    assert load.call_args.kwargs["extra_group_sources"] == suite_plot.BASELINE_SOURCES


def test_plot_order_contains_only_baseline_and_new_action_distributions() -> None:
    assert action_dist_plot_common.GROUP_ORDER == (
        "tmasac_baseline",
        "tmasac_baseline_bernstein_6",
        "tmasac_baseline_bernstein_8",
        "tmasac_baseline_bernstein_12",
        "tmasac_baseline_bernstein_18",
        "tmasac_baseline_rqs_4",
        "tmasac_baseline_rqs_6",
    )
