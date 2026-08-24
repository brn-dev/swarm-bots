from pathlib import Path
from typing import Any, cast

import pytest

import experiments.thesis_mjw_po_wall_disconnected_finetune.plot_results as plot_results
import experiments.thesis_mjw_po_wall_disconnected_finetune.scripts.common as common
import experiments.thesis_mjw_po_wall_disconnected_finetune.scripts.recording_common as recording_common


def test_disconnected_finetune_wires_tmasac_continuation(monkeypatch: Any) -> None:
    invocation: dict[str, object] = {}
    checkpoint_path = Path(__file__)

    def fake_run_tmasac_experiment(**kwargs: object) -> None:
        invocation.update(kwargs)

    monkeypatch.setattr(common, "run_tmasac_experiment", fake_run_tmasac_experiment)

    common.run_experiment(checkpoint_path=checkpoint_path, entrypoint_path=Path(__file__))

    assert invocation["variant"] == "tmasac_baseline"
    assert invocation["scenario_name"] == "wall"
    assert invocation["load_path"] == checkpoint_path.resolve()
    assert invocation["additional_timesteps"] == 20_000_000
    assert invocation["evaluation_milestones"] == (50.0, 100.0)
    assert invocation["evaluation_recording_episodes"] == 0
    assert invocation["live_recording_schedule"] == {}

    training_kwargs = cast(dict[str, object], invocation["scenario_kwargs"])
    evaluation_kwargs = cast(dict[str, object], invocation["evaluation_scenario_kwargs"])
    training_pool = cast(Any, training_kwargs["unit_start_locations"])
    evaluation_pool = cast(Any, evaluation_kwargs["unit_start_locations"])
    assert training_pool.unconnected_prob == 1.0
    assert evaluation_pool.unconnected_prob == 1.0
    assert training_pool.num_unit_probs == {4: 0.5, 5: 0.5}
    assert evaluation_pool.num_unit_probs == {4: 0.5, 5: 0.5}
    assert set(training_pool.pool_seeds).isdisjoint(evaluation_pool.pool_seeds)


def test_50m_tmasac_launcher_uses_separate_run_group(monkeypatch: Any) -> None:
    invocation: dict[str, object] = {}

    def fake_run_tmasac_experiment(**kwargs: object) -> None:
        invocation.update(kwargs)

    monkeypatch.setattr(common, "run_tmasac_experiment", fake_run_tmasac_experiment)

    common.run_tmasac_50m(checkpoint_path=Path(__file__), entrypoint_path=Path(__file__))

    assert invocation["experiment_run_name"] == common.LONG_EXPERIMENT_RUN_NAME
    assert invocation["variant"] == "tmasac_baseline"
    assert invocation["variant_name"] == "tmasac_baseline"
    assert invocation["additional_timesteps"] == 50_000_000
    assert invocation["evaluation_milestones"] == (50.0, 100.0)
    assert invocation["evaluation_recording_episodes"] == 0
    assert invocation["live_recording_schedule"] == {}


def test_50m_no_connectors_launcher_removes_connector_policy_actions(
    monkeypatch: Any,
) -> None:
    invocation: dict[str, object] = {}

    def fake_run_tmasac_experiment(**kwargs: object) -> None:
        invocation.update(kwargs)

    monkeypatch.setattr(common, "run_tmasac_experiment", fake_run_tmasac_experiment)

    common.run_tmasac_no_connectors_50m(
        checkpoint_path=Path(__file__),
        entrypoint_path=Path(__file__),
    )

    assert invocation["experiment_run_name"] == common.LONG_EXPERIMENT_RUN_NAME
    assert invocation["variant_name"] == "tmasac_no_connectors"
    assert invocation["additional_timesteps"] == 50_000_000
    assert invocation["disable_connector_actions"] is True
    assert invocation["migrate_removed_connector_actions"] is True


def test_50m_mat_qcx_launcher_uses_ppo_continuation(monkeypatch: Any) -> None:
    invocation: dict[str, object] = {}

    def fake_run_thesis_ppo_experiment(**kwargs: object) -> None:
        invocation.update(kwargs)

    monkeypatch.setattr(
        common,
        "run_thesis_ppo_experiment",
        fake_run_thesis_ppo_experiment,
    )

    common.run_mat_qcx_50m(checkpoint_path=Path(__file__), entrypoint_path=Path(__file__))

    assert invocation["experiment_run_name"] == common.LONG_EXPERIMENT_RUN_NAME
    assert invocation["variant"] == "mat_qcx"
    assert invocation["variant_name"] == "mat_qcx"
    assert invocation["additional_timesteps"] == 50_000_000
    assert invocation["evaluation_milestones"] == (50.0, 100.0)
    assert invocation["evaluation_recording_episodes"] == 0
    assert invocation["live_recording_schedule"] == {}


def test_50m_recording_uses_disconnected_four_and_five_unit_pools(
    monkeypatch: Any,
) -> None:
    invocation: dict[str, object] = {}
    checkpoints = [Path("first-final.pt"), Path("second-best.pt")]

    def fake_record_main(argv: object, **kwargs: object) -> int:
        invocation.update(argv=argv, **kwargs)
        return 41

    monkeypatch.setattr(recording_common, "_record_main", fake_record_main)
    monkeypatch.setattr(
        recording_common,
        "discover_evaluation_checkpoints",
        lambda group_dirs: checkpoints,
    )

    result = recording_common.run_recording(
        ["--episodes", "2"],
        disable_policy_connector_actions=True,
        variant_name="tmasac_no_connectors",
    )

    assert result == 41
    assert invocation["argv"] == [
        "--target",
        "po_wall_tmasac",
        "--unit-counts",
        "4",
        "5",
        "--pool-seed-base",
        "3000000",
        "--po-wall-checkpoint",
        "first-final.pt",
        "--po-wall-checkpoint",
        "second-best.pt",
        "--episodes",
        "2",
    ]
    assert invocation["unconnected_prob"] == 1.0
    assert invocation["disable_policy_connector_actions"] is True
    assert invocation["default_output_root"] == (
        recording_common.RECORDING_ROOT / "tmasac_no_connectors"
    )


def test_50m_recording_fails_when_its_run_group_has_no_models(
    monkeypatch: Any,
) -> None:
    monkeypatch.setattr(
        recording_common,
        "discover_evaluation_checkpoints",
        lambda group_dirs: [],
    )

    with pytest.raises(FileNotFoundError) as exc_info:
        recording_common.run_recording(
            [],
            disable_policy_connector_actions=False,
            variant_name="tmasac_baseline",
        )

    assert str(recording_common.EXPERIMENT_RUN_DIR / "tmasac_baseline") in str(
        exc_info.value
    )


def test_50m_plot_compares_all_launcher_groups(monkeypatch: Any) -> None:
    invocation: dict[str, object] = {}

    class FakeResult:
        output_paths: list[Path] = []

    def fake_plot_experiment_results(*args: object, **kwargs: object) -> FakeResult:
        invocation.update(args=args, **kwargs)
        return FakeResult()

    monkeypatch.setattr(
        plot_results,
        "plot_experiment_results",
        fake_plot_experiment_results,
    )

    assert plot_results.main() == 0
    assert invocation["group_order"] == (
        "tmasac_baseline",
        "tmasac_no_connectors",
        "mat_qcx",
    )
    assert invocation["run_length_limit"] == 150_000_000
    assert invocation["cut_at_limit"] is True
