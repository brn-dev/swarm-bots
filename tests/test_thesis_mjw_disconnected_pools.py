from pathlib import Path
from typing import Any

import experiments.evaluate_thesis_mjw_disconnected_pools as evaluate_disconnected
import experiments.evaluate_thesis_mjw_disconnected_pools_no_connections as evaluate_no_connections
import experiments.record_thesis_mjw_disconnected_pools as record_disconnected


def test_evaluation_entry_point_uses_fully_disconnected_pool_config(
    monkeypatch: Any,
) -> None:
    invocation: dict[str, object] = {}

    def fake_main(argv: object, **kwargs: object) -> int:
        invocation.update(argv=argv, **kwargs)
        return 17

    monkeypatch.setattr(evaluate_disconnected, "_main", fake_main)

    result = evaluate_disconnected.main(["--dry-run"])

    assert result == 17
    assert invocation["argv"] == ["--dry-run"]
    assert invocation["unconnected_prob"] == 1.0
    assert invocation["default_output_path"] == evaluate_disconnected.DEFAULT_OUTPUT_PATH
    assert "disconnected" in str(invocation["morphology_description"])


def test_recording_entry_point_uses_fully_disconnected_pool_config(
    monkeypatch: Any,
) -> None:
    invocation: dict[str, object] = {}

    def fake_main(argv: object, **kwargs: object) -> int:
        invocation.update(argv=argv, **kwargs)
        return 23

    monkeypatch.setattr(record_disconnected, "_main", fake_main)

    result = record_disconnected.main(["--dry-run"])

    assert result == 23
    assert invocation["argv"] == ["--dry-run"]
    assert invocation["unconnected_prob"] == 1.0
    assert invocation["default_output_root"] == record_disconnected.DEFAULT_OUTPUT_ROOT
    assert "disconnected" in str(invocation["morphology_description"])


def test_disconnected_outputs_do_not_overlap_preconnected_outputs() -> None:
    assert evaluate_disconnected.DEFAULT_OUTPUT_PATH == Path(
        evaluate_disconnected.REPO_ROOT,
        "experiments",
        "thesis_mjw_disconnected_pools",
        "results",
        "evaluation.json",
    )
    assert record_disconnected.DEFAULT_OUTPUT_ROOT == Path(
        record_disconnected.REPO_ROOT,
        "runs",
        "thesis_mjw_disconnected_pools",
        "recordings",
    )
    assert evaluate_no_connections.DEFAULT_OUTPUT_PATH != evaluate_disconnected.DEFAULT_OUTPUT_PATH


def test_no_connection_ablation_forces_disabled_connector_actions(monkeypatch: Any) -> None:
    invocation: dict[str, object] = {}

    def fake_main(argv: object, **kwargs: object) -> int:
        invocation.update(argv=argv, **kwargs)
        return 31

    monkeypatch.setattr(evaluate_no_connections, "_main", fake_main)

    result = evaluate_no_connections.main(["--dry-run"])

    assert result == 31
    assert invocation["argv"] == [
        "--dry-run",
        "--target",
        "po_wall_tmasac",
        "--disable-connector-actions",
    ]
    assert invocation["unconnected_prob"] == 1.0
    assert invocation["default_output_path"] == evaluate_no_connections.DEFAULT_OUTPUT_PATH
