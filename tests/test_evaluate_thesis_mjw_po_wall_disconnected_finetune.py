from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import experiments.thesis_mjw_po_wall_disconnected_finetune.evaluate_tmasac_50m as evaluation


def test_evaluator_dispatches_selected_medium_and_hard_connector_variants(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    invocations: list[tuple[list[str], dict[str, Any]]] = []
    output_root = Path("custom-output")

    def fake_discover(group_dirs: tuple[Path, ...]) -> list[Path]:
        group_dir = group_dirs[0]
        return [group_dir / "run-1" / "models" / "model_50000000_steps_final.pt"]

    def fake_evaluate_main(argv: list[str], **kwargs: Any) -> int:
        invocations.append((argv, kwargs))
        return 0

    monkeypatch.setattr(evaluation, "discover_evaluation_checkpoints", fake_discover)
    monkeypatch.setattr(evaluation, "_evaluate_main", fake_evaluate_main)

    result = evaluation.main(
        [
            "--case",
            "medium_connectors",
            "--case",
            "hard_no_connectors",
            "--episodes",
            "16",
            "--max-parallel-envs",
            "16",
            "--output-root",
            str(output_root),
            "--dry-run",
        ]
    )

    assert result == 0
    assert len(invocations) == 2

    medium_argv, medium_kwargs = invocations[0]
    assert medium_kwargs["unconnected_prob"] == 1.0
    assert medium_kwargs["disable_policy_connector_actions"] is False
    assert medium_kwargs["scenario_kwargs_overrides"] is None
    assert "--dry-run" in medium_argv
    assert str(output_root / "medium_connectors.json") in medium_argv

    hard_argv, hard_kwargs = invocations[1]
    assert hard_kwargs["disable_policy_connector_actions"] is True
    assert hard_kwargs["scenario_kwargs_overrides"] == {"wall_height": 0.4}
    assert str(output_root / "hard_no_connectors.json") in hard_argv
    assert hard_argv[hard_argv.index("--unit-counts") + 1 :][:2] == ["4", "5"]


def test_evaluator_requires_a_discovered_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        evaluation,
        "discover_evaluation_checkpoints",
        lambda _group_dirs: [],
    )

    with pytest.raises(FileNotFoundError, match="No final or best checkpoints"):
        evaluation.main(["--case", "medium_connectors", "--dry-run"])
