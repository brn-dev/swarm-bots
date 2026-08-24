from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

import experiments.thesis_mjw_po_wall_disconnected_finetune.evaluate_tmasac_50m as evaluation


def _metric_summary(mean: float, std: float = 0.0) -> dict[str, float]:
    return {"mean": mean, "std": std, "min": mean - std, "max": mean + std}


def _connection_usage(
    *,
    episode_count: int,
    never_connected_count: int,
    no_connection_count: int,
) -> dict[str, int | float | None]:
    return {
        "episode_count": episode_count,
        "episodes_with_never_connected_unit_count": never_connected_count,
        "episodes_with_never_connected_unit_rate_percent": (
            100.0 * never_connected_count / episode_count if episode_count else None
        ),
        "episodes_without_any_successful_connection_count": no_connection_count,
        "episodes_without_any_successful_connection_rate_percent": (
            100.0 * no_connection_count / episode_count if episode_count else None
        ),
    }


def _evaluation_result(
    *,
    run_id: str,
    episode_count: int,
    success_count: int,
    episode_return_mean: float,
    connections_mean: float,
    successful_connections_mean: float,
    unsuccessful_connections_mean: float,
    successful_never_connected_count: int,
    successful_no_connection_count: int,
) -> dict[str, object]:
    unsuccessful_count = episode_count - success_count
    return {
        "run_id": run_id,
        "unit_count": 4,
        "summary": {
            "episode_count": episode_count,
            "success_count": success_count,
            "success_rate_percent": 100.0 * success_count / episode_count,
            "episode_return": _metric_summary(episode_return_mean, 1.0),
            "episode_length": _metric_summary(100.0, 10.0),
            "progress_reward": _metric_summary(2.0, 0.5),
            "guidance_reward": _metric_summary(1.0, 0.25),
            "successful_connections_per_unit": _metric_summary(connections_mean, 1.0),
            "successful_connections_per_unit_successful_episodes": _metric_summary(
                successful_connections_mean,
                0.5,
            ),
            "successful_connections_per_unit_unsuccessful_episodes": _metric_summary(
                unsuccessful_connections_mean,
                0.25,
            ),
            "connection_usage_by_outcome": {
                "successful_episodes": _connection_usage(
                    episode_count=success_count,
                    never_connected_count=successful_never_connected_count,
                    no_connection_count=successful_no_connection_count,
                ),
                "unsuccessful_episodes": _connection_usage(
                    episode_count=unsuccessful_count,
                    never_connected_count=unsuccessful_count,
                    no_connection_count=0,
                ),
            },
        },
    }


def _case_payload(results: list[dict[str, object]]) -> dict[str, object]:
    return {
        "config": {
            "unit_counts": [4],
            "pool_size": 50,
            "episodes": 10,
            "pool_seed_base": 3_000_000,
            "rollout_seed": 2_000_000,
            "deterministic": True,
            "episode_length": 512,
            "unconnected_prob": 1.0,
        },
        "created_at": "2026-08-24T00:00:00+00:00",
        "completed_at": "2026-08-24T01:00:00+00:00",
        "results": results,
    }


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


def test_summary_only_rebuilds_existing_results_without_checkpoint_discovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output_root = Path("existing-results")
    monkeypatch.setattr(
        evaluation,
        "write_global_summary",
        lambda received_output_root: received_output_root / evaluation.GLOBAL_SUMMARY_FILENAME,
    )
    monkeypatch.setattr(
        evaluation,
        "discover_evaluation_checkpoints",
        lambda _group_dirs: pytest.fail("summary-only must not discover checkpoints"),
    )

    assert evaluation.main(["--summary-only", "--output-root", str(output_root)]) == 0


def test_global_summary_pools_runs_and_builds_thesis_comparisons() -> None:
    connector_payload = _case_payload(
        [
            _evaluation_result(
                run_id="connector-run-1",
                episode_count=10,
                success_count=8,
                episode_return_mean=4.0,
                connections_mean=2.0,
                successful_connections_mean=2.5,
                unsuccessful_connections_mean=0.5,
                successful_never_connected_count=2,
                successful_no_connection_count=1,
            ),
            _evaluation_result(
                run_id="connector-run-2",
                episode_count=10,
                success_count=6,
                episode_return_mean=2.0,
                connections_mean=4.0,
                successful_connections_mean=4.0,
                unsuccessful_connections_mean=1.0,
                successful_never_connected_count=1,
                successful_no_connection_count=0,
            ),
        ]
    )
    no_connector_payload = _case_payload(
        [
            _evaluation_result(
                run_id="no-connector-run",
                episode_count=20,
                success_count=4,
                episode_return_mean=1.0,
                connections_mean=0.0,
                successful_connections_mean=0.0,
                unsuccessful_connections_mean=0.0,
                successful_never_connected_count=4,
                successful_no_connection_count=4,
            )
        ]
    )

    summary = evaluation.build_global_summary(
        {
            "medium_connectors": connector_payload,
            "medium_no_connectors": no_connector_payload,
        },
        source_paths={
            "medium_connectors": Path("medium_connectors.json"),
            "medium_no_connectors": Path("medium_no_connectors.json"),
        },
    )

    aggregate = summary["cases"]["medium_connectors"]["by_unit_count"]["4"]
    assert aggregate["run_count"] == 2
    assert aggregate["episode_count"] == 20
    assert aggregate["success_rate_percent"] == 70.0
    assert aggregate["episode_return"] == pytest.approx(
        {
            "observation_count": 20,
            "mean": 3.0,
            "std": 2.0**0.5,
            "min": 1.0,
            "max": 5.0,
        }
    )
    assert aggregate["successful_connections_per_unit"] == pytest.approx(
        {
            "observation_count": 80,
            "mean": 3.0,
            "std": 2.0**0.5,
            "min": 1.0,
            "max": 5.0,
        }
    )
    assert aggregate["connection_usage_by_outcome"]["successful_episodes"] == pytest.approx(
        {
            "episode_count": 14,
            "observed_episode_count": 14,
            "observation_coverage_percent": 100.0,
            "episodes_with_never_connected_unit_count": 3,
            "episodes_with_never_connected_unit_rate_percent": 100.0 * 3 / 14,
            "episodes_without_any_successful_connection_count": 1,
            "episodes_without_any_successful_connection_rate_percent": 100.0 / 14,
        }
    )

    comparison = summary["comparisons"]["connector_enabled_vs_disabled"]["medium"]
    assert comparison["comparable_evaluation_design"] is True
    difference = comparison["by_unit_count"]["4"][
        "connector_enabled_minus_connector_disabled"
    ]
    assert difference["success_rate_percent"] == 50.0
    assert difference["episode_return_mean"] == 2.0
    assert summary["missing_cases"] == ["hard_connectors", "hard_no_connectors"]
