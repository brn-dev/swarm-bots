from pathlib import Path
from typing import Any, cast

import experiments.thesis_mjw_po_wall_disconnected_finetune.scripts.common as common


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

    training_kwargs = cast(dict[str, object], invocation["scenario_kwargs"])
    evaluation_kwargs = cast(dict[str, object], invocation["evaluation_scenario_kwargs"])
    training_pool = cast(Any, training_kwargs["unit_start_locations"])
    evaluation_pool = cast(Any, evaluation_kwargs["unit_start_locations"])
    assert training_pool.unconnected_prob == 1.0
    assert evaluation_pool.unconnected_prob == 1.0
    assert training_pool.num_unit_probs == {4: 0.5, 5: 0.5}
    assert evaluation_pool.num_unit_probs == {4: 0.5, 5: 0.5}
    assert set(training_pool.pool_seeds).isdisjoint(evaluation_pool.pool_seeds)
