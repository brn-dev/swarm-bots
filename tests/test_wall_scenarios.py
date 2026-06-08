from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from swarmbots.mj_env.float_or_dist_params import UniformDistParams
from swarmbots.mj_env.scenarios.scenario_presets import default_wall as default_mj_wall
from swarmbots.mjw_env.scenarios.mjw_wall_runtime import WallMJWScenarioRuntime
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_wall as default_mjw_wall
from swarmbots.scenario_presets.scenario_presets_kwargs import PO_WALL_MEDIUM_SCENARIO_KWARGS


def test_po_wall_medium_samples_wall_y_into_hidden_global_obs_only_in_mj() -> None:
    scenario = default_mj_wall(
        seed=123,
        reset_settle_time=0.0,
        **PO_WALL_MEDIUM_SCENARIO_KWARGS,
    )

    sampled_wall_positions: list[float] = []
    for _ in range(8):
        state, connections = scenario.reset_scenario(scenario.dummy_model, scenario.dummy_data, settle=False)
        obs = scenario.get_obs(scenario.dummy_model, scenario.dummy_data, state, connections)

        assert obs["global_obs"].shape == (0,)
        assert obs["hidden_global_vars"].shape == (1,)
        assert np.isclose(obs["hidden_global_vars"][0], state["wall_y"])
        sampled_wall_positions.append(float(obs["hidden_global_vars"][0]))

    assert all(0.75 <= wall_y <= 1.25 for wall_y in sampled_wall_positions)
    assert len({round(wall_y, 6) for wall_y in sampled_wall_positions}) > 1


def test_po_wall_medium_mjw_space_keeps_wall_position_hidden() -> None:
    scenario = default_mjw_wall(**PO_WALL_MEDIUM_SCENARIO_KWARGS)
    obs_space = scenario.get_single_observation_space()

    assert obs_space["global_obs"].shape == (0,)
    assert obs_space["hidden_global_vars"].shape == (1,)
    assert scenario.get_settings()["first_wall_distance"] == UniformDistParams(0.75, 1.25)


def test_po_wall_medium_mjw_reset_sampling_puts_wall_y_in_hidden_global_vars() -> None:
    runtime = object.__new__(WallMJWScenarioRuntime)
    runtime.bindings = SimpleNamespace(device=torch.device("cpu"))
    runtime.hidden_global_vars = torch.zeros((1, 1), dtype=torch.float32)
    runtime.wall_pass_absolute_thresholds = torch.zeros((1, 2), dtype=torch.float32)
    runtime._num_wall_thresholds = 2
    runtime._threshold_values = torch.tensor([0.2, 0.5], dtype=torch.float32)
    runtime.scenario = SimpleNamespace(
        first_wall_distance=UniformDistParams(0.75, 1.25),
    )

    hidden_global_vars, wall_pass_thresholds = runtime._sample_wall_configuration_values(
        n_reset=16,
        rng=torch.Generator(device="cpu").manual_seed(123),
    )

    wall_y = hidden_global_vars[:, 0]
    assert hidden_global_vars.shape == (16, 1)
    assert torch.all((wall_y >= 0.75) & (wall_y <= 1.25))
    assert torch.unique(wall_y).numel() > 1
    assert torch.allclose(wall_pass_thresholds, wall_y.unsqueeze(1) + runtime._threshold_values.unsqueeze(0))


