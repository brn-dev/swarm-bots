from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import call, patch

import numpy as np
import torch

from swarmbots.mj_env.float_or_dist_params import UniformDistParams
from swarmbots.mj_env.scenarios.scenario_presets import default_wall as default_mj_wall
from swarmbots.mjw_env.scenarios.mjw_wall_runtime import WallMJWScenarioRuntime
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_wall as default_mjw_wall
from swarmbots.scenario_presets.scenario_presets_kwargs import (
    PO_WALL_EASY_SCENARIO_KWARGS,
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    po_wall_scenario_kwargs_with_difficulty,
)
from swarmbots.utils.mujoco_render_geoms import add_wall_y_reference_line_geoms


def test_po_wall_difficulties_set_wall_height() -> None:
    assert PO_WALL_EASY_SCENARIO_KWARGS["wall_height"] == 0.25
    assert PO_WALL_MEDIUM_SCENARIO_KWARGS["wall_height"] == 0.3
    assert po_wall_scenario_kwargs_with_difficulty("medium")["wall_height"] == 0.3
    assert po_wall_scenario_kwargs_with_difficulty()["wall_height"] == 0.25


def test_po_wall_easy_samples_wall_y_into_hidden_global_obs_only_in_mj() -> None:
    scenario = default_mj_wall(
        seed=123,
        reset_settle_time=0.0,
        **PO_WALL_EASY_SCENARIO_KWARGS,
    )

    sampled_wall_positions: list[float] = []
    for _ in range(8):
        state, connections = scenario.reset_scenario(scenario.dummy_model, scenario.dummy_data, settle=False)
        obs = scenario.get_obs(scenario.dummy_model, scenario.dummy_data, state, connections)

        assert obs["global_obs"].shape == (0,)
        assert obs["hidden_global_vars"].shape == (1,)
        assert np.isclose(obs["hidden_global_vars"][0], state["wall_y"])
        sampled_wall_positions.append(float(obs["hidden_global_vars"][0]))

    first_wall_distance = scenario.first_wall_distance
    assert isinstance(first_wall_distance, UniformDistParams)
    assert all(first_wall_distance.low <= wall_y <= first_wall_distance.high for wall_y in sampled_wall_positions)
    assert len({round(wall_y, 6) for wall_y in sampled_wall_positions}) > 1


def test_po_wall_easy_mjw_space_keeps_wall_position_hidden() -> None:
    scenario = default_mjw_wall(**PO_WALL_EASY_SCENARIO_KWARGS)
    obs_space = scenario.get_single_observation_space()

    assert obs_space["global_obs"].shape == (0,)
    assert obs_space["hidden_global_vars"].shape == (1,)
    assert isinstance(scenario.get_settings()["first_wall_distance"], UniformDistParams)


def test_po_wall_easy_mjw_reset_sampling_puts_wall_y_in_hidden_global_vars() -> None:
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


def test_mjw_wall_configuration_moves_wall_mocap_for_selected_worlds() -> None:
    runtime = object.__new__(WallMJWScenarioRuntime)
    runtime.bindings = SimpleNamespace(mocap_pos=torch.zeros((3, 1, 3), dtype=torch.float32))
    runtime._wall_mocap_id = 0

    runtime._apply_wall_configuration(
        world_idx=torch.tensor([0, 2]),
        hidden_global_vars=torch.tensor([[0.5], [1.5]], dtype=torch.float32),
    )

    assert torch.allclose(runtime.bindings.mocap_pos[0, 0], torch.tensor([0.0, 0.5, 0.0]))
    assert torch.allclose(runtime.bindings.mocap_pos[1, 0], torch.zeros(3))
    assert torch.allclose(runtime.bindings.mocap_pos[2, 0], torch.tensor([0.0, 1.5, 0.0]))


def test_wall_scenarios_add_wall_y_reference_lines() -> None:
    mj_scenario = default_mj_wall(
        seed=123,
        reset_settle_time=0.0,
        **PO_WALL_EASY_SCENARIO_KWARGS,
    )
    mjw_scenario = default_mjw_wall(**PO_WALL_EASY_SCENARIO_KWARGS)
    scene = object()

    with (
        patch("swarmbots.mj_env.scenarios.wall_scenario.add_wall_y_reference_line_geoms") as add_mj_lines,
        patch("swarmbots.mjw_env.scenarios.mjw_wall_scenario.add_wall_y_reference_line_geoms") as add_mjw_lines,
    ):
        mj_scenario.add_render_geoms(scene)
        mjw_scenario.add_render_geoms(scene)

    add_mj_lines.assert_called_once_with(scene)
    add_mjw_lines.assert_called_once_with(scene)


def test_wall_y_reference_lines_show_bounds() -> None:
    scene = object()

    with patch("swarmbots.utils.mujoco_render_geoms.add_y_reference_line_geom") as add_line:
        add_wall_y_reference_line_geoms(scene)

    bounds_rgba = (0.1, 0.8, 1.0, 1.0)
    assert add_line.call_args_list == [
        call(scene, y=0.5, rgba=bounds_rgba),
        call(scene, y=1.5, rgba=bounds_rgba),
    ]


