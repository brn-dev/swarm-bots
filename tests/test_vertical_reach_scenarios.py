from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.mj_env.scenarios.scenario_presets import default_vertical_reach as default_mj_vertical_reach
from swarmbots.mj_env.scenarios.vertical_reach_scenario import _compute_vertical_reach_progress_baselines_np
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_vertical_reach as default_mjw_vertical_reach
from swarmbots.mjw_env.scenarios.mjw_vertical_reach_runtime import (
    VerticalReachMJWScenarioRuntime,
    _compute_reach_column_progress_baseline_torch,
    _compute_vertical_reach_goal_success_terminations,
    _compute_vertical_reach_progress_baselines_torch,
    _compute_vertical_reach_reward_kernel,
)
from swarmbots.scenario_presets.scenario_obs_layouts import VERTICAL_REACH_GOAL_XYZ_GLOBAL_OBS_LAYOUT
from swarmbots.scenario_presets.scenario_presets_kwargs import VERTICAL_REACH_SCENARIO_KWARGS


def test_vertical_reach_presets_expose_goal_box_center_global_obs_in_both_backends() -> None:
    mj_scenario = default_mj_vertical_reach(reset_settle_time=0.0)
    mjw_scenario = default_mjw_vertical_reach()
    state, connections = mj_scenario.reset_scenario(mj_scenario.dummy_model, mj_scenario.dummy_data, settle=False)
    mj_obs = mj_scenario.get_obs(mj_scenario.dummy_model, mj_scenario.dummy_data, state, connections)

    expected_goal = np.array(
        [
            VERTICAL_REACH_SCENARIO_KWARGS["wall_center_x"],
            (
                VERTICAL_REACH_SCENARIO_KWARGS["wall_y"]
                - VERTICAL_REACH_SCENARIO_KWARGS["wall_thickness"] / 2.0
                - VERTICAL_REACH_SCENARIO_KWARGS["goal_box_depth"] / 2.0
            ),
            VERTICAL_REACH_SCENARIO_KWARGS["goal_center_z"],
        ],
        dtype=float,
    )

    assert mj_obs["global_obs"].shape == (3,)
    assert mj_obs["hidden_global_vars"].shape == (0,)
    assert np.allclose(mj_obs["global_obs"], expected_goal)
    assert mjw_scenario.get_single_observation_space()["global_obs"].shape == (3,)
    assert mjw_scenario.get_single_observation_space()["hidden_global_vars"].shape == (0,)
    assert np.allclose(np.asarray(mjw_scenario.goal_position, dtype=float), expected_goal)
    assert mj_scenario.get_settings()["global_obs_layout"] == VERTICAL_REACH_GOAL_XYZ_GLOBAL_OBS_LAYOUT
    assert mjw_scenario.get_settings()["global_obs_layout"] == VERTICAL_REACH_GOAL_XYZ_GLOBAL_OBS_LAYOUT


def test_vertical_reach_horizontal_goal_defaults_to_wall_box_contact_in_both_backends() -> None:
    mj_scenario = default_mj_vertical_reach(reset_settle_time=0.0)
    mjw_scenario = default_mjw_vertical_reach()
    expected_wall_contact_goal = np.array(
        [
            VERTICAL_REACH_SCENARIO_KWARGS["wall_center_x"],
            VERTICAL_REACH_SCENARIO_KWARGS["wall_y"] - VERTICAL_REACH_SCENARIO_KWARGS["wall_thickness"] / 2.0,
            VERTICAL_REACH_SCENARIO_KWARGS["goal_center_z"],
        ],
        dtype=float,
    )

    assert mj_scenario.horizontal_goal_at_wall_contact
    assert mjw_scenario.horizontal_goal_at_wall_contact
    assert np.allclose(mj_scenario.horizontal_goal_position, expected_wall_contact_goal)
    assert np.allclose(np.asarray(mjw_scenario.horizontal_goal_position, dtype=float), expected_wall_contact_goal)
    assert mj_scenario.get_settings()["horizontal_goal_at_wall_contact"] is True
    assert mjw_scenario.get_settings()["horizontal_goal_at_wall_contact"] is True


def test_vertical_reach_horizontal_goal_can_use_goal_box_center_in_both_backends() -> None:
    mj_scenario = default_mj_vertical_reach(reset_settle_time=0.0, horizontal_goal_at_wall_contact=False)
    mjw_scenario = default_mjw_vertical_reach(horizontal_goal_at_wall_contact=False)

    assert not mj_scenario.horizontal_goal_at_wall_contact
    assert not mjw_scenario.horizontal_goal_at_wall_contact
    assert np.allclose(mj_scenario.horizontal_goal_position, mj_scenario.goal_position)
    assert np.allclose(
        np.asarray(mjw_scenario.horizontal_goal_position, dtype=float),
        np.asarray(mjw_scenario.goal_position, dtype=float),
    )


def test_vertical_reach_global_obs_indices_are_scalar_goal_coordinates_in_both_backends() -> None:
    mj_env = SwarmBotsEnv(default_mj_vertical_reach(reset_settle_time=0.0))
    mj_obs_indices = build_obs_indices(
        mj_env.get_settings(),
        mj_env.observation_space["local_obs"].shape[-1],
        mj_env.observation_space["global_obs"].shape[-1],
        mj_env.observation_space["hidden_local_vars"].shape[-1],
        mj_env.observation_space["hidden_global_vars"].shape[-1],
    )

    mjw_scenario = default_mjw_vertical_reach()
    mjw_space = mjw_scenario.get_single_observation_space()
    mjw_obs_indices = build_obs_indices(
        {"scenario": mjw_scenario.get_settings()},
        mjw_space["local_obs"].shape[-1],
        mjw_space["global_obs"].shape[-1],
        mjw_space["hidden_local_vars"].shape[-1],
        mjw_space["hidden_global_vars"].shape[-1],
    )

    for obs_indices in (mj_obs_indices, mjw_obs_indices):
        assert obs_indices.global_scalar_indices == [0, 1, 2]
        assert obs_indices.global_rot6d_indices == []
        assert obs_indices.global_quaternion_indices == []


def test_vertical_reach_wall_and_goal_box_geometry_in_both_backends() -> None:
    mj_scenario = default_mj_vertical_reach(
        reset_settle_time=0.0,
        wall_width=8.0,
        wall_thickness=0.5,
        wall_height=7.0,
        wall_center_x=0.25,
        wall_y=3.0,
        goal_box_width=1.2,
        goal_box_depth=0.6,
        goal_box_height=0.8,
        goal_center_z=2.0,
    )
    mjw_scenario = default_mjw_vertical_reach(
        wall_width=8.0,
        wall_thickness=0.5,
        wall_height=7.0,
        wall_center_x=0.25,
        wall_y=3.0,
        goal_box_width=1.2,
        goal_box_depth=0.6,
        goal_box_height=0.8,
        goal_center_z=2.0,
    )

    for scenario, model in ((mj_scenario, mj_scenario.dummy_model), (mjw_scenario, mjw_scenario.build_model())):
        wall_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "VerticalReachWall")
        goal_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "VerticalReachGoal")

        assert wall_id >= 0
        assert goal_id >= 0
        assert int(model.geom_type[wall_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        assert int(model.geom_type[goal_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        assert np.allclose(model.geom_size[wall_id], np.array([4.0, 0.25, 3.5], dtype=float))
        assert np.allclose(model.geom_pos[wall_id], np.array([0.25, 3.0, 3.5], dtype=float))
        assert np.allclose(model.geom_size[goal_id], np.array([0.6, 0.3, 0.4], dtype=float))
        assert np.allclose(model.geom_pos[goal_id], np.asarray(scenario.goal_position, dtype=float))
        assert int(model.geom_contype[goal_id]) == 0
        assert int(model.geom_conaffinity[goal_id]) == 0


def test_vertical_reach_goal_box_must_fit_between_ground_and_wall_top_in_both_backends() -> None:
    for default_vertical_reach in (default_mj_vertical_reach, default_mjw_vertical_reach):
        with pytest.raises(ValueError, match="fit within the wall height"):
            default_vertical_reach(goal_center_z=0.1, goal_box_height=0.4)

        with pytest.raises(ValueError, match="fit within the wall height"):
            default_vertical_reach(wall_height=1.0, goal_center_z=0.9, goal_box_height=0.4)


def test_vertical_reach_success_terminates_when_any_active_unit_enters_goal_in_mj_env() -> None:
    scenario = default_mj_vertical_reach(
        reset_settle_time=0.0,
        goal_success_reward=7.0,
        progress_reward_weight=2.0,
        units_without_connections_reward_weight=0.0,
    )
    state, connections = scenario.reset_scenario(scenario.dummy_model, scenario.dummy_data, settle=False)
    active_mask = np.ones((scenario.num_units,), dtype=bool)
    active_mask[-1] = False
    state["units_active_mask"] = active_mask
    scenario.dummy_data.qpos[scenario._qpos_indices[:, :3]] = scenario.goal_position + np.array([10.0, 0.0, 0.0])
    scenario.dummy_data.qpos[scenario._qpos_indices[1, :3]] = scenario.goal_position
    scenario.dummy_data.qpos[scenario._qpos_indices[-1, :3]] = scenario.goal_position
    scenario._reset_progress_baselines(scenario.dummy_data, state)

    action = {
        "actuators": np.zeros(scenario.get_actuator_action_shape(), dtype=np.float32),
        "connectors": np.zeros(scenario.get_connector_action_shape(), dtype=bool),
    }
    reward, terminated = scenario.evaluate_step(
        action,
        scenario.dummy_model,
        scenario.dummy_data,
        state,
        connections,
    )

    assert terminated
    assert state["success"]
    assert state["goal_success_reward"] == 7.0
    assert state["weighted_goal_success_reward"] == 14.0
    assert state["reward_terms"]["success"] == 14.0
    assert reward == 14.0


def test_vertical_reach_success_requires_any_active_unit_inside_goal_in_mjw() -> None:
    unit_position = torch.tensor(
        [
            [[0.0, 0.0, 1.0], [3.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[2.0, 0.0, 1.0], [3.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
            [[0.0, 0.0, 1.0], [3.0, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    goal_position = torch.tensor([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=torch.float32)
    goal_box_half_size = torch.tensor([0.5, 0.5, 0.2], dtype=torch.float32)
    stable_mask = torch.tensor([True, True, False], dtype=torch.bool)
    units_active_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, False],
            [True, True, False],
        ],
        dtype=torch.bool,
    )

    terminations = _compute_vertical_reach_goal_success_terminations(
        unit_position=unit_position,
        goal_position=goal_position,
        goal_box_half_size=goal_box_half_size,
        stable_mask=stable_mask,
        units_active_mask=units_active_mask,
    )

    assert terminations.tolist() == [True, False, False]


def test_vertical_reach_progress_uses_best_active_unit_with_vertical_heavy_weight() -> None:
    unit_position = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [1.1, 0.0, 1.0], [0.0, 0.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    goal_position = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32)
    goal_box_half_size = torch.tensor([0.5, 0.5, 0.1], dtype=torch.float32)
    units_active_mask = torch.tensor([[True, True, False]], dtype=torch.bool)

    horizontal_progress, height_progress = _compute_vertical_reach_progress_baselines_torch(
        unit_position=unit_position,
        goal_position=goal_position,
        horizontal_goal_position=goal_position,
        goal_box_half_size=goal_box_half_size,
        units_active_mask=units_active_mask,
        horizontal_reward_weight=1.0,
        height_reward_weight=8.0,
    )

    assert torch.allclose(horizontal_progress, torch.tensor([-0.6]), atol=1e-6)
    assert torch.allclose(height_progress, torch.tensor([0.0]), atol=1e-6)


def test_vertical_reach_horizontal_progress_uses_wall_contact_target() -> None:
    goal_position = np.array([0.0, 0.8, 1.0], dtype=float)
    horizontal_goal_position = np.array([0.0, 1.0, 1.0], dtype=float)
    goal_box_half_size = np.array([0.5, 0.2, 0.1], dtype=float)
    unit_positions = np.array([[0.0, 0.6, 1.0]], dtype=float)

    horizontal_progress, height_progress = _compute_vertical_reach_progress_baselines_np(
        unit_positions=unit_positions,
        goal_position=goal_position,
        horizontal_goal_position=horizontal_goal_position,
        goal_box_half_size=goal_box_half_size,
        active_units_mask=np.array([True], dtype=bool),
        horizontal_reward_weight=1.0,
        height_reward_weight=1.0,
    )

    assert horizontal_progress == pytest.approx(-0.2)
    assert height_progress == 0.0


def test_vertical_reach_column_progress_rewards_max_height_only_in_front_wall_column() -> None:
    unit_position = torch.tensor(
        [
            [[0.0, 1.0, 0.4], [0.2, 1.3, 0.7], [0.0, 0.2, 2.0]],
            [[1.6, 1.0, 2.0], [0.2, 1.1, 0.1], [0.0, 0.2, 2.0]],
        ],
        dtype=torch.float32,
    )
    goal_position = torch.tensor([[0.0, 1.0, 1.0], [0.0, 1.0, 1.0]], dtype=torch.float32)
    units_active_mask = torch.tensor([[True, True, True], [True, True, False]], dtype=torch.bool)

    progress = _compute_reach_column_progress_baseline_torch(
        unit_position=unit_position,
        goal_position=goal_position,
        units_active_mask=units_active_mask,
        reach_column_half_width=1.0,
        reach_column_min_y=0.5,
        reach_column_max_y=1.5,
    )

    assert torch.allclose(progress, torch.tensor([-0.3, -0.9]), atol=1e-6)


def test_vertical_reach_success_reward_is_added_once_to_weighted_progress_reward_in_mjw() -> None:
    runtime = object.__new__(VerticalReachMJWScenarioRuntime)
    runtime.scenario = SimpleNamespace(
        goal_success_reward=3.0,
        progress_reward_weight=2.0,
        horizontal_reward_weight=0.0,
        height_reward_weight=0.0,
        reach_column_reward_weight=0.0,
        reach_column_half_width=1.0,
        reach_column_min_y=-1.0,
        reach_column_max_y=1.0,
        potential_reward_discount_factor=1.0,
        units_without_connections_reward_weight=0.0,
        guidance_reward_weight=1.0,
    )
    runtime.bindings = SimpleNamespace(
        units_active_mask=torch.tensor([[True, True]], dtype=torch.bool),
        partner_unit=torch.full((1, 2, 1), -1, dtype=torch.long),
    )
    runtime._global_obs = torch.zeros((1, 3), dtype=torch.float32)
    runtime._horizontal_goal_position = torch.zeros((3,), dtype=torch.float32)
    runtime._goal_box_half_size = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    runtime.horizontal_progress = torch.zeros((1,), dtype=torch.float32)
    runtime.height_progress = torch.zeros((1,), dtype=torch.float32)
    runtime.reach_column_progress = torch.zeros((1,), dtype=torch.float32)
    runtime._reward_kernel = _compute_vertical_reach_reward_kernel
    runtime._get_unit_position = lambda: torch.tensor(
        [[[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )

    result = runtime.compute_step_rewards(stable_mask=torch.tensor([True], dtype=torch.bool))

    assert bool(result.terminations[0])
    assert bool(result.info["success"][0])
    assert float(result.info["goal_success_reward"][0]) == 6.0
    assert float(result.info["reward_terms"]["success"][0]) == 6.0
    assert float(result.info["progress_reward"][0]) == 6.0
    assert float(result.reward[0]) == 6.0
