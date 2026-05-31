from __future__ import annotations

import mujoco
import numpy as np
import torch

from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv
from swarmbots.mj_env.scenarios.scenario_presets import default_climb as default_mj_climb
from swarmbots.mjw_env.scenarios.mjw_climb_runtime import _compute_climb_reward_kernel
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_climb as default_mjw_climb
from swarmbots.scenario_presets.scenario_presets_kwargs import CLIMB_SCENARIO_KWARGS


def test_climb_presets_expose_top_center_goal_global_obs_in_both_backends() -> None:
    mj_scenario = default_mj_climb(reset_settle_time=0.0)
    mjw_scenario = default_mjw_climb()
    state, connections = mj_scenario.reset_scenario(mj_scenario.dummy_model, mj_scenario.dummy_data, settle=False)
    mj_obs = mj_scenario.get_obs(mj_scenario.dummy_model, mj_scenario.dummy_data, state, connections)

    expected_goal = np.array(
        [
            CLIMB_SCENARIO_KWARGS["cuboid_center_x"],
            CLIMB_SCENARIO_KWARGS["cuboid_center_y"],
            CLIMB_SCENARIO_KWARGS["cuboid_size_z"] + mj_scenario.goal_height_offset,
        ],
        dtype=float,
    )

    assert mj_obs["global_obs"].shape == (3,)
    assert mj_obs["hidden_global_vars"].shape == (0,)
    assert np.allclose(mj_obs["global_obs"], expected_goal)
    assert mjw_scenario.get_single_observation_space()["global_obs"].shape == (3,)
    assert mjw_scenario.get_single_observation_space()["hidden_global_vars"].shape == (0,)
    assert np.allclose(np.asarray(mjw_scenario.goal_position, dtype=float), expected_goal)
    assert np.isclose(mj_scenario.goal_height_offset, mj_scenario.swarm.max_unit_extent / 2.0)
    assert np.isclose(mjw_scenario.goal_height_offset, mjw_scenario.swarm.max_unit_extent / 2.0)


def test_climb_global_obs_indices_are_scalar_goal_coordinates_in_both_backends() -> None:
    mj_env = SwarmBotsEnv(default_mj_climb(reset_settle_time=0.0))
    mj_obs_indices = build_obs_indices(
        mj_env.get_settings(),
        mj_env.observation_space["local_obs"].shape[-1],
        mj_env.observation_space["global_obs"].shape[-1],
        mj_env.observation_space["hidden_local_vars"].shape[-1],
        mj_env.observation_space["hidden_global_vars"].shape[-1],
    )

    mjw_scenario = default_mjw_climb()
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


def test_climb_cuboid_is_static_box_at_requested_top_height_in_both_backends() -> None:
    mj_scenario = default_mj_climb(
        reset_settle_time=0.0,
        cuboid_size_x=4.0,
        cuboid_size_y=2.0,
        cuboid_size_z=1.5,
        cuboid_center_x=0.25,
        cuboid_center_y=3.0,
    )
    mjw_scenario = default_mjw_climb(
        cuboid_size_x=4.0,
        cuboid_size_y=2.0,
        cuboid_size_z=1.5,
        cuboid_center_x=0.25,
        cuboid_center_y=3.0,
    )

    for model in (mj_scenario.dummy_model, mjw_scenario.build_model()):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ClimbCuboid")

        assert geom_id >= 0
        assert int(model.geom_type[geom_id]) == int(mujoco.mjtGeom.mjGEOM_BOX)
        assert np.allclose(model.geom_size[geom_id], np.array([2.0, 1.0, 0.75], dtype=float))
        assert np.allclose(model.geom_pos[geom_id], np.array([0.25, 3.0, 0.75], dtype=float))
        assert int(model.body_mocapid[model.geom_bodyid[geom_id]]) == -1


def test_climb_mjw_reward_kernel_exposes_horizontal_and_height_rewards() -> None:
    unit_position = torch.tensor(
        [
            [[0.0, 0.0, 0.0], [0.0, 2.0, 0.7]],
            [[0.0, 0.0, 0.0], [0.0, 2.0, 1.0]],
        ],
        dtype=torch.float32,
    )
    goal_position = torch.tensor([[0.0, 2.0, 1.0], [0.0, 2.0, 1.0]], dtype=torch.float32)
    stable_mask = torch.tensor([True, False], dtype=torch.bool)
    active_mask = torch.tensor([[True, True], [True, True]], dtype=torch.bool)
    partner_unit = torch.full((2, 2, 1), -1, dtype=torch.long)
    horizontal_progress = torch.tensor([-1.5, -1.8], dtype=torch.float32)
    height_progress = torch.tensor([-0.8, -0.8], dtype=torch.float32)

    (
        new_horizontal_progress,
        new_height_progress,
        progress_reward,
        horizontal_reward,
        height_reward,
        guidance_reward,
    ) = _compute_climb_reward_kernel(
        unit_position,
        goal_position,
        stable_mask,
        active_mask,
        partner_unit,
        horizontal_progress,
        height_progress,
        0.2,
        0.2,
        2.0,
        10.0,
        3.0,
        1.0,
        -0.25,
        3.0,
    )

    assert torch.allclose(new_horizontal_progress, torch.tensor([-0.9, -1.8]))
    assert torch.allclose(new_height_progress, torch.tensor([-0.45, -0.8]))
    assert torch.allclose(horizontal_reward, torch.tensor([12.0, 0.0]), atol=1e-5)
    assert torch.allclose(height_reward, torch.tensor([2.1, 0.0]), atol=1e-5)
    assert torch.allclose(progress_reward, torch.tensor([14.1, 0.0]), atol=1e-5)
    assert torch.allclose(guidance_reward, torch.tensor([-0.75, -0.75]))
