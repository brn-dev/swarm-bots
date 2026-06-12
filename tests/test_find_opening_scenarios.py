from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
import torch

from swarmbots.mj_env.float_or_dist_params import SplitUniformDistParams
from swarmbots.mj_env.scenarios.scenario_presets import (
    default_find_opening as default_mj_find_opening,
)
from swarmbots.mjw_env.mjw_model_metadata import build_model_metadata
from swarmbots.mjw_env.scenarios.mjw_find_opening_runtime import (
    FindOpeningMJWScenarioRuntime,
    _compute_find_opening_reward_kernel,
)
from swarmbots.mjw_env.scenarios.mjw_find_opening_scenario import (
    MJWFindOpeningRuntimeMetadata,
)
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import (
    default_find_opening as default_mjw_find_opening,
)


def test_find_opening_mj_samples_opening_once_per_episode_and_keeps_it_hidden() -> None:
    scenario = default_mj_find_opening(
        seed=123,
        reset_settle_time=0.0,
    )

    sampled_opening_x: list[float] = []
    for _ in range(64):
        state, connections = scenario.reset_scenario(
            scenario.dummy_model,
            scenario.dummy_data,
            settle=False,
        )
        first_obs = scenario.get_obs(
            scenario.dummy_model,
            scenario.dummy_data,
            state,
            connections,
        )
        second_obs = scenario.get_obs(
            scenario.dummy_model,
            scenario.dummy_data,
            state,
            connections,
        )

        assert first_obs["global_obs"].shape == (0,)
        assert first_obs["hidden_global_vars"].shape == (1,)
        assert first_obs["hidden_global_vars"][0] == pytest.approx(state["opening_x"])
        assert np.array_equal(first_obs["hidden_global_vars"], second_obs["hidden_global_vars"])
        sampled_opening_x.append(float(first_obs["hidden_global_vars"][0]))

    assert isinstance(scenario.opening_x_param, SplitUniformDistParams)
    assert all(-3.0 <= opening_x <= 3.0 for opening_x in sampled_opening_x)
    assert all(abs(opening_x) >= 1.0 for opening_x in sampled_opening_x)
    assert any(opening_x < 0.0 for opening_x in sampled_opening_x)
    assert any(opening_x > 0.0 for opening_x in sampled_opening_x)
    assert len({round(opening_x, 6) for opening_x in sampled_opening_x}) > 1


def test_find_opening_mj_barrier_geometry_matches_sampled_opening() -> None:
    opening_x = 1.25
    opening_width = 1.5
    wall_segment_width = 100.0
    scenario = default_mj_find_opening(
        reset_settle_time=0.0,
        opening_x=opening_x,
        opening_width=opening_width,
        wall_segment_width=wall_segment_width,
    )
    scenario.reset_scenario(scenario.dummy_model, scenario.dummy_data, settle=False)

    left_geom_id = mujoco.mj_name2id(
        scenario.dummy_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "FindOpeningBarrier-left",
    )
    right_geom_id = mujoco.mj_name2id(
        scenario.dummy_model,
        mujoco.mjtObj.mjOBJ_GEOM,
        "FindOpeningBarrier-right",
    )
    left_right_edge = (
        scenario.dummy_data.geom_xpos[left_geom_id, 0]
        + scenario.dummy_model.geom_size[left_geom_id, 0]
    )
    right_left_edge = (
        scenario.dummy_data.geom_xpos[right_geom_id, 0]
        - scenario.dummy_model.geom_size[right_geom_id, 0]
    )

    assert left_right_edge == pytest.approx(opening_x - opening_width / 2.0)
    assert right_left_edge == pytest.approx(opening_x + opening_width / 2.0)
    assert scenario.dummy_model.geom_size[left_geom_id, 0] == pytest.approx(
        wall_segment_width / 2.0
    )
    assert scenario.dummy_model.geom_size[right_geom_id, 0] == pytest.approx(
        wall_segment_width / 2.0
    )


def test_find_opening_mj_potential_rewards_xy_progress_towards_post_wall_waypoint() -> None:
    scenario = default_mj_find_opening(
        reset_settle_time=0.0,
        opening_x=2.0,
        wall_y=3.0,
        opening_y_margin=1.0,
    )
    state, _ = scenario.reset_scenario(
        scenario.dummy_model,
        scenario.dummy_data,
        settle=False,
    )
    active_mask = state.get("units_active_mask")
    if active_mask is None:
        active_mask = np.ones(scenario.num_units, dtype=bool)

    scenario.dummy_data.qpos[scenario._qpos_indices[active_mask, 0]] = 0.0
    scenario.dummy_data.qpos[scenario._qpos_indices[active_mask, 1]] = 4.0
    far_potential = scenario._compute_opening_potential(
        scenario.dummy_data,
        active_mask,
        opening_x=2.0,
    )
    scenario.dummy_data.qpos[scenario._qpos_indices[active_mask, 0]] = 1.0
    closer_potential = scenario._compute_opening_potential(
        scenario.dummy_data,
        active_mask,
        opening_x=2.0,
    )

    assert far_potential == pytest.approx(-2.0)
    assert closer_potential == pytest.approx(-1.0)


def test_find_opening_mjw_space_and_metadata_keep_opening_actor_hidden() -> None:
    scenario = default_mjw_find_opening()
    obs_space = scenario.get_single_observation_space()
    host_model = scenario.build_model()

    generic_metadata = build_model_metadata(host_model, scenario)
    runtime_metadata = scenario.build_runtime_metadata(host_model=host_model)

    assert obs_space["global_obs"].shape == (0,)
    assert obs_space["hidden_global_vars"].shape == (1,)
    assert isinstance(runtime_metadata, MJWFindOpeningRuntimeMetadata)
    assert not hasattr(generic_metadata, "barrier_mocap_id")


def test_find_opening_mjw_camera_views_wall_diagonally_from_swarm_start_side() -> None:
    scenario = default_mjw_find_opening()

    camera_config = scenario.get_default_recording_camera_config()

    assert camera_config is not None
    assert camera_config.lookat == pytest.approx((0.0, scenario.wall_y * 0.5, 0.8))
    assert camera_config.azimuth == 45.0
    assert camera_config.elevation == -35.0


def test_find_opening_mjw_reset_sampling_respects_opening_distribution() -> None:
    runtime = object.__new__(FindOpeningMJWScenarioRuntime)
    device = torch.device("cpu")
    runtime.bindings = SimpleNamespace(device=device)
    runtime.scenario = SimpleNamespace(
        opening_x=SplitUniformDistParams(-0.75, 0.5, margin=0.2),
        swarm=SimpleNamespace(get_active_pool_size=lambda: 2, max_unit_extent=0.2),
        swarm_start_x=0.0,
        swarm_start_y=0.0,
        randomize_initial_swarm_z_rotation=False,
    )

    samples = runtime.sample_reset_batch(
        n_reset=32,
        rng=torch.Generator(device=device).manual_seed(123),
    )

    assert samples.opening_x.shape == (32,)
    assert torch.all(samples.opening_x >= -0.75)
    assert torch.all(samples.opening_x <= 0.5)
    assert torch.all(torch.abs(samples.opening_x) >= 0.2)
    assert torch.any(samples.opening_x < 0.0)
    assert torch.any(samples.opening_x > 0.0)
    assert torch.unique(samples.opening_x).numel() > 1


def test_find_opening_mjw_reward_kernel_terminates_only_when_all_active_units_pass() -> None:
    unit_x = torch.tensor([[1.0, 1.0, -5.0], [0.0, 2.0, -5.0]], dtype=torch.float32)
    unit_y = torch.tensor([[4.5, 4.5, -5.0], [4.0, 4.0, -5.0]], dtype=torch.float32)
    opening_x = torch.tensor([1.0, 1.0], dtype=torch.float32)
    stable_mask = torch.tensor([True, True], dtype=torch.bool)
    active_mask = torch.tensor([[True, True, False], [True, True, False]], dtype=torch.bool)
    partner_unit = torch.full((2, 3, 1), -1, dtype=torch.long)
    opening_potential = torch.tensor([-1.0, -2.0], dtype=torch.float32)

    new_potential, opening_reward, success_reward, guidance_reward, success = (
        _compute_find_opening_reward_kernel(
            unit_x,
            unit_y,
            opening_x,
            stable_mask,
            active_mask,
            partner_unit,
            opening_potential,
            4.0,
            2.0,
            5.0,
            1.0,
            1.0,
            -0.25,
            1.0,
        )
    )

    assert torch.allclose(new_potential, torch.tensor([-0.5, -1.0]))
    assert torch.allclose(opening_reward, torch.tensor([1.0, 2.0]))
    assert torch.allclose(success_reward, torch.tensor([5.0, 0.0]))
    assert torch.allclose(guidance_reward, torch.tensor([-0.25, -0.25]))
    assert torch.equal(success, torch.tensor([True, False]))
