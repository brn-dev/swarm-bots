from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from swarmbots.mj_env.scenarios.payload_step_scenario import PayloadStepScenario
from swarmbots.mj_env.scenarios.scenario_presets import default_payload_step as default_mj_payload_step
from swarmbots.mjw_env.scenarios.mjw_payload_step_runtime import _compute_payload_step_reward_extras_kernel
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_payload_step as default_mjw_payload_step
from swarmbots.scenario_presets.scenario_presets_kwargs import PAYLOAD_STEP_SCENARIO_KWARGS


def test_payload_step_height_reward_uses_payload_lift_near_step_in_mj_env() -> None:
    scenario = object.__new__(PayloadStepScenario)
    scenario.step_start_y = 1.5
    scenario.step_height = 0.1
    scenario.payload_radius = 0.2
    scenario.payload_step_height_reward_weight = 2.0
    scenario.payload_step_height_reward_distance = 0.5
    scenario.potential_reward_discount_factor = 1.0
    scenario._payload_qpos_indices = np.arange(7, dtype=int)

    state = {
        "payload_step_height_potential": 0.0,
        "payload_step_height_done": False,
    }
    data = SimpleNamespace(qpos=np.array([0.0, 1.25, 0.24, 1.0, 0.0, 0.0, 0.0], dtype=float))

    reward = scenario._compute_payload_step_height_reward(data, state)

    expected_potential = np.sqrt(0.5) * 0.4
    assert np.isclose(state["payload_step_height_potential"], expected_potential)
    assert np.isclose(reward, expected_potential * 2.0)
    assert state["payload_step_height_done"] is False


def test_payload_step_height_reward_does_not_penalize_crossing_step_front_in_mj_env() -> None:
    scenario = object.__new__(PayloadStepScenario)
    scenario.step_start_y = 1.5
    scenario.step_height = 0.1
    scenario.payload_radius = 0.2
    scenario.payload_step_height_reward_weight = 2.0
    scenario.payload_step_height_reward_distance = 0.5
    scenario.potential_reward_discount_factor = 1.0
    scenario._payload_qpos_indices = np.arange(7, dtype=int)

    state = {
        "payload_step_height_potential": 0.75,
        "payload_step_height_done": False,
    }
    data = SimpleNamespace(qpos=np.array([0.0, 1.51, 0.3, 1.0, 0.0, 0.0, 0.0], dtype=float))

    reward = scenario._compute_payload_step_height_reward(data, state)

    assert np.isclose(reward, 0.0)
    assert np.isclose(state["payload_step_height_potential"], 0.0)
    assert state["payload_step_height_done"] is True


def test_payload_step_success_requires_y_and_elevated_payload_in_mj_env() -> None:
    scenario = object.__new__(PayloadStepScenario)
    scenario.payload_success_y = 3.0
    scenario.payload_radius = 0.2
    scenario.step_height = 0.1
    scenario.payload_success_height_tolerance = 0.02
    scenario._payload_qpos_indices = np.arange(7, dtype=int)

    too_low = SimpleNamespace(qpos=np.array([0.0, 3.1, 0.25, 1.0, 0.0, 0.0, 0.0], dtype=float))
    high_enough = SimpleNamespace(qpos=np.array([0.0, 3.1, 0.28, 1.0, 0.0, 0.0, 0.0], dtype=float))

    assert not scenario._compute_payload_success_termination(too_low, {})
    assert scenario._compute_payload_success_termination(high_enough, {})


def test_payload_step_reward_extras_kernel_adds_height_reward_and_success_in_mjw_runtime() -> None:
    payload_position = torch.tensor(
        [
            [0.0, 1.25, 0.24],
            [0.0, 3.2, 0.28],
        ],
        dtype=torch.float32,
    )

    (
        height_potential,
        height_done,
        height_reward,
        success_reward,
        success,
    ) = _compute_payload_step_reward_extras_kernel(
        payload_position=payload_position,
        stable_mask=torch.tensor([True, True], dtype=torch.bool),
        payload_step_height_potential=torch.zeros((2,), dtype=torch.float32),
        payload_step_height_done=torch.zeros((2,), dtype=torch.bool),
        step_start_y=1.5,
        step_height=0.1,
        payload_ground_z=0.2,
        payload_step_height_reward_weight=2.0,
        payload_step_height_reward_distance=0.5,
        payload_success_y=3.0,
        payload_success_min_z=0.28,
        payload_success_reward_value=5.0,
        progress_reward_weight=3.0,
        potential_reward_discount_factor=1.0,
    )

    assert torch.allclose(height_potential, torch.tensor([np.sqrt(0.5) * 0.4, 0.0], dtype=torch.float32))
    assert torch.equal(height_done, torch.tensor([False, True]))
    assert torch.allclose(
        height_reward,
        torch.tensor([np.sqrt(0.5) * 0.4 * 2.0 * 3.0, 0.0], dtype=torch.float32),
    )
    assert torch.allclose(success_reward, torch.tensor([0.0, 15.0], dtype=torch.float32))
    assert torch.equal(success, torch.tensor([False, True]))


def test_payload_step_presets_and_geometry_are_available_in_both_backends() -> None:
    mj_scenario = default_mj_payload_step(reset_settle_time=0.0)
    mjw_scenario = default_mjw_payload_step(compile_reward_kernel=False)

    assert mj_scenario.get_obs_space()["global_obs"].shape == (9,)
    assert mjw_scenario.get_single_observation_space()["global_obs"].shape == (9,)
    assert mj_scenario.get_settings()["scenario_type"] == "payload_step"
    assert mjw_scenario.get_settings()["scenario_type"] == "payload_step"
    assert np.isclose(mj_scenario.step_height, PAYLOAD_STEP_SCENARIO_KWARGS["step_height"])
    assert np.isclose(mjw_scenario.step_height, PAYLOAD_STEP_SCENARIO_KWARGS["step_height"])
    assert _model_has_step_geom(mj_scenario.dummy_model, mj_scenario)
    assert _model_has_step_geom(mjw_scenario.build_model(), mjw_scenario)


def _model_has_step_geom(model: mujoco.MjModel, scenario: object) -> bool:
    expected_pos = np.array(
        [
            0.0,
            float(scenario.step_start_y) + (float(scenario.step_length) / 2.0),
            float(scenario.step_height) / 2.0,
        ],
        dtype=float,
    )
    expected_size = np.array(
        [
            float(scenario.step_width) / 2.0,
            float(scenario.step_length) / 2.0,
            float(scenario.step_height) / 2.0,
        ],
        dtype=float,
    )
    for geom_idx in range(model.ngeom):
        if int(model.geom_type[geom_idx]) != int(mujoco.mjtGeom.mjGEOM_BOX):
            continue
        if np.allclose(model.geom_pos[geom_idx], expected_pos) and np.allclose(
            model.geom_size[geom_idx, :3],
            expected_size,
        ):
            return True
    return False
