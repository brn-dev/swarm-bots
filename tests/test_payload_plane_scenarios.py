from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import torch

from swarmbots.mj_env.scenarios.payload_plane_scenario import PayloadPlaneScenario
from swarmbots.mj_env.scenarios.dual_payload_plane_scenario import DualPayloadPlaneScenario
from swarmbots.mj_env.scenarios.scenario_presets import default_payload_plane as default_mj_payload_plane
from swarmbots.mj_env.scenarios.scenario_presets import default_dual_payload_plane as default_mj_dual_payload_plane
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv
from swarmbots.mjw_env.mjw_model_metadata import build_model_metadata
from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWCommonResetBatch
from swarmbots.mjw_env.scenarios.mjw_dual_payload_plane_runtime import (
    DualPayloadPlaneMJWScenarioRuntime,
    _compute_dual_payload_plane_reward_kernel,
)
from swarmbots.mjw_env.scenarios.mjw_payload_plane_scenario import MJWPayloadPlaneRuntimeMetadata
from swarmbots.mjw_env.scenarios.mjw_payload_plane_runtime import (
    PayloadPlaneMJWScenarioRuntime,
    _compute_payload_plane_reward_kernel,
)
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_payload_plane as default_mjw_payload_plane
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_dual_payload_plane as default_mjw_dual_payload_plane
from swarmbots.scenario_presets.scenario_presets_kwargs import PAYLOAD_PLANE_SCENARIO_KWARGS


def test_payload_plane_reward_uses_payload_progress_and_x_penalty_in_mj_env() -> None:
    scenario = object.__new__(PayloadPlaneScenario)
    scenario.forward_reward_weight = 1.5
    scenario.forward_reward_max_y = None
    scenario.towards_payload_reward_weight = 0.0
    scenario.payload_centering_penalty_weight = 0.5
    scenario.payload_centering_penalty_power = 2.0
    scenario.payload_centering_tolerance = 0.0
    scenario._payload_qpos_indices = np.arange(7, dtype=int)

    state = {"progress": 1.0}
    data = SimpleNamespace(qpos=np.array([0.3, 1.4, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=float))

    reward = scenario.compute_progress_reward(data, state)

    assert np.isclose(state["forward_reward"], 0.4)
    assert np.isclose(state["payload_x_penalty"], -(0.3 ** 2) * 0.5)
    assert np.isclose(reward, (0.4 * 1.5) - ((0.3 ** 2) * 0.5))


def test_payload_plane_x_penalty_uses_tolerance_in_mj_env() -> None:
    scenario = object.__new__(PayloadPlaneScenario)
    scenario.payload_centering_penalty_weight = 0.5
    scenario.payload_centering_penalty_power = 2.0
    scenario.payload_centering_tolerance = 0.1
    scenario._payload_qpos_indices = np.arange(7, dtype=int)

    inside_tolerance = SimpleNamespace(qpos=np.array([0.08, 1.4, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=float))
    outside_tolerance = SimpleNamespace(qpos=np.array([0.3, 1.4, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=float))

    assert np.isclose(scenario._compute_payload_x_penalty(inside_tolerance), 0.0)
    assert np.isclose(scenario._compute_payload_x_penalty(outside_tolerance), -((0.3 - 0.1) ** 2) * 0.5)


def test_payload_plane_reset_exposes_payload_pose_as_global_obs_in_mj_env() -> None:
    scenario = default_mj_payload_plane(
        reset_settle_time=0.0,
        swarm_start_x=0.25,
        swarm_start_y=-0.5,
        payload_offset_x=0.4,
        payload_offset_y=1.2,
    )
    env = SwarmBotsEnv(scenario=scenario, episode_length=8)
    obs, _ = env.reset(seed=123)

    assert np.allclose(
        obs["global_obs"],
        np.array([0.65, 0.7, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0], dtype=float),
    )
    assert obs["hidden_global_vars"].shape == (0,)
    env.close()


def test_payload_plane_forward_reward_cap_is_applied_in_mj_env() -> None:
    scenario = object.__new__(PayloadPlaneScenario)
    scenario.forward_reward_weight = 1.0
    scenario.forward_reward_max_y = 0.5
    scenario.towards_payload_reward_weight = 0.0
    scenario.payload_centering_penalty_weight = 0.0
    scenario.payload_centering_penalty_power = 1.0
    scenario.payload_centering_tolerance = 0.0
    scenario._payload_qpos_indices = np.arange(7, dtype=int)

    state = {"progress": 0.45}
    data = SimpleNamespace(qpos=np.array([0.2, 0.7, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=float))

    reward = scenario.compute_progress_reward(data, state)

    assert np.isclose(reward, 0.05)
    assert np.isclose(state["forward_reward"], 0.05)
    assert np.isclose(state["progress"], 0.5)


def test_payload_plane_forward_reward_applies_discount_factor_in_mj_env() -> None:
    scenario = object.__new__(PayloadPlaneScenario)
    scenario.forward_reward_weight = 1.5
    scenario.forward_reward_max_y = None
    scenario.towards_payload_reward_weight = 0.0
    scenario.payload_centering_penalty_weight = 0.0
    scenario.payload_centering_penalty_power = 1.0
    scenario.payload_centering_tolerance = 0.0
    scenario.potential_reward_discount_factor = 0.9
    scenario._payload_qpos_indices = np.arange(7, dtype=int)

    state = {"progress": 1.0}
    data = SimpleNamespace(qpos=np.array([0.0, 1.4, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=float))

    reward = scenario.compute_progress_reward(data, state)

    assert np.isclose(state["forward_reward"], 0.26)
    assert np.isclose(reward, 0.26 * 1.5)


def test_payload_plane_evaluate_step_reports_weighted_reward_terms_in_mj_env() -> None:
    scenario = object.__new__(PayloadPlaneScenario)
    scenario.num_units = 1
    scenario.forward_reward_weight = 2.0
    scenario.forward_reward_max_y = None
    scenario.towards_payload_reward_weight = 0.0
    scenario.payload_centering_penalty_weight = 0.5
    scenario.payload_centering_penalty_power = 1.0
    scenario.payload_centering_tolerance = 0.0
    scenario.reward_weights = {
        "progress_reward_weight": 3.0,
        "guidance_reward_weight": 4.0,
        "units_without_connections_reward_weight": 0.0,
    }
    scenario._payload_qpos_indices = np.arange(7, dtype=int)

    state = {
        "progress": 1.0,
        "units_active_mask": np.array([True], dtype=bool),
    }
    data = SimpleNamespace(qpos=np.array([0.4, 1.3, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=float))
    connections = SimpleNamespace(get_is_active_mask=lambda: np.array([[False]], dtype=bool))

    reward, terminated = scenario.evaluate_step(
        action={"actuators": np.zeros((0, 0), dtype=float), "connectors": np.zeros((0, 0), dtype=bool)},
        model=None,
        data=data,
        state=state,
        connections=connections,
    )

    assert not terminated
    assert np.isclose(state["weighted_forward_reward"], 1.8)
    assert np.isclose(state["weighted_towards_payload_reward"], 0.0)
    assert np.isclose(state["weighted_payload_x_penalty"], -0.6)
    assert np.isclose(state["weighted_guidance_reward"], 0.0)
    assert np.isclose(state["weighted_progress_reward"], 1.2)
    assert np.isclose(state["reward_terms"]["forward"], 1.8)
    assert np.isclose(state["reward_terms"]["towards_payload"], 0.0)
    assert np.isclose(state["reward_terms"]["payload_x"], -0.6)
    assert np.isclose(state["reward_terms"]["guidance"], 0.0)
    assert np.isclose(reward, 1.2)


def test_payload_plane_step_exposes_payload_reward_terms_in_mj_env() -> None:
    scenario = default_mj_payload_plane(
        reset_settle_time=0.0,
        payload_offset_x=0.4,
        payload_offset_y=1.0,
    )
    env = SwarmBotsEnv(scenario=scenario, episode_length=8)
    _obs, _ = env.reset(seed=123)
    action = {
        "actuators": np.zeros(env.action_space["actuators"].shape, dtype=np.float32),
        "connectors": np.zeros(env.action_space["connectors"].shape, dtype=bool),
    }

    _next_obs, reward, terminated, truncated, info = env.step(action)

    assert not terminated
    assert not truncated
    assert np.isfinite(float(reward))
    assert "payload_x" in info["reward_terms"]
    assert "towards_payload" in info["reward_terms"]
    assert "forward_reward" in info
    assert "guidance_reward" in info
    env.close()


def test_payload_plane_towards_payload_reward_uses_virtual_goal_radius_in_mj_env() -> None:
    scenario = object.__new__(PayloadPlaneScenario)
    scenario.payload_radius = 0.3
    scenario.towards_payload_goal_radius = 0.2
    scenario.towards_payload_reward_weight = 2.0
    scenario._payload_qpos_indices = np.arange(7, dtype=int)
    scenario._qpos_indices = np.array([[7, 8]], dtype=int)

    state = {
        "towards_payload_progress": -0.5,
        "units_active_mask": np.array([True], dtype=bool),
    }
    data = SimpleNamespace(
        qpos=np.array(
            [
                0.0,
                1.3,
                0.2,
                1.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.8,
            ],
            dtype=float,
        )
    )

    reward = scenario._compute_towards_payload_reward(data, state)

    assert np.isclose(state["towards_payload_progress"], -0.0)
    assert np.isclose(reward, 1.0)


def test_dual_payload_plane_forward_reward_blends_individual_and_lagging_payload_progress_in_mj_env() -> None:
    scenario = object.__new__(DualPayloadPlaneScenario)
    scenario.forward_reward_weight = 2.0
    scenario.forward_reward_max_y = None
    scenario.lagging_payload_weight = 0.75
    scenario.towards_payload_reward_weight = 0.0
    scenario.payload_centering_penalty_weight = 0.5
    scenario.payload_centering_penalty_power = 1.0
    scenario.payload_centering_tolerance = 0.0
    scenario._payload_qpos_indices = np.array(
        [
            np.arange(7, dtype=int),
            np.arange(7, 14, dtype=int),
        ]
    )

    state = {
        "payload_progress": np.array([1.0, 1.1], dtype=float),
        "back_payload_progress": 1.0,
    }
    data = SimpleNamespace(
        qpos=np.array(
            [
                0.3, 1.4, 0.2, 1.0, 0.0, 0.0, 0.0,
                -0.2, 1.2, 0.2, 1.0, 0.0, 0.0, 0.0,
            ],
            dtype=float,
        )
    )

    reward = scenario.compute_progress_reward(data, state)

    assert np.isclose(state["forward_reward"], 0.425)
    assert np.isclose(state["payload_x_penalty"], -0.25)
    assert np.isclose(reward, (0.425 * 2.0) - 0.25)


def test_dual_payload_plane_towards_payload_reward_uses_nearest_units_per_payload_in_mj_env() -> None:
    scenario = object.__new__(DualPayloadPlaneScenario)
    scenario.payload_radius = 0.3
    scenario.towards_payload_goal_radius = 0.0
    scenario.towards_payload_reward_weight = 2.0
    scenario.towards_payload_units_per_payload = 2
    scenario._payload_qpos_indices = np.array(
        [
            np.arange(7, dtype=int),
            np.arange(7, 14, dtype=int),
        ]
    )
    scenario._qpos_indices = np.array([[14, 15], [16, 17], [18, 19], [20, 21]], dtype=int)

    state = {
        "towards_payload_progress": np.array([-2.0, -4.0], dtype=float),
        "units_active_mask": np.array([True, True, True, True], dtype=bool),
    }
    data = SimpleNamespace(
        qpos=np.array(
            [
                0.0, 1.0, 0.2, 1.0, 0.0, 0.0, 0.0,
                10.0, 1.0, 0.2, 1.0, 0.0, 0.0, 0.0,
                0.0, 0.7,
                0.0, 1.7,
                10.0, 0.7,
                10.0, 1.7,
            ],
            dtype=float,
        )
    )

    reward = scenario._compute_towards_payload_reward(data, state)

    assert np.allclose(state["towards_payload_progress"], np.array([-0.5, -0.5]))
    assert np.isclose(reward, 10.0)


def test_payload_plane_reward_uses_payload_progress_and_x_penalty_in_mjw_runtime() -> None:
    runtime = object.__new__(PayloadPlaneMJWScenarioRuntime)
    device = torch.device("cpu")
    payload_qpos = torch.tensor([[0.3, 1.4, 0.2, 1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    runtime.bindings = SimpleNamespace(
        units_active_mask=torch.tensor([[True]], device=device, dtype=torch.bool),
        partner_unit=torch.tensor([[[-1]]], device=device, dtype=torch.long),
        qpos=payload_qpos,
    )
    runtime.scenario = SimpleNamespace(
        progress_reward_weight=1.0,
        forward_reward_weight=1.5,
        forward_reward_max_y=None,
        potential_reward_discount_factor=1.0,
        payload_radius=0.2,
        towards_payload_reward_weight=0.0,
        towards_payload_goal_radius=0.2,
        payload_centering_penalty_weight=0.5,
        payload_centering_penalty_power=2.0,
        payload_centering_tolerance=0.0,
        units_without_connections_reward_weight=0.0,
        guidance_reward_weight=1.0,
        payload_pos_observable=True,
    )
    runtime.payload_position = torch.zeros((1, 3), device=device, dtype=torch.float32)
    runtime._global_obs = torch.zeros((1, 9), device=device, dtype=torch.float32)
    runtime._payload_qpos_indices = torch.arange(7, device=device, dtype=torch.long)
    runtime._unit_qpos_adr = torch.tensor([0], device=device, dtype=torch.long)
    runtime._xy_offsets = torch.tensor([0, 1], device=device, dtype=torch.long)
    runtime.progress = torch.tensor([1.0], device=device, dtype=torch.float32)
    runtime.towards_payload_progress = torch.tensor([0.0], device=device, dtype=torch.float32)
    runtime._reward_kernel = _compute_payload_plane_reward_kernel

    result = runtime.compute_step_rewards(stable_mask=torch.tensor([True], device=device, dtype=torch.bool))

    assert np.isclose(float(result.info["forward_reward"][0]), 0.6)
    assert np.isclose(float(result.info["payload_x_penalty"][0]), -((0.3 ** 2) * 0.5))
    assert np.isclose(float(result.reward[0]), 0.6 - ((0.3 ** 2) * 0.5))
    assert np.isclose(float(runtime.progress[0]), 1.4)


def test_payload_plane_x_penalty_uses_tolerance_in_mjw_runtime() -> None:
    payload_position = torch.tensor([[0.08, 1.0, 0.2], [0.3, 1.0, 0.2]], dtype=torch.float32)
    stable_mask = torch.tensor([True, True], dtype=torch.bool)
    units_active_mask = torch.tensor([[True], [True]], dtype=torch.bool)
    partner_unit = torch.tensor([[[-1]], [[-1]]], dtype=torch.long)
    progress = torch.tensor([1.0, 1.0], dtype=torch.float32)

    (
        _new_progress,
        _new_towards_payload_progress,
        progress_reward,
        forward_reward,
        towards_payload_reward,
        payload_x_penalty,
        _guidance_reward,
    ) = _compute_payload_plane_reward_kernel(
        torch.zeros((2, 1, 2), dtype=torch.float32),
        payload_position,
        stable_mask,
        units_active_mask,
        partner_unit,
        progress,
        torch.zeros_like(progress),
        1.0,
        1.0,
        float("inf"),
        1.0,
        0.2,
        0.0,
        0.2,
        0.5,
        2.0,
        0.1,
        0.0,
        1.0,
    )

    assert torch.allclose(forward_reward, torch.zeros_like(forward_reward))
    assert torch.allclose(towards_payload_reward, torch.zeros_like(towards_payload_reward))
    assert torch.allclose(payload_x_penalty, torch.tensor([0.0, -((0.3 - 0.1) ** 2) * 0.5]))
    assert torch.allclose(progress_reward, payload_x_penalty)


def test_payload_plane_towards_payload_reward_uses_virtual_goal_radius_in_mjw_runtime() -> None:
    unit_xy = torch.tensor([[[0.0, 0.8]]], dtype=torch.float32)
    payload_position = torch.tensor([[0.0, 1.3, 0.2]], dtype=torch.float32)
    stable_mask = torch.tensor([True], dtype=torch.bool)
    units_active_mask = torch.tensor([[True]], dtype=torch.bool)
    partner_unit = torch.tensor([[[-1]]], dtype=torch.long)
    progress = torch.tensor([1.3], dtype=torch.float32)
    towards_payload_progress = torch.tensor([-0.5], dtype=torch.float32)

    (
        _new_progress,
        new_towards_payload_progress,
        progress_reward,
        forward_reward,
        towards_payload_reward,
        payload_x_penalty,
        _guidance_reward,
    ) = _compute_payload_plane_reward_kernel(
        unit_xy,
        payload_position,
        stable_mask,
        units_active_mask,
        partner_unit,
        progress,
        towards_payload_progress,
        3.0,
        1.0,
        float("inf"),
        1.0,
        0.3,
        2.0,
        0.2,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
    )

    assert torch.allclose(new_towards_payload_progress, torch.tensor([-0.0]))
    assert torch.allclose(towards_payload_reward, torch.tensor([3.0]))
    assert torch.allclose(forward_reward, torch.zeros_like(forward_reward))
    assert torch.allclose(payload_x_penalty, torch.zeros_like(payload_x_penalty))
    assert torch.allclose(progress_reward, torch.tensor([3.0]))


def test_payload_plane_forward_reward_cap_is_applied_in_mjw_runtime() -> None:
    runtime = object.__new__(PayloadPlaneMJWScenarioRuntime)
    device = torch.device("cpu")
    payload_qpos = torch.tensor([[0.1, 0.7, 0.2, 1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
    runtime.bindings = SimpleNamespace(
        units_active_mask=torch.tensor([[True]], device=device, dtype=torch.bool),
        partner_unit=torch.tensor([[[-1]]], device=device, dtype=torch.long),
        qpos=payload_qpos,
    )
    runtime.scenario = SimpleNamespace(
        progress_reward_weight=1.0,
        forward_reward_weight=1.0,
        forward_reward_max_y=0.5,
        potential_reward_discount_factor=1.0,
        payload_radius=0.2,
        towards_payload_reward_weight=0.0,
        towards_payload_goal_radius=0.2,
        payload_centering_penalty_weight=0.0,
        payload_centering_penalty_power=1.0,
        payload_centering_tolerance=0.0,
        units_without_connections_reward_weight=0.0,
        guidance_reward_weight=1.0,
        payload_pos_observable=True,
    )
    runtime.payload_position = torch.zeros((1, 3), device=device, dtype=torch.float32)
    runtime._global_obs = torch.zeros((1, 9), device=device, dtype=torch.float32)
    runtime._payload_qpos_indices = torch.arange(7, device=device, dtype=torch.long)
    runtime._unit_qpos_adr = torch.tensor([0], device=device, dtype=torch.long)
    runtime._xy_offsets = torch.tensor([0, 1], device=device, dtype=torch.long)
    runtime.progress = torch.tensor([0.45], device=device, dtype=torch.float32)
    runtime.towards_payload_progress = torch.tensor([0.0], device=device, dtype=torch.float32)
    runtime._reward_kernel = _compute_payload_plane_reward_kernel

    first = runtime.compute_step_rewards(stable_mask=torch.tensor([True], device=device, dtype=torch.bool))
    assert np.isclose(float(first.info["forward_reward"][0]), 0.05)

    runtime.bindings.qpos[:, :3] = torch.tensor([[0.1, 0.9, 0.2]], device=device, dtype=torch.float32)
    second = runtime.compute_step_rewards(stable_mask=torch.tensor([True], device=device, dtype=torch.bool))
    assert np.isclose(float(second.info["forward_reward"][0]), 0.0)
    assert np.isclose(float(runtime.progress[0]), 0.5)


def test_payload_plane_forward_reward_applies_discount_factor_in_mjw_runtime() -> None:
    progress = torch.tensor([1.0], dtype=torch.float32)

    (
        _new_progress,
        _new_towards_payload_progress,
        progress_reward,
        forward_reward,
        _towards_payload_reward,
        _payload_x_penalty,
        _guidance_reward,
    ) = _compute_payload_plane_reward_kernel(
        unit_xy=torch.zeros((1, 1, 2), dtype=torch.float32),
        payload_position=torch.tensor([[0.0, 1.4, 0.2]], dtype=torch.float32),
        stable_mask=torch.tensor([True], dtype=torch.bool),
        units_active_mask=torch.tensor([[True]], dtype=torch.bool),
        partner_unit=torch.zeros((1, 1, 1), dtype=torch.long),
        progress=progress,
        towards_payload_progress=torch.zeros_like(progress),
        progress_reward_weight=1.0,
        forward_reward_weight=1.5,
        forward_reward_max_y=float("inf"),
        potential_reward_discount_factor=0.9,
        payload_radius=0.2,
        towards_payload_reward_weight=0.0,
        towards_payload_goal_radius=0.2,
        payload_centering_penalty_weight=0.0,
        payload_centering_penalty_power=1.0,
        payload_centering_tolerance=0.0,
        units_without_connections_reward_weight=0.0,
        guidance_reward_weight=1.0,
    )

    assert torch.allclose(forward_reward, torch.tensor([0.39]))
    assert torch.allclose(progress_reward, torch.tensor([0.39]))


def test_dual_payload_plane_reward_kernel_combines_forward_terms_in_mjw_runtime() -> None:
    (
        new_payload_progress,
        new_back_payload_progress,
        _new_towards_payload_progress,
        progress_reward,
        forward_reward,
        towards_payload_reward,
        payload_x_penalty,
        _guidance_reward,
    ) = _compute_dual_payload_plane_reward_kernel(
        unit_xy=torch.zeros((1, 1, 2), dtype=torch.float32),
        payload_position=torch.tensor([[[0.3, 1.4, 0.2], [-0.2, 1.2, 0.2]]], dtype=torch.float32),
        stable_mask=torch.tensor([True], dtype=torch.bool),
        units_active_mask=torch.tensor([[True]], dtype=torch.bool),
        partner_unit=torch.tensor([[[-1]]], dtype=torch.long),
        payload_progress=torch.tensor([[1.0, 1.1]], dtype=torch.float32),
        back_payload_progress=torch.tensor([1.0], dtype=torch.float32),
        towards_payload_progress=torch.zeros((1, 2), dtype=torch.float32),
        progress_reward_weight=1.0,
        forward_reward_weight=2.0,
        forward_reward_max_y=float("inf"),
        potential_reward_discount_factor=1.0,
        lagging_payload_weight=0.75,
        payload_radius=0.2,
        towards_payload_reward_weight=0.0,
        towards_payload_goal_radius=0.2,
        towards_payload_units_per_payload=1,
        payload_centering_penalty_weight=0.5,
        payload_centering_penalty_power=1.0,
        payload_centering_tolerance=0.0,
        units_without_connections_reward_weight=0.0,
        guidance_reward_weight=1.0,
    )

    assert torch.allclose(new_payload_progress, torch.tensor([[1.4, 1.2]]))
    assert torch.allclose(new_back_payload_progress, torch.tensor([1.2]))
    assert torch.allclose(forward_reward, torch.tensor([0.85]))
    assert torch.allclose(towards_payload_reward, torch.zeros_like(towards_payload_reward))
    assert torch.allclose(payload_x_penalty, torch.tensor([-0.25]))
    assert torch.allclose(progress_reward, torch.tensor([0.6]))


def test_dual_payload_plane_towards_reward_kernel_uses_nearest_units_per_payload_in_mjw_runtime() -> None:
    (
        _new_payload_progress,
        _new_back_payload_progress,
        new_towards_payload_progress,
        progress_reward,
        forward_reward,
        towards_payload_reward,
        payload_x_penalty,
        _guidance_reward,
    ) = _compute_dual_payload_plane_reward_kernel(
        unit_xy=torch.tensor([[[0.0, 0.7], [0.0, 1.7], [10.0, 0.7], [10.0, 1.7]]], dtype=torch.float32),
        payload_position=torch.tensor([[[0.0, 1.0, 0.2], [10.0, 1.0, 0.2]]], dtype=torch.float32),
        stable_mask=torch.tensor([True], dtype=torch.bool),
        units_active_mask=torch.tensor([[True, True, True, True]], dtype=torch.bool),
        partner_unit=torch.tensor([[[-1], [-1], [-1], [-1]]], dtype=torch.long),
        payload_progress=torch.tensor([[1.0, 1.0]], dtype=torch.float32),
        back_payload_progress=torch.tensor([1.0], dtype=torch.float32),
        towards_payload_progress=torch.tensor([[-2.0, -4.0]], dtype=torch.float32),
        progress_reward_weight=3.0,
        forward_reward_weight=1.0,
        forward_reward_max_y=float("inf"),
        potential_reward_discount_factor=1.0,
        lagging_payload_weight=0.75,
        payload_radius=0.3,
        towards_payload_reward_weight=2.0,
        towards_payload_goal_radius=0.0,
        towards_payload_units_per_payload=2,
        payload_centering_penalty_weight=0.0,
        payload_centering_penalty_power=1.0,
        payload_centering_tolerance=0.0,
        units_without_connections_reward_weight=0.0,
        guidance_reward_weight=1.0,
    )

    assert torch.allclose(new_towards_payload_progress, torch.tensor([[-0.5, -0.5]]))
    assert torch.allclose(towards_payload_reward, torch.tensor([30.0]))
    assert torch.allclose(forward_reward, torch.zeros_like(forward_reward))
    assert torch.allclose(payload_x_penalty, torch.zeros_like(payload_x_penalty))
    assert torch.allclose(progress_reward, torch.tensor([30.0]))


def test_payload_plane_mjw_reset_sampling_respects_offsets_and_height() -> None:
    runtime = object.__new__(PayloadPlaneMJWScenarioRuntime)
    device = torch.device("cpu")
    runtime.bindings = SimpleNamespace(device=device)
    runtime.scenario = SimpleNamespace(
        payload_offset_x=0.4,
        payload_offset_y=1.2,
        payload_radius=0.2,
    )
    common_reset_batch = MJWCommonResetBatch(
        pool_idx=torch.tensor([0, 1], device=device, dtype=torch.long),
        swarm_start=torch.tensor([[0.25, -0.5, 0.3], [1.0, 2.0, 0.3]], device=device, dtype=torch.float32),
        initial_z_rotation=torch.zeros((2,), device=device, dtype=torch.float32),
    )

    payload_position = runtime._sample_payload_positions(
        common_reset_batch=common_reset_batch,
        rng=torch.Generator(device=device).manual_seed(123),
    )

    expected = torch.tensor([[0.65, 0.7, 0.2], [1.4, 3.2, 0.2]], device=device, dtype=torch.float32)
    assert torch.allclose(payload_position, expected)


def test_payload_plane_presets_expose_payload_pose_global_obs_in_both_backends() -> None:
    mj_scenario = default_mj_payload_plane(reset_settle_time=0.0)
    mjw_scenario = default_mjw_payload_plane()

    assert mj_scenario.get_obs_space()["global_obs"].shape == (9,)
    assert mj_scenario.get_obs_space()["hidden_global_vars"].shape == (0,)
    assert np.isclose(
        mj_scenario.payload_centering_penalty_weight,
        PAYLOAD_PLANE_SCENARIO_KWARGS["payload_centering_penalty_weight"],
    )
    assert np.isclose(
        mj_scenario.payload_centering_tolerance,
        PAYLOAD_PLANE_SCENARIO_KWARGS["payload_centering_tolerance"],
    )
    assert np.isclose(
        mj_scenario.towards_payload_reward_weight,
        PAYLOAD_PLANE_SCENARIO_KWARGS["towards_payload_reward_weight"],
    )
    assert np.isclose(
        mj_scenario.towards_payload_goal_radius,
        PAYLOAD_PLANE_SCENARIO_KWARGS["towards_payload_goal_radius"],
    )
    assert mj_scenario.payload_shape == PAYLOAD_PLANE_SCENARIO_KWARGS["payload_shape"]
    assert mjw_scenario.get_single_observation_space()["global_obs"].shape == (9,)
    assert mjw_scenario.get_single_observation_space()["hidden_global_vars"].shape == (0,)
    assert np.isclose(
        mjw_scenario.payload_centering_penalty_weight,
        PAYLOAD_PLANE_SCENARIO_KWARGS["payload_centering_penalty_weight"],
    )
    assert np.isclose(
        mjw_scenario.payload_centering_tolerance,
        PAYLOAD_PLANE_SCENARIO_KWARGS["payload_centering_tolerance"],
    )
    assert np.isclose(
        mjw_scenario.towards_payload_reward_weight,
        PAYLOAD_PLANE_SCENARIO_KWARGS["towards_payload_reward_weight"],
    )
    assert np.isclose(
        mjw_scenario.towards_payload_goal_radius,
        PAYLOAD_PLANE_SCENARIO_KWARGS["towards_payload_goal_radius"],
    )
    assert mjw_scenario.payload_shape == PAYLOAD_PLANE_SCENARIO_KWARGS["payload_shape"]


def test_dual_payload_plane_presets_expose_two_payload_poses_in_global_obs() -> None:
    mj_scenario = default_mj_dual_payload_plane(reset_settle_time=0.0)
    mjw_scenario = default_mjw_dual_payload_plane(compile_reward_kernel=False)

    assert mj_scenario.get_obs_space()["global_obs"].shape == (18,)
    assert mj_scenario.get_obs_space()["hidden_global_vars"].shape == (0,)
    assert mj_scenario.get_settings()["payload_pos_observable"]
    assert mj_scenario.get_settings()["num_payloads"] == 2
    assert mj_scenario.towards_payload_units_per_payload == 2
    assert mjw_scenario.get_single_observation_space()["global_obs"].shape == (18,)
    assert mjw_scenario.get_single_observation_space()["hidden_global_vars"].shape == (0,)
    assert mjw_scenario.get_settings()["payload_pos_observable"]
    assert mjw_scenario.get_settings()["num_payloads"] == 2
    assert mjw_scenario.towards_payload_units_per_payload == 2


def test_dual_payload_plane_payload_pose_can_be_hidden_in_mj_env() -> None:
    scenario = default_mj_dual_payload_plane(
        reset_settle_time=0.0,
        swarm_start_x=0.25,
        swarm_start_y=-0.5,
        payload_offset_x=(-0.4, 0.6),
        payload_offset_y=(1.2, 1.4),
        payload_pos_observable=False,
    )
    state, connections = scenario.reset_scenario(scenario.dummy_model, scenario.dummy_data, settle=False)
    obs = scenario.get_obs(scenario.dummy_model, scenario.dummy_data, state, connections)

    expected_payload_obs = np.array(
        [
            -0.15, 0.7, scenario.payload_radius, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.85, 0.9, scenario.payload_radius, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ],
        dtype=float,
    )
    assert obs["global_obs"].shape == (0,)
    assert obs["hidden_global_vars"].shape == (18,)
    assert np.allclose(obs["hidden_global_vars"], expected_payload_obs)
    assert scenario.get_obs_space()["global_obs"].shape == (0,)
    assert scenario.get_obs_space()["hidden_global_vars"].shape == (18,)


def test_dual_payload_plane_payload_pose_can_be_hidden_in_mjw_space_and_runtime() -> None:
    scenario = default_mjw_dual_payload_plane(
        compile_reward_kernel=False,
        payload_pos_observable=False,
    )
    obs_space = scenario.get_single_observation_space()

    assert obs_space["global_obs"].shape == (0,)
    assert obs_space["hidden_global_vars"].shape == (18,)
    assert not scenario.get_settings()["payload_pos_observable"]

    runtime = object.__new__(DualPayloadPlaneMJWScenarioRuntime)
    device = torch.device("cpu")
    runtime.scenario = SimpleNamespace(payload_pos_observable=False)
    runtime.bindings = SimpleNamespace(
        qpos=torch.tensor(
            [[
                -0.15, 0.7, 0.3, 1.0, 0.0, 0.0, 0.0,
                0.85, 0.9, 0.3, 1.0, 0.0, 0.0, 0.0,
            ]],
            device=device,
            dtype=torch.float32,
        ),
    )
    runtime.payload_position = torch.zeros((1, 2, 3), device=device, dtype=torch.float32)
    runtime._global_obs = torch.zeros((1, 0), device=device, dtype=torch.float32)
    runtime._hidden_global_obs = torch.zeros((1, 18), device=device, dtype=torch.float32)
    runtime._payload_qpos_indices = torch.tensor(
        [list(range(7)), list(range(7, 14))],
        device=device,
        dtype=torch.long,
    )

    runtime._update_payload_obs()

    expected_payload_obs = torch.tensor(
        [[
            -0.15, 0.7, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
            0.85, 0.9, 0.3, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        ]],
        device=device,
        dtype=torch.float32,
    )
    assert runtime._global_obs.shape == (1, 0)
    assert torch.allclose(runtime._hidden_global_obs, expected_payload_obs)


def test_payload_plane_shape_selects_payload_geom_type_in_both_backends() -> None:
    expected_types = {
        "sphere": mujoco.mjtGeom.mjGEOM_SPHERE,
        "box": mujoco.mjtGeom.mjGEOM_BOX,
        "capsule": mujoco.mjtGeom.mjGEOM_CAPSULE,
    }

    for payload_shape, expected_type in expected_types.items():
        mj_scenario = default_mj_payload_plane(reset_settle_time=0.0, payload_shape=payload_shape)
        mjw_scenario = default_mjw_payload_plane(payload_shape=payload_shape)

        assert _payload_geom_type(mj_scenario.dummy_model) == expected_type
        assert _payload_geom_type(mjw_scenario.build_model()) == expected_type


def test_payload_plane_mjw_recording_camera_is_farther_than_old_tight_default() -> None:
    scenario = default_mjw_payload_plane()

    camera_config = scenario.get_default_recording_camera_config()
    old_distance = max(3.0, min(8.0, scenario.swarm.max_unit_extent * 6.0))

    assert camera_config is not None
    assert camera_config.distance > old_distance
    assert camera_config.distance >= 6.0
    assert camera_config.lookat[1] == scenario.payload_offset_y


def test_payload_plane_mjw_runtime_metadata_hook_owns_payload_indices() -> None:
    scenario = default_mjw_payload_plane()
    host_model = scenario.build_model()

    generic_metadata = build_model_metadata(host_model, scenario)
    runtime_metadata = scenario.build_runtime_metadata(host_model=host_model)

    assert isinstance(runtime_metadata, MJWPayloadPlaneRuntimeMetadata)
    assert runtime_metadata.payload_qpos_indices.shape == (7,)
    assert runtime_metadata.payload_qpos_indices.dtype == np.int64
    assert not hasattr(generic_metadata, "payload_qpos_indices")


def _payload_geom_type(model: mujoco.MjModel) -> int:
    payload_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Payload")
    assert payload_body_id >= 0
    payload_geom_ids = np.flatnonzero(model.geom_bodyid == payload_body_id)
    assert payload_geom_ids.shape == (1,)
    return int(model.geom_type[int(payload_geom_ids[0])])
