from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from swarmbots.mj_env.scenarios.payload_plane_scenario import PayloadPlaneScenario
from swarmbots.mj_env.scenarios.scenario_presets import default_payload_plane as default_mj_payload_plane
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv
from swarmbots.mjw_env.mjw_model_metadata import build_model_metadata
from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWCommonResetBatch
from swarmbots.mjw_env.scenarios.mjw_payload_plane_scenario import MJWPayloadPlaneRuntimeMetadata
from swarmbots.mjw_env.scenarios.mjw_payload_plane_runtime import (
    PayloadPlaneMJWScenarioRuntime,
    _compute_payload_plane_reward_kernel,
)
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_payload_plane as default_mjw_payload_plane


def test_payload_plane_reward_uses_payload_progress_and_x_penalty_in_mj_env() -> None:
    scenario = object.__new__(PayloadPlaneScenario)
    scenario.forward_reward_weight = 1.5
    scenario.forward_reward_max_y = None
    scenario.payload_centering_penalty_weight = 0.5
    scenario.payload_centering_penalty_power = 2.0
    scenario._payload_qpos_indices = np.arange(7, dtype=int)

    state = {"progress": 1.0}
    data = SimpleNamespace(qpos=np.array([0.3, 1.4, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=float))

    reward = scenario.compute_progress_reward(data, state)

    assert np.isclose(state["forward_reward"], 0.4)
    assert np.isclose(state["payload_x_penalty"], -(0.3 ** 2) * 0.5)
    assert np.isclose(reward, (0.4 * 1.5) - ((0.3 ** 2) * 0.5))


def test_payload_plane_reset_exposes_payload_position_as_global_obs_in_mj_env() -> None:
    scenario = default_mj_payload_plane(
        reset_settle_time=0.0,
        swarm_start_x=0.25,
        swarm_start_y=-0.5,
        payload_offset_x=0.4,
        payload_offset_y=1.2,
    )
    env = SwarmBotsEnv(scenario=scenario, episode_length=8)
    obs, _ = env.reset(seed=123)

    assert np.allclose(obs["global_obs"], np.array([0.65, 0.7, 0.2], dtype=float))
    assert obs["hidden_global_vars"].shape == (0,)
    env.close()


def test_payload_plane_forward_reward_cap_is_applied_in_mj_env() -> None:
    scenario = object.__new__(PayloadPlaneScenario)
    scenario.forward_reward_weight = 1.0
    scenario.forward_reward_max_y = 0.5
    scenario.payload_centering_penalty_weight = 0.0
    scenario.payload_centering_penalty_power = 1.0
    scenario._payload_qpos_indices = np.arange(7, dtype=int)

    state = {"progress": 0.45}
    data = SimpleNamespace(qpos=np.array([0.2, 0.7, 0.2, 1.0, 0.0, 0.0, 0.0], dtype=float))

    reward = scenario.compute_progress_reward(data, state)

    assert np.isclose(reward, 0.05)
    assert np.isclose(state["forward_reward"], 0.05)
    assert np.isclose(state["progress"], 0.5)


def test_payload_plane_evaluate_step_reports_weighted_reward_terms_in_mj_env() -> None:
    scenario = object.__new__(PayloadPlaneScenario)
    scenario.num_units = 1
    scenario.forward_reward_weight = 2.0
    scenario.forward_reward_max_y = None
    scenario.payload_centering_penalty_weight = 0.5
    scenario.payload_centering_penalty_power = 1.0
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
    assert np.isclose(state["weighted_payload_x_penalty"], -0.6)
    assert np.isclose(state["weighted_guidance_reward"], 0.0)
    assert np.isclose(state["weighted_progress_reward"], 1.2)
    assert np.isclose(state["reward_terms"]["forward"], 1.8)
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
    assert "forward_reward" in info
    assert "guidance_reward" in info
    env.close()


def test_payload_plane_reward_uses_payload_progress_and_x_penalty_in_mjw_runtime() -> None:
    runtime = object.__new__(PayloadPlaneMJWScenarioRuntime)
    device = torch.device("cpu")
    runtime.bindings = SimpleNamespace(
        units_active_mask=torch.tensor([[True]], device=device, dtype=torch.bool),
        partner_unit=torch.tensor([[[-1]]], device=device, dtype=torch.long),
    )
    runtime.scenario = SimpleNamespace(
        progress_reward_weight=1.0,
        forward_reward_weight=1.5,
        forward_reward_max_y=None,
        payload_centering_penalty_weight=0.5,
        payload_centering_penalty_power=2.0,
        units_without_connections_reward_weight=0.0,
        guidance_reward_weight=1.0,
    )
    runtime.payload_position = torch.zeros((1, 3), device=device, dtype=torch.float32)
    runtime.progress = torch.tensor([1.0], device=device, dtype=torch.float32)
    runtime._reward_kernel = _compute_payload_plane_reward_kernel
    runtime._get_payload_position = lambda: torch.tensor([[0.3, 1.4, 0.2]], device=device, dtype=torch.float32)

    result = runtime.compute_step_rewards(stable_mask=torch.tensor([True], device=device, dtype=torch.bool))

    assert np.isclose(float(result.info["forward_reward"][0]), 0.6)
    assert np.isclose(float(result.info["payload_x_penalty"][0]), -((0.3 ** 2) * 0.5))
    assert np.isclose(float(result.reward[0]), 0.6 - ((0.3 ** 2) * 0.5))
    assert np.isclose(float(runtime.progress[0]), 1.4)


def test_payload_plane_forward_reward_cap_is_applied_in_mjw_runtime() -> None:
    runtime = object.__new__(PayloadPlaneMJWScenarioRuntime)
    device = torch.device("cpu")
    runtime.bindings = SimpleNamespace(
        units_active_mask=torch.tensor([[True]], device=device, dtype=torch.bool),
        partner_unit=torch.tensor([[[-1]]], device=device, dtype=torch.long),
    )
    runtime.scenario = SimpleNamespace(
        progress_reward_weight=1.0,
        forward_reward_weight=1.0,
        forward_reward_max_y=0.5,
        payload_centering_penalty_weight=0.0,
        payload_centering_penalty_power=1.0,
        units_without_connections_reward_weight=0.0,
        guidance_reward_weight=1.0,
    )
    runtime.payload_position = torch.zeros((1, 3), device=device, dtype=torch.float32)
    runtime.progress = torch.tensor([0.45], device=device, dtype=torch.float32)
    runtime._reward_kernel = _compute_payload_plane_reward_kernel

    runtime._get_payload_position = lambda: torch.tensor([[0.1, 0.7, 0.2]], device=device, dtype=torch.float32)
    first = runtime.compute_step_rewards(stable_mask=torch.tensor([True], device=device, dtype=torch.bool))
    assert np.isclose(float(first.info["forward_reward"][0]), 0.05)

    runtime._get_payload_position = lambda: torch.tensor([[0.1, 0.9, 0.2]], device=device, dtype=torch.float32)
    second = runtime.compute_step_rewards(stable_mask=torch.tensor([True], device=device, dtype=torch.bool))
    assert np.isclose(float(second.info["forward_reward"][0]), 0.0)
    assert np.isclose(float(runtime.progress[0]), 0.5)


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


def test_payload_plane_presets_expose_payload_global_obs_in_both_backends() -> None:
    mj_scenario = default_mj_payload_plane(reset_settle_time=0.0)
    mjw_scenario = default_mjw_payload_plane()

    assert mj_scenario.get_obs_space()["global_obs"].shape == (3,)
    assert mj_scenario.get_obs_space()["hidden_global_vars"].shape == (0,)
    assert mjw_scenario.get_single_observation_space()["global_obs"].shape == (3,)
    assert mjw_scenario.get_single_observation_space()["hidden_global_vars"].shape == (0,)


def test_payload_plane_mjw_runtime_metadata_hook_owns_payload_indices() -> None:
    scenario = default_mjw_payload_plane()
    host_model = scenario.build_model()

    generic_metadata = build_model_metadata(host_model, scenario)
    runtime_metadata = scenario.build_runtime_metadata(host_model=host_model)

    assert isinstance(runtime_metadata, MJWPayloadPlaneRuntimeMetadata)
    assert runtime_metadata.payload_qpos_indices.shape == (7,)
    assert runtime_metadata.payload_qpos_indices.dtype == np.int64
    assert not hasattr(generic_metadata, "payload_qpos_indices")
