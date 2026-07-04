from __future__ import annotations

from types import SimpleNamespace

import mujoco
import numpy as np
import pytest

from swarmbots.mj_env.scenarios.multi_payload_goal_scenario import MultiPayloadGoalScenario
from swarmbots.mj_env.scenarios.scenario_presets import default_multi_payload_goal
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv
from swarmbots.scenario_presets.multi_payload_goal import PAYLOAD_GOAL_IDENTITY_ROT6D


def _import_mjw_multi_payload_goal() -> tuple[object, object, object, object]:
    try:
        import torch
    except Exception as error:
        pytest.skip(f"MJW multi-payload goal test requires working torch/MJW imports: {error!r}")
    from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_runtime import (
        _compute_multi_payload_goal_reward_kernel,
    )
    from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_scenario import (
        MJWMultiPayloadGoalRuntimeMetadata,
    )
    from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_multi_payload_goal as default_mjw

    return torch, default_mjw, MJWMultiPayloadGoalRuntimeMetadata, _compute_multi_payload_goal_reward_kernel


def test_multi_payload_goal_reset_samples_active_payloads_goals_and_obs() -> None:
    scenario = default_multi_payload_goal(
        reset_settle_time=0.0,
        max_payloads=4,
        active_payload_count_probs={2: 1.0},
        payload_spawn_y=1.0,
        payload_spawn_margin=0.8,
        goal_rect=(-1.0, 1.0, 2.0, 3.0),
    )
    env = SwarmBotsEnv(scenario=scenario, episode_length=8)
    obs, _ = env.reset(seed=123)

    payload_records = obs["global_obs"].reshape(4, 12)
    active_mask = env.scenario_state["active_payload_mask"]
    goal_positions = env.scenario_state["goal_positions"]
    spawn_positions = env.scenario_state["payload_spawn_position"]
    swarm_start_location = env.scenario_state["swarm_start_location"]

    assert int(active_mask.sum()) == 2
    assert payload_records[:, 0].tolist() == active_mask.astype(float).tolist()
    relative_spawn_x = spawn_positions[active_mask, 0] - swarm_start_location[0]
    assert np.allclose(np.sort(relative_spawn_x), np.array([-0.4, 0.4]))
    assert np.allclose(spawn_positions[active_mask, 1], swarm_start_location[1] + 1.0)
    inactive_payload_records = payload_records[~active_mask]
    assert np.all(inactive_payload_records[:, 0] == 0.0)
    assert np.all(inactive_payload_records[:, 1:4] == 0.0)
    assert np.allclose(inactive_payload_records[:, 4:10], np.asarray(PAYLOAD_GOAL_IDENTITY_ROT6D))
    assert np.all(inactive_payload_records[:, 10:12] == 0.0)
    assert np.all(goal_positions[:, 0] >= -1.0)
    assert np.all(goal_positions[:, 0] <= 1.0)
    assert np.all(goal_positions[:, 1] >= 2.0)
    assert np.all(goal_positions[:, 1] <= 3.0)
    assert np.allclose(payload_records[active_mask, 10:12], goal_positions[active_mask])
    env.close()


def test_multi_payload_goal_active_payload_subset_is_not_prefix_locked() -> None:
    scenario = default_multi_payload_goal(
        reset_settle_time=0.0,
        max_payloads=5,
        active_payload_count_probs={3: 1.0},
    )

    active_masks = np.stack([scenario._sample_active_payload_mask() for _ in range(100)], axis=0)

    assert np.all(active_masks.sum(axis=1) == 3)
    assert np.any(~active_masks[:, :3])
    assert np.any(active_masks[:, 3:])


def test_multi_payload_goal_goal_mocaps_follow_active_goals() -> None:
    scenario = default_multi_payload_goal(
        reset_settle_time=0.0,
        max_payloads=3,
        active_payload_count_probs={2: 1.0},
        goal_rect=(-1.0, 1.0, 2.0, 3.0),
    )
    state, _connections = scenario.reset_scenario(scenario.dummy_model, scenario.dummy_data, settle=False)

    for payload_idx in np.flatnonzero(state["active_payload_mask"]):
        goal_body_id = mujoco.mj_name2id(
            scenario.dummy_model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"PayloadGoal{payload_idx}",
        )
        mocap_id = int(scenario.dummy_model.body_mocapid[goal_body_id])
        assert np.allclose(scenario.dummy_data.mocap_pos[mocap_id, :2], state["goal_positions"][payload_idx])

    for payload_idx in np.flatnonzero(~state["active_payload_mask"]):
        inactive_goal_body_id = mujoco.mj_name2id(
            scenario.dummy_model,
            mujoco.mjtObj.mjOBJ_BODY,
            f"PayloadGoal{payload_idx}",
        )
        inactive_mocap_id = int(scenario.dummy_model.body_mocapid[inactive_goal_body_id])
        assert np.allclose(scenario.dummy_data.mocap_pos[inactive_mocap_id], scenario.inactive_area_location)


def test_multi_payload_goal_success_requires_all_active_payloads_inside_goals() -> None:
    scenario = object.__new__(MultiPayloadGoalScenario)
    scenario.max_payloads = 3
    scenario.payload_radius = 0.3
    scenario.goal_radius = 0.2
    scenario._payload_qpos_indices = np.array(
        [
            np.arange(7, dtype=int),
            np.arange(7, 14, dtype=int),
            np.arange(14, 21, dtype=int),
        ],
    )
    state = {
        "active_payload_mask": np.array([True, True, False], dtype=bool),
        "goal_positions": np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 10.0]], dtype=float),
    }
    data = SimpleNamespace(
        qpos=np.array(
            [
                1.1, 2.0, 0.3, 1.0, 0.0, 0.0, 0.0,
                3.0, 4.19, 0.3, 1.0, 0.0, 0.0, 0.0,
                100.0, 100.0, 0.3, 1.0, 0.0, 0.0, 0.0,
            ],
            dtype=float,
        ),
    )

    assert scenario._compute_success_termination(data, state)

    data.qpos[7:9] = np.array([3.0, 4.21], dtype=float)
    assert not scenario._compute_success_termination(data, state)


def test_multi_payload_goal_evaluate_step_terminates_with_success_reward() -> None:
    scenario = object.__new__(MultiPayloadGoalScenario)
    scenario.max_payloads = 3
    scenario.payload_radius = 0.3
    scenario.goal_radius = 0.2
    scenario.forward_reward_weight = 3.0
    scenario.success_reward = 5.0
    scenario.potential_reward_discount_factor = 1.0
    scenario.reward_weights = {
        "progress_reward_weight": 2.0,
        "guidance_reward_weight": 4.0,
        "units_without_connections_reward_weight": 0.0,
    }
    scenario.num_units = 1
    scenario._payload_qpos_indices = np.array(
        [
            np.arange(7, dtype=int),
            np.arange(7, 14, dtype=int),
            np.arange(14, 21, dtype=int),
        ],
    )
    state = {
        "active_payload_mask": np.array([True, True, False], dtype=bool),
        "goal_positions": np.array([[1.0, 2.0], [3.0, 4.0], [10.0, 10.0]], dtype=float),
        "payload_progress": np.zeros(3, dtype=float),
    }
    data = SimpleNamespace(
        qpos=np.array(
            [
                1.1, 2.0, 0.3, 1.0, 0.0, 0.0, 0.0,
                3.0, 4.19, 0.3, 1.0, 0.0, 0.0, 0.0,
                100.0, 100.0, 0.3, 1.0, 0.0, 0.0, 0.0,
            ],
            dtype=float,
        ),
    )
    connections = SimpleNamespace(get_is_active_mask=lambda: np.array([[False]], dtype=bool))

    reward, terminated = scenario.evaluate_step(
        action={"actuators": np.zeros((0, 0), dtype=float), "connectors": np.zeros((0, 0), dtype=bool)},
        model=None,
        data=data,
        state=state,
        connections=connections,
    )

    assert terminated
    assert state["success"]
    assert state["goal_success_reward"] == 5.0
    assert state["weighted_goal_success_reward"] == 10.0
    assert state["reward_terms"]["success"] == 10.0
    assert reward == 10.0


def test_multi_payload_goal_preset_exposes_fixed_width_payload_records() -> None:
    scenario = default_multi_payload_goal(
        reset_settle_time=0.0,
        max_payloads=3,
        active_payload_count_probs={3: 1.0},
    )

    assert scenario.get_obs_space()["global_obs"].shape == (36,)
    assert scenario.get_obs_space()["hidden_global_vars"].shape == (0,)
    settings = scenario.get_settings()
    assert settings["scenario_type"] == "multi_payload_goal"
    assert settings["num_payloads"] == 3
    assert settings["global_obs_layout"] == "multi_payload_goal"


def test_mjw_multi_payload_goal_preset_exposes_fixed_width_payload_records_and_metadata() -> None:
    _torch, default_mjw, metadata_cls, _reward_kernel = _import_mjw_multi_payload_goal()

    scenario = default_mjw(
        reset_settle_time=0.0,
        max_payloads=3,
        active_payload_count_probs={3: 1.0},
        compile_reward_kernel=False,
    )
    obs_space = scenario.get_single_observation_space()
    model = scenario.build_model()
    runtime_metadata = scenario.build_runtime_metadata(host_model=model)

    assert obs_space["global_obs"].shape == (36,)
    assert obs_space["hidden_global_vars"].shape == (0,)
    assert scenario.get_settings()["scenario_type"] == "multi_payload_goal"
    assert isinstance(runtime_metadata, metadata_cls)
    assert runtime_metadata.payload_qpos_indices.shape == (3, 7)
    assert runtime_metadata.goal_mocap_ids.shape == (3,)
    assert np.all(runtime_metadata.goal_mocap_ids >= 0)


def test_mjw_multi_payload_goal_reward_kernel_terminates_when_all_active_payloads_reach_goals() -> None:
    torch, _default_mjw, _metadata_cls, reward_kernel = _import_mjw_multi_payload_goal()

    payload_position = torch.tensor([[[0.0, 1.0, 0.3], [2.0, 3.0, 0.3], [10.0, 10.0, 0.3]]])
    goal_position = torch.tensor([[[0.0, 1.0], [2.1, 3.0], [0.0, 0.0]]])
    active_payload_mask = torch.tensor([[True, True, False]])
    stable_mask = torch.tensor([True])
    units_active_mask = torch.tensor([[True, True]])
    partner_unit = torch.full((1, 2, 1), -1, dtype=torch.long)
    payload_progress = torch.tensor([[-1.0, -1.0, 0.0]])

    (
        new_payload_progress,
        progress_reward,
        forward_reward,
        goal_success_reward,
        guidance_reward,
        success,
    ) = reward_kernel(
        payload_position,
        goal_position,
        active_payload_mask,
        stable_mask,
        units_active_mask,
        partner_unit,
        payload_progress,
        2.0,
        3.0,
        1.0,
        0.3,
        0.2,
        5.0,
        -0.5,
        4.0,
    )

    assert torch.allclose(new_payload_progress, torch.tensor([[-0.0, -0.0, 0.0]]))
    assert torch.allclose(forward_reward, torch.tensor([12.0]))
    assert torch.allclose(goal_success_reward, torch.tensor([10.0]))
    assert torch.allclose(progress_reward, torch.tensor([22.0]))
    assert torch.allclose(guidance_reward, torch.tensor([-2.0]))
    assert bool(success[0])
