from __future__ import annotations

from contextlib import ExitStack
from typing import Any
from unittest.mock import Mock

import mujoco
import mujoco_warp as mjw
import numpy as np
import pytest
import torch
import warp as wp

from swarmbots import make_env
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.parameter_distributions import FloatOrDistParams, NormalDistParams
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


def _small_swarm(*, num_units: int = 2) -> MJWPreConnectedUnitLocationsConfig:
    return MJWPreConnectedUnitLocationsConfig(num_units=num_units, pool_seeds=(42000,), z_pos=3.0)


def _make_payload_capacity_env(*, device: str, capacity: int | None) -> MJWSwarmBotsVectorEnv:
    return make_env(
        "SwarmBots-MultiPayloadGoal-v0", num_envs=1, device=device,
        nconmax=capacity, compile_tensor_operations=False, settled_reset_buffer_size=0,
        scenario_kwargs={"compile_reward_kernel": False},
    )


def _contacts_at_geometry_state(
    env: MJWSwarmBotsVectorEnv, geometry_state: dict[str, torch.Tensor],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with wp.ScopedDevice(env._wp_device), env._warp_stream_scope():
        for name, value in geometry_state.items():
            wp.to_torch(getattr(env._data, name)).copy_(value)
        mjw.kinematics(env._model, env._data)
        mjw.collision(env._model, env._data)
        contact_count = int(env._data.nacon.numpy()[0])
        assert 0 < contact_count <= env._data.naconmax
        contacts = env._data.contact
        contact_keys = np.column_stack((
            np.sort(contacts.geom.numpy()[:contact_count], axis=1),
            contacts.geomcollisionid.numpy()[:contact_count],
        ))
        # GPU contact writes use atomics; the pair and local contact ID identify
        # the same contact independently of its slot in the output buffer.
        order = np.lexsort((contact_keys[:, 2], contact_keys[:, 1], contact_keys[:, 0]))
        return (
            contact_keys[order],
            contacts.pos.numpy()[:contact_count][order],
            contacts.dist.numpy()[:contact_count][order],
        )


def test_negative_first_episode_length_is_rejected_before_building_physics() -> None:
    scenario = Mock()
    scenario.build_model.side_effect = AssertionError("invalid limits must not allocate physics")
    with pytest.raises(ValueError, match="first_episode_length must be >= 0"):
        MJWSwarmBotsVectorEnv(scenario, num_envs=1, episode_length=3, first_episode_length=-1, device="cpu")
    scenario.build_model.assert_not_called()


@pytest.mark.integration
def test_default_multi_payload_cpu_dynamics_match_a_generous_capacity() -> None:
    trajectories = []
    for capacity in (None, 128):
        env = _make_payload_capacity_env(device="cpu", capacity=capacity)
        try:
            env.reset(seed=0)
            actions = {key: torch.zeros(space.shape, device=env.device) for key, space in env.action_space.items()}
            trajectory = []
            for _ in range(5):
                _, rewards, terminated, truncated, _ = env.step(actions)
                assert int(env._data.nacon.numpy()[0]) <= env._data.naconmax
                trajectory.append((env._qpos.clone(), rewards.clone(), terminated.clone(), truncated.clone()))
            trajectories.append(trajectory)
        finally:
            env.close()
    torch.testing.assert_close(trajectories[0], trajectories[1], rtol=1e-4, atol=1e-5)


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("seed", [2, 4])
def test_default_multi_payload_contacts_match_a_generous_capacity_at_identical_poses(device: str, seed: int) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    with ExitStack() as cleanup:
        env = _make_payload_capacity_env(device=device, capacity=None)
        cleanup.callback(env.close)
        reference = _make_payload_capacity_env(device=device, capacity=128)
        cleanup.callback(reference.close)
        env.reset(seed=seed)
        reference.reset(seed=seed)
        actions = {key: torch.zeros(space.shape, device=reference.device) for key, space in reference.action_space.items()}
        for _ in range(5):
            reference.step(actions)
            geometry_state = {
                name: wp.to_torch(getattr(reference._data, name)).clone()
                for name in ("qpos", "mocap_pos", "mocap_quat")
            }
            expected_keys, expected_positions, expected_distances = _contacts_at_geometry_state(reference, geometry_state)
            keys, positions, distances = _contacts_at_geometry_state(env, geometry_state)
            np.testing.assert_array_equal(keys, expected_keys)
            np.testing.assert_allclose(positions, expected_positions, rtol=1e-6, atol=1e-6)
            np.testing.assert_allclose(distances, expected_distances, rtol=1e-6, atol=1e-6)


@pytest.mark.integration
def test_mjw_close_is_idempotent_even_with_a_pending_step(monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(
        "SwarmBots-MoveToGoal-v0", num_envs=1, device="cpu", compile_tensor_operations=False,
        scenario_kwargs=dict(unit_start_locations=_small_swarm(num_units=1), reset_settle_time=0, compile_reward_kernel=False),
    )
    try:
        env.reset(seed=7)
        actions = {key: torch.zeros(space.shape) for key, space in env.action_space.items()}
        env.begin_step(actions)
        assert env.has_pending_step
        close_recorder = Mock(wraps=env._live_episode_recorder.close)
        monkeypatch.setattr(env._live_episode_recorder, "close", close_recorder)
        env.close()
        assert env.closed
        assert not env.has_pending_step
        env.close()
        close_recorder.assert_called_once_with()
    finally:
        env.close()


@pytest.mark.integration
def test_small_radius_population_without_full_swarms_retains_every_model_slot() -> None:
    num_units = 3
    config = MJWPreConnectedUnitLocationsConfig(
        num_units=num_units, num_unit_probs={1: 1.0, num_units: 0.0}, max_radius=0.05,
        pool_seeds=(42000, 42001), z_pos=3.0,
    )
    env = make_env(
        "SwarmBots-MoveToGoal-v0", num_envs=2, device="cpu", compile_tensor_operations=False,
        scenario_kwargs=dict(unit_start_locations=config, reset_settle_time=0, compile_reward_kernel=False),
    )
    try:
        observations, _ = env.reset(seed=7)
        assert observations["local_obs"].shape[1] == num_units
        assert observations["agent_mask"].sum(dim=1).tolist() == [1, 1]
        assert env._pool.active_mask.sum(dim=1).tolist() == [1, 1]
        root_bodies = [env._host_model.body(f"Unit{index}--main_body") for index in range(num_units)]
        assert len({int(body.id) for body in root_bodies}) == num_units
        assert all(env._host_model.jnt_type[int(body.jntadr[0])] == mujoco.mjtJoint.mjJNT_FREE for body in root_bodies)
        actions = {key: torch.zeros(space.shape) for key, space in env.action_space.items()}
        next_observations, rewards, terminated, _, _ = env.step(actions)
        assert torch.isfinite(rewards).all()
        assert not terminated.any()
        assert next_observations["agent_mask"].sum(dim=1).tolist() == [1, 1]
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("payload_radius", [None, 0.4])
@pytest.mark.parametrize("spawn_x", [
    0.0, -10_000.0, 10_000.0,
    pytest.param(NormalDistParams(mean=10_000.0, std=100.0), id="sampled_x"),
])
@pytest.mark.parametrize("device", [
    "cpu",
    pytest.param("cuda", marks=[
        pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
    ]),
])
def test_capsule_payload_reset_positions_have_no_penetrating_payload_contacts(
    payload_radius: float | None, spawn_x: FloatOrDistParams, device: str,
) -> None:
    overrides: dict[str, Any] = dict(
        payload_shape="capsule", max_payloads=4, active_payload_count_probs={4: 1.0},
        swarm_start_x=spawn_x,
        unit_start_locations=_small_swarm(num_units=1), reset_settle_time=0, compile_reward_kernel=False,
    )
    if payload_radius is not None:
        overrides.update(payload_radius=payload_radius, payload_spawn_margin=2.0 * payload_radius)
    env = make_env(
        "SwarmBots-MultiPayloadGoal-v0", num_envs=2, device=device, compile_tensor_operations=False,
        scenario_kwargs=overrides,
    )
    try:
        env.reset(seed=7)
        host_data = mujoco.MjData(env._host_model)
        payload_geoms = {
            int(env._host_model.body_geomadr[env._host_model.body(f"Payload{index}").id])
            for index in range(4)
        }
        for qpos in env._qpos.cpu().numpy():
            host_data.qpos[:] = qpos
            mujoco.mj_forward(env._host_model, host_data)
            for contact in host_data.contact:
                if all(int(geom) in payload_geoms for geom in contact.geom):
                    assert contact.dist >= -1e-6
        payload_indices = env._scenario_runtime_metadata.payload_qpos_indices
        positions = env._qpos[:, payload_indices[:, :3]]
        ordered_x = positions[:, :, 0].sort(dim=1).values
        assert ((ordered_x[:, 1:] - ordered_x[:, :-1]) > 4.0 * env.scenario.payload_radius).all()
    finally:
        env.close()


def _snapshot_step(env: MJWSwarmBotsVectorEnv, result: tuple[Any, ...]) -> dict[str, Any]:
    observations, rewards, terminated, truncated, infos = result
    return {
        "observations": {key: value.clone() for key, value in observations.items()},
        "rewards": rewards.clone(), "terminated": terminated.clone(), "truncated": truncated.clone(),
        "final_obs": {key: value.clone() for key, value in infos.get("final_obs", {}).items()},
        "qpos": env._qpos.clone(), "qvel": env._qvel.clone(), "xpos": env._xpos.clone(),
        "eq_active": env._eq_active.clone(), "partner_unit": env.partner_unit.clone(),
        "current_step": env.current_step.clone(),
    }


@pytest.mark.integration
@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("compile_operations", [False, True])
@pytest.mark.parametrize("capture_physics", [False, True])
def test_custom_cuda_stream_matches_default_through_reset_deferred_steps_and_autoreset(
    monkeypatch: pytest.MonkeyPatch, compile_operations: bool, capture_physics: bool,
) -> None:
    if not capture_physics:
        monkeypatch.setattr("swarmbots.mjw_env.mjw_swarm_bots_vector_env._capture_step_graph", lambda *args: None)
    torch.compiler.reset()
    stream = torch.cuda.Stream()

    def make_stream_env() -> MJWSwarmBotsVectorEnv:
        return make_env(
            "SwarmBots-MoveToGoal-v0", num_envs=2, device="cuda", episode_length=3,
            first_episode_lengths=[1, 3], compile_tensor_operations=compile_operations,
            scenario_kwargs=dict(
                unit_start_locations=_small_swarm(), reset_settle_time=0, compile_reward_kernel=compile_operations,
                action_repeat=1, continuous_connector_actions=True, disconnect_potential_threshold=0.5,
                connection_dist_threshold=0.05,
            ),
        )

    reference = make_stream_env()
    actual: MJWSwarmBotsVectorEnv | None = None
    try:
        with torch.cuda.stream(stream):
            torch.cuda._sleep(5_000_000)
            actual = make_stream_env()
        stream.synchronize()
        assert (reference._physics_graph is not None) is capture_physics
        assert (actual._physics_graph is not None) is capture_physics
        reference_obs, _ = reference.reset(seed=7)
        reference_obs = {key: value.clone() for key, value in reference_obs.items()}
        with torch.cuda.stream(stream):
            torch.cuda._sleep(5_000_000)
            actual_obs, _ = actual.reset(seed=7)
            actual_obs = {key: value.clone() for key, value in actual_obs.items()}
        stream.synchronize()
        torch.testing.assert_close(actual_obs, reference_obs, atol=1e-5, rtol=1e-5)

        for step in range(3):
            reference_actions = {
                "actuators": torch.full(reference.action_space["actuators"].shape, 0.2, device="cuda"),
                "connectors": torch.full(reference.action_space["connectors"].shape, -1.0 if step == 0 else 1.0, device="cuda"),
            }
            reference._qvel[:, 0] = 2.0
            reference.begin_step(reference_actions)
            expected = _snapshot_step(reference, reference.end_step())
            with torch.cuda.stream(stream):
                torch.cuda._sleep(5_000_000)
                actual_actions = {key: value.clone() for key, value in reference_actions.items()}
                actual._qvel[:, 0] = 2.0
                actual.begin_step(actual_actions)
                assert actual.has_pending_step
                torch.cuda._sleep(5_000_000)
                observed = _snapshot_step(actual, actual.end_step())
            stream.synchronize()
            torch.testing.assert_close(observed, expected, atol=1e-5, rtol=1e-5)
            if step == 0:
                assert observed["truncated"].tolist() == [True, False]
                assert (observed["partner_unit"][1] < 0).all()
                mask = torch.tensor([True, False], device="cuda")
                reference_obs, _ = reference.reset(seed=8, options={"reset_mask": mask})
                reference_obs = {key: value.clone() for key, value in reference_obs.items()}
                with torch.cuda.stream(stream):
                    torch.cuda._sleep(5_000_000)
                    actual_obs, _ = actual.reset(seed=8, options={"reset_mask": mask})
                    actual_obs = {key: value.clone() for key, value in actual_obs.items()}
                stream.synchronize()
                torch.testing.assert_close(actual_obs, reference_obs, atol=1e-5, rtol=1e-5)
            if step == 2:
                assert observed["truncated"].tolist() == [False, True]
    finally:
        if actual is not None:
            actual.close()
        reference.close()
        torch.compiler.reset()
