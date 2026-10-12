from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import mujoco
import pytest
import torch

from swarmbots import ALL_BENCHMARK_IDS, make_env, make_scenario
from swarmbots.mjw_env.mjw_torch_utils import sample_float_or_dist
from swarmbots.mjw_env.parameter_distributions import TruncatedNormalDistParams, UniformDistParams, eval_truncated_normal
from swarmbots.mjw_env.scenarios.mjw_payload_step_runtime import _compute_payload_step_reward_extras_kernel
from swarmbots.mjw_env.scenarios.mjw_bridge_runtime import _compute_bridge_reward_kernel
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_episode_limit_is_rejected_before_scenario_creation(limit, monkeypatch):
    factory = Mock(side_effect=AssertionError("Scenario must not be constructed"))
    monkeypatch.setattr("swarmbots.benchmark.registry.make_scenario", factory)
    with pytest.raises(ValueError, match="episode_length > 0"):
        make_env("SwarmBots-MoveToGoal-v0", num_envs=1, episode_length=limit)
    factory.assert_not_called()


def test_dual_payload_broadcasts_distribution_offsets():
    distribution = UniformDistParams(1.2, 1.4)
    scenario = make_scenario(
        "SwarmBots-DualPayloadPlane-v0", payload_offset_x=distribution,
        payload_offset_y=distribution, compile_reward_kernel=False,
    )
    assert scenario.payload_offset_x == (distribution, distribution)
    assert scenario.payload_offset_y == (distribution, distribution)


def test_payload_step_success_requires_position_over_the_step():
    positions = torch.tensor([
        [3., 2.6, 0.25 * 2**0.5],  # Tilted cube beside the step.
        [0., 5., 0.35],           # Beyond the finite platform.
        [0., 2.6, 0.35],          # On the platform at the goal.
        [0., 2.6, 0.25],          # Too low.
    ])
    result = _compute_payload_step_reward_extras_kernel(
        positions, torch.ones(4, dtype=torch.bool), torch.zeros(4), torch.zeros(4, dtype=torch.bool),
        1.5, 0.1, 4., 3., 0.25, 4., 0.7, 2.5, 0.32, 5., 1., 0.99,
    )
    assert result[-1].tolist() == [False, False, True, False]
    torch.testing.assert_close(result[-2], torch.tensor([0., 0., 5., 0.]))


@pytest.mark.parametrize("positions, climbed", [
    ([(3., 1.4, .35), (3., 1.5, .35), (3., 1., .25)], False),
    ([(0., 1.4, .35), (0., 1.5, .25), (0., 1., .25)], False),
    ([(0., 1.4, .35), (0., 1.6, .35), (0., 2., .35)], True),
    ([(0., 4.6, .35)], False),
])
def test_payload_step_climb_reward_requires_an_actual_ascent(
    positions: list[tuple[float, float, float]], climbed: bool,
) -> None:
    scenario = make_scenario("SwarmBots-PayloadStep-v0", compile_reward_kernel=False)
    potential = torch.zeros(1)
    done = torch.zeros(1, dtype=torch.bool)
    total_reward = torch.zeros(1)
    for position in positions:
        potential, done, height_reward, _, success = _compute_payload_step_reward_extras_kernel(
            torch.tensor([position]), torch.ones(1, dtype=torch.bool), potential, done,
            scenario.step_start_y, scenario.step_height, scenario.step_width, scenario.step_length,
            scenario.payload_radius, scenario.payload_step_height_reward_weight,
            scenario.payload_step_height_reward_distance, scenario.payload_success_y,
            scenario.payload_radius + scenario.step_height - scenario.payload_success_height_tolerance,
            scenario.payload_success_reward, scenario.progress_reward_weight,
            scenario.potential_reward_discount_factor,
        )
        total_reward += height_reward
        assert not success.any()
    assert done.item() is climbed
    assert (total_reward.item() > 0.0) is climbed


@pytest.mark.parametrize("shape", ["box", "sphere", "capsule"])
@pytest.mark.parametrize("radius", [.3, .6])
def test_dual_payload_default_offsets_keep_reset_geometry_separate(shape: str, radius: float) -> None:
    scenario = make_scenario(
        "SwarmBots-DualPayloadPlane-v0", payload_shape=shape, payload_radius=radius,
        unit_start_locations=MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=[42000]),
        compile_reward_kernel=False,
    )
    model = scenario.build_model()
    data = mujoco.MjData(model)
    payload_indices = scenario.build_runtime_metadata(host_model=model).payload_qpos_indices
    payload_bodies = {model.body("Payload0").id, model.body("Payload1").id}
    for index in range(2):
        data.qpos[payload_indices[index, :3]] = (
            scenario.payload_offset_x[index], scenario.payload_offset_y[index], scenario.payload_radius,
        )
    mujoco.mj_forward(model, data)
    for contact in data.contact:
        if all(int(model.geom_bodyid[geom]) in payload_bodies for geom in contact.geom):
            assert contact.dist >= 0.0


@pytest.mark.integration
@pytest.mark.parametrize("settle_time", [0.0, .01])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_payload_step_side_spawns_do_not_complete_the_climb_on_reset_or_autoreset(
    settle_time: float, device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    env = make_env(
        "SwarmBots-PayloadStep-v0", num_envs=1, device=device, episode_length=1,
        compile_tensor_operations=False,
        scenario_kwargs=dict(
            unit_start_locations=MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=[42000], z_pos=3.),
            compile_reward_kernel=False, reset_settle_time=settle_time,
            payload_offset_x=3., payload_offset_y=1.6,
        ),
    )
    try:
        env.reset(seed=7)
        assert not env._scenario_runtime.payload_step_height_done.any()
        actions = {key: torch.zeros(space.shape, device=env.device) for key, space in env.action_space.items()}
        _, _, _, truncated, infos = env.step(actions)
        assert truncated.all()
        assert not env._scenario_runtime.payload_step_height_done.any()
        assert not infos["payload_step_height_reward"].any()
    finally:
        env.close()


@pytest.mark.parametrize("benchmark_id", ["SwarmBots-Climb-v0", "SwarmBots-MultiPayloadGoal-v0"])
@pytest.mark.parametrize("radius", [0., -1., 1e-7, float("nan"), float("inf"), -float("inf")])
def test_goal_radius_is_finite_and_large_enough_for_the_visual_geometry(benchmark_id, radius):
    with pytest.raises(ValueError, match="finite goal_radius >= 1e-06"):
        make_scenario(benchmark_id, goal_radius=radius, compile_reward_kernel=False)


@pytest.mark.parametrize("benchmark_id", ["SwarmBots-Climb-v0", "SwarmBots-MultiPayloadGoal-v0"])
def test_minimum_goal_radius_builds_a_model(benchmark_id):
    cfg = MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=[42000])
    scenario = make_scenario(
        benchmark_id, goal_radius=1e-6, unit_start_locations=cfg, compile_reward_kernel=False,
    )
    assert scenario.goal_radius == 1e-6
    assert scenario.build_model().ngeom > 0


@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
@pytest.mark.parametrize("scale", [0., -1., float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("settle_time", [0., 1.])
def test_invalid_settling_scale_is_rejected_even_when_settling_is_disabled(benchmark_id, scale, settle_time):
    with pytest.raises(ValueError, match="finite reset_settle_timestep_scale > 0"):
        make_scenario(
            benchmark_id, reset_settle_timestep_scale=scale, reset_settle_time=settle_time,
            compile_reward_kernel=False,
        )


def test_bridge_finish_requires_every_active_root_at_or_above_the_platform() -> None:
    unit_y = torch.tensor([[6.6, 6.6], [6.6, 6.6], [6.6, 6.6], [6.5, 6.6], [6.6, 0.], [6.6, 6.6]])
    unit_z = torch.tensor([[-1., -1.], [-.01, .2], [0., .2], [.1, .2], [.1, -10.], [-2., .2]])
    active = torch.tensor([[True, True]] * 4 + [[True, False], [True, True]])
    result = _compute_bridge_reward_kernel(
        unit_y, unit_z, torch.ones(6, dtype=torch.bool), active,
        torch.zeros(6), -1.5, -.5, 6.5, 10., 1., 1.,
    )
    assert result[2].tolist() == [False, False, True, False, True, False]
    assert result[4].tolist() == [False, False, False, False, False, True]
    torch.testing.assert_close(result[3], torch.tensor([0., 0., 10., 0., 10., 0.]))
    scenario = make_scenario("SwarmBots-Bridge-v0", compile_reward_kernel=False)
    assert scenario.get_settings()["success_min_z"] == scenario.platform_top_z == 0.


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_rejection_sampling_stops_at_attempt_limit(backend):
    distribution = TruncatedNormalDistParams(0., 1., 1e6, 1., "rejection", max_rejection_attempts=3)
    with pytest.raises(RuntimeError, match="exceeded 3 attempts"):
        if backend == "numpy":
            rng = Mock()
            rng.normal.return_value = 1e6
            try:
                eval_truncated_normal(distribution, rng)
            finally:
                assert rng.normal.call_count == 3
        else:
            sample_float_or_dist(
                distribution, shape=(2,), device=torch.device("cpu"), generator=torch.Generator().manual_seed(1),
            )


@pytest.mark.parametrize("backend", ["numpy", "torch"])
def test_rejection_sampling_accepts_valid_samples(backend):
    distribution = TruncatedNormalDistParams(-2., 2., 0., 1., "rejection")
    if backend == "numpy":
        samples = np.array([eval_truncated_normal(distribution, np.random.default_rng(i)) for i in range(20)])
        assert np.all((samples >= -2) & (samples <= 2))
    else:
        samples = sample_float_or_dist(
            distribution, shape=(20,), device=torch.device("cpu"), generator=torch.Generator().manual_seed(1),
        )
        assert ((samples >= -2) & (samples <= 2)).all()


def test_impossible_rejection_distribution_is_rejected():
    with pytest.raises(ValueError, match="mean within"):
        TruncatedNormalDistParams(0., 1., 2., 0., "rejection")


@pytest.mark.integration
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
@pytest.mark.parametrize("action_repeat", [1, 3])
def test_connector_geometry_is_current_and_independent_of_other_world_resets(device, action_repeat):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    cfg = MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=[42000], z_pos=3.)
    surviving_observations = []
    for limits in ([10, 10], [1, 10]):
        env = make_env(
            "SwarmBots-MoveToGoal-v0", num_envs=2, device=device, seed=2,
            episode_length=10, first_episode_lengths=limits, compile_tensor_operations=False,
            scenario_kwargs=dict(unit_start_locations=cfg, reset_settle_time=0,
                                 compile_reward_kernel=False, action_repeat=action_repeat),
        )
        try:
            env.reset(seed=2)
            env._qvel[:, 0] = 3.
            env._qvel[:, 5] = 2.
            actions = {key: torch.zeros(space.shape, device=env.device) for key, space in env.action_space.items()}
            observations, *_ = env.step(actions)
            body = env._host_model.body("Unit0--main_body").id
            torch.testing.assert_close(env._xpos[1, body], env._qpos[1, :3])
            torch.testing.assert_close(env._xquat[1, body], env._qpos[1, 3:7])
            surviving_observations.append(observations["local_obs"][1].clone())
        finally:
            env.close()
    torch.testing.assert_close(*surviving_observations)


@pytest.mark.integration
def test_settling_preserves_live_physics_options_in_an_independent_model():
    cfg = MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=[42000])
    env = make_env(
        "SwarmBots-MoveToGoal-v0", num_envs=1, device="cpu", ccd_iterations=1,
        compile_tensor_operations=False,
        scenario_kwargs=dict(unit_start_locations=cfg, compile_reward_kernel=False, reset_settle_time=0.01),
    )
    try:
        settler_model = env._scenario_runtime._cpu_settler.model
        assert settler_model.opt.ccd_iterations == env._host_model.opt.ccd_iterations == 1
        settler_model.opt.ccd_iterations = 3
        assert env._host_model.opt.ccd_iterations == 1
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("settle_time", [0.0, .01])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_wall_threshold_observations_retain_custom_order_and_latch_crossings(
    settle_time: float, device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    env = make_env(
        "SwarmBots-WallMedium-v0", num_envs=1, device=device, compile_tensor_operations=False,
        scenario_kwargs=dict(
            unit_start_locations=MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=[42000], z_pos=3.),
            compile_reward_kernel=False, reset_settle_time=settle_time,
            swarm_start_y=1.3, wall_pass_thresholds=[.5, .2],
        ),
    )
    try:
        observations, _ = env.reset(seed=7)
        assert observations["hidden_local_vars"][0, 0].tolist() == [0.0, 1.0]
        actions = {key: torch.zeros(space.shape, device=env.device) for key, space in env.action_space.items()}
        root_y = int(env._metadata.unit_qpos_adr[0]) + 1
        env._qpos[0, root_y] = 1.6
        observations, *_ = env.step(actions)
        assert observations["hidden_local_vars"][0, 0].tolist() == [1.0, 1.0]
        env._qpos[0, root_y] = 1.1
        observations, *_ = env.step(actions)
        assert observations["hidden_local_vars"][0, 0].tolist() == [1.0, 1.0]
        observations, _ = env.reset(seed=7)
        assert observations["hidden_local_vars"][0, 0].tolist() == [0.0, 1.0]
    finally:
        env.close()
