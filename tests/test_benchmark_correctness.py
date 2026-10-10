from __future__ import annotations

from unittest.mock import Mock

import numpy as np
import pytest
import torch

from swarmbots import make_env, make_scenario
from swarmbots.mjw_env.mjw_torch_utils import sample_float_or_dist
from swarmbots.mjw_env.parameter_distributions import TruncatedNormalDistParams, UniformDistParams, eval_truncated_normal
from swarmbots.mjw_env.scenarios.mjw_payload_step_runtime import _compute_payload_step_reward_extras_kernel
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


@pytest.mark.parametrize("limit", [0, -1])
def test_invalid_episode_limit_is_rejected_before_scenario_creation(limit, monkeypatch):
    factory = Mock(side_effect=AssertionError("Scenario must not be constructed"))
    monkeypatch.setattr("swarmbots.benchmark.registry.make_scenario", factory)
    with pytest.raises(ValueError, match="episode_length > 0"):
        make_env("SwarmBots-MoveTo-v0", num_envs=1, episode_length=limit)
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
            "SwarmBots-MoveTo-v0", num_envs=2, device=device, seed=2,
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
        "SwarmBots-MoveTo-v0", num_envs=1, device="cpu", ccd_iterations=1,
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
