from __future__ import annotations

from collections.abc import Iterator
import math

import pytest
import torch

from swarmbots import get_benchmark_spec, make_env, make_scenario
from swarmbots.benchmark.metadata import serialize_settings
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.scenarios.mjw_payload_goal_runtime import (
    _compute_payload_goal_approach_progress,
    _compute_payload_goal_approach_reward_kernel,
    _sample_arc_goal_positions,
)
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


def test_registered_payload_goal_always_has_one_payload_and_a_public_goal() -> None:
    scenario = make_scenario("SwarmBots-PayloadGoal-v0", compile_reward_kernel=False)
    assert scenario.max_payloads == 1
    assert scenario.active_payload_count_probs == {1: 1.0}
    assert scenario.get_single_observation_space()["global_obs"].shape == (12,)
    assert scenario.forward_reward_weight == 10.0
    assert scenario.towards_payload_reward_weight == 1.0
    assert scenario.success_reward == 5.0
    assert scenario.progress_reward_weight == 1.0
    assert get_benchmark_spec("SwarmBots-PayloadGoal-v0").supports_success_metric
    sampling = serialize_settings(scenario.get_settings())["goal_sampling"]
    assert sampling["radius"] == 3.0
    assert sampling["arc_degrees"] == 120.0
    assert sampling["noise_std_xy"] == .15
    assert sampling["origin"] == "payload_spawn_xy_before_settling"
    multi_payload = make_scenario("SwarmBots-MultiPayloadGoal-v0", compile_reward_kernel=False)
    assert multi_payload.active_payload_count_probs == {2: 1 / 3, 3: 1 / 3, 4: 1 / 3}
    assert "towards_payload_reward_weight" not in multi_payload.get_settings()


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_goals_cover_a_uniform_forward_arc_at_the_requested_distance(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    origin = torch.tensor([[1., -2.]], device=device).expand(32768, -1)
    goals = _sample_arc_goal_positions(
        origin=origin, radius=3., arc_degrees=120., noise_std=0.,
        rng=torch.Generator(device=device).manual_seed(7),
    )
    offset = goals - origin
    torch.testing.assert_close(offset.norm(dim=-1), torch.full((32768,), 3., device=device))
    angle = torch.atan2(offset[:, 0], offset[:, 1])
    assert (angle.abs() <= math.pi / 3).all()
    quartile_edges = torch.linspace(-math.pi / 3, math.pi / 3, 5, device=device)
    for low, high in zip(quartile_edges[:-1], quartile_edges[1:]):
        frequency = ((angle >= low) & (angle < high)).float().mean().item()
        assert frequency == pytest.approx(.25, abs=.01)
    translated = _sample_arc_goal_positions(
        origin=origin + 10., radius=3., arc_degrees=120., noise_std=0.,
        rng=torch.Generator(device=device).manual_seed(7),
    )
    torch.testing.assert_close(translated, goals + 10.)


def test_arc_noise_has_small_independent_gaussian_offsets() -> None:
    goals = _sample_arc_goal_positions(
        origin=torch.zeros((32768, 2)), radius=3., arc_degrees=1e-5, noise_std=.15,
        rng=torch.Generator().manual_seed(42),
    )
    noise = goals - torch.tensor([0., 3.])
    torch.testing.assert_close(noise.mean(dim=0), torch.zeros(2), atol=.003, rtol=0.)
    torch.testing.assert_close(noise.std(dim=0), torch.full((2,), .15), atol=.003, rtol=0.)
    covariance = (noise[:, 0] * noise[:, 1]).mean()
    assert covariance.abs().item() < .0006
    assert (noise.abs() > .45).any()


@pytest.mark.parametrize("kwargs", [
    {"goal_arc_radius": 0.}, {"goal_arc_radius": float("nan")},
    {"goal_arc_degrees": 0.}, {"goal_arc_degrees": 361.},
    {"goal_noise_std": -.1}, {"goal_noise_std": float("inf")},
])
def test_invalid_arc_settings_are_rejected(kwargs: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        make_scenario("SwarmBots-PayloadGoal-v0", compile_reward_kernel=False, **kwargs)


@pytest.mark.parametrize("direction", [(1., 0.), (-1., 0.), (0., 1.), (0., -1.), (.6, .8)])
def test_approach_zone_is_on_the_side_opposite_the_goal(direction: tuple[float, float]) -> None:
    payload = torch.tensor([[1., 2., .3]])
    heading = torch.tensor([direction])
    goal = payload[:, :2] + 5.0 * heading
    pushing_side = payload[:, :2] - .3 * heading
    goal_side = payload[:, :2] + .3 * heading
    progress = _compute_payload_goal_approach_progress(
        torch.stack((pushing_side, goal_side), dim=0),
        payload.expand(2, -1),
        goal.expand(2, -1),
        torch.ones((2, 1), dtype=torch.bool),
        .3,
        .1,
    )
    torch.testing.assert_close(progress, torch.tensor([0., -.5]))


def test_default_approach_zone_can_accommodate_five_spaced_units_behind_payload() -> None:
    scenario = make_scenario("SwarmBots-PayloadGoal-v0", compile_reward_kernel=False)
    angles = torch.linspace(math.pi, 2 * math.pi, 5)
    unit_xy = torch.stack((angles.cos(), angles.sin() - scenario.payload_radius), dim=-1)
    separation = torch.pdist(unit_xy)
    assert (separation > 2 * scenario.swarm.max_unit_extent).all()
    assert (unit_xy[:, 1] <= -scenario.payload_radius + 1e-6).all()
    payload_clearance = math.sqrt(2) * scenario.payload_radius + scenario.swarm.max_unit_extent
    assert (unit_xy.norm(dim=-1) > payload_clearance).all()
    progress = _compute_payload_goal_approach_progress(
        unit_xy[None], torch.tensor([[0., 0., scenario.payload_radius]]), torch.tensor([[0., 5.]]),
        torch.ones((1, 5), dtype=torch.bool), scenario.payload_radius, scenario.towards_payload_goal_radius,
    )
    torch.testing.assert_close(progress, torch.zeros(1))


@pytest.mark.parametrize("device, compiled", [
    ("cpu", False),
    pytest.param("cuda", False, marks=pytest.mark.cuda),
    pytest.param("cuda", True, marks=pytest.mark.cuda),
])
def test_approach_rewards_progress_and_ignores_inactive_units(device: str, compiled: bool) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    kernel = _compute_payload_goal_approach_reward_kernel
    if compiled:
        kernel = torch.compile(kernel, fullgraph=True)
    try:
        unit_xy = torch.tensor([
            [[0., 0.], [100., 100.]],
            [[0., -1.], [100., 100.]],
            [[0., -.5], [100., 100.]],
            [[0., .7], [100., 100.]],
        ], device=device)
        progress, reward = kernel(
            unit_xy,
            torch.tensor([[0., 1., .3]], device=device).expand(4, -1),
            torch.tensor([[0., 3.]], device=device).expand(4, -1),
            torch.ones(4, dtype=torch.bool, device=device),
            torch.tensor([[True, False]], device=device).expand(4, -1),
            torch.tensor([-1., -1., -1., 0.], device=device),
            .3, .2, 2., 3., 1.,
        )
        torch.testing.assert_close(progress, torch.tensor([-.5, -1.5, -1., 0.], device=device))
        torch.testing.assert_close(reward, torch.tensor([3., -3., 0., 0.], device=device))
    finally:
        if compiled:
            torch.compiler.reset()


@pytest.mark.parametrize("discount, expected_reward", [(1., 2.), (.995, 2.01)])
def test_approach_rewards_average_progress_over_active_units(discount: float, expected_reward: float) -> None:
    progress, reward = _compute_payload_goal_approach_reward_kernel(
        torch.tensor([[[0., -.5], [0., -2.5], [100., 100.]]]),
        torch.tensor([[0., 1., .3]]),
        torch.tensor([[0., 4.]]),
        torch.tensor([True]),
        torch.tensor([[True, True, False]]),
        torch.tensor([-4.]),
        .3, .2, 1., 1., discount,
    )
    torch.testing.assert_close(progress, torch.tensor([-2.]))
    torch.testing.assert_close(reward, torch.tensor([expected_reward]))


def test_coincident_goal_empty_swarm_and_unstable_world_are_finite() -> None:
    positions = torch.tensor([[[0., 0.]], [[10., 10.]], [[float("nan"), 0.]]])
    payloads = torch.tensor([[0., 0., .3], [0., 0., .3], [float("nan"), 0., .3]])
    progress, reward = _compute_payload_goal_approach_reward_kernel(
        positions, payloads, torch.zeros((3, 2)), torch.tensor([True, True, False]),
        torch.tensor([[True], [False], [True]]), torch.zeros(3), .3, .2, 1., 1., .995,
    )
    assert torch.isfinite(progress).all()
    torch.testing.assert_close(reward, torch.zeros(3))


@pytest.fixture(params=[0.0, 0.06], ids=["unsettled", "settled"])
def payload_goal_env(request: pytest.FixtureRequest) -> Iterator[MJWSwarmBotsVectorEnv]:
    env = make_env(
        "SwarmBots-PayloadGoal-v0", num_envs=2, device="cpu", seed=7,
        compile_tensor_operations=False,
        scenario_kwargs={
            "unit_start_locations": MJWPreConnectedUnitLocationsConfig(num_units=2, pool_seeds=(42000,)),
            "compile_reward_kernel": False,
            "reset_settle_time": request.param,
            "payload_spawn_y": 2.5,
            "potential_reward_discount_factor": 1.0,
        },
    )
    try:
        yield env
    finally:
        env.close()


@pytest.mark.integration
def test_reset_samples_one_goal_and_initializes_approach_history(payload_goal_env: MJWSwarmBotsVectorEnv) -> None:
    env = payload_goal_env
    observations, _ = env.reset(seed=7)
    records = observations["global_obs"].reshape(2, 1, 12)
    assert (records[..., 0] == 1).all()
    original_goals = records[..., 10:12].clone()
    runtime = env._scenario_runtime
    original_payload_position = runtime.payload_position[:, 0, :2].clone()
    spread = (original_goals[:, 0] - original_payload_position).norm(dim=-1)
    assert ((spread > 2.3) & (spread < 3.7)).all()
    stable = torch.ones(2, dtype=torch.bool)
    assert (runtime.towards_payload_progress < 0.).all()
    initial = runtime.compute_step_rewards(stable_mask=stable)
    torch.testing.assert_close(initial.info["towards_payload_reward"], torch.zeros(2))
    torch.testing.assert_close(runtime.goal_position, original_goals)

    runtime.bindings.qpos[:, runtime._unit_xy_qpos_indices] += .2
    runtime.compute_step_rewards(stable_mask=stable)
    preserved_progress = runtime.towards_payload_progress[1].clone()
    assert preserved_progress < 0.
    reset_obs, _ = env.reset(seed=8, options={"reset_mask": torch.tensor([True, False])})
    new_goals = reset_obs["global_obs"].reshape(2, 1, 12)[..., 10:12]
    assert not torch.equal(new_goals[0], original_goals[0])
    torch.testing.assert_close(new_goals[1], original_goals[1])
    torch.testing.assert_close(runtime.towards_payload_progress[1], preserved_progress)
    after_reset = runtime.compute_step_rewards(stable_mask=stable)
    torch.testing.assert_close(after_reset.info["towards_payload_reward"], torch.zeros(2))


@pytest.mark.integration
def test_goal_is_anchored_before_contact_settling_and_stays_fixed_during_steps() -> None:
    env = make_env(
        "SwarmBots-PayloadGoal-v0", num_envs=2, device="cpu", seed=7,
        compile_tensor_operations=False, settled_reset_buffer_size=0,
        scenario_kwargs={
            "unit_start_locations": MJWPreConnectedUnitLocationsConfig(num_units=1, pool_seeds=(42000,)),
            "compile_reward_kernel": False,
            "reset_settle_time": .2,
            "swarm_start_x": 2.5,
            "swarm_start_y": -1.,
            "payload_spawn_y": .2,
            "goal_noise_std": 0.,
        },
    )
    try:
        observations, _ = env.reset(seed=7)
        goals = observations["global_obs"].reshape(2, 1, 12)[..., 10:12].clone()
        spawn_xy = torch.tensor([2.5, -.8])
        settled_xy = env._scenario_runtime.payload_position[:, 0, :2]
        assert ((settled_xy - spawn_xy).norm(dim=-1) > .001).all()
        torch.testing.assert_close((goals[:, 0] - spawn_xy).norm(dim=-1), torch.full((2,), 3.))
        actions = {key: torch.zeros(space.shape) for key, space in env.action_space.items()}
        for _ in range(5):
            observations, _, terminated, truncated, _ = env.step(actions)
            assert not (terminated | truncated).any()
            torch.testing.assert_close(
                observations["global_obs"].reshape(2, 1, 12)[..., 10:12], goals, rtol=0., atol=0.,
            )
    finally:
        env.close()


@pytest.mark.integration
def test_runtime_applies_default_progress_weights(payload_goal_env: MJWSwarmBotsVectorEnv) -> None:
    env = payload_goal_env
    env.reset(seed=7)
    runtime = env._scenario_runtime
    stable = torch.ones(2, dtype=torch.bool)
    runtime.goal_position[:, 0] = runtime.payload_position[:, 0, :2] + torch.tensor([0., 3.])
    runtime.compute_step_rewards(stable_mask=stable)
    runtime.bindings.qpos[:, runtime._payload_position_qpos_indices[0, 1]] += .1
    forward = runtime.compute_step_rewards(stable_mask=stable)
    torch.testing.assert_close(forward.info["forward_reward"], torch.ones(2), atol=1e-5, rtol=0.)

    zone_center = runtime.payload_position[:, 0, :2] - torch.tensor([0., env.scenario.payload_radius])
    unit_xy = zone_center[:, None, :] + torch.tensor([-env.scenario.towards_payload_goal_radius - 1., 0.])
    runtime.bindings.qpos[:, runtime._unit_xy_qpos_indices] = unit_xy
    runtime.compute_step_rewards(stable_mask=stable)
    runtime.bindings.qpos[:, runtime._unit_xy_qpos_indices[:, 0]] += .25
    approach = runtime.compute_step_rewards(stable_mask=stable)
    torch.testing.assert_close(approach.info["towards_payload_reward"], torch.full((2,), .25))


@pytest.mark.integration
def test_payload_goal_keeps_success_and_reward_accounting(payload_goal_env: MJWSwarmBotsVectorEnv) -> None:
    env = payload_goal_env
    env.reset(seed=7)
    runtime = env._scenario_runtime
    runtime.bindings.qpos[:, runtime._payload_position_qpos_indices[0, :2]] = runtime.goal_position[:, 0]
    runtime.bindings.qpos[:, runtime._payload_position_qpos_indices[0, 2]] = env.scenario.payload_radius
    result = runtime.compute_step_rewards(stable_mask=torch.ones(2, dtype=torch.bool))
    assert result.terminations.all()
    assert result.info["success"].all()
    torch.testing.assert_close(result.info["goal_success_reward"], torch.full((2,), 5.))
    torch.testing.assert_close(result.reward, sum(result.info["reward_terms"].values()))


@pytest.mark.integration
def test_unstable_world_does_not_overwrite_approach_history(payload_goal_env: MJWSwarmBotsVectorEnv) -> None:
    env = payload_goal_env
    env.reset(seed=7)
    runtime = env._scenario_runtime
    previous = runtime.towards_payload_progress.clone()
    assert (previous < 0.).all()
    runtime.bindings.qpos[0, runtime._unit_xy_qpos_indices] = float("nan")
    result = runtime.compute_step_rewards(stable_mask=torch.tensor([False, True]))
    torch.testing.assert_close(runtime.towards_payload_progress, previous)
    assert torch.isfinite(result.info["towards_payload_reward"]).all()
