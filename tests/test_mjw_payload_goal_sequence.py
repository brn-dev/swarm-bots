from collections.abc import Iterator
from contextlib import ExitStack
import math
from typing import Any
from unittest.mock import patch

from gymnasium.vector import AutoresetMode
import pytest
import torch

from swarmbots import get_benchmark_spec, make_env, make_scenario
from swarmbots.learn.presets.nop import make_next_obs_pred_config
from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWStepResult
from swarmbots.mjw_env.scenarios.mjw_payload_goal_sequence_runtime import (
    PayloadGoalSequenceMJWScenarioRuntime,
    _rotate_xy,
    _sample_goal_sequence,
)
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig

TASK = "SwarmBots-PayloadGoalSequence-v0"


def _env_kwargs() -> dict[str, Any]:
    return {
        "num_envs": 2, "device": "cpu", "compile_tensor_operations": False,
        "scenario_kwargs": {
            "unit_start_locations": MJWPreConnectedUnitLocationsConfig(2, pool_seeds=(42000,)),
            "compile_reward_kernel": False, "reset_settle_time": 0.,
            "goal_hold_time": .06, "randomize_initial_swarm_z_rotation": False,
            "potential_reward_discount_factor": 1.,
        },
    }


@pytest.fixture(
    params=[(settle, delta, potential) for settle in (0., .06) for delta in (False, True)
            for potential in ("zone", "pseudo_huber")],
)
def sequence_env(request: pytest.FixtureRequest) -> Iterator[MJWSwarmBotsVectorEnv]:
    kwargs = _env_kwargs()
    settle_time, include_delta, potential = request.param
    kwargs["scenario_kwargs"].update(
        reset_settle_time=settle_time, include_goal_delta_obs=include_delta, towards_payload_potential=potential,
    )
    env = make_env(TASK, **kwargs)
    try:
        env.reset(seed=7)
        yield env
    finally:
        env.close()


def _place_payload(runtime: PayloadGoalSequenceMJWScenarioRuntime, xy: torch.Tensor) -> None:
    runtime.bindings.qpos[:, runtime._payload_position_qpos_indices[0, :2]] = xy
    runtime.bindings.qpos[:, runtime._payload_position_qpos_indices[0, 2]] = runtime.scenario.payload_radius


def _reward_step(env: MJWSwarmBotsVectorEnv) -> MJWStepResult:
    return env._scenario_runtime.compute_step_rewards(stable_mask=torch.ones(env.num_envs, dtype=torch.bool))


def test_defaults_and_public_observation_contract() -> None:
    scenario = make_scenario(TASK, compile_reward_kernel=False)
    assert get_benchmark_spec(TASK).episode_length == 1000
    assert scenario.goal_radius == .15
    assert scenario.goal_hold_steps == 17
    assert scenario.num_goals == 3
    assert scenario.randomize_initial_swarm_z_rotation
    settings = scenario.get_settings()
    assert settings["effective_goal_hold_time"] == pytest.approx(.51)
    assert settings["goal_sampling"]["algorithm"] == "ordered_relative_turns_v1"
    spaces = scenario.get_single_observation_space()
    assert spaces["global_obs"].shape == (22,)
    indices = build_obs_indices(
        env_settings={"scenario": settings}, local_obs_dim=spaces["local_obs"].shape[-1], global_obs_dim=22,
        hidden_local_vars_dim=spaces["hidden_local_vars"].shape[-1], hidden_global_vars_dim=1,
    )
    assert indices.global_scalar_indices == [1, 2, 3, 10, 11, 12, 13, 14, 15, 16, 17, 21]
    assert indices.global_rot6d_indices == [4]
    targets = make_next_obs_pred_config(indices)
    assert targets.global_scalar_active_mask_indices == [0] * 12
    assert targets.global_rot6d_active_mask_indices == [0]
    assert set(indices.global_scalar_indices).isdisjoint({0, 18, 19, 20})


@pytest.mark.parametrize("num_goals", [1, 3, 4])
def test_optional_goal_delta_is_public_normalized_and_predicted(num_goals: int) -> None:
    scenario = make_scenario(
        TASK, compile_reward_kernel=False, include_goal_delta_obs=True,
        goal_distance_ranges=((1., 2.),) * num_goals, goal_turn_arc_degrees=(90.,) * num_goals,
    )
    spaces = scenario.get_single_observation_space()
    base_dim = 13 + 3 * num_goals
    assert spaces["global_obs"].shape == (base_dim + 2,)
    indices = build_obs_indices(
        env_settings={"scenario": scenario.get_settings()}, local_obs_dim=spaces["local_obs"].shape[-1],
        global_obs_dim=base_dim + 2, hidden_local_vars_dim=spaces["hidden_local_vars"].shape[-1],
        hidden_global_vars_dim=1,
    )
    assert indices.global_scalar_indices[-3:] == [base_dim - 1, base_dim, base_dim + 1]
    targets = make_next_obs_pred_config(indices)
    assert targets.global_scalar_active_mask_indices == [0] * len(indices.global_scalar_indices)


@pytest.mark.integration
def test_goal_delta_tracks_motion_switches_and_partial_resets(sequence_env: MJWSwarmBotsVectorEnv) -> None:
    env = sequence_env
    runtime = env._scenario_runtime
    if not runtime.scenario.include_goal_delta_obs:
        assert runtime.global_obs.shape[1] == 22
        return

    def assert_delta() -> None:
        torch.testing.assert_close(
            runtime.global_obs[:, 22:24], runtime.goal_position[:, 0] - runtime.payload_position[:, 0, :2],
        )

    assert_delta()
    _place_payload(runtime, runtime.goal_sequence[:, 0] + torch.tensor([.4, -.3]))
    _reward_step(env)
    assert_delta()
    torch.testing.assert_close(runtime.global_obs[:, 22:24], torch.tensor([[-.4, .3], [-.4, .3]]))
    _place_payload(runtime, runtime.goal_sequence[:, 0])
    for _ in range(runtime.scenario.goal_hold_steps + 1):
        result = _reward_step(env)
    assert (runtime.completed_goals == 1).all()
    assert not result.info["goal_hold_progress"].any()
    assert_delta()
    other = runtime.global_obs[1].clone()
    env.reset(options={"reset_mask": torch.tensor([True, False])})
    assert_delta()
    torch.testing.assert_close(runtime.global_obs[1], other, rtol=0., atol=0.)


@pytest.mark.parametrize("kwargs", [
    {"goal_distance_ranges": ()}, {"goal_distance_ranges": ((.2, 1.),)},
    {"goal_turn_arc_degrees": (0.,)}, {"goal_turn_arc_degrees": (0., 90., 361.)},
    {"goal_hold_time": -.1}, {"goal_hold_time": float("nan")}, {"waypoint_reward": float("inf")},
])
def test_invalid_sequence_settings_fail_at_construction(kwargs: dict[str, object]) -> None:
    with pytest.raises(ValueError):
        make_scenario(TASK, **kwargs)


def test_sampled_routes_follow_local_distances_turns_and_joint_rotation() -> None:
    n = 8192
    origin = torch.tensor([2., -3.]).expand(n, -1)
    ranges = ((.9, 1.1), (1., 2.), (1., 2.))
    arcs = (10., 90., 180.)
    goals = _sample_goal_sequence(
        origin=origin, yaw=torch.zeros(n), distance_ranges=ranges, turn_arc_degrees=arcs,
        rng=torch.Generator().manual_seed(42),
    )
    legs = torch.diff(torch.cat((origin[:, None], goals), dim=1), dim=1)
    headings = torch.atan2(-legs[..., 0], legs[..., 1])
    turns = torch.diff(torch.cat((torch.zeros((n, 1)), headings), dim=1), dim=1)
    turns = torch.atan2(turns.sin(), turns.cos())
    for i, ((low, high), arc) in enumerate(zip(ranges, arcs, strict=True)):
        assert (legs[:, i].norm(dim=-1) >= low - 1e-6).all()
        assert (legs[:, i].norm(dim=-1) <= high + 1e-6).all()
        assert (turns[:, i].abs() <= math.radians(arc / 2.) + 1e-6).all()
        assert legs[:, i].norm(dim=-1).mean().item() == pytest.approx((low + high) / 2., abs=.02)
    rotated = _sample_goal_sequence(
        origin=origin, yaw=torch.full((n,), math.pi / 2.), distance_ranges=ranges, turn_arc_degrees=arcs,
        rng=torch.Generator().manual_seed(42),
    )
    expected = origin[:, None] + _rotate_xy(goals - origin[:, None], torch.full((n, 1), math.pi / 2.))
    torch.testing.assert_close(rotated, expected)


@pytest.mark.integration
def test_order_hold_interruption_and_final_only_success(sequence_env: MJWSwarmBotsVectorEnv) -> None:
    env = sequence_env
    runtime = env._scenario_runtime
    _place_payload(runtime, runtime.goal_sequence[:, 1])
    for _ in range(3):
        result = _reward_step(env)
        assert not result.terminations.any()
        assert not runtime.completed_goals.any()
    _place_payload(runtime, runtime.goal_sequence[:, 0])
    _reward_step(env)
    torch.testing.assert_close(runtime.global_obs[:, 21], torch.full((2,), 1. / 3.))
    _place_payload(runtime, runtime.goal_sequence[:, 0] + torch.tensor([.151, 0.]))
    _reward_step(env)
    assert not runtime.goal_inside_samples.any()
    _place_payload(runtime, runtime.goal_sequence[:, 0])
    runtime.bindings.qpos[:, runtime._payload_position_qpos_indices[0, 2]] += .151
    _reward_step(env)
    assert not runtime.goal_inside_samples.any()
    for goal_idx in range(3):
        _place_payload(runtime, runtime.goal_sequence[:, goal_idx])
        first = _reward_step(env)
        assert not first.terminations.any()
        assert not first.info["waypoint_reward"].any()
        second = _reward_step(env)
        assert not second.terminations.any()
        result = _reward_step(env)
        assert (runtime.completed_goals == goal_idx + 1).all()
        assert result.terminations.all().item() == (goal_idx == 2)
        assert result.info["success"].all().item() == (goal_idx == 2)
        torch.testing.assert_close(result.info["waypoint_reward"], torch.full((2,), 2. if goal_idx < 2 else 0.))
        torch.testing.assert_close(result.info["goal_success_reward"], torch.full((2,), 5. if goal_idx == 2 else 0.))
        torch.testing.assert_close(result.reward, sum(result.info["reward_terms"].values()))
        if goal_idx < 2:
            torch.testing.assert_close(runtime.global_obs[:, 10:12], runtime.goal_sequence[:, goal_idx + 1])
            assert (runtime.global_obs[:, 18:21].argmax(dim=-1) == goal_idx + 1).all()
            assert not runtime.global_obs[:, 21].any()
    assert not runtime.global_obs[:, 18:21].any()
    assert (runtime.global_obs[:, 21] == 1.).all()
    repeated = _reward_step(env)
    assert not repeated.info["goal_success_reward"].any()


@pytest.mark.integration
@pytest.mark.parametrize("hold_time", [0., .06, .5])
def test_hold_covers_the_required_elapsed_time_after_first_confirmed_entry(hold_time: float) -> None:
    kwargs = _env_kwargs()
    kwargs["scenario_kwargs"]["goal_hold_time"] = hold_time
    env = make_env(TASK, **kwargs)
    try:
        env.reset(seed=7)
        runtime = env._scenario_runtime
        _place_payload(runtime, runtime.goal_sequence[:, 0])
        control_dt = env.scenario.timestep * env.scenario.action_repeat
        for sample in range(env.scenario.goal_hold_steps + 1):
            _reward_step(env)
            elapsed_since_entry = sample * control_dt
            if elapsed_since_entry < hold_time:
                assert not runtime.completed_goals.any()
            else:
                assert (runtime.completed_goals == 1).all()
    finally:
        env.close()


@pytest.mark.integration
def test_target_switch_does_not_create_forward_or_approach_reward(sequence_env: MJWSwarmBotsVectorEnv) -> None:
    env = sequence_env
    runtime = env._scenario_runtime
    runtime.goal_sequence[:] = torch.tensor([[[0., 2.], [0., -2.], [2., 0.]]])
    runtime.goal_position[:] = runtime.goal_sequence[:, :1]
    runtime.bindings.qpos[:, runtime._unit_xy_qpos_indices] = torch.tensor([3., 3.])
    _place_payload(runtime, runtime.goal_sequence[:, 0])
    _reward_step(env)
    _reward_step(env)
    switched = _reward_step(env)
    torch.testing.assert_close(switched.info["forward_reward"], torch.zeros(2))
    torch.testing.assert_close(switched.info["towards_payload_reward"], torch.zeros(2))
    stationary = _reward_step(env)
    torch.testing.assert_close(stationary.info["forward_reward"], torch.zeros(2))
    torch.testing.assert_close(stationary.info["towards_payload_reward"], torch.zeros(2))


@pytest.mark.integration
def test_reset_restores_selected_world_sequence_and_preserves_other_world(sequence_env: MJWSwarmBotsVectorEnv) -> None:
    env = sequence_env
    runtime = env._scenario_runtime
    _place_payload(runtime, runtime.goal_sequence[:, 0])
    _reward_step(env)
    _reward_step(env)
    _reward_step(env)
    _place_payload(runtime, runtime.goal_sequence[:, 1])
    _reward_step(env)
    other_obs = runtime.global_obs[1].clone()
    other_qpos = runtime.bindings.qpos[1].clone()
    other_potential = runtime.towards_payload_progress[1].clone()
    env.reset(seed=8, options={"reset_mask": torch.tensor([True, False])})
    assert runtime.completed_goals[0] == runtime.goal_inside_samples[0] == 0
    assert runtime.global_obs[0, 18] == 1.
    torch.testing.assert_close(runtime.global_obs[1], other_obs, rtol=0., atol=0.)
    torch.testing.assert_close(runtime.bindings.qpos[1], other_qpos, rtol=0., atol=0.)
    torch.testing.assert_close(runtime.towards_payload_progress[1], other_potential, rtol=0., atol=0.)


@pytest.mark.integration
def test_whole_reset_arrangement_rotates_and_double_apply_is_idempotent() -> None:
    kwargs = _env_kwargs()
    kwargs["scenario_kwargs"].update(reset_settle_time=.06, payload_shape="capsule")
    env = make_env(TASK, **kwargs)
    try:
        runtime = env._scenario_runtime
        batch = runtime.sample_reset_batch(n_reset=2, rng=torch.Generator().manual_seed(42))
        batch.payload.common.swarm_start[:, :2] = torch.tensor([2., -3.])
        batch.payload.common.initial_z_rotation[:] = torch.tensor([0., math.pi / 2.])
        # Use the same local scene in both worlds, with a quarter turn about a translated pivot.
        batch.payload.common.pool_idx[1] = batch.payload.common.pool_idx[0]
        pivot = batch.payload.common.swarm_start[0, :2]
        batch.payload.payload_position[:, 0, :2] = pivot + torch.tensor([[0., 1.], [-1., 0.]])
        route = torch.tensor([[0., 2.], [1., 3.], [1., 4.]])
        batch.goals[0] = pivot + route
        batch.goals[1] = pivot + _rotate_xy(route, torch.tensor(math.pi / 2.))
        batch.payload.goal_position = batch.goals[:, :1]
        worlds = torch.arange(2)
        runtime.apply_reset_batch(world_idx=worlds, reset_batch=batch)
        qpos = runtime.bindings.qpos.clone()
        obs = runtime.global_obs.clone()
        units = runtime.bindings.qpos[:, runtime._unit_xy_qpos_indices]
        torch.testing.assert_close(units[1] - pivot, _rotate_xy(units[0] - pivot, torch.tensor(math.pi / 2.)))
        quats = runtime.bindings.qpos[:, runtime._payload_quat_qpos_indices[0]]
        torch.testing.assert_close(quats[0], torch.tensor([1., 0., 0., 0.]))
        torch.testing.assert_close(quats[1], torch.tensor([2 ** -.5, 0., 0., 2 ** -.5]))
        runtime.apply_reset_batch(world_idx=worlds, reset_batch=batch)
        torch.testing.assert_close(runtime.bindings.qpos, qpos, rtol=0., atol=0.)
        torch.testing.assert_close(runtime.global_obs, obs, rtol=0., atol=0.)
        snapshots = runtime.settle_cpu_reset_specs(specs=runtime.build_cpu_reset_specs(reset_batch=batch))
        quaternion_indices = runtime._payload_quat_qpos_indices[0].numpy()
        tilted = torch.tensor([math.cos(.2), math.sin(.2), 0., 0.])
        snapshots[0].payload.common.qpos[quaternion_indices] = tilted.numpy()
        runtime.apply_settled_reset_batch(world_idx=worlds, snapshots=snapshots)
        settled = runtime.bindings.qpos.clone()
        settled_obs = runtime.global_obs.clone()
        runtime.apply_settled_reset_batch(world_idx=worlds, snapshots=snapshots)
        torch.testing.assert_close(runtime.bindings.qpos, settled, rtol=0., atol=0.)
        torch.testing.assert_close(runtime.global_obs, settled_obs, rtol=0., atol=0.)
        torch.testing.assert_close(runtime.bindings.qpos[0, runtime._payload_quat_qpos_indices[0]], tilted)
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize("mode", [AutoresetMode.SAME_STEP, AutoresetMode.NEXT_STEP])
@pytest.mark.parametrize("settle_time", [0., .06])
def test_final_success_returns_terminal_sequence_then_a_clean_reset(mode: AutoresetMode, settle_time: float) -> None:
    kwargs = _env_kwargs()
    kwargs["scenario_kwargs"]["reset_settle_time"] = settle_time
    env = make_env(TASK, **kwargs, autoreset_mode=mode)
    try:
        env.reset(seed=7)
        runtime = env._scenario_runtime
        runtime.completed_goals[0] = 2
        runtime.goal_position[0] = runtime.goal_sequence[0, 2:3]
        runtime.goal_inside_samples[0] = 2
        runtime.bindings.qpos[0, runtime._payload_position_qpos_indices[0, :2]] = runtime.goal_sequence[0, 2]
        actions = {key: torch.zeros(space.shape) for key, space in env.action_space.items()}
        observations, _, terminated, truncated, info = env.step(actions)
        assert terminated.tolist() == [True, False]
        assert not truncated.any()
        assert info["success"].tolist() == [True, False]
        terminal = info["final_obs"]["global_obs"] if mode == AutoresetMode.SAME_STEP else observations["global_obs"]
        assert not terminal[0, 18:21].any()
        assert terminal[0, 21] == 1.
        if mode == AutoresetMode.NEXT_STEP:
            observations, rewards, terminated, truncated, _ = env.step(actions)
            assert rewards[0] == 0.
            assert not terminated[0] and not truncated[0]
        assert observations["global_obs"][0, 18] == 1.
        assert observations["global_obs"][0, 21] == 0.
        assert runtime.completed_goals[0] == 0
    finally:
        env.close()


@pytest.mark.integration
def test_unstable_lane_preserves_sequence_history_and_prior_step_info(sequence_env: MJWSwarmBotsVectorEnv) -> None:
    env = sequence_env
    runtime = env._scenario_runtime
    runtime.bindings.qpos[:, runtime._unit_xy_qpos_indices] = torch.tensor([3., 3.])
    _place_payload(runtime, runtime.goal_sequence[:, 0])
    _reward_step(env)
    previous_info = _reward_step(env).info
    previous_completed = previous_info["goals_completed"].clone()
    previous_hold = previous_info["goal_hold_progress"].clone()
    history = [value[0].clone() for value in (
        runtime.completed_goals, runtime.goal_inside_samples, runtime.payload_progress, runtime.towards_payload_progress,
    )]
    runtime.bindings.qpos[0, runtime._payload_position_qpos_indices[0]] = float("nan")
    runtime.bindings.qpos[0, runtime._unit_xy_qpos_indices] = float("nan")
    result = runtime.compute_step_rewards(stable_mask=torch.tensor([False, True]))
    for value, expected in zip((
        runtime.completed_goals, runtime.goal_inside_samples, runtime.payload_progress, runtime.towards_payload_progress,
    ), history, strict=True):
        torch.testing.assert_close(value[0], expected)
    assert not result.terminations[0]
    assert result.info["waypoint_reward"][0] == 0.
    assert runtime.completed_goals[1] == 1
    torch.testing.assert_close(previous_info["goals_completed"], previous_completed)
    torch.testing.assert_close(previous_info["goal_hold_progress"], previous_hold)


@pytest.mark.integration
def test_completed_milestones_survive_simulation_failure(sequence_env: MJWSwarmBotsVectorEnv) -> None:
    env = sequence_env
    runtime = env._scenario_runtime
    _place_payload(runtime, runtime.goal_sequence[:, 0])
    for _ in range(3):
        _reward_step(env)
    assert (runtime.completed_goals == 1).all()
    run_physics = env._run_physics

    def inject_failure() -> None:
        run_physics()
        env._qvel[0, 0] = float("nan")

    actions = {key: torch.zeros(space.shape) for key, space in env.action_space.items()}
    with patch.object(env, "_run_physics", inject_failure):
        _, _, terminated, _, info = env.step(actions)
    assert terminated[0]
    assert not info["success"][0]
    assert info["goals_completed"][0] == 1
    assert info["goal_1_reached"][0]
    assert not info["goal_2_reached"][0]
    assert runtime.completed_goals[0] == 0


@pytest.mark.integration
def test_sequence_progress_only_counts_sequence_episodes_in_mixed_training() -> None:
    from swarmbots.learn.env_wrappers.multi_scenario_vector_env import MultiScenarioVectorEnv
    from swarmbots.learn.presets.policy_factory import wrap_vec_env
    from swarmbots.learn.rollout_utils import append_episode_infos
    from swarmbots.learn.training import _observation_indices

    wall = "SwarmBots-WallMedium-v0"
    env_options = dict(num_envs=1, device="cpu", episode_length=1, compile_tensor_operations=False)
    scenario_options = dict(
        reset_settle_time=0., compile_reward_kernel=False,
        unit_start_locations=MJWPreConnectedUnitLocationsConfig(2, pool_seeds=(42000,)),
    )
    with ExitStack() as cleanup:
        wall_env = make_env(wall, scenario_kwargs=scenario_options, **env_options)
        cleanup.callback(wall_env.close)
        sequence_env = make_env(TASK, scenario_kwargs={**scenario_options, "goal_hold_time": 0.}, **env_options)
        cleanup.callback(sequence_env.close)
        env = wrap_vec_env(
            vector_env=MultiScenarioVectorEnv({wall: wall_env, TASK: sequence_env}),
            obs_indices=_observation_indices(sequence_env), gamma=.995,
            use_popart=True, rollout_device=torch.device("cpu"),
        )
        cleanup.callback(env.close)
        env.reset(seed=7)
        runtime = sequence_env._scenario_runtime
        _place_payload(runtime, runtime.goal_sequence[:, 0])
        _reward_step(sequence_env)
        actions = torch.zeros((env.num_envs, env.n_agents, env.action_space.total_agent_action_dim))
        _, _, terminated, truncated, info = env.step(actions)
        episodes = []
        append_episode_infos(episode_infos=episodes, infos=info, dones=terminated | truncated)

    assert [episode["scenario_name"] for episode in episodes] == [wall, TASK]
    assert not any(key.startswith("task/") for key in episodes[0])
    assert {key: value for key, value in episodes[1].items() if key.startswith("task/")} == {
        "task/goals_completed": 1., "task/goal_1_reached": 1., "task/goal_2_reached": 0., "task/goal_3_reached": 0.,
    }


@pytest.mark.integration
@pytest.mark.parametrize("include_delta", [False, True])
def test_tmasac_actors_consume_the_sequence_observations_and_training_logs_progress(include_delta: bool) -> None:
    from swarmbots.learn import make_training
    from swarmbots.learn.task_episode_metrics import summarize_task_episode_metrics

    kwargs = _env_kwargs()
    kwargs["scenario_kwargs"]["include_goal_delta_obs"] = include_delta
    algorithm = make_training(
        TASK, "tmasac", num_envs=2, device="cpu", seed=7, model_scale="2.5M", episode_length=4,
        scenario_kwargs=kwargs["scenario_kwargs"], env_kwargs={"compile_tensor_operations": False},
        algorithm_kwargs={"batch_size": 2},
    )
    try:
        observations, _ = algorithm.env.reset(seed=7)
        with torch.no_grad():
            actions = algorithm.policy.act(
                local_obs=observations["local_obs"], global_obs=observations["global_obs"],
                agent_mask=observations["agent_mask"], deterministic=True,
            )
        assert torch.isfinite(actions).all()
        runtime = algorithm.env.unwrapped._scenario_runtime
        _place_payload(runtime, runtime.goal_sequence[:, 0])
        for _ in range(3):
            runtime.compute_step_rewards(stable_mask=torch.ones(2, dtype=torch.bool))
        episode_records = []
        for _ in range(4):
            _, _, terminated, truncated, info = algorithm.env.step(torch.zeros_like(actions))
            if (terminated | truncated).any():
                assert "task/goals_completed" in info["episode"]
                for lane in (terminated | truncated).nonzero(as_tuple=True)[0].tolist():
                    episode_records.append({
                        key: values[lane].item() for key, values in info["episode"].items() if key.startswith("task/")
                    })
        metrics = summarize_task_episode_metrics(episode_records)
        assert metrics["ep_task_goals_completed"].n == 2
        assert metrics["ep_task_goals_completed"].mean == 1.
        assert metrics["ep_task_goal_1_reached"].mean == 1.
        assert metrics["ep_task_goal_2_reached"].mean == 0.
    finally:
        algorithm.env.close()


@pytest.mark.integration
def test_sequence_transition_kernel_is_fullgraph_traceable(monkeypatch: pytest.MonkeyPatch) -> None:
    env = make_env(TASK, **_env_kwargs())
    try:
        env.reset(seed=7)
        runtime = env._scenario_runtime
        monkeypatch.setattr(runtime, "_sequence_kernel", torch.compile(
            runtime._sequence_kernel, backend="eager", fullgraph=True,
        ))
        for goal_idx in range(3):
            _place_payload(runtime, runtime.goal_sequence[:, goal_idx])
            for _ in range(3):
                result = _reward_step(env)
            assert (runtime.completed_goals == goal_idx + 1).all()
        assert result.terminations.all()
    finally:
        env.close()
        torch.compiler.reset()
