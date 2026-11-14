from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from swarmbots.learn import make_training
from swarmbots.learn.training import _make_policy_env, _staggered_first_episode_lengths, _variant_options
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


def _training_kwargs() -> dict[str, Any]:
    return {
        "num_envs": 6,
        "device": "cpu",
        "seed": 42,
        "episode_length": 8,
        "model_scale": None,
        "use_nop": False,
        "scenario_kwargs": {
            "unit_start_locations": MJWPreConnectedUnitLocationsConfig(1, pool_seeds=(42000,)),
            "reset_settle_time": 0,
            "compile_reward_kernel": False,
        },
        "env_kwargs": {"compile_tensor_operations": False},
        "policy_kwargs": {"enc_d_model": 16, "dec_d_model": 16},
    }


def _episode_end_steps(env: MJWSwarmBotsVectorEnv, *, num_steps: int) -> list[list[int]]:
    env.reset(seed=42)
    actions = {key: torch.zeros(space.shape, device=env.device) for key, space in env.action_space.items()}
    episode_ends: list[list[int]] = [[] for _ in range(env.num_envs)]
    for step in range(1, num_steps + 1):
        _, _, terminated, truncated, _ = env.step(actions)
        assert not terminated.any()
        for lane in torch.nonzero(truncated, as_tuple=True)[0].tolist():
            episode_ends[lane].append(step)
    return episode_ends


@pytest.mark.parametrize("num_envs, episode_length", [
    (1024, 500), (1024, 512), (17, 8), (6, 8), (1, 500), (7, 1),
])
def test_initial_episode_endings_are_balanced_across_steps(num_envs: int, episode_length: int) -> None:
    lengths = _staggered_first_episode_lengths(num_envs=num_envs, episode_length=episode_length)
    assert len(lengths) == num_envs
    assert all(1 <= length <= episode_length for length in lengths)
    completions = torch.bincount(torch.tensor(lengths), minlength=episode_length + 1)[1:]
    minimum = num_envs // episode_length
    maximum = minimum + int(num_envs % episode_length != 0)
    assert (completions >= minimum).all()
    assert (completions <= maximum).all()
    cumulative = completions.cumsum(0)
    ideal_cumulative = torch.arange(1, episode_length + 1, dtype=torch.float64) * num_envs / episode_length
    assert ((cumulative - ideal_cumulative).abs() < 1.0).all()


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["ppo", "masac__deepset_critic"])
@pytest.mark.parametrize("num_envs, episode_length", [(6, 8), (18, 8), (16, 8)])
def test_training_staggers_initial_episodes_and_then_uses_full_lengths(
    variant: str, num_envs: int, episode_length: int,
) -> None:
    kwargs = _training_kwargs()
    kwargs.update(num_envs=num_envs, episode_length=episode_length)
    if variant == "masac__deepset_critic":
        kwargs["algorithm_kwargs"] = {"batch_size": num_envs}
    algorithm = make_training("SwarmBots-MoveToGoal-v0", variant, **kwargs)
    try:
        episode_ends = _episode_end_steps(algorithm.env.unwrapped, num_steps=3 * episode_length)
        first_ends = [ends[0] for ends in episode_ends]
        counts = torch.bincount(torch.tensor(first_ends), minlength=episode_length + 1)[1:]
        assert (counts >= num_envs // episode_length).all()
        assert (counts <= (num_envs + episode_length - 1) // episode_length).all()
        for ends in episode_ends:
            assert len(ends) >= 3
            assert all(second - first == episode_length for first, second in zip(ends, ends[1:]))
        assert kwargs["env_kwargs"] == {"compile_tensor_operations": False}
    finally:
        algorithm.env.close()


@pytest.mark.integration
def test_early_success_consumes_first_episode_limit_and_restores_full_horizon() -> None:
    kwargs = _training_kwargs()
    kwargs["num_envs"] = 2
    algorithm = make_training("SwarmBots-PayloadGoal-v0", "ppo", **kwargs)
    try:
        env = algorithm.env.unwrapped
        env.reset(seed=42)
        runtime = env._scenario_runtime
        runtime.bindings.qpos[0, runtime._payload_position_qpos_indices[0, :2]] = runtime.goal_position[0, 0]
        runtime.bindings.qpos[0, runtime._payload_position_qpos_indices[0, 2]] = env.scenario.payload_radius
        actions = {key: torch.zeros(space.shape, device=env.device) for key, space in env.action_space.items()}
        episode_ends: list[list[int]] = [[], []]
        for step in range(1, 18):
            _, _, terminated, truncated, info = env.step(actions)
            torch.testing.assert_close(terminated, torch.tensor([step == 1, False]))
            if step == 1:
                assert not truncated.any()
                torch.testing.assert_close(info["success"], torch.tensor([True, False]))
            for lane in torch.nonzero(terminated | truncated, as_tuple=True)[0].tolist():
                episode_ends[lane].append(step)
        assert episode_ends == [[1, 9, 17], [8, 16]]
    finally:
        algorithm.env.close()


@pytest.mark.integration
def test_staggering_is_reproducible_and_preserves_policy_initialization() -> None:
    first_limits: list[list[int]] = []
    states: list[Mapping[str, torch.Tensor]] = []
    for disable_staggering in (False, False, True):
        kwargs = _training_kwargs()
        if disable_staggering:
            kwargs["env_kwargs"]["first_episode_lengths"] = None
        algorithm = make_training("SwarmBots-MoveToGoal-v0", "ppo", **kwargs)
        try:
            env = algorithm.env.unwrapped
            if not disable_staggering:
                first_limits.append(env.first_episode_lengths)
            states.append({key: value.clone() for key, value in algorithm.policy.state_dict().items()})
        finally:
            algorithm.env.close()
    assert first_limits[0] == first_limits[1]
    for state in states[1:]:
        assert state.keys() == states[0].keys()
        for key in state:
            torch.testing.assert_close(state[key], states[0][key], rtol=0, atol=0)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("overrides", "expected_first_ends"),
    [
        ({"first_episode_length": 3}, [3] * 6),
        ({"first_episode_lengths": [1, 2, 3, 4, 5, 6]}, [1, 2, 3, 4, 5, 6]),
        ({"first_episode_length": None}, [8] * 6),
        ({"first_episode_lengths": None}, [8] * 6),
    ],
)
def test_explicit_first_episode_settings_override_training_staggering(
    overrides: dict[str, Any], expected_first_ends: list[int],
) -> None:
    kwargs = _training_kwargs()
    kwargs["env_kwargs"].update(overrides)
    algorithm = make_training("SwarmBots-MoveToGoal-v0", "ppo", **kwargs)
    try:
        episode_ends = _episode_end_steps(algorithm.env.unwrapped, num_steps=8)
        assert [ends[0] for ends in episode_ends] == expected_first_ends
    finally:
        algorithm.env.close()


@pytest.mark.integration
def test_policy_environment_construction_keeps_full_initial_episodes() -> None:
    kwargs = _training_kwargs()
    policy_options, _ = _variant_options("ppo", model_scale=None)
    policy_options.update(kwargs.pop("policy_kwargs"), use_nop=False)
    kwargs.pop("model_scale")
    kwargs.pop("use_nop")
    env, _ = _make_policy_env("SwarmBots-MoveToGoal-v0", policy_options=policy_options, **kwargs)
    try:
        episode_ends = _episode_end_steps(env.unwrapped, num_steps=8)
        assert [ends[0] for ends in episode_ends] == [8] * 6
    finally:
        env.close()
