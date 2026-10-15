from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import pytest
import torch

from swarmbots.learn import make_training
from swarmbots.learn.training import _make_policy_env, _variant_options
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


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["ppo", "masac__deepset_critic"])
def test_training_staggers_initial_episodes_and_then_uses_full_lengths(variant: str) -> None:
    kwargs = _training_kwargs()
    algorithm = make_training("SwarmBots-MoveToGoal-v0", variant, **kwargs)
    try:
        episode_ends = _episode_end_steps(algorithm.env.unwrapped, num_steps=24)
        first_ends = [ends[0] for ends in episode_ends]
        assert all(1 <= step <= 8 for step in first_ends)
        assert len(set(first_ends)) > 1
        for ends in episode_ends:
            assert len(ends) >= 3
            assert all(second - first == 8 for first, second in zip(ends, ends[1:]))
        assert kwargs["env_kwargs"] == {"compile_tensor_operations": False}
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
