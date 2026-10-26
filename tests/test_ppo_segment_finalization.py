from collections.abc import Iterator

import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPORolloutBuffer
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as SwarmTestEnv


DEVICES = ["cpu", pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])]


@pytest.fixture
def buffer(request: pytest.FixtureRequest) -> Iterator[PPORolloutBuffer]:
    device = request.param
    env = SwarmBotsLearnEnvWrapper(SyncVectorEnv(
        [lambda: SwarmTestEnv(2, 3, 1, 1, 1, max_steps=8) for _ in range(5)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    ))
    try:
        yield PPORolloutBuffer(
            max_episode_length=4, observation_space=env.observation_space, action_space=env.action_space,
            gamma=.9, gae_lambda=.8, rollout_device=device, train_device=device,
        )
    finally:
        env.close()


@pytest.mark.parametrize("buffer", DEVICES, indirect=True)
@pytest.mark.parametrize("clone_tensors", [False, True])
def test_partial_segments_preserve_lambda_returns_metadata_and_storage(
    buffer: PPORolloutBuffer, clone_tensors: bool,
) -> None:
    accumulator = buffer.accumulator
    device = accumulator.rewards.device
    lengths = [4, 0, 1, 3, 4]
    accumulator.step.copy_(torch.tensor(lengths, device=device))
    accumulator.is_true_episode_start.copy_(torch.tensor([True, False, False, True, False], device=device))
    accumulator.rollout_start_steps.copy_(torch.tensor([30, 34, 33, 31, 30], device=device))
    accumulator.initial_temporal_state = {
        "actor": torch.arange(30., device=device).reshape(5, 2, 3).requires_grad_(),
        "critic": (torch.arange(10., device=device).reshape(5, 2),),
    }
    accumulator.rewards.copy_(torch.arange(20., device=device).reshape(5, 4) / 7)
    accumulator.values.copy_(torch.arange(20., device=device).reshape(5, 4) / 3)
    for env_idx, length in enumerate(lengths):
        accumulator.rewards[env_idx, length:] = float("nan")
        accumulator.values[env_idx, length:] = float("nan")
    final_values = torch.arange(10., 15., device=device)
    final_obs = {
        name: getattr(accumulator, name)[:, -1]
        for name in ("local_obs", "global_obs", "hidden_local_vars", "hidden_global_vars")
    }

    episodes = buffer.dump_partial_episodes(
        final_obs=final_obs, final_values=final_values, clone_tensors=clone_tensors,
    )
    assert [episode.rollout_env_idx for episode in episodes] == [0, 2, 3, 4]
    for episode in episodes:
        env_idx = episode.rollout_env_idx
        length = lengths[env_idx]
        rewards = accumulator.rewards[env_idx, :length]
        values = accumulator.values[env_idx, :length]
        successors = torch.cat((values[1:], final_values[env_idx:env_idx + 1]))
        deltas = rewards + buffer.gamma * successors - values
        expected_advantages = torch.stack([
            sum((buffer.gamma * buffer.gae_lambda) ** offset * deltas[step + offset]
                for offset in range(length - step))
            for step in range(length)
        ])
        torch.testing.assert_close(episode.advantages, expected_advantages)
        torch.testing.assert_close(episode.returns, expected_advantages + values)
        assert episode.is_true_episode_start == (env_idx in (0, 3))
        assert episode.rollout_start_step == [30, 34, 33, 31, 30][env_idx]
        torch.testing.assert_close(
            episode.initial_temporal_state["actor"],
            accumulator.initial_temporal_state["actor"][env_idx:env_idx + 1],
        )
        assert not episode.initial_temporal_state["actor"].requires_grad
        assert (episode.local_obs.data_ptr() != accumulator.local_obs[env_idx].data_ptr()) == clone_tensors
        assert (episode.final_local_obs.data_ptr() != final_obs["local_obs"][env_idx].data_ptr()) == clone_tensors

    saved_returns = [episode.returns.clone() for episode in episodes]
    saved_actor_states = [episode.initial_temporal_state["actor"].clone() for episode in episodes]
    buffer.reset()
    accumulator.rewards.fill_(100.)
    accumulator.values.fill_(200.)
    for episode, returns, actor_state in zip(episodes, saved_returns, saved_actor_states, strict=True):
        torch.testing.assert_close(episode.returns, returns)
        torch.testing.assert_close(episode.initial_temporal_state["actor"], actor_state)


@pytest.mark.parametrize("buffer", DEVICES, indirect=True)
def test_empty_partial_segments_produce_no_episodes(buffer: PPORolloutBuffer) -> None:
    accumulator = buffer.accumulator
    final_obs = {
        name: getattr(accumulator, name)[:, -1]
        for name in ("local_obs", "global_obs", "hidden_local_vars", "hidden_global_vars")
    }
    assert buffer.dump_partial_episodes(
        final_obs=final_obs, final_values=torch.zeros(5, device=accumulator.rewards.device),
    ) == []
