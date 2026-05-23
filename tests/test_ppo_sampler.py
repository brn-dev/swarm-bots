import unittest
from unittest.mock import patch

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSampler, PPOSamplerConfig


def _make_episode(
        *,
        base_value: float,
        num_steps: int,
        with_agent_mask: bool,
) -> PPOEpisodeSegment:
    n_agents = 2
    n_local_obs = 3
    n_global_obs = 2
    n_hidden_local_vars = 1
    n_hidden_global_vars = 1
    n_actions = 2

    step_values = torch.arange(num_steps, dtype=torch.float32)
    local_obs = (
        base_value
        + step_values.view(num_steps, 1, 1) * 100.0
        + torch.arange(n_agents, dtype=torch.float32).view(1, n_agents, 1) * 10.0
        + torch.arange(n_local_obs, dtype=torch.float32).view(1, 1, n_local_obs)
    )
    global_obs = base_value + 1000.0 + step_values.view(num_steps, 1) + torch.arange(n_global_obs)
    hidden_local_vars = (
        base_value
        + 2000.0
        + step_values.view(num_steps, 1, 1)
        + torch.arange(n_agents * n_hidden_local_vars, dtype=torch.float32).view(1, n_agents, n_hidden_local_vars)
    )
    hidden_global_vars = base_value + 3000.0 + step_values.view(num_steps, 1)
    actions = (
        base_value
        + 4000.0
        + step_values.view(num_steps, 1, 1) * 100.0
        + torch.arange(n_agents, dtype=torch.float32).view(1, n_agents, 1) * 10.0
        + torch.arange(n_actions, dtype=torch.float32).view(1, 1, n_actions)
    )
    log_probs = base_value + 5000.0 + step_values.view(num_steps, 1) + torch.arange(n_agents)
    values = base_value + 6000.0 + step_values
    rewards = base_value + 7000.0 + step_values
    returns = base_value + 8000.0 + step_values
    advantages = base_value + 9000.0 + step_values

    agent_mask = None
    final_agent_mask = None
    if with_agent_mask:
        agent_mask = torch.tensor(
            [[True, step_idx % 2 == 0] for step_idx in range(num_steps)],
            dtype=torch.bool,
        )
        final_agent_mask = torch.tensor([True, num_steps % 2 == 0], dtype=torch.bool)

    return PPOEpisodeSegment(
        local_obs=local_obs,
        global_obs=global_obs,
        hidden_local_vars=hidden_local_vars,
        hidden_global_vars=hidden_global_vars,
        agent_mask=agent_mask,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
        values=values,
        final_local_obs=base_value + torch.zeros(n_agents, n_local_obs),
        final_global_obs=base_value + torch.zeros(n_global_obs),
        final_hidden_local_vars=base_value + torch.zeros(n_agents, n_hidden_local_vars),
        final_hidden_global_vars=base_value + torch.zeros(n_hidden_global_vars),
        final_agent_mask=final_agent_mask,
        final_value=torch.tensor(base_value),
        initial_previous_actions=base_value + 10000.0 + torch.arange(
            n_agents * n_actions,
            dtype=torch.float32,
        ).view(n_agents, n_actions),
        is_true_episode_start=True,
        returns=returns,
        advantages=advantages,
    )


class PPOSamplerTests(unittest.TestCase):
    def test_samples_keep_fields_aligned_after_permutation(self) -> None:
        first_episode = _make_episode(base_value=10.0, num_steps=2, with_agent_mask=True)
        second_episode = _make_episode(base_value=20.0, num_steps=1, with_agent_mask=True)
        sampler = PPOSampler(
            episodes=[first_episode, second_episode],
            config=PPOSamplerConfig(batch_size=2),
            requires_previous_actions=True,
        )

        with patch("torch.randperm", return_value=torch.tensor([2, 0, 1])):
            first_batch, second_batch = list(sampler.sample(drop_last=False))

        torch.testing.assert_close(
            first_batch.local_obs[:, 0, 0],
            torch.tensor([second_episode.local_obs[0, 0, 0], first_episode.local_obs[0, 0, 0]]),
        )
        torch.testing.assert_close(
            first_batch.actions[:, 0, 0],
            torch.tensor([second_episode.actions[0, 0, 0], first_episode.actions[0, 0, 0]]),
        )
        torch.testing.assert_close(
            first_batch.previous_actions[:, 0, 0],
            torch.tensor([
                second_episode.initial_previous_actions[0, 0],
                first_episode.initial_previous_actions[0, 0],
            ]),
        )
        torch.testing.assert_close(
            first_batch.returns,
            torch.tensor([second_episode.returns[0], first_episode.returns[0]]),
        )
        self.assertIsNotNone(first_batch.agent_mask)
        torch.testing.assert_close(
            first_batch.agent_mask,
            torch.stack((second_episode.agent_mask[0], first_episode.agent_mask[0])),
        )

        torch.testing.assert_close(second_batch.local_obs[0], first_episode.local_obs[1])
        torch.testing.assert_close(second_batch.previous_actions[0], first_episode.actions[0])
        torch.testing.assert_close(second_batch.advantages, first_episode.advantages[1:2])

    def test_optional_agent_mask_and_previous_actions_stay_none_when_unused(self) -> None:
        episode = _make_episode(base_value=30.0, num_steps=2, with_agent_mask=False)
        sampler = PPOSampler(
            episodes=[episode],
            config=PPOSamplerConfig(batch_size=2),
            requires_previous_actions=False,
        )

        samples = next(sampler.sample())

        self.assertIsNone(samples.agent_mask)
        self.assertIsNone(samples.previous_actions)
        torch.testing.assert_close(samples.local_obs, episode.local_obs)
        torch.testing.assert_close(samples.advantages, episode.advantages)

    def test_rejects_mixed_agent_mask_episodes(self) -> None:
        with self.assertRaisesRegex(ValueError, "agent_mask must be provided for all episodes or none"):
            PPOSampler(
                episodes=[
                    _make_episode(base_value=10.0, num_steps=1, with_agent_mask=True),
                    _make_episode(base_value=20.0, num_steps=1, with_agent_mask=False),
                ],
                config=PPOSamplerConfig(batch_size=1),
            )


if __name__ == "__main__":
    unittest.main()
