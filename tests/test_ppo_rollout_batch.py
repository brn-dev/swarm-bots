import unittest

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_batch import PPORolloutBatch, compute_step_rollout_gae
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMBatchSampler, PPOWMSamplerConfig
from swarmbots.learn.algos.world_modeling.wm_sampler_helper import build_wm_episode_windows


def _make_rollout_batch(*, with_agent_mask: bool) -> PPORolloutBatch:
    n_envs = 2
    n_steps = 5
    n_agents = 2
    n_local_obs = 3
    n_global_obs = 2
    n_hidden_local = 1
    n_hidden_global = 1
    n_actions = 2

    step_values = torch.arange(n_envs * n_steps, dtype=torch.float32).view(n_envs, n_steps)
    local_obs = (
        step_values.view(n_envs, n_steps, 1, 1) * 100.0
        + torch.arange(n_agents, dtype=torch.float32).view(1, 1, n_agents, 1) * 10.0
        + torch.arange(n_local_obs, dtype=torch.float32).view(1, 1, 1, n_local_obs)
    )
    global_obs = (
        1000.0
        + step_values.view(n_envs, n_steps, 1)
        + torch.arange(n_global_obs, dtype=torch.float32).view(1, 1, n_global_obs)
    )
    hidden_local_vars = (
        2000.0
        + step_values.view(n_envs, n_steps, 1, 1)
        + torch.arange(n_agents, dtype=torch.float32).view(1, 1, n_agents, 1)
    )
    hidden_global_vars = 3000.0 + step_values.view(n_envs, n_steps, 1)
    actions = (
        4000.0
        + step_values.view(n_envs, n_steps, 1, 1) * 100.0
        + torch.arange(n_agents, dtype=torch.float32).view(1, 1, n_agents, 1) * 10.0
        + torch.arange(n_actions, dtype=torch.float32).view(1, 1, 1, n_actions)
    )
    previous_actions = actions - 50.0
    log_probs = 5000.0 + step_values.view(n_envs, n_steps, 1) + torch.arange(n_agents, dtype=torch.float32)
    values = 10.0 + step_values
    rewards = 1.0 + step_values / 10.0
    bootstrap_values = 20.0 + step_values / 7.0

    dones = torch.tensor(
        [
            [False, True, False, False, False],
            [True, False, False, True, False],
        ],
        dtype=torch.bool,
    )
    terminations = torch.tensor(
        [
            [False, True, False, False, False],
            [False, False, False, False, False],
        ],
        dtype=torch.bool,
    )
    truncations = dones & ~terminations
    bootstrap_values = bootstrap_values.masked_fill(terminations, 0.0)
    episode_start_mask = torch.tensor(
        [
            [True, False, False, True, False],
            [True, False, True, False, False],
        ],
        dtype=torch.bool,
    )

    bootstrap_local_obs = local_obs + 0.5
    bootstrap_global_obs = global_obs + 0.5
    bootstrap_hidden_local_vars = hidden_local_vars + 0.5
    bootstrap_hidden_global_vars = hidden_global_vars + 0.5
    nonterminal_with_next_step = ~dones[:, :-1]
    bootstrap_local_obs[:, :-1] = torch.where(
        nonterminal_with_next_step.view(n_envs, n_steps - 1, 1, 1),
        local_obs[:, 1:],
        bootstrap_local_obs[:, :-1],
    )
    bootstrap_global_obs[:, :-1] = torch.where(
        nonterminal_with_next_step.view(n_envs, n_steps - 1, 1),
        global_obs[:, 1:],
        bootstrap_global_obs[:, :-1],
    )
    bootstrap_hidden_local_vars[:, :-1] = torch.where(
        nonterminal_with_next_step.view(n_envs, n_steps - 1, 1, 1),
        hidden_local_vars[:, 1:],
        bootstrap_hidden_local_vars[:, :-1],
    )
    bootstrap_hidden_global_vars[:, :-1] = torch.where(
        nonterminal_with_next_step.view(n_envs, n_steps - 1, 1),
        hidden_global_vars[:, 1:],
        bootstrap_hidden_global_vars[:, :-1],
    )

    agent_mask = None
    bootstrap_agent_mask = None
    if with_agent_mask:
        agent_mask = torch.tensor(
            [
                [[True, True], [True, False], [True, True], [True, True], [True, False]],
                [[True, False], [True, True], [True, True], [True, False], [True, True]],
            ],
            dtype=torch.bool,
        )
        bootstrap_agent_mask = ~agent_mask
        bootstrap_agent_mask[:, :-1] = torch.where(
            nonterminal_with_next_step.view(n_envs, n_steps - 1, 1),
            agent_mask[:, 1:],
            bootstrap_agent_mask[:, :-1],
        )

    advantages = compute_step_rollout_gae(
        rewards=rewards,
        values=values,
        bootstrap_values=bootstrap_values,
        dones=dones,
        gamma=0.9,
        gae_lambda=0.8,
    )
    return PPORolloutBatch(
        local_obs=local_obs,
        global_obs=global_obs,
        hidden_local_vars=hidden_local_vars,
        hidden_global_vars=hidden_global_vars,
        agent_mask=agent_mask,
        previous_actions=previous_actions,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
        values=values,
        bootstrap_local_obs=bootstrap_local_obs,
        bootstrap_global_obs=bootstrap_global_obs,
        bootstrap_hidden_local_vars=bootstrap_hidden_local_vars,
        bootstrap_hidden_global_vars=bootstrap_hidden_global_vars,
        bootstrap_agent_mask=bootstrap_agent_mask,
        bootstrap_values=bootstrap_values,
        terminations=terminations,
        truncations=truncations,
        dones=dones,
        episode_start_mask=episode_start_mask,
        returns=advantages + values,
        advantages=advantages,
    )


def _split_rollout_batch(batch: PPORolloutBatch) -> list[tuple[int, int, PPOEpisodeSegment]]:
    episodes: list[tuple[int, int, PPOEpisodeSegment]] = []
    for env_idx in range(batch.n_envs):
        start_idx = 0
        for step_idx in range(batch.n_steps):
            if not bool(batch.dones[env_idx, step_idx]) and step_idx != batch.n_steps - 1:
                continue
            end_idx = step_idx + 1
            episode = PPOEpisodeSegment(
                local_obs=batch.local_obs[env_idx, start_idx:end_idx],
                global_obs=batch.global_obs[env_idx, start_idx:end_idx],
                hidden_local_vars=batch.hidden_local_vars[env_idx, start_idx:end_idx],
                hidden_global_vars=batch.hidden_global_vars[env_idx, start_idx:end_idx],
                agent_mask=None if batch.agent_mask is None else batch.agent_mask[env_idx, start_idx:end_idx],
                actions=batch.actions[env_idx, start_idx:end_idx],
                rewards=batch.rewards[env_idx, start_idx:end_idx],
                log_probs=batch.log_probs[env_idx, start_idx:end_idx],
                values=batch.values[env_idx, start_idx:end_idx],
                final_local_obs=batch.bootstrap_local_obs[env_idx, step_idx],
                final_global_obs=batch.bootstrap_global_obs[env_idx, step_idx],
                final_hidden_local_vars=batch.bootstrap_hidden_local_vars[env_idx, step_idx],
                final_hidden_global_vars=batch.bootstrap_hidden_global_vars[env_idx, step_idx],
                final_agent_mask=(
                    None if batch.bootstrap_agent_mask is None else batch.bootstrap_agent_mask[env_idx, step_idx]
                ),
                final_value=batch.bootstrap_values[env_idx, step_idx],
                initial_previous_actions=batch.previous_actions[env_idx, start_idx],
                is_true_episode_start=bool(batch.episode_start_mask[env_idx, start_idx]),
            )
            episode.compute_gae(gamma=0.9, gae_lambda=0.8)
            episodes.append((env_idx, start_idx, episode))
            start_idx = end_idx
    return episodes


class PPORolloutBatchTests(unittest.TestCase):
    def test_gae_bootstraps_truncation_without_crossing_into_reset_episode(self) -> None:
        rewards = torch.tensor([
            [0.0, 100.0],
            [0.0, 100.0],
        ])
        values = torch.zeros_like(rewards)
        bootstrap_values = torch.tensor([
            [5.0, 0.0],
            [0.0, 0.0],
        ])
        dones = torch.tensor([
            [True, False],
            [True, False],
        ])

        advantages = compute_step_rollout_gae(
            rewards=rewards,
            values=values,
            bootstrap_values=bootstrap_values,
            dones=dones,
            gamma=1.0,
            gae_lambda=1.0,
        )

        torch.testing.assert_close(
            advantages,
            torch.tensor([
                [5.0, 100.0],
                [0.0, 100.0],
            ]),
        )

    def test_vectorized_gae_matches_episode_segments(self) -> None:
        batch = _make_rollout_batch(with_agent_mask=True)
        for env_idx, start_idx, episode in _split_rollout_batch(batch):
            end_idx = start_idx + int(episode.rewards.shape[0])
            torch.testing.assert_close(batch.advantages[env_idx, start_idx:end_idx], episode.advantages)
            torch.testing.assert_close(batch.returns[env_idx, start_idx:end_idx], episode.returns)

    def test_wm_batch_sampler_matches_episode_windows_at_boundaries(self) -> None:
        for with_agent_mask in (False, True):
            with self.subTest(with_agent_mask=with_agent_mask):
                batch = _make_rollout_batch(with_agent_mask=with_agent_mask)
                num_next_steps = 4
                sampler = PPOWMBatchSampler(
                    rollout_batch=batch,
                    config=PPOWMSamplerConfig(batch_size=3, num_next_steps=num_next_steps),
                    requires_previous_actions=True,
                )

                expected_actions = torch.zeros_like(sampler.multi_step_actions)
                expected_next_local_obs = torch.zeros_like(sampler.next_local_obs)
                expected_time_mask = torch.zeros_like(sampler.wm_target_time_mask)
                expected_next_global_obs = torch.zeros_like(sampler.next_global_obs)
                expected_agent_mask = None if sampler.wm_agent_mask is None else torch.ones_like(sampler.wm_agent_mask)
                expected_loss_agent_mask = (
                    None if sampler.wm_loss_agent_mask is None else torch.ones_like(sampler.wm_loss_agent_mask)
                )

                for env_idx, start_idx, episode in _split_rollout_batch(batch):
                    windows = build_wm_episode_windows(episode, num_next_steps=num_next_steps)
                    length = int(episode.actions.shape[0])
                    flat_start = env_idx * batch.n_steps + start_idx
                    flat_end = flat_start + length
                    expected_actions[flat_start:flat_end] = windows.multi_step_actions
                    expected_next_local_obs[flat_start:flat_end] = windows.next_local_obs
                    expected_time_mask[flat_start:flat_end] = windows.wm_target_time_mask
                    expected_next_global_obs[flat_start:flat_end] = windows.next_global_obs
                    if expected_agent_mask is not None:
                        assert windows.wm_agent_mask is not None
                        expected_agent_mask[flat_start:flat_end] = windows.wm_agent_mask
                    if expected_loss_agent_mask is not None:
                        assert windows.wm_loss_agent_mask is not None
                        expected_loss_agent_mask[flat_start:flat_end] = windows.wm_loss_agent_mask

                self.assertTrue(torch.equal(sampler.multi_step_actions, expected_actions))
                self.assertTrue(torch.equal(sampler.next_local_obs, expected_next_local_obs))
                self.assertTrue(torch.equal(sampler.wm_target_time_mask, expected_time_mask))
                self.assertTrue(torch.equal(sampler.next_global_obs, expected_next_global_obs))
                if with_agent_mask:
                    self.assertTrue(torch.equal(sampler.wm_agent_mask, expected_agent_mask))
                    self.assertTrue(torch.equal(sampler.wm_loss_agent_mask, expected_loss_agent_mask))
                else:
                    self.assertIsNone(sampler.wm_agent_mask)
                    self.assertIsNone(sampler.wm_loss_agent_mask)


if __name__ == "__main__":
    unittest.main()
