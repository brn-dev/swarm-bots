import unittest

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.r_mat.r_ppo_wm_sampler import RPPOWMSampler, RPPOWMSamplerConfig
from swarmbots.learn.algos.world_modeling.wm_recurrent_batch import (
    build_wm_target_time_mask,
    flatten_recurrent_wm_batch,
)


def _make_episode(
        *,
        num_steps: int,
        is_true_episode_start: bool = True,
        with_agent_mask: bool = True,
) -> PPOEpisodeSegment:
    n_agents = 2
    n_local_obs = 3
    n_global_obs = 2
    n_hidden_local = 1
    n_hidden_global = 1
    n_actions = 2

    step_values = torch.arange(num_steps, dtype=torch.float32)
    local_obs = (
        step_values.view(num_steps, 1, 1) * 100.0
        + torch.arange(n_agents, dtype=torch.float32).view(1, n_agents, 1) * 10.0
        + torch.arange(n_local_obs, dtype=torch.float32).view(1, 1, n_local_obs)
    )
    global_obs = step_values.view(num_steps, 1) * 100.0 + torch.arange(n_global_obs, dtype=torch.float32)
    hidden_local_vars = step_values.view(num_steps, 1, 1) + torch.arange(n_agents).view(1, n_agents, 1)
    hidden_global_vars = step_values.view(num_steps, 1)
    actions = (
        1000.0
        + step_values.view(num_steps, 1, 1) * 100.0
        + torch.arange(n_agents, dtype=torch.float32).view(1, n_agents, 1) * 10.0
        + torch.arange(n_actions, dtype=torch.float32).view(1, 1, n_actions)
    )
    log_probs = 2000.0 + step_values.view(num_steps, 1) + torch.arange(n_agents, dtype=torch.float32)
    values = 3000.0 + step_values
    rewards = 4000.0 + step_values
    returns = 5000.0 + step_values
    advantages = 6000.0 + step_values

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
        final_local_obs=9000.0 + torch.arange(n_agents * n_local_obs, dtype=torch.float32).view(n_agents, n_local_obs),
        final_global_obs=9100.0 + torch.arange(n_global_obs, dtype=torch.float32),
        final_hidden_local_vars=9200.0 + torch.arange(n_agents * n_hidden_local, dtype=torch.float32).view(
            n_agents,
            n_hidden_local,
        ),
        final_hidden_global_vars=9300.0 + torch.arange(n_hidden_global, dtype=torch.float32),
        final_agent_mask=final_agent_mask,
        final_value=torch.tensor(9400.0),
        initial_previous_actions=9500.0 + torch.arange(n_agents * n_actions, dtype=torch.float32).view(
            n_agents,
            n_actions,
        ),
        is_true_episode_start=is_true_episode_start,
        returns=returns,
        advantages=advantages,
    )


class RPPOWMSamplerTests(unittest.TestCase):
    def test_burn_in_chunks_overlap_and_mask_only_train_steps(self) -> None:
        episode = _make_episode(num_steps=7)

        sampler = RPPOWMSampler(
            episodes=[episode],
            config=RPPOWMSamplerConfig(batch_size=2, num_next_steps=3, sequence_length=4, burn_in_length=2),
            requires_previous_actions=True,
        )

        self.assertEqual(sampler.local_obs.shape[:2], (3, 4))
        self.assertTrue(torch.equal(sampler.local_obs[:, :, 0, 0], torch.tensor([
            [0.0, 100.0, 200.0, 300.0],
            [200.0, 300.0, 400.0, 500.0],
            [400.0, 500.0, 600.0, 0.0],
        ])))
        self.assertTrue(torch.equal(sampler.time_mask, torch.tensor([
            [True, True, True, True],
            [True, True, True, True],
            [True, True, True, False],
        ])))
        self.assertTrue(torch.equal(sampler.time_loss_mask, torch.tensor([
            [True, True, True, True],
            [False, False, True, True],
            [False, False, True, False],
        ])))
        self.assertTrue(torch.equal(sampler.is_true_episode_start, torch.tensor([True, False, False])))

        assert sampler.previous_actions is not None
        self.assertTrue(torch.equal(sampler.previous_actions[0, 0], episode.initial_previous_actions))
        self.assertTrue(torch.equal(sampler.previous_actions[0, 1], episode.actions[0]))
        self.assertTrue(torch.equal(sampler.previous_actions[1, 0], episode.actions[1]))
        self.assertTrue(torch.equal(sampler.previous_actions[2, 3], torch.zeros_like(episode.actions[0])))

        assert sampler.agent_mask is not None
        assert sampler.wm_agent_mask is not None
        assert sampler.wm_loss_agent_mask is not None
        self.assertTrue(torch.equal(sampler.agent_mask[2], torch.tensor([
            [True, True],
            [True, False],
            [True, True],
            [True, True],
        ])))
        self.assertTrue(torch.equal(sampler.wm_target_time_mask[2], torch.tensor([
            [True, True, True],
            [True, True, False],
            [True, False, False],
            [False, False, False],
        ])))
        self.assertTrue(torch.equal(sampler.wm_actions[2, 2, 0], episode.actions[6]))
        self.assertTrue(torch.equal(sampler.next_local_obs[2, 2, 0], episode.final_local_obs))
        self.assertTrue(torch.equal(sampler.next_global_obs[2, 2, 0], episode.final_global_obs))

    def test_mid_episode_first_chunk_treats_burn_in_as_state_warmup(self) -> None:
        episode = _make_episode(num_steps=3, is_true_episode_start=False)

        sampler = RPPOWMSampler(
            episodes=[episode],
            config=RPPOWMSamplerConfig(batch_size=1, num_next_steps=1, sequence_length=4, burn_in_length=2),
        )

        self.assertTrue(torch.equal(sampler.time_mask, torch.tensor([[True, True, True, False]])))
        self.assertTrue(torch.equal(sampler.time_loss_mask, torch.tensor([[False, False, True, False]])))
        self.assertTrue(torch.equal(sampler.is_true_episode_start, torch.tensor([False])))

    def test_rejects_empty_segments_and_mixed_agent_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one non-empty"):
            RPPOWMSampler(
                episodes=[_make_episode(num_steps=0)],
                config=RPPOWMSamplerConfig(batch_size=1, num_next_steps=1, sequence_length=2),
            )

        with self.assertRaisesRegex(ValueError, "agent_mask must be provided for all episodes or none"):
            RPPOWMSampler(
                episodes=[
                    _make_episode(num_steps=1, with_agent_mask=True),
                    _make_episode(num_steps=1, with_agent_mask=False),
                ],
                config=RPPOWMSamplerConfig(batch_size=1, num_next_steps=1, sequence_length=2),
            )

    def test_rejects_invalid_recurrent_sampler_config(self) -> None:
        valid_episode = _make_episode(num_steps=1)
        invalid_configs = [
            (RPPOWMSamplerConfig(batch_size=1, num_next_steps=1, sequence_length=0), "sequence_length"),
            (RPPOWMSamplerConfig(batch_size=1, num_next_steps=1, sequence_length=2, burn_in_length=-1), "burn_in"),
            (RPPOWMSamplerConfig(batch_size=1, num_next_steps=1, sequence_length=2, burn_in_length=2), "burn_in"),
            (RPPOWMSamplerConfig(batch_size=1, num_next_steps=0, sequence_length=2), "num_next_steps"),
        ]
        for config, message in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, message):
                    RPPOWMSampler(episodes=[valid_episode], config=config)


class RecurrentWMBatchHelperTests(unittest.TestCase):
    def test_flatten_recurrent_wm_batch_preserves_row_major_order_and_optional_masks(self) -> None:
        local_latents = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
        next_local_obs = torch.arange(2 * 3 * 2 * 2 * 5, dtype=torch.float32).reshape(2, 3, 2, 2, 5)
        actions = torch.arange(2 * 3 * 2 * 2 * 1, dtype=torch.float32).reshape(2, 3, 2, 2, 1)
        local_obs = torch.arange(2 * 3 * 2 * 6, dtype=torch.float32).reshape(2, 3, 2, 6)
        next_global_obs = torch.arange(2 * 3 * 2 * 7, dtype=torch.float32).reshape(2, 3, 2, 7)
        agent_mask = torch.tensor([
            [[True, True], [True, False], [False, False]],
            [[True, False], [True, True], [False, True]],
        ])
        loss_agent_mask = ~agent_mask
        time_mask = torch.tensor([
            [[True, True], [True, False], [False, False]],
            [[True, False], [True, True], [False, True]],
        ])

        flattened = flatten_recurrent_wm_batch(
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            actions=actions,
            local_obs=local_obs,
            next_global_obs=next_global_obs,
            agent_mask=agent_mask,
            loss_agent_mask=loss_agent_mask,
            time_mask=time_mask,
        )

        self.assertIsNotNone(flattened)
        assert flattened is not None
        self.assertTrue(torch.equal(flattened.local_latents[0], local_latents[0, 0]))
        self.assertTrue(torch.equal(flattened.local_latents[3], local_latents[1, 0]))
        self.assertTrue(torch.equal(flattened.next_local_obs[5], next_local_obs[1, 2]))
        self.assertTrue(torch.equal(flattened.actions[4], actions[1, 1]))
        self.assertTrue(torch.equal(flattened.local_obs[2], local_obs[0, 2]))
        self.assertTrue(torch.equal(flattened.next_global_obs[1], next_global_obs[0, 1]))
        self.assertTrue(torch.equal(flattened.agent_mask, agent_mask.reshape(6, 2)))
        self.assertTrue(torch.equal(flattened.loss_agent_mask, loss_agent_mask.reshape(6, 2)))
        self.assertTrue(torch.equal(flattened.time_mask, time_mask.reshape(6, 2)))

    def test_flatten_recurrent_wm_batch_returns_none_for_flat_batches(self) -> None:
        result = flatten_recurrent_wm_batch(
            local_latents=torch.zeros(2, 3, 4),
            next_local_obs=torch.zeros(2, 1, 3, 5),
            actions=torch.zeros(2, 1, 3, 2),
        )
        self.assertIsNone(result)

    def test_build_wm_target_time_mask_combines_window_and_loss_masks(self) -> None:
        wm_target_time_mask = torch.tensor([
            [[True, True, False], [True, False, False], [True, True, True]],
            [[True, True, True], [False, False, False], [True, False, True]],
        ])
        time_loss_mask = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])

        actual = build_wm_target_time_mask(
            wm_target_time_mask=wm_target_time_mask,
            time_loss_mask=time_loss_mask,
        )

        self.assertTrue(torch.equal(actual, torch.tensor([
            [[True, True, False], [False, False, False], [True, True, True]],
            [[False, False, False], [False, False, False], [True, False, True]],
        ])))

    def test_recurrent_wm_helpers_reject_bad_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_mask dtype"):
            flatten_recurrent_wm_batch(
                local_latents=torch.zeros(1, 2, 3, 4),
                next_local_obs=torch.zeros(1, 2, 1, 3, 5),
                actions=torch.zeros(1, 2, 1, 3, 2),
                time_mask=torch.ones(1, 2, 1),
            )

        with self.assertRaisesRegex(ValueError, "agent_mask ndim"):
            flatten_recurrent_wm_batch(
                local_latents=torch.zeros(1, 2, 3, 4),
                next_local_obs=torch.zeros(1, 2, 1, 3, 5),
                actions=torch.zeros(1, 2, 1, 3, 2),
                agent_mask=torch.ones(1, 2, 1, 3, 1, dtype=torch.bool),
            )

        with self.assertRaisesRegex(ValueError, "prefix shape"):
            build_wm_target_time_mask(
                wm_target_time_mask=torch.ones(2, 3, 1, dtype=torch.bool),
                time_loss_mask=torch.ones(2, 4, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
