import unittest

import torch
from torch import nn

from swarmbots.learn.algos.ppo.ppo import PPO
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples


def _make_samples() -> PPOSamples:
    return PPOSamples(
        local_obs=torch.empty(2, 3, 2, 0),
        global_obs=torch.empty(2, 3, 0),
        hidden_local_vars=torch.empty(2, 3, 2, 0),
        hidden_global_vars=torch.empty(2, 3, 0),
        agent_mask=torch.tensor([
            [[True, True], [True, False], [False, False]],
            [[True, False], [True, True], [False, True]],
        ]),
        previous_actions=None,
        actions=torch.arange(2 * 3 * 2 * 2, dtype=torch.float32).reshape(2, 3, 2, 2),
        log_probs=torch.zeros(2, 3, 2),
        values=torch.zeros(2, 3),
        returns=torch.zeros(2, 3),
        advantages=torch.tensor([
            [1.0, 100.0, 3.0],
            [200.0, 5.0, 6.0],
        ]),
    )


def _ppo_with_reduction(agent_logprob_reduction: str | None) -> PPO:
    ppo = PPO.__new__(PPO)
    ppo.agent_logprob_reduction = agent_logprob_reduction
    ppo.value_loss_fn = nn.MSELoss(reduction="none")
    return ppo


class PPOMaskingTests(unittest.TestCase):
    def test_time_loss_mask_takes_precedence_over_time_mask_for_valid_items(self) -> None:
        batch = _make_samples()
        batch.time_mask = torch.tensor([
            [True, True, True],
            [True, True, True],
        ])
        batch.time_loss_mask = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])
        target = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)

        valid_mask = PPO._combine_valid_masks(
            agent_mask=batch.agent_mask,
            time_mask=PPO._get_time_loss_mask(batch),
            target=target,
        )

        self.assertTrue(torch.equal(valid_mask, torch.tensor([
            [[True, True], [False, False], [False, False]],
            [[False, False], [True, True], [False, True]],
        ])))

    def test_normalize_advantages_uses_only_unmasked_loss_steps_and_zeroes_masked_steps(self) -> None:
        batch = _make_samples()
        batch.time_mask = torch.ones(2, 3, dtype=torch.bool)
        batch.time_loss_mask = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])
        ppo = _ppo_with_reduction(None)

        normalized = ppo._normalize_advantages(batch, batch.advantages)

        valid_values = torch.tensor([1.0, 3.0, 5.0, 6.0])
        mean = valid_values.mean()
        variance = ((valid_values - mean) ** 2).mean()
        expected = torch.zeros_like(batch.advantages)
        expected[batch.time_loss_mask] = (valid_values - mean) / torch.sqrt(variance + 1e-8)
        self.assertTrue(torch.allclose(normalized, expected))

    def test_extra_loss_reduction_masks_agents_and_burn_in_without_agent_reduction(self) -> None:
        batch = _make_samples()
        batch.time_loss_mask = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])
        value = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
        ppo = _ppo_with_reduction(None)

        reduced = ppo._reduce_extra_loss_value(batch, value)

        expected_mask = batch.agent_mask & batch.time_loss_mask.unsqueeze(-1)
        self.assertTrue(torch.equal(value[expected_mask], torch.tensor([0.0, 1.0, 8.0, 9.0, 11.0])))
        self.assertTrue(torch.equal(reduced, value[expected_mask].mean()))

    def test_extra_loss_reduction_averages_active_agents_before_time_mask(self) -> None:
        batch = _make_samples()
        batch.time_loss_mask = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])
        value = torch.tensor([
            [[1.0, 3.0], [10.0, 100.0], [1000.0, 2000.0]],
            [[20.0, 200.0], [5.0, 7.0], [300.0, 9.0]],
        ])
        ppo = _ppo_with_reduction("mean")

        reduced = ppo._reduce_extra_loss_value(batch, value)

        per_step_agent_mean = torch.tensor([
            [2.0, 10.0, 0.0],
            [20.0, 6.0, 9.0],
        ])
        valid_step_mask = torch.tensor([
            [True, False, False],
            [False, True, True],
        ])
        self.assertTrue(torch.equal(reduced, per_step_agent_mean[valid_step_mask].mean()))

    def test_action_metrics_drop_padded_burn_in_and_inactive_agents(self) -> None:
        batch = _make_samples()
        batch.time_mask = torch.ones(2, 3, dtype=torch.bool)
        batch.time_loss_mask = torch.tensor([
            [False, True, False],
            [True, True, False],
        ])

        actions = PPO._get_action_metrics_actions(batch)

        expected_valid_mask = batch.agent_mask & batch.time_loss_mask.unsqueeze(-1)
        self.assertTrue(torch.equal(actions, batch.actions[expected_valid_mask]))

    def test_value_loss_requires_none_reduction_and_applies_valid_mask(self) -> None:
        with self.assertRaisesRegex(ValueError, "reduction='none'"):
            PPO._validate_value_loss_fn(nn.MSELoss(reduction="mean"))

        ppo = _ppo_with_reduction(None)
        values_pred = torch.tensor([[1.0, 10.0], [2.0, 20.0]])
        value_targets = torch.zeros_like(values_pred)
        valid_mask = torch.tensor([[True, False], [True, False]])

        loss = ppo._compute_masked_value_loss(
            values_pred=values_pred,
            value_targets=value_targets,
            valid_mask=valid_mask,
        )

        self.assertTrue(torch.equal(loss, torch.tensor(2.5)))


if __name__ == "__main__":
    unittest.main()
