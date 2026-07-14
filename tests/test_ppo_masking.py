import unittest

import torch
from torch import nn

from swarmbots.learn.algos.ppo.ppo import PPO
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples
from swarmbots.learn.metrics_list import MetricsLists


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


class _LinearEvalPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.tensor(0.1))

    def evaluate_actions(
            self,
            batch: PPOSamples,
            action_splitter: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        _ = action_splitter
        x = batch.local_obs[:, 0, 0]
        log_probs = self.weight * x
        values = torch.zeros_like(batch.values)
        return log_probs, values, {}, {}


class _MaskedGradientPolicy(nn.Module):
    def __init__(self, batch: PPOSamples) -> None:
        super().__init__()
        self.log_prob_delta = nn.Parameter(torch.zeros_like(batch.log_probs))
        self.value_delta = nn.Parameter(
            torch.arange(batch.values.numel(), dtype=torch.float32).reshape_as(batch.values) + 1.0
        )

    def evaluate_actions(
            self,
            batch: PPOSamples,
            action_splitter: object = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        _ = batch
        _ = action_splitter
        return self.log_prob_delta, self.value_delta, {}, {}


def _ppo_for_virtual_gradient(policy: _LinearEvalPolicy, *, virtual_mini_batches: int = 1) -> PPO:
    ppo = _ppo_with_reduction(None)
    ppo.policy = policy
    ppo.metrics_action_splitters = []
    ppo.normalize_advantage = True
    ppo.virtual_mini_batches = virtual_mini_batches
    ppo.clip_range = 100.0
    ppo.clip_range_vf = None
    ppo.use_popart = False
    ppo.vf_coef = 0.0
    ppo.mc_ent_coef = 0.0
    ppo.optimizer = torch.optim.SGD(policy.parameters(), lr=0.1)
    return ppo


def _make_flat_samples() -> PPOSamples:
    batch_size = 4
    return PPOSamples(
        local_obs=torch.arange(batch_size, dtype=torch.float32).reshape(batch_size, 1, 1),
        global_obs=torch.empty(batch_size, 0),
        hidden_local_vars=torch.empty(batch_size, 1, 0),
        hidden_global_vars=torch.empty(batch_size, 0),
        agent_mask=None,
        previous_actions=None,
        actions=torch.zeros(batch_size, 1, 1),
        log_probs=torch.zeros(batch_size),
        values=torch.zeros(batch_size),
        returns=torch.zeros(batch_size),
        advantages=torch.tensor([1.0, 2.0, 4.0, 8.0]),
    )


class PPOMaskingTests(unittest.TestCase):
    def test_compute_loss_does_not_backpropagate_through_padded_steps_or_inactive_agents(self) -> None:
        batch = _make_samples()
        batch.time_mask = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])
        policy = _MaskedGradientPolicy(batch)
        ppo = _ppo_with_reduction(None)
        ppo.policy = policy
        ppo.metrics_action_splitters = []
        ppo.normalize_advantage = False
        ppo.clip_range = 100.0
        ppo.clip_range_vf = None
        ppo.use_popart = False
        ppo.vf_coef = 1.0
        ppo.mc_ent_coef = 0.0

        loss, _approx_kl, _metrics = ppo.compute_loss(batch)
        loss.backward()

        expected_log_prob_grad_mask = batch.agent_mask & batch.time_mask.unsqueeze(-1)
        expected_value_grad_mask = batch.agent_mask.any(dim=-1) & batch.time_mask
        self.assertTrue(torch.equal(policy.log_prob_delta.grad != 0.0, expected_log_prob_grad_mask))
        self.assertTrue(torch.equal(policy.value_delta.grad != 0.0, expected_value_grad_mask))

    def test_time_mask_controls_valid_items(self) -> None:
        batch = _make_samples()
        batch.time_mask = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])
        target = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)

        valid_mask = PPO._combine_valid_masks(
            agent_mask=batch.agent_mask,
            time_mask=PPO._get_time_mask(batch),
            target=target,
        )

        self.assertTrue(torch.equal(valid_mask, torch.tensor([
            [[True, True], [False, False], [False, False]],
            [[False, False], [True, True], [False, True]],
        ])))

    def test_popart_statistics_exclude_padded_and_all_inactive_steps(self) -> None:
        batch = _make_samples()
        batch.returns = torch.tensor([
            [1.0, 100.0, 200.0],
            [300.0, 5.0, 6.0],
        ])
        batch.time_mask = torch.tensor([
            [True, True, True],
            [False, True, True],
        ])
        ppo = _ppo_with_reduction(None)

        valid_returns = ppo._select_valid_value_items(batch, batch.returns)

        torch.testing.assert_close(valid_returns, torch.tensor([1.0, 100.0, 5.0, 6.0]))

    def test_normalize_advantages_uses_only_unmasked_loss_steps_and_zeroes_masked_steps(self) -> None:
        batch = _make_samples()
        batch.time_mask = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])
        ppo = _ppo_with_reduction(None)

        normalized = ppo._normalize_advantages(batch, batch.advantages)

        valid_values = torch.tensor([1.0, 3.0, 5.0, 6.0])
        mean = valid_values.mean()
        variance = ((valid_values - mean) ** 2).mean()
        expected = torch.zeros_like(batch.advantages)
        expected[batch.time_mask] = (valid_values - mean) / torch.sqrt(variance + 1e-8)
        self.assertTrue(torch.allclose(normalized, expected))

    def test_extra_loss_reduction_masks_agents_and_padded_steps_without_agent_reduction(self) -> None:
        batch = _make_samples()
        batch.time_mask = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])
        value = torch.arange(2 * 3 * 2, dtype=torch.float32).reshape(2, 3, 2)
        ppo = _ppo_with_reduction(None)

        reduced = ppo._reduce_extra_loss_value(batch, value)

        expected_mask = batch.agent_mask & batch.time_mask.unsqueeze(-1)
        self.assertTrue(torch.equal(value[expected_mask], torch.tensor([0.0, 1.0, 8.0, 9.0, 11.0])))
        self.assertTrue(torch.equal(reduced, value[expected_mask].mean()))

    def test_extra_loss_reduction_averages_active_agents_before_time_mask(self) -> None:
        batch = _make_samples()
        batch.time_mask = torch.tensor([
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

    def test_action_metrics_drop_padded_steps_and_inactive_agents(self) -> None:
        batch = _make_samples()
        batch.time_mask = torch.tensor([
            [False, True, False],
            [True, True, False],
        ])

        actions = PPO._get_action_metrics_actions(batch)

        expected_valid_mask = batch.agent_mask & batch.time_mask.unsqueeze(-1)
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

    def test_virtual_batch_split_preserves_sample_type_and_slices_fields(self) -> None:
        batch = _make_flat_samples()

        chunks = PPO._split_batch(batch, n_chunks=2)

        self.assertEqual(len(chunks), 2)
        self.assertIsInstance(chunks[0], PPOSamples)
        self.assertTrue(torch.equal(chunks[0].advantages, torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.equal(chunks[1].advantages, torch.tensor([4.0, 8.0])))
        self.assertIsNone(chunks[0].agent_mask)

    def test_virtual_batch_field_slicing_supports_nested_temporal_state(self) -> None:
        temporal_state = [
            (
                torch.arange(8, dtype=torch.float32).reshape(4, 2),
                {"normalizer": torch.arange(4, dtype=torch.float32).reshape(4, 1)},
            )
        ]

        first_chunk = PPO._slice_batch_field(temporal_state, 0, 2)
        second_chunk = PPO._slice_batch_field(temporal_state, 2, 4)

        torch.testing.assert_close(
            first_chunk[0][0],
            torch.tensor([[0.0, 1.0], [2.0, 3.0]]),
        )
        torch.testing.assert_close(
            second_chunk[0][1]["normalizer"],
            torch.tensor([[2.0], [3.0]]),
        )

    def test_virtual_batch_gradient_matches_full_batch_with_global_advantage_normalization(self) -> None:
        batch = _make_flat_samples()

        full_policy = _LinearEvalPolicy()
        full_ppo = _ppo_for_virtual_gradient(full_policy)
        full_loss, _, _ = full_ppo.compute_loss(batch)
        full_loss.backward()
        full_grad = full_policy.weight.grad.detach().clone()

        virtual_policy = _LinearEvalPolicy()
        virtual_ppo = _ppo_for_virtual_gradient(virtual_policy, virtual_mini_batches=2)
        virtual_ppo._compute_virtual_batch_gradients(
            batch=batch,
            epoch=0,
            batch_idx=0,
            loss_metrics=MetricsLists[float](),
        )
        virtual_grad = virtual_policy.weight.grad.detach().clone()

        self.assertTrue(torch.allclose(virtual_grad, full_grad))


if __name__ == "__main__":
    unittest.main()
