import math
import unittest

import numpy as np
import torch
from gymnasium import spaces
from torch import distributions as torchdist
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionGradientEstimator
from swarmbots.learn.action_dists.hybrid_action_dist import (
    continuous_action_gradient_estimator,
    make_proba_distribution,
)
from swarmbots.learn.action_dists.ternary_sign_magnitude_beta_action_dist import (
    TernarySignMagnitudeBetaActionDist,
    TernarySignMagnitudeBetaConfig,
)


def _zero_init(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)


def _make_distribution(**kwargs: float) -> TernarySignMagnitudeBetaActionDist:
    return TernarySignMagnitudeBetaActionDist(
        latent_dim=4,
        action_dim=3,
        action_net_initialization=_zero_init,
        **kwargs,
    )


class TestTernarySignMagnitudeBetaActionDist(unittest.TestCase):
    def test_default_starts_with_balanced_component_probabilities(self) -> None:
        dist = _make_distribution()

        dist.update_latent_features(torch.zeros(2, 5, 4))

        assert dist.weight_logits is not None
        torch.testing.assert_close(
            torch.softmax(dist.weight_logits, dim=-1),
            torch.full((2, 5, 3, 3), 1.0 / 3.0),
        )
        assert dist.negative_beta_dist is not None
        assert dist.positive_beta_dist is not None
        torch.testing.assert_close(
            dist.negative_beta_dist.mean,
            torch.full((2, 5, 3), 0.5),
        )
        torch.testing.assert_close(
            dist.positive_beta_dist.mean,
            torch.full((2, 5, 3), 0.5),
        )

    def test_initial_component_probabilities_are_marginal_probabilities(self) -> None:
        dist = _make_distribution(initial_zero_prob=0.2, initial_positive_prob=0.3)

        dist.update_latent_features(torch.zeros(1, 1, 4))

        assert dist.weight_logits is not None
        torch.testing.assert_close(
            torch.softmax(dist.weight_logits, dim=-1),
            torch.tensor([0.5, 0.2, 0.3]).expand(1, 1, 3, 3),
        )

    def test_sampling_can_select_negative_zero_and_positive_actions(self) -> None:
        dist = _make_distribution()
        latent = torch.zeros(16, 1, 4)

        for component_index in range(3):
            with self.subTest(component_index=component_index):
                with torch.no_grad():
                    bias = dist.output_net.bias.view(dist.action_dim, 7)
                    bias[:, :3].fill_(-100.0)
                    bias[:, component_index] = 100.0
                actions = dist.update_latent_features(latent).sample()

                if component_index == dist._NEGATIVE_INDEX:
                    self.assertTrue(((actions > -1.0) & (actions < 0.0)).all())
                elif component_index == dist._ZERO_INDEX:
                    torch.testing.assert_close(actions, torch.zeros_like(actions))
                else:
                    self.assertTrue(((actions > 0.0) & (actions < 1.0)).all())

    def test_log_prob_uses_the_zero_point_mass_and_beta_densities(self) -> None:
        config = TernarySignMagnitudeBetaConfig()
        dist = _make_distribution()
        actions = torch.tensor([[[-0.75, 0.0, 0.25]]])

        dist.update_latent_features(torch.zeros(1, 1, 4))
        actual_log_prob = dist.log_prob(actions)

        beta_dist = torchdist.Beta(
            concentration1=torch.tensor(config.negative_alpha),
            concentration0=torch.tensor(config.negative_beta),
        )
        beta_component_log_prob = math.log(1.0 / 3.0) + beta_dist.log_prob(
            torch.tensor(0.25)
        )
        expected_log_prob = 2.0 * beta_component_log_prob + math.log(1.0 / 3.0)
        torch.testing.assert_close(actual_log_prob, torch.tensor([[expected_log_prob]]))

    def test_gumbel_rsample_is_hard_with_exact_selected_log_prob(self) -> None:
        torch.manual_seed(11)
        dist = _make_distribution(gumbel_temperature=0.7)
        latent = torch.ones(4, 8, 4)
        dist.update_latent_features(latent)

        sample = dist._rsample_with_selection()
        _actions, selection, _negative_magnitudes, _positive_magnitudes = sample
        actions, sampled_log_probs = dist.get_actions_with_log_probs(
            latent,
            use_rsample=True,
        )

        self.assertTrue(((selection == 0.0) | (selection == 1.0)).all())
        self.assertTrue((selection.sum(dim=-1) == 1.0).all())
        torch.testing.assert_close(sampled_log_probs, dist.log_prob(actions))

    def test_rsample_backpropagates_through_selection_and_both_betas(self) -> None:
        torch.manual_seed(17)
        dist = _make_distribution()
        latent = torch.ones(16, 8, 4)

        actions, log_probs = dist.get_actions_with_log_probs(latent, use_rsample=True)
        (actions.square().sum() + 0.2 * log_probs.sum()).backward()

        assert dist.output_net.bias.grad is not None
        output_grad = dist.output_net.bias.grad.view(dist.action_dim, 7)
        self.assertTrue(torch.isfinite(output_grad).all())
        self.assertTrue((output_grad[:, :3].abs().sum(dim=-1) > 0.0).all())
        self.assertTrue((output_grad[:, 3:].abs().sum(dim=-1) > 0.0).all())

    def test_factory_registers_straight_through_distribution(self) -> None:
        config = TernarySignMagnitudeBetaConfig()

        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32),
            action_space_dim=3,
            action_net_initialization=_zero_init,
            continuous_config=config,
        )

        self.assertIsInstance(dist, TernarySignMagnitudeBetaActionDist)
        self.assertIs(
            continuous_action_gradient_estimator(config),
            ActionGradientEstimator.STRAIGHT_THROUGH,
        )

    def test_rejects_invalid_component_probabilities(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be < 1"):
            _make_distribution(initial_zero_prob=0.6, initial_positive_prob=0.4)


if __name__ == "__main__":
    unittest.main()
