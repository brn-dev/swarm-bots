import math
import unittest

import numpy as np
import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.hybrid_action_dist import make_proba_distribution
from swarmbots.learn.action_dists.reparameterized_squashed_gaussian_mixture_action_dist import (
    ReparameterizedSquashedGaussianMixtureActionDist,
    ReparameterizedSquashedGaussianMixtureConfig,
    _SquashedGaussianMixtureICDF,
)
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class ReparameterizedSquashedGaussianMixtureActionDistTests(unittest.TestCase):
    def test_factory_creates_distribution_and_computes_log_prob_and_entropy_loss(self) -> None:
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32),
            action_space_dim=3,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=ReparameterizedSquashedGaussianMixtureConfig(ent_loss_coef=1e-3),
        )
        latent = torch.zeros(5, 2, 4)

        dist.update_latent_features(latent)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        losses, metrics = dist.compute_extra_losses()

        self.assertIsInstance(dist, ReparameterizedSquashedGaussianMixtureActionDist)
        self.assertEqual(actions.shape, (5, 2, 3))
        self.assertEqual(log_probs.shape, (5, 2))
        self.assertTrue(torch.all(actions > -1.0))
        self.assertTrue(torch.all(actions < 1.0))
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertIn("entropy", losses)
        self.assertIn("ent_categorical", metrics)
        self.assertIn("ent_gaussian", metrics)
        self.assertFalse(dist.compile_friendly)

    def test_rsample_is_reparameterized_through_distribution_head(self) -> None:
        torch.manual_seed(0)
        dist = ReparameterizedSquashedGaussianMixtureActionDist(
            latent_dim=4,
            action_dim=3,
            action_net_initialization=init_linear_orthogonal,
        )
        latent = torch.zeros(32, 2, 4, requires_grad=True)

        actions = dist.update_latent_features(latent).rsample()
        actions.sum().backward()
        bias_grad = dist.output_net.bias.grad.view(
            dist.action_dim,
            dist.num_components,
            dist._OUTPUTS_PER_COMPONENT,
        )

        self.assertTrue(actions.requires_grad)
        self.assertGreater(bias_grad.abs().sum().item(), 0.0)
        self.assertGreater(bias_grad[:, :, 0].abs().sum().item(), 0.0)
        self.assertGreater(bias_grad[:, :, 1].abs().sum().item(), 0.0)
        self.assertGreater(bias_grad[:, :, 2].abs().sum().item(), 0.0)

    def test_sample_does_not_track_distribution_head_gradient(self) -> None:
        dist = ReparameterizedSquashedGaussianMixtureActionDist(
            latent_dim=4,
            action_dim=3,
            action_net_initialization=init_linear_orthogonal,
            inverse_cdf_iterations=8,
        )
        latent = torch.zeros(5, 2, 4, requires_grad=True)

        actions = dist.update_latent_features(latent).sample()
        log_probs = dist.log_prob(actions)

        self.assertFalse(actions.requires_grad)
        self.assertTrue(log_probs.requires_grad)

    def test_initial_mode_is_near_zero_with_symmetric_components(self) -> None:
        dist = ReparameterizedSquashedGaussianMixtureActionDist(
            latent_dim=4,
            action_dim=3,
            action_net_initialization=init_linear_orthogonal,
        )
        latent = torch.zeros(5, 2, 4)

        actions = dist.update_latent_features(latent).mode()

        self.assertLess(actions.abs().max().item(), 0.05)

    def test_implicit_backward_matches_finite_difference_for_component_mean(self) -> None:
        u = torch.tensor([[0.35]], dtype=torch.float64)
        means = torch.tensor([[[-0.6, 0.4]]], dtype=torch.float64, requires_grad=True)
        log_stds = torch.full((1, 1, 2), math.log(0.35), dtype=torch.float64, requires_grad=True)
        logits = torch.zeros(1, 1, 2, dtype=torch.float64, requires_grad=True)

        action = _SquashedGaussianMixtureICDF.apply(u, means, log_stds, logits, 1e-8, 64)
        action.sum().backward()
        analytic_grad = means.grad[0, 0, 0].item()

        delta = 1e-4
        with torch.no_grad():
            means_plus = means.detach().clone()
            means_minus = means.detach().clone()
            means_plus[0, 0, 0] += delta
            means_minus[0, 0, 0] -= delta
            action_plus = _SquashedGaussianMixtureICDF.apply(u, means_plus, log_stds.detach(), logits.detach(), 1e-8, 64)
            action_minus = _SquashedGaussianMixtureICDF.apply(
                u,
                means_minus,
                log_stds.detach(),
                logits.detach(),
                1e-8,
                64,
            )
            finite_difference_grad = ((action_plus - action_minus) / (2.0 * delta)).item()

        self.assertAlmostEqual(analytic_grad, finite_difference_grad, places=4)

    def test_log_prob_density_integrates_to_one_for_initial_distribution(self) -> None:
        dist = ReparameterizedSquashedGaussianMixtureActionDist(
            latent_dim=4,
            action_dim=1,
            action_net_initialization=init_linear_orthogonal,
        )
        dist.update_latent_features(torch.zeros(1, 1, 4))
        actions = torch.linspace(-1.0 + 1e-4, 1.0 - 1e-4, 20_000).view(20_000, 1, 1)

        density = dist.log_prob(actions).exp().squeeze()
        integral = torch.trapezoid(density, actions.squeeze()).item()

        self.assertAlmostEqual(integral, 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
