import unittest

import numpy as np
import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.hybrid_action_dist import make_proba_distribution
from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyActionDist,
    ReparameterizedSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class ReparameterizedSignMagnitudeKumaraswamyActionDistTests(unittest.TestCase):
    def test_factory_creates_distribution_and_computes_log_prob_and_entropy_loss(self) -> None:
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32),
            action_space_dim=3,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=ReparameterizedSignMagnitudeKumaraswamyConfig(ent_loss_coef=1e-3),
        )
        latent = torch.zeros(5, 2, 4)

        dist.update_latent_features(latent)
        actions = dist.sample()
        log_probs = dist.log_prob(actions)
        losses, metrics = dist.compute_extra_losses()

        self.assertIsInstance(dist, ReparameterizedSignMagnitudeKumaraswamyActionDist)
        self.assertEqual(actions.shape, (5, 2, 3))
        self.assertEqual(log_probs.shape, (5, 2))
        self.assertTrue(torch.all(actions >= -1.0))
        self.assertTrue(torch.all(actions <= 1.0))
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertIn("entropy", losses)
        self.assertIn("ent_categorical", metrics)
        self.assertIn("ent_kumaraswamy", metrics)
        self.assertTrue(dist.compile_friendly)

    def test_rsample_is_reparameterized_through_distribution_head(self) -> None:
        dist = ReparameterizedSignMagnitudeKumaraswamyActionDist(
            latent_dim=4,
            action_dim=3,
            action_net_initialization=init_linear_orthogonal,
        )
        latent = torch.zeros(5, 2, 4, requires_grad=True)

        actions = dist.update_latent_features(latent).rsample()
        loss = actions.sum()
        loss.backward()

        self.assertTrue(actions.requires_grad)
        self.assertIsNotNone(dist.output_net.bias.grad)
        self.assertGreater(dist.output_net.bias.grad.abs().sum().item(), 0.0)

    def test_sample_does_not_track_distribution_head_gradient(self) -> None:
        dist = ReparameterizedSignMagnitudeKumaraswamyActionDist(
            latent_dim=4,
            action_dim=3,
            action_net_initialization=init_linear_orthogonal,
        )
        latent = torch.zeros(5, 2, 4, requires_grad=True)

        actions = dist.update_latent_features(latent).sample()
        log_probs = dist.log_prob(actions)

        self.assertFalse(actions.requires_grad)
        self.assertTrue(log_probs.requires_grad)

    def test_initial_mode_is_near_zero_with_symmetric_probabilities(self) -> None:
        dist = ReparameterizedSignMagnitudeKumaraswamyActionDist(
            latent_dim=4,
            action_dim=3,
            action_net_initialization=init_linear_orthogonal,
        )
        latent = torch.zeros(5, 2, 4)

        actions = dist.update_latent_features(latent).mode()

        self.assertLess(actions.abs().max().item(), 0.05)


if __name__ == "__main__":
    unittest.main()
