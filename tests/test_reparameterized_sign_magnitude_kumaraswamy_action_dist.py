import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.entropy_utils import EntropyLossConfig
from swarmbots.learn.action_dists.hybrid_action_dist import make_proba_distribution
from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyActionDist,
    ReparameterizedSignMagnitudeKumaraswamyConfig,
    _kumaraswamy_icdf,
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

    def test_rsample_negates_negative_magnitude(self) -> None:
        dist = ReparameterizedSignMagnitudeKumaraswamyActionDist(
            latent_dim=4,
            action_dim=3,
            action_net_initialization=init_linear_orthogonal,
        )
        dist.update_latent_features(torch.zeros(2, 1, 4))
        assert dist.weight_logits is not None
        with torch.no_grad():
            dist.weight_logits[..., dist._NEGATIVE_INDEX] = 100.0
            dist.weight_logits[..., dist._POSITIVE_INDEX] = -100.0

        with patch(
                "swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist."
                "kumaraswamy_icdf",
                return_value=torch.full((2, 1, 3), 0.2),
        ):
            actions = dist.rsample()

        torch.testing.assert_close(actions, torch.full((2, 1, 3), -0.2))

    def test_inverse_cdf_preserves_float32_tail_value_and_gradient(self) -> None:
        u = torch.tensor(1e-6, dtype=torch.float32)
        a = torch.tensor(2.0, dtype=torch.float32)
        b = torch.tensor(100.0, dtype=torch.float32, requires_grad=True)

        sample = _kumaraswamy_icdf(u, a, b, epsilon=1e-6)
        sample.backward()

        self.assertGreater(sample.item(), 1e-5)
        self.assertIsNotNone(b.grad)
        assert b.grad is not None
        self.assertTrue(torch.isfinite(b.grad))
        self.assertNotEqual(b.grad.item(), 0.0)

    def test_sample_does_not_track_distribution_head_gradient(self) -> None:
        dist = ReparameterizedSignMagnitudeKumaraswamyActionDist(
            latent_dim=4,
            action_dim=3,
            action_net_initialization=init_linear_orthogonal,
        )
        latent = torch.zeros(5, 2, 4, requires_grad=True)

        dist.update_latent_features(latent)
        with patch.object(
                ReparameterizedSignMagnitudeKumaraswamyActionDist,
                "rsample",
                side_effect=AssertionError("ordinary sampling must not use the inverse-CDF mixture estimator"),
        ):
            actions = dist.sample()
        log_probs = dist.log_prob(actions)

        self.assertFalse(actions.requires_grad)
        self.assertTrue(log_probs.requires_grad)

    def test_initial_mode_uses_negative_component_on_symmetric_density_tie(self) -> None:
        dist = ReparameterizedSignMagnitudeKumaraswamyActionDist(
            latent_dim=4,
            action_dim=3,
            action_net_initialization=init_linear_orthogonal,
        )
        latent = torch.zeros(5, 2, 4)

        actions = dist.update_latent_features(latent).mode()

        torch.testing.assert_close(actions, torch.full_like(actions, -0.5))

    def test_loss_only_path_skips_kumaraswamy_entropy_when_scale_is_zero(self) -> None:
        dist = ReparameterizedSignMagnitudeKumaraswamyActionDist(
            latent_dim=4,
            action_dim=3,
            action_net_initialization=init_linear_orthogonal,
            ent_loss_coef=1e-3,
            kumaraswamy_ent_scale=0.0,
            categorical_ent_loss_config=EntropyLossConfig(entropy_floor=0.35),
        )
        latent = torch.zeros(5, 2, 4)
        dist.update_latent_features(latent)

        with patch(
                "torch.distributions.Kumaraswamy.entropy",
                side_effect=AssertionError("kumaraswamy entropy should not be computed"),
        ):
            losses = dist.compute_extra_losses_without_metrics()

        self.assertIn("entropy", losses)


if __name__ == "__main__":
    unittest.main()
