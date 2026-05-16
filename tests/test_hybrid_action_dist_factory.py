import unittest

import numpy as np
import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.beta_action_dist import BetaActionDist, BetaConfig
from swarmbots.learn.action_dists.bang_zero_bang_action_dist import BangZeroBangActionDist, BangZeroBangConfig
from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution, make_proba_distribution
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdActionDist, PredictedStdConfig
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import (
    SquashedDiagGaussianActionDist,
    SquashedDiagGaussianConfig,
)
from swarmbots.learn.action_dists.sticky_bang_zero_bang_action_dist import (
    StickyBangZeroBangActionDist,
    StickyBangZeroBangConfig,
)
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class HybridActionDistFactoryTests(unittest.TestCase):
    def test_sticky_bang_config_creates_sticky_distribution_before_base_config_match(self) -> None:
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32),
            action_space_dim=3,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=StickyBangZeroBangConfig(stickiness=0.25, zero_sticky=True),
        )

        self.assertIsInstance(dist, StickyBangZeroBangActionDist)
        self.assertTrue(dist.requires_previous_actions())
        self.assertAlmostEqual(dist.get_stickiness(), 0.25)
        self.assertTrue(dist.zero_sticky)

    def test_hybrid_distribution_splits_previous_actions_per_subspace(self) -> None:
        action_space = HybridActionSpace([
            ("left", spaces.Box(-1.0, 1.0, shape=(2, 1), dtype=np.float32)),
            ("right", spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32)),
        ])
        dist = HybridActionDistribution(
            latent_dim=4,
            action_space=action_space,
            continuous_config=[
                StickyBangZeroBangConfig(stickiness=0.25),
                StickyBangZeroBangConfig(stickiness=0.50),
            ],
            action_net_initialization=init_linear_orthogonal,
        )
        latent = torch.zeros(5, 2, 4)
        previous_actions = torch.tensor([
            [[-1.0, 0.0, 1.0], [1.0, -1.0, 0.0]],
            [[0.0, 1.0, -1.0], [-1.0, 0.0, 1.0]],
            [[1.0, -1.0, 0.0], [0.0, 1.0, -1.0]],
            [[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0], [-1.0, -1.0, -1.0]],
        ])

        dist.update_latent_features(latent)
        actions = dist.mode(previous_actions=previous_actions)
        log_probs = dist.log_prob(actions, previous_actions=previous_actions)

        self.assertEqual(actions.shape, previous_actions.shape)
        self.assertEqual(log_probs.shape, previous_actions.shape[:-1])
        self.assertTrue(dist.requires_previous_actions())
        self.assertTrue(dist.compile_friendly)
        self.assertTrue(all(isinstance(sub_dist, StickyBangZeroBangActionDist) for sub_dist in dist.distributions))

    def test_factory_rejects_non_unit_box_and_missing_continuous_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "low=-1 and high=1"):
            make_proba_distribution(
                latent_dim=4,
                action_space=spaces.Box(0.0, 1.0, shape=(2, 1), dtype=np.float32),
                action_space_dim=1,
                action_net_initialization=init_linear_orthogonal,
                continuous_config=StickyBangZeroBangConfig(),
            )

        with self.assertRaisesRegex(ValueError, "Supply a ContinuousActionDistConfig"):
            make_proba_distribution(
                latent_dim=4,
                action_space=spaces.Box(-1.0, 1.0, shape=(2, 1), dtype=np.float32),
                action_space_dim=1,
                action_net_initialization=init_linear_orthogonal,
                continuous_config=None,
            )

    def test_plain_bang_config_still_creates_non_sticky_distribution(self) -> None:
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32),
            action_space_dim=2,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=BangZeroBangConfig(bang=0.5),
        )

        self.assertIsInstance(dist, BangZeroBangActionDist)
        self.assertNotIsInstance(dist, StickyBangZeroBangActionDist)
        self.assertFalse(dist.requires_previous_actions())

    def test_beta_config_creates_regular_beta_distribution(self) -> None:
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32),
            action_space_dim=2,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=BetaConfig(ent_loss_coef=1e-3),
        )
        latent = torch.zeros(3, 2, 4)

        dist.update_latent_features(latent)
        actions = dist.mode()
        log_probs = dist.log_prob(actions)
        losses, metrics = dist.compute_extra_losses()

        self.assertIsInstance(dist, BetaActionDist)
        self.assertEqual(actions.shape, (3, 2, 2))
        self.assertEqual(log_probs.shape, (3, 2))
        self.assertIn("entropy", losses)
        self.assertIn("ent", metrics)
        self.assertTrue(dist.compile_friendly)

    def test_squashed_diag_gaussian_config_computes_entropy_loss(self) -> None:
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32),
            action_space_dim=2,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=SquashedDiagGaussianConfig(
                std=0.25,
                std_learnable=True,
                ent_loss_coef=1e-3,
            ),
        )
        latent = torch.zeros(3, 2, 4)

        dist.update_latent_features(latent)
        losses, metrics = dist.compute_extra_losses()

        self.assertIsInstance(dist, SquashedDiagGaussianActionDist)
        self.assertIn("entropy", losses)
        self.assertIn("ent", metrics)

    def test_predicted_std_config_computes_entropy_loss_when_squashed_by_factory(self) -> None:
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32),
            action_space_dim=2,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=PredictedStdConfig(
                base_std=0.25,
                ent_loss_coef=1e-3,
            ),
        )
        latent = torch.zeros(3, 2, 4)

        dist.update_latent_features(latent)
        losses, metrics = dist.compute_extra_losses()

        self.assertIsInstance(dist, PredictedStdActionDist)
        self.assertTrue(dist.squash_output)
        self.assertIn("entropy", losses)
        self.assertIn("ent", metrics)


if __name__ == "__main__":
    unittest.main()
