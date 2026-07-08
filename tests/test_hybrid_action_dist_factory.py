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
from swarmbots.learn.hybrid_action_space import HybridActionSpace, VectorHybridActionSpace
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class HybridActionDistFactoryTests(unittest.TestCase):
    def test_hybrid_action_space_preserves_declared_sequence_order_for_split_concat(self) -> None:
        action_space = HybridActionSpace([
            ("actuators", spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32)),
            ("connectors", spaces.MultiBinary((2, 1))),
        ])
        action_parts = {
            "actuators": np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32),
            "connectors": np.array([[1], [0]], dtype=np.int8),
        }

        concatenated = action_space.concat_actions(action_parts)
        split = action_space.split_actions(concatenated)

        self.assertEqual(action_space.key_order, ["actuators", "connectors"])
        np.testing.assert_allclose(
            concatenated,
            np.array([[0.1, 0.2, 1.0], [0.3, 0.4, 0.0]], dtype=np.float32),
        )
        np.testing.assert_allclose(split["actuators"], action_parts["actuators"])
        np.testing.assert_array_equal(split["connectors"], action_parts["connectors"])

    def test_vector_hybrid_action_space_preserves_declared_sequence_order_for_split_concat(self) -> None:
        action_space = VectorHybridActionSpace([
            ("actuators", spaces.Box(-1.0, 1.0, shape=(2, 3, 2), dtype=np.float32)),
            ("connectors", spaces.MultiBinary((2, 3, 1))),
        ])
        action_parts = {
            "actuators": np.array(
                [
                    [[0.1, 0.2], [0.3, 0.4], [0.5, 0.6]],
                    [[1.1, 1.2], [1.3, 1.4], [1.5, 1.6]],
                ],
                dtype=np.float32,
            ),
            "connectors": np.array(
                [
                    [[1], [0], [1]],
                    [[0], [1], [0]],
                ],
                dtype=np.int8,
            ),
        }

        concatenated = action_space.concat_actions(action_parts)
        split = action_space.split_actions(concatenated)

        self.assertEqual(action_space.key_order, ["actuators", "connectors"])
        self.assertEqual(concatenated.shape, (2, 3, 3))
        np.testing.assert_allclose(concatenated[..., :2], action_parts["actuators"])
        np.testing.assert_array_equal(concatenated[..., 2:], action_parts["connectors"])
        np.testing.assert_allclose(split["actuators"], action_parts["actuators"])
        np.testing.assert_array_equal(split["connectors"], action_parts["connectors"])

    def test_hybrid_action_space_rejects_subspaces_with_different_agent_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "share n_agents"):
            HybridActionSpace([
                ("actuators", spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32)),
                ("connectors", spaces.MultiBinary((3, 1))),
            ])

    def test_vector_hybrid_action_space_rejects_subspaces_with_different_env_or_agent_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "share n_envs"):
            VectorHybridActionSpace([
                ("actuators", spaces.Box(-1.0, 1.0, shape=(2, 3, 2), dtype=np.float32)),
                ("connectors", spaces.MultiBinary((4, 3, 1))),
            ])

        with self.assertRaisesRegex(ValueError, "share n_agents"):
            VectorHybridActionSpace([
                ("actuators", spaces.Box(-1.0, 1.0, shape=(2, 3, 2), dtype=np.float32)),
                ("connectors", spaces.MultiBinary((2, 4, 1))),
            ])

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

    def test_beta_rsample_keeps_gradient_for_sac_actor_loss(self) -> None:
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32),
            action_space_dim=2,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=BetaConfig(),
        )
        latent = torch.randn(3, 2, 4, requires_grad=True)

        actions, _log_probs = dist.get_actions_with_log_probs(
            latent,
            deterministic=False,
            use_rsample=True,
        )
        actor_surrogate = actions.square().mean()
        actor_surrogate.backward()

        self.assertTrue(actions.requires_grad)
        self.assertIsNotNone(latent.grad)
        assert latent.grad is not None
        self.assertGreater(latent.grad.abs().sum().item(), 0.0)

    def test_beta_sample_does_not_track_sample_gradient(self) -> None:
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32),
            action_space_dim=2,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=BetaConfig(),
        )
        latent = torch.randn(3, 2, 4, requires_grad=True)

        actions, log_probs = dist.get_actions_with_log_probs(
            latent,
            deterministic=False,
        )

        self.assertFalse(actions.requires_grad)
        self.assertTrue(log_probs.requires_grad)

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
