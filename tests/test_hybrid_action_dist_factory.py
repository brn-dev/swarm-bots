import math
import sys
import unittest
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
from torch import distributions as torchdist
from gymnasium import spaces

from swarmbots.learn.action_dists.beta_action_dist import BetaActionDist, BetaConfig
from swarmbots.learn.action_dists.bang_zero_bang_action_dist import BangZeroBangActionDist, BangZeroBangConfig
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaConfig,
    GumbelSoftmaxSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution, make_proba_distribution
from swarmbots.learn.action_dists.left_middle_right_beta_action_dist import LeftMiddleRightBetaConfig
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdActionDist, PredictedStdConfig
from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import (
    SignMagnitudeBetaActionDist,
    SignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import (
    SquashedDiagGaussianActionDist,
    SquashedDiagGaussianConfig,
)
from swarmbots.learn.action_dists.sticky_left_middle_right_beta_action_dist import (
    StickyLeftMiddleRightBetaConfig,
)
from swarmbots.learn.action_dists.sticky_sign_magnitude_beta_action_dist import (
    StickySignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.sticky_bang_zero_bang_action_dist import (
    StickyBangZeroBangActionDist,
    StickyBangZeroBangConfig,
)
from swarmbots.learn.action_dists.ternary_sign_magnitude_beta_action_dist import (
    TernarySignMagnitudeBetaConfig,
)
from swarmbots.learn.hybrid_action_space import HybridActionSpace, VectorHybridActionSpace
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


_REAL_TORCH_COMPILE = torch.compile


def _compile_with_eager_backend(
        function: Callable[..., Any],
        **kwargs: Any,
) -> Callable[..., Any]:
    return _REAL_TORCH_COMPILE(function, backend="eager", **kwargs)


class HybridActionDistFactoryTests(unittest.TestCase):
    def test_inductor_actor_refreshes_distribution_state_for_external_entropy_loss(self) -> None:
        if sys.platform == "win32":
            self.skipTest("TorchInductor's C++ backend requires a complete OpenMP toolchain on Windows")
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)

        action_space = HybridActionSpace([
            ("actions", spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32)),
        ])
        config = PredictedStdConfig(base_std=0.5, ent_loss_coef=0.2)
        torch.manual_seed(321)
        eager_dist = HybridActionDistribution(
            latent_dim=4,
            action_space=action_space,
            continuous_config=config,
            action_net_initialization=init_linear_orthogonal,
        )
        torch.manual_seed(321)
        compiled_dist = HybridActionDistribution(
            latent_dim=4,
            action_space=action_space,
            continuous_config=config,
            action_net_initialization=init_linear_orthogonal,
        )

        def compiled_actor(latent_pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return compiled_dist.get_actions_with_log_probs(
                latent_pi,
                deterministic=True,
                use_rsample=False,
            )

        compiled_actor_fn = _REAL_TORCH_COMPILE(
            compiled_actor,
            mode="default",
            fullgraph=True,
            dynamic=False,
        )
        latent_pi = torch.randn(3, 2, 4)
        for latent_offset in (0.0, 0.25):
            current_latent = latent_pi + latent_offset
            eager_actions, eager_log_probs = eager_dist.get_actions_with_log_probs(
                current_latent,
                deterministic=True,
                use_rsample=False,
            )
            compiled_actions, compiled_log_probs = compiled_actor_fn(current_latent)
            torch.testing.assert_close(compiled_actions, eager_actions)
            torch.testing.assert_close(compiled_log_probs, eager_log_probs)
            eager_losses = eager_dist.compute_extra_losses_without_metrics()
            compiled_losses = compiled_dist.compute_extra_losses_without_metrics()
            self.assertEqual(compiled_losses.keys(), eager_losses.keys())
            self.assertTrue(compiled_losses)
            for name in compiled_losses:
                torch.testing.assert_close(compiled_losses[name], eager_losses[name])

    def test_compile_friendly_continuous_distributions_with_external_losses_refresh_state(self) -> None:
        configs = (
            SquashedDiagGaussianConfig(std=0.5, std_learnable=True),
            PredictedStdConfig(base_std=0.5),
            BetaConfig(),
            GumbelSoftmaxSignMagnitudeBetaConfig(),
            GumbelSoftmaxSignMagnitudeKumaraswamyConfig(),
            TernarySignMagnitudeBetaConfig(),
            ReparameterizedSignMagnitudeKumaraswamyConfig(),
            StickySignMagnitudeBetaConfig(stickiness=0.25),
            SignMagnitudeBetaConfig(),
            LeftMiddleRightBetaConfig(eps_c=0.1),
            StickyLeftMiddleRightBetaConfig(eps_c=0.1, stickiness=0.25),
            BangZeroBangConfig(),
            StickyBangZeroBangConfig(stickiness=0.25),
        )
        action_space = HybridActionSpace([
            ("actions", spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32)),
        ])
        previous_actions = torch.zeros(3, 2, 2)
        agent_mask = torch.tensor([
            [True, True],
            [True, False],
            [True, True],
        ])

        for config in configs:
            with self.subTest(config=type(config).__name__):
                torch._dynamo.reset()
                try:
                    torch.manual_seed(123)
                    eager_dist = HybridActionDistribution(
                        latent_dim=4,
                        action_space=action_space,
                        continuous_config=config,
                        action_net_initialization=init_linear_orthogonal,
                    )
                    torch.manual_seed(123)
                    compiled_dist = HybridActionDistribution(
                        latent_dim=4,
                        action_space=action_space,
                        continuous_config=config,
                        action_net_initialization=init_linear_orthogonal,
                    )
                    self.assertTrue(eager_dist.compile_friendly)
                    self.assertTrue(compiled_dist.compile_friendly)
                    eager_dist.set_all_ent_loss_coefs(0.2)
                    compiled_dist.set_all_ent_loss_coefs(0.2)

                    def compiled_actor(latent_pi: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
                        return compiled_dist.get_actions_with_log_probs(
                            latent_pi,
                            deterministic=True,
                            previous_actions=previous_actions,
                            use_rsample=False,
                        )

                    compiled_actor_fn = _compile_with_eager_backend(
                        compiled_actor,
                        fullgraph=True,
                        dynamic=False,
                    )
                    latent_pi = torch.randn(3, 2, 4)
                    for latent_offset in (0.0, 0.25):
                        current_latent = latent_pi + latent_offset
                        eager_actions, eager_log_probs = eager_dist.get_actions_with_log_probs(
                            current_latent,
                            deterministic=True,
                            previous_actions=previous_actions,
                            use_rsample=False,
                        )
                        compiled_actions, compiled_log_probs = compiled_actor_fn(current_latent)
                        torch.testing.assert_close(compiled_actions, eager_actions)
                        torch.testing.assert_close(compiled_log_probs, eager_log_probs)

                        eager_losses = eager_dist.compute_extra_losses_without_metrics(
                            agent_mask=agent_mask,
                        )
                        compiled_losses = compiled_dist.compute_extra_losses_without_metrics(
                            agent_mask=agent_mask,
                        )
                        self.assertEqual(compiled_losses.keys(), eager_losses.keys())
                        self.assertTrue(compiled_losses)
                        for name in compiled_losses:
                            torch.testing.assert_close(compiled_losses[name], eager_losses[name])
                finally:
                    torch._dynamo.reset()

    def test_entropy_update_handles_neutral_kumaraswamy_config_family(self) -> None:
        action_space = HybridActionSpace([
            ("actuators", spaces.Box(-1.0, 1.0, shape=(2, 3), dtype=np.float32)),
        ])
        dist = HybridActionDistribution(
            latent_dim=4,
            action_space=action_space,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=GumbelSoftmaxSignMagnitudeKumaraswamyConfig(),
        )

        dist.set_all_ent_loss_coefs(0.25)

        updated_config = dist.continuous_configs[0]
        self.assertIsInstance(updated_config, GumbelSoftmaxSignMagnitudeKumaraswamyConfig)
        self.assertEqual(updated_config.ent_loss_coef, 0.25)
        self.assertEqual(dist.distributions[0].ent_loss_coef, 0.25)

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

    def test_sign_magnitude_beta_default_starts_balanced_on_zero_latent(self) -> None:
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32),
            action_space_dim=2,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=SignMagnitudeBetaConfig(),
        )

        dist.update_latent_features(torch.zeros(3, 2, 4))

        self.assertIsInstance(dist, SignMagnitudeBetaActionDist)
        assert isinstance(dist, SignMagnitudeBetaActionDist)
        assert dist.weight_logits is not None
        torch.testing.assert_close(
            torch.softmax(dist.weight_logits, dim=-1),
            torch.full((3, 2, 2, 2), 0.5),
        )
        assert dist.negative_beta_dist is not None
        assert dist.positive_beta_dist is not None
        torch.testing.assert_close(dist.negative_beta_dist.mean, torch.full((3, 2, 2), 0.5))
        torch.testing.assert_close(dist.positive_beta_dist.mean, torch.full((3, 2, 2), 0.5))

    def test_sign_magnitude_beta_log_prob_matches_disjoint_mixture_density(self) -> None:
        config = SignMagnitudeBetaConfig(negative_alpha=2.0, negative_beta=5.0)
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(1, 2), dtype=np.float32),
            action_space_dim=2,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=config,
        )
        actions = torch.tensor([[[-0.75, 0.25]]])

        dist.update_latent_features(torch.zeros(1, 1, 4))
        actual_log_prob = dist.log_prob(actions)

        negative_magnitude_dist = torchdist.Beta(
            concentration1=torch.tensor(config.negative_alpha),
            concentration0=torch.tensor(config.negative_beta),
        )
        positive_magnitude_dist = torchdist.Beta(
            concentration1=torch.tensor(config.positive_alpha),
            concentration0=torch.tensor(config.positive_beta),
        )
        expected_log_prob = (
            math.log(0.5)
            + negative_magnitude_dist.log_prob(torch.tensor(0.75))
            + math.log(0.5)
            + positive_magnitude_dist.log_prob(torch.tensor(0.25))
        ).reshape(1, 1)
        torch.testing.assert_close(actual_log_prob, expected_log_prob)

    def test_probability_clamp_epsilon_must_leave_a_nonempty_interval(self) -> None:
        action_space = spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32)
        for config in (
                BetaConfig(epsilon=0.5),
                GumbelSoftmaxSignMagnitudeKumaraswamyConfig(epsilon=0.5),
        ):
            with self.subTest(config=type(config).__name__):
                with self.assertRaisesRegex(ValueError, r"epsilon must be in \(0, 0.5\)"):
                    make_proba_distribution(
                        latent_dim=4,
                        action_space=action_space,
                        action_space_dim=2,
                        action_net_initialization=init_linear_orthogonal,
                        continuous_config=config,
                    )

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

    def test_predicted_std_metrics_include_predicted_stds(self) -> None:
        dist = make_proba_distribution(
            latent_dim=4,
            action_space=spaces.Box(-1.0, 1.0, shape=(2, 2), dtype=np.float32),
            action_space_dim=2,
            action_net_initialization=init_linear_orthogonal,
            continuous_config=PredictedStdConfig(base_std=0.25),
        )
        latent = torch.zeros(3, 2, 4)

        actions = dist.update_latent_features(latent).sample()
        metrics = dist.get_metrics(actions)

        self.assertIsInstance(dist, PredictedStdActionDist)
        self.assertIn("std", metrics)

    def test_action_net_initialization_applies_when_latent_dim_matches_action_dim(self) -> None:
        def constant_init(module: torch.nn.Linear) -> None:
            torch.nn.init.constant_(module.weight, 0.0)
            torch.nn.init.constant_(module.bias, 0.25)

        dist = make_proba_distribution(
            latent_dim=2,
            action_space=spaces.Box(-1.0, 1.0, shape=(1, 2), dtype=np.float32),
            action_space_dim=2,
            action_net_initialization=constant_init,
            continuous_config=SquashedDiagGaussianConfig(
                std=0.25,
                std_learnable=True,
            ),
        )
        latent = torch.zeros(3, 1, 2)

        dist.update_latent_features(latent)

        self.assertIsInstance(dist.action_net, torch.nn.Linear)
        self.assertTrue(torch.allclose(dist.distribution.mean, torch.full((3, 1, 2), 0.25)))


if __name__ == "__main__":
    unittest.main()
