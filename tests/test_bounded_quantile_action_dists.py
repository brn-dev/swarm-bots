import math
import unittest
from unittest.mock import patch

import numpy as np
import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.action_dist import ActionGradientEstimator
from swarmbots.learn.action_dists.bernstein_quantile_action_dist import (
    BernsteinQuantileActionDist,
    BernsteinQuantileConfig,
)
from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    continuous_action_gradient_estimator,
    make_proba_distribution,
)
from swarmbots.learn.action_dists.rational_quadratic_spline_quantile_action_dist import (
    RationalQuadraticSplineQuantileActionDist,
    RationalQuadraticSplineQuantileConfig,
)
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


def _finite_difference_bias_gradient(
        distribution: BernsteinQuantileActionDist | RationalQuadraticSplineQuantileActionDist,
        latent: torch.Tensor,
        actions: torch.Tensor,
        epsilon: float = 1e-3,
) -> torch.Tensor:
    original_bias = distribution.action_net.bias.detach().clone()
    gradient = torch.empty_like(original_bias)
    with torch.no_grad():
        for parameter_index in range(original_bias.numel()):
            distribution.action_net.bias.copy_(original_bias)
            distribution.action_net.bias[parameter_index] += epsilon
            distribution.update_latent_features(latent)
            positive_log_prob = distribution.log_prob(actions).sum()

            distribution.action_net.bias.copy_(original_bias)
            distribution.action_net.bias[parameter_index] -= epsilon
            distribution.update_latent_features(latent)
            negative_log_prob = distribution.log_prob(actions).sum()
            gradient[parameter_index] = (
                positive_log_prob - negative_log_prob
            ) / (2.0 * epsilon)

        distribution.action_net.bias.copy_(original_bias)
        distribution.update_latent_features(latent)
    return gradient


def _reference_rqs_log_prob_float64(
        distribution: RationalQuadraticSplineQuantileActionDist,
        actions: torch.Tensor,
) -> torch.Tensor:
    widths, heights, derivatives, _x_knots, y_knots = distribution._spline_parameters()
    widths = widths.double()
    heights = heights.double()
    derivatives = derivatives.double()
    y_knots = y_knots.double()
    actions = actions.double()
    bin_indices = distribution._find_bins(actions, y_knots)
    y_left = distribution._gather(y_knots, bin_indices)
    widths_in_bin = distribution._gather(widths, bin_indices)
    heights_in_bin = distribution._gather(heights, bin_indices)
    derivative_left = distribution._gather(derivatives, bin_indices)
    derivative_right = distribution._gather(derivatives, bin_indices + 1)

    lower = torch.zeros_like(actions)
    upper = torch.ones_like(actions)
    for _ in range(80):
        theta = 0.5 * (lower + upper)
        reconstructed_actions, _log_det = distribution._evaluate_bin(
            theta=theta,
            y_left=y_left,
            widths=widths_in_bin,
            heights=heights_in_bin,
            derivative_left=derivative_left,
            derivative_right=derivative_right,
        )
        lower = torch.where(reconstructed_actions < actions, theta, lower)
        upper = torch.where(reconstructed_actions >= actions, theta, upper)

    _reconstructed_actions, log_det = distribution._evaluate_bin(
        theta=0.5 * (lower + upper),
        y_left=y_left,
        widths=widths_in_bin,
        heights=heights_in_bin,
        derivative_left=derivative_left,
        derivative_right=derivative_right,
    )
    return -log_det.sum(dim=-1)


class BoundedQuantileActionDistTests(unittest.TestCase):
    def test_factory_registers_both_distributions_as_pathwise(self) -> None:
        configs_and_types = (
            (BernsteinQuantileConfig(), BernsteinQuantileActionDist),
            (RationalQuadraticSplineQuantileConfig(), RationalQuadraticSplineQuantileActionDist),
        )
        for config, expected_type in configs_and_types:
            with self.subTest(config=type(config).__name__):
                distribution = make_proba_distribution(
                    latent_dim=4,
                    action_space=spaces.Box(-1.0, 1.0, shape=(2, 3)),
                    action_space_dim=3,
                    action_net_initialization=init_linear_orthogonal,
                    continuous_config=config,
                )
                self.assertIsInstance(distribution, expected_type)
                self.assertIs(
                    continuous_action_gradient_estimator(config),
                    ActionGradientEstimator.PATHWISE,
                )

    def test_zero_initialized_heads_are_exact_uniform_quantile_maps(self) -> None:
        distributions = (
            BernsteinQuantileActionDist(latent_dim=5, action_dim=2, degree=8),
            RationalQuadraticSplineQuantileActionDist(latent_dim=5, action_dim=2, num_bins=6),
        )
        u = torch.tensor([[0.0, 0.1], [0.25, 0.5], [0.75, 1.0]])
        expected_actions = 2.0 * u - 1.0
        expected_log_det = torch.full_like(u, math.log(2.0))

        for distribution in distributions:
            with self.subTest(distribution=type(distribution).__name__):
                distribution.update_latent_features(torch.randn(3, 5))
                actions, log_det = distribution._transform_forward_and_log_det(u)
                torch.testing.assert_close(actions, expected_actions, atol=2e-6, rtol=2e-6)
                torch.testing.assert_close(log_det, expected_log_det, atol=2e-6, rtol=2e-6)
                self.assertEqual(torch.count_nonzero(distribution.action_net.weight).item(), 0)
                self.assertEqual(torch.count_nonzero(distribution.action_net.bias).item(), 0)

    def test_hot_path_does_not_call_arbitrary_action_inverse(self) -> None:
        distributions = (
            BernsteinQuantileActionDist(latent_dim=3, action_dim=2),
            RationalQuadraticSplineQuantileActionDist(latent_dim=3, action_dim=2),
        )
        latent = torch.randn(4, 3)
        for distribution in distributions:
            with self.subTest(distribution=type(distribution).__name__):
                with patch.object(distribution, "log_prob", side_effect=AssertionError("inverse used")):
                    actions, log_probs = distribution.get_actions_with_log_probs(latent, use_rsample=True)
                self.assertEqual(tuple(actions.shape), (4, 2))
                self.assertEqual(tuple(log_probs.shape), (4,))

    def test_no_grad_non_rsample_hot_path_does_not_call_arbitrary_action_inverse(self) -> None:
        distributions = (
            BernsteinQuantileActionDist(latent_dim=3, action_dim=2),
            RationalQuadraticSplineQuantileActionDist(latent_dim=3, action_dim=2),
        )
        latent = torch.randn(4, 3)
        for distribution in distributions:
            with self.subTest(distribution=type(distribution).__name__):
                with torch.no_grad(), patch.object(
                        distribution,
                        "log_prob",
                        side_effect=AssertionError("inverse used"),
                ):
                    actions, log_probs = distribution.get_actions_with_log_probs(
                        latent,
                        use_rsample=False,
                    )
                self.assertEqual(tuple(actions.shape), (4, 2))
                self.assertEqual(tuple(log_probs.shape), (4,))

    def test_hot_path_action_and_log_prob_use_the_same_uniform_sample(self) -> None:
        distributions = (
            BernsteinQuantileActionDist(latent_dim=3, action_dim=2),
            RationalQuadraticSplineQuantileActionDist(latent_dim=3, action_dim=2),
        )
        latent = torch.randn(4, 3)
        u = torch.tensor([[0.05, 0.15], [0.25, 0.35], [0.55, 0.75], [0.85, 0.95]])
        for distribution in distributions:
            with self.subTest(distribution=type(distribution).__name__):
                distribution.update_latent_features(latent)
                expected_actions, expected_log_det = distribution._transform_forward_and_log_det(u)
                with patch(
                    "swarmbots.learn.action_dists.bounded_quantile_action_dist.torch.rand",
                    return_value=u,
                ):
                    actions, log_probs = distribution.get_actions_with_log_probs(latent, use_rsample=True)

                torch.testing.assert_close(actions, expected_actions)
                torch.testing.assert_close(log_probs, -expected_log_det.sum(dim=-1))

    def test_closed_form_jacobians_match_autograd_for_nonuniform_maps(self) -> None:
        distributions = (
            BernsteinQuantileActionDist(latent_dim=3, action_dim=2, degree=8),
            RationalQuadraticSplineQuantileActionDist(latent_dim=3, action_dim=2, num_bins=6),
        )
        for seed, distribution in enumerate(distributions, start=41):
            with self.subTest(distribution=type(distribution).__name__):
                torch.manual_seed(seed)
                with torch.no_grad():
                    distribution.action_net.weight.normal_(std=0.2)
                    distribution.action_net.bias.normal_(std=0.2)
                distribution.update_latent_features(torch.randn(16, 3))
                u = torch.rand(16, 2, requires_grad=True)
                actions, log_det = distribution._transform_forward_and_log_det(u)
                autograd_derivative, = torch.autograd.grad(actions.sum(), u)

                torch.testing.assert_close(autograd_derivative, log_det.exp(), atol=2e-5, rtol=2e-5)

    def test_sample_detaches_actions_while_rsample_is_pathwise(self) -> None:
        distributions = (
            BernsteinQuantileActionDist(latent_dim=3, action_dim=2),
            RationalQuadraticSplineQuantileActionDist(latent_dim=3, action_dim=2),
        )
        for distribution in distributions:
            with self.subTest(distribution=type(distribution).__name__):
                distribution.update_latent_features(torch.randn(4, 3))
                self.assertFalse(distribution.sample().requires_grad)
                actions = distribution.rsample()
                self.assertTrue(actions.requires_grad)
                actions.sum().backward()
                self.assertGreater(distribution.action_net.weight.grad.abs().sum().item(), 0.0)

    def test_combined_non_rsample_uses_fixed_action_log_prob_gradients(self) -> None:
        distributions = (
            BernsteinQuantileActionDist(latent_dim=3, action_dim=2),
            RationalQuadraticSplineQuantileActionDist(latent_dim=3, action_dim=2),
        )
        latent = torch.randn(4, 3)
        u = torch.rand(4, 2)
        for seed, distribution in enumerate(distributions, start=31):
            with self.subTest(distribution=type(distribution).__name__):
                torch.manual_seed(seed)
                with torch.no_grad():
                    distribution.action_net.weight.normal_(std=0.4)
                    distribution.action_net.bias.normal_(std=0.4)
                with patch(
                        "swarmbots.learn.action_dists.bounded_quantile_action_dist.torch.rand",
                        return_value=u,
                ):
                    actions, log_probs = distribution.get_actions_with_log_probs(
                        latent,
                        use_rsample=False,
                    )
                self.assertFalse(actions.requires_grad)
                self.assertTrue(log_probs.requires_grad)
                combined_gradient, = torch.autograd.grad(
                    log_probs.sum(),
                    distribution.action_net.bias,
                )

                distribution.update_latent_features(latent)
                reference_log_probs = distribution.log_prob(actions)
                reference_gradient, = torch.autograd.grad(
                    reference_log_probs.sum(),
                    distribution.action_net.bias,
                )

                torch.testing.assert_close(log_probs, reference_log_probs)
                torch.testing.assert_close(combined_gradient, reference_gradient)

    def test_rollout_log_prob_is_recomputed_from_the_stored_float32_action(self) -> None:
        torch.manual_seed(0)
        distribution = RationalQuadraticSplineQuantileActionDist(
            latent_dim=3,
            action_dim=4,
            num_bins=6,
        )
        with torch.no_grad():
            distribution.action_net.weight.normal_(std=5.0)
            distribution.action_net.bias.normal_(std=5.0)
        latent = torch.randn(32, 3)
        u = torch.rand(32, 4)
        distribution.update_latent_features(latent)
        expected_actions, known_sample_log_det = distribution._transform_forward_and_log_det(u)
        known_sample_log_probs = -known_sample_log_det.sum(dim=-1)

        with patch(
                "swarmbots.learn.action_dists.bounded_quantile_action_dist.torch.rand",
                return_value=u,
        ):
            actions, rollout_log_probs = distribution.get_on_policy_actions_with_log_probs(latent)

        recomputed_log_probs = distribution.log_prob(actions)
        torch.testing.assert_close(actions, expected_actions)
        torch.testing.assert_close(rollout_log_probs, recomputed_log_probs)
        self.assertGreater(
            (known_sample_log_probs - recomputed_log_probs).abs().max().item(),
            0.2,
        )

    def test_rollout_computes_inverse_likelihood_once_with_gradients_enabled(self) -> None:
        distributions = (
            BernsteinQuantileActionDist(latent_dim=3, action_dim=2),
            RationalQuadraticSplineQuantileActionDist(latent_dim=3, action_dim=2),
        )
        latent = torch.randn(4, 3)
        for distribution in distributions:
            with self.subTest(distribution=type(distribution).__name__), patch.object(
                    distribution,
                    "log_prob",
                    wraps=distribution.log_prob,
            ) as log_prob:
                actions, rollout_log_probs = distribution.get_on_policy_actions_with_log_probs(latent)

            self.assertEqual(log_prob.call_count, 1)
            self.assertFalse(actions.requires_grad)
            self.assertTrue(rollout_log_probs.requires_grad)

    def test_hybrid_distribution_delegates_on_policy_log_prob_handling(self) -> None:
        distribution = HybridActionDistribution(
            latent_dim=3,
            action_space=HybridActionSpace([
                ("continuous", spaces.Box(-1.0, 1.0, shape=(2, 4), dtype=np.float32)),
            ]),
            continuous_config=RationalQuadraticSplineQuantileConfig(),
        )
        latent = torch.randn(8, 3)

        with patch.object(distribution, "log_prob", side_effect=AssertionError("aggregate inverse used")):
            actions, rollout_log_probs = distribution.get_on_policy_actions_with_log_probs(latent)

        distribution.update_latent_features(latent)
        torch.testing.assert_close(rollout_log_probs, distribution.log_prob(actions))

    def test_hybrid_distribution_updates_quantile_entropy_coefficient(self) -> None:
        distribution = HybridActionDistribution(
            latent_dim=3,
            action_space=HybridActionSpace([
                ("continuous", spaces.Box(-1.0, 1.0, shape=(2, 4), dtype=np.float32)),
            ]),
            continuous_config=RationalQuadraticSplineQuantileConfig(),
        )

        distribution.set_all_ent_loss_coefs(0.25)

        quantile_distribution = distribution.distributions[0]
        self.assertIsInstance(quantile_distribution, RationalQuadraticSplineQuantileActionDist)
        assert isinstance(quantile_distribution, RationalQuadraticSplineQuantileActionDist)
        self.assertEqual(quantile_distribution.ent_loss_coef, 0.25)
        updated_config = distribution.continuous_configs[0]
        self.assertIsInstance(updated_config, RationalQuadraticSplineQuantileConfig)
        assert isinstance(updated_config, RationalQuadraticSplineQuantileConfig)
        self.assertEqual(updated_config.ent_loss_coef, 0.25)

    def test_entropy_loss_uses_current_quantile_jacobian_sample(self) -> None:
        coefficient = 0.25
        distribution = RationalQuadraticSplineQuantileActionDist(
            latent_dim=3,
            action_dim=2,
            ent_loss_coef=coefficient,
        )
        distribution.update_latent_features(torch.randn(4, 3))
        u = torch.tensor([[0.1, 0.2], [0.3, 0.4], [0.6, 0.7], [0.8, 0.9]])
        _actions, log_det = distribution._transform_forward_and_log_det(u)

        with patch(
                "swarmbots.learn.action_dists.bounded_quantile_action_dist.torch.rand",
                return_value=u,
        ):
            losses, metrics = distribution.compute_extra_losses()

        torch.testing.assert_close(losses["entropy"], -coefficient * log_det.sum(dim=-1))
        self.assertAlmostEqual(metrics["ent"], log_det.mean().item())
        losses["entropy"].mean().backward()
        self.assertTrue(torch.isfinite(distribution.action_net.weight.grad).all())

    def test_transform_arithmetic_stays_float32_for_low_precision_head(self) -> None:
        distributions = (
            BernsteinQuantileActionDist(latent_dim=3, action_dim=2).bfloat16(),
            RationalQuadraticSplineQuantileActionDist(latent_dim=3, action_dim=2).bfloat16(),
        )
        latent = torch.randn(4, 3, dtype=torch.bfloat16)
        for distribution in distributions:
            with self.subTest(distribution=type(distribution).__name__):
                actions, log_probs = distribution.get_actions_with_log_probs(latent, use_rsample=True)
                self.assertEqual(actions.dtype, torch.float32)
                self.assertEqual(log_probs.dtype, torch.float32)

    def test_uniform_density_mode_prefers_the_median_among_equal_candidates(self) -> None:
        distributions = (
            BernsteinQuantileActionDist(latent_dim=3, action_dim=2),
            RationalQuadraticSplineQuantileActionDist(latent_dim=3, action_dim=2),
        )
        for distribution in distributions:
            with self.subTest(distribution=type(distribution).__name__):
                distribution.update_latent_features(torch.randn(4, 3))
                torch.testing.assert_close(distribution.mode(), torch.zeros(4, 2), atol=2e-6, rtol=0.0)

    def test_nonuniform_densities_integrate_to_one(self) -> None:
        distributions = (
            BernsteinQuantileActionDist(latent_dim=2, action_dim=1, degree=6),
            RationalQuadraticSplineQuantileActionDist(latent_dim=2, action_dim=1, num_bins=6),
        )
        actions = torch.linspace(-1.0, 1.0, 4001).unsqueeze(-1)
        latent = torch.zeros(actions.shape[0], 2)
        for distribution in distributions:
            with self.subTest(distribution=type(distribution).__name__):
                with torch.no_grad():
                    distribution.action_net.bias.copy_(
                        torch.linspace(-0.5, 0.5, distribution.action_net.bias.numel())
                    )
                distribution.update_latent_features(latent)
                density = distribution.log_prob(actions).exp()
                integral = torch.trapezoid(density, actions.squeeze(-1))
                torch.testing.assert_close(integral, torch.tensor(1.0), atol=2e-4, rtol=2e-4)


class BernsteinQuantileActionDistTests(unittest.TestCase):
    def test_deterministic_action_uses_high_density_mode_instead_of_median(self) -> None:
        distribution = BernsteinQuantileActionDist(latent_dim=1, action_dim=1, degree=3)
        with torch.no_grad():
            distribution.action_net.bias.copy_(torch.tensor([-8.0, 8.0, -8.0]))
        distribution.update_latent_features(torch.zeros(1, 1))
        median, _median_log_det = distribution._transform_forward_and_log_det(torch.full((1, 1), 0.5))

        mode = distribution.mode()

        self.assertLess(mode.item(), -0.9)
        self.assertLess(abs(median.item()), 1e-6)
        self.assertGreater(distribution.log_prob(mode).item(), distribution.log_prob(median).item())

    def test_inverse_log_prob_parameter_gradient_matches_finite_difference(self) -> None:
        distribution = BernsteinQuantileActionDist(latent_dim=1, action_dim=1, degree=3)
        latent = torch.ones(1, 1)
        actions = torch.tensor([[0.23]])
        with torch.no_grad():
            distribution.action_net.weight.zero_()
            distribution.action_net.bias.copy_(torch.tensor([0.4, -0.2, 0.7]))

        distribution.update_latent_features(latent)
        distribution.log_prob(actions).sum().backward()
        autograd_gradient = distribution.action_net.bias.grad.detach().clone()
        finite_difference_gradient = _finite_difference_bias_gradient(
            distribution,
            latent,
            actions,
        )

        torch.testing.assert_close(
            autograd_gradient,
            finite_difference_gradient,
            atol=5e-4,
            rtol=5e-3,
        )

    def test_inverse_log_prob_matches_forward_jacobian_for_nonuniform_map(self) -> None:
        distribution = BernsteinQuantileActionDist(latent_dim=2, action_dim=2, degree=6)
        with torch.no_grad():
            distribution.action_net.weight.normal_(std=0.5)
            distribution.action_net.bias.normal_(std=0.5)
        distribution.update_latent_features(torch.randn(7, 2))
        actions, log_det = distribution._transform_forward_and_log_det(torch.rand(7, 2))

        torch.testing.assert_close(
            distribution.log_prob(actions),
            -log_det.sum(dim=-1),
            atol=2e-5,
            rtol=2e-5,
        )

    def test_derivative_respects_structural_lower_bound(self) -> None:
        degree = 8
        min_increment = 0.01
        distribution = BernsteinQuantileActionDist(
            latent_dim=2,
            action_dim=2,
            degree=degree,
            min_normalized_increment=min_increment,
        )
        with torch.no_grad():
            distribution.action_net.weight.normal_(std=4.0)
            distribution.action_net.bias.normal_(std=4.0)
        distribution.update_latent_features(torch.randn(64, 2))
        _actions, log_det = distribution._transform_forward_and_log_det(torch.rand(64, 2))
        self.assertGreaterEqual(log_det.exp().amin().item(), 2.0 * degree * min_increment - 1e-6)

    def test_out_of_support_action_has_zero_density(self) -> None:
        distribution = BernsteinQuantileActionDist(latent_dim=2, action_dim=2)
        distribution.update_latent_features(torch.zeros(2, 2))
        log_probs = distribution.log_prob(torch.tensor([[1.01, 0.0], [-1.0, 1.0]]))
        self.assertTrue(torch.isneginf(log_probs[0]))
        torch.testing.assert_close(log_probs[1], torch.tensor(-2.0 * math.log(2.0)))

    def test_sharp_map_keeps_exact_endpoints_and_consistent_coefficient_increments(self) -> None:
        torch.manual_seed(7)
        distribution = BernsteinQuantileActionDist(latent_dim=1, action_dim=1, degree=8)
        with torch.no_grad():
            distribution.action_net.bias.normal_(std=20.0)
        distribution.update_latent_features(torch.zeros(2, 1))
        deltas, coefficients = distribution._deltas_and_coefficients()
        actions, _log_det = distribution._transform_forward_and_log_det(
            torch.tensor([[0.0], [1.0]])
        )

        torch.testing.assert_close(deltas, coefficients[..., 1:] - coefficients[..., :-1])
        torch.testing.assert_close(actions, torch.tensor([[-1.0], [1.0]]))


class RationalQuadraticSplineQuantileActionDistTests(unittest.TestCase):
    def test_deterministic_action_uses_high_density_mode_instead_of_median(self) -> None:
        num_bins = 4
        distribution = RationalQuadraticSplineQuantileActionDist(
            latent_dim=1,
            action_dim=1,
            num_bins=num_bins,
        )
        with torch.no_grad():
            distribution.action_net.bias.zero_()
            distribution.action_net.bias[2 * num_bins:] = torch.tensor([-12.0, 4.0, 4.0, 4.0, -12.0])
        distribution.update_latent_features(torch.zeros(1, 1))
        median, _median_log_det = distribution._transform_forward_and_log_det(torch.full((1, 1), 0.5))

        mode = distribution.mode()

        self.assertEqual(mode.item(), -1.0)
        self.assertLess(abs(median.item()), 1e-6)
        self.assertGreater(distribution.log_prob(mode).item(), distribution.log_prob(median).item())

    def test_comparison_bin_lookup_matches_right_searchsorted_including_knots(self) -> None:
        torch.manual_seed(5)
        widths = torch.softmax(torch.randn(3, 2, 6), dim=-1)
        knots = torch.cat(
            (
                torch.zeros_like(widths[..., :1]),
                torch.cumsum(widths, dim=-1),
            ),
            dim=-1,
        )
        random_values = torch.rand(3, 2, 20)
        exact_knots = knots[..., 1:-1]
        values = torch.cat((random_values, exact_knots), dim=-1)
        expanded_knots = knots.unsqueeze(-2).expand(3, 2, values.shape[-1], 7)

        actual = RationalQuadraticSplineQuantileActionDist._find_bins(values, expanded_knots)
        expected = torch.searchsorted(
            expanded_knots[..., 1:-1].contiguous(),
            values.unsqueeze(-1),
            right=True,
        ).squeeze(-1)

        torch.testing.assert_close(actual, expected)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA compilation")
    def test_deterministic_rqs_compiles_for_checkpoint_evaluation_shapes(self) -> None:
        latent = torch.randn(32, 8, 16, device="cuda")
        for num_bins in (4, 6):
            with self.subTest(num_bins=num_bins):
                distribution = RationalQuadraticSplineQuantileActionDist(
                    latent_dim=16,
                    action_dim=4,
                    num_bins=num_bins,
                ).cuda()

                def deterministic_actions(features: torch.Tensor) -> torch.Tensor:
                    actions, _log_probs = distribution.get_actions_with_log_probs(
                        features,
                        deterministic=True,
                    )
                    return actions

                compiled_actions = torch.compile(deterministic_actions, fullgraph=True)
                actions = compiled_actions(latent)

                self.assertEqual(tuple(actions.shape), (32, 8, 4))
                self.assertTrue(torch.isfinite(actions).all())

    def test_inverse_log_prob_parameter_gradient_matches_finite_difference(self) -> None:
        distribution = RationalQuadraticSplineQuantileActionDist(
            latent_dim=1,
            action_dim=1,
            num_bins=3,
        )
        latent = torch.ones(1, 1)
        actions = torch.tensor([[0.23]])
        with torch.no_grad():
            distribution.action_net.weight.zero_()
            distribution.action_net.bias.copy_(
                torch.linspace(-0.4, 0.6, distribution.action_net.bias.numel())
            )

        distribution.update_latent_features(latent)
        distribution.log_prob(actions).sum().backward()
        autograd_gradient = distribution.action_net.bias.grad.detach().clone()
        finite_difference_gradient = _finite_difference_bias_gradient(
            distribution,
            latent,
            actions,
        )

        torch.testing.assert_close(
            autograd_gradient,
            finite_difference_gradient,
            atol=5e-4,
            rtol=5e-3,
        )

    def test_analytic_inverse_log_prob_matches_forward_jacobian_for_nonuniform_map(self) -> None:
        for fixed_boundary_derivatives in (False, True):
            with self.subTest(fixed_boundary_derivatives=fixed_boundary_derivatives):
                torch.manual_seed(123 + int(fixed_boundary_derivatives))
                distribution = RationalQuadraticSplineQuantileActionDist(
                    latent_dim=2,
                    action_dim=2,
                    num_bins=6,
                    fixed_boundary_derivatives=fixed_boundary_derivatives,
                )
                with torch.no_grad():
                    distribution.action_net.weight.normal_(std=0.2)
                    distribution.action_net.bias.normal_(std=0.2)
                distribution.update_latent_features(torch.randn(64, 2))
                actions, log_det = distribution._transform_forward_and_log_det(torch.rand(64, 2))

                torch.testing.assert_close(
                    distribution.log_prob(actions),
                    -log_det.sum(dim=-1),
                    atol=2e-5,
                    rtol=2e-5,
                )

    def test_sharp_inverse_likelihood_matches_high_precision_reference(self) -> None:
        torch.manual_seed(4)
        distribution = RationalQuadraticSplineQuantileActionDist(
            latent_dim=3,
            action_dim=4,
            num_bins=6,
        )
        with torch.no_grad():
            distribution.action_net.weight.normal_(std=5.0)
            distribution.action_net.bias.normal_(std=5.0)
        distribution.update_latent_features(torch.randn(256, 3))
        actions, _log_det = distribution._transform_forward_and_log_det(torch.rand(256, 4))

        reference_log_probs = _reference_rqs_log_prob_float64(distribution, actions)

        torch.testing.assert_close(
            distribution.log_prob(actions).double(),
            reference_log_probs,
            atol=5e-3,
            rtol=1e-4,
        )

    def test_spline_constraints_and_fixed_boundary_derivatives_are_structural(self) -> None:
        distribution = RationalQuadraticSplineQuantileActionDist(
            latent_dim=2,
            action_dim=2,
            num_bins=4,
            min_bin_width=0.02,
            min_bin_height=0.03,
            min_derivative=0.04,
            fixed_boundary_derivatives=True,
        )
        with torch.no_grad():
            distribution.action_net.weight.normal_(std=5.0)
            distribution.action_net.bias.normal_(std=5.0)
        distribution.update_latent_features(torch.randn(32, 2))
        widths, heights, derivatives, x_knots, y_knots = distribution._spline_parameters()

        self.assertGreaterEqual(widths.amin().item(), 0.02 - 1e-6)
        self.assertGreaterEqual(heights.amin().item(), 0.06 - 1e-6)
        self.assertGreaterEqual(derivatives.amin().item(), 0.04 - 1e-6)
        torch.testing.assert_close(widths.sum(dim=-1), torch.ones_like(widths[..., 0]))
        torch.testing.assert_close(heights.sum(dim=-1), torch.full_like(heights[..., 0], 2.0))
        torch.testing.assert_close(x_knots[..., 0], torch.zeros_like(x_knots[..., 0]))
        torch.testing.assert_close(x_knots[..., -1], torch.ones_like(x_knots[..., -1]))
        torch.testing.assert_close(y_knots[..., 0], -torch.ones_like(y_knots[..., 0]))
        torch.testing.assert_close(y_knots[..., -1], torch.ones_like(y_knots[..., -1]))
        torch.testing.assert_close(derivatives[..., 0], torch.full_like(derivatives[..., 0], 2.0))
        torch.testing.assert_close(derivatives[..., -1], torch.full_like(derivatives[..., -1], 2.0))

    def test_out_of_support_action_has_zero_density(self) -> None:
        distribution = RationalQuadraticSplineQuantileActionDist(latent_dim=2, action_dim=2)
        distribution.update_latent_features(torch.zeros(2, 2))
        log_probs = distribution.log_prob(torch.tensor([[0.0, -1.01], [-1.0, 1.0]]))
        self.assertTrue(torch.isneginf(log_probs[0]))
        torch.testing.assert_close(log_probs[1], torch.tensor(-2.0 * math.log(2.0)))

    def test_sharp_map_keeps_exact_endpoints_and_finite_endpoint_likelihoods(self) -> None:
        torch.manual_seed(0)
        distribution = RationalQuadraticSplineQuantileActionDist(
            latent_dim=1,
            action_dim=1,
            num_bins=6,
        )
        with torch.no_grad():
            distribution.action_net.bias.normal_(std=20.0)
        distribution.update_latent_features(torch.zeros(2, 1))
        actions, _log_det = distribution._transform_forward_and_log_det(
            torch.tensor([[0.0], [1.0]])
        )
        log_probs = distribution.log_prob(torch.tensor([[-1.0], [1.0]]))

        torch.testing.assert_close(actions, torch.tensor([[-1.0], [1.0]]), atol=1e-7, rtol=0.0)
        self.assertTrue(torch.isfinite(log_probs).all())
        log_probs.sum().backward()
        self.assertTrue(torch.isfinite(distribution.action_net.bias.grad).all())

    def test_sharp_map_forward_samples_have_finite_inverse_likelihoods(self) -> None:
        torch.manual_seed(13)
        distribution = RationalQuadraticSplineQuantileActionDist(
            latent_dim=2,
            action_dim=3,
            num_bins=6,
        )
        with torch.no_grad():
            distribution.action_net.weight.normal_(std=10.0)
            distribution.action_net.bias.normal_(std=10.0)
        distribution.update_latent_features(torch.randn(32, 2))
        actions, _log_det = distribution._transform_forward_and_log_det(torch.rand(32, 3))

        log_probs = distribution.log_prob(actions)

        self.assertTrue(torch.isfinite(log_probs).all())


if __name__ == "__main__":
    unittest.main()
