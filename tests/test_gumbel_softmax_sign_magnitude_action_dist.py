import unittest
from unittest.mock import patch

import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionGradientEstimator
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaActionDist,
    GumbelSoftmaxSignMagnitudeBetaConfig,
    GumbelSoftmaxSignMagnitudeKumaraswamyActionDist,
    GumbelSoftmaxSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.action_dists.hybrid_action_dist import continuous_action_gradient_estimator
from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyActionDist,
)
from swarmbots.learn.action_dists.sign_magnitude_action_dist import SignMagnitudeActionDist
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaActionDist
from swarmbots.learn.action_dists.sign_magnitude_kumaraswamy_action_dist import (
    SignMagnitudeKumaraswamyActionDist,
)


def _zero_init(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)


def _make_distributions() -> list[
        GumbelSoftmaxSignMagnitudeBetaActionDist | GumbelSoftmaxSignMagnitudeKumaraswamyActionDist
]:
    common_kwargs = {
        "latent_dim": 4,
        "action_dim": 3,
        "action_net_initialization": _zero_init,
        "initial_positive_prob": 0.4,
        "gumbel_temperature": 0.7,
    }
    return [
        GumbelSoftmaxSignMagnitudeBetaActionDist(**common_kwargs),
        GumbelSoftmaxSignMagnitudeKumaraswamyActionDist(**common_kwargs),
    ]


class TestGumbelSoftmaxSignMagnitudeActionDist(unittest.TestCase):
    def test_beta_and_kumaraswamy_families_share_sign_magnitude_base(self) -> None:
        self.assertTrue(issubclass(SignMagnitudeBetaActionDist, SignMagnitudeActionDist))
        self.assertTrue(issubclass(SignMagnitudeKumaraswamyActionDist, SignMagnitudeActionDist))

    def test_kumaraswamy_variant_uses_neutral_magnitude_family(self) -> None:
        self.assertTrue(
            issubclass(GumbelSoftmaxSignMagnitudeKumaraswamyActionDist, SignMagnitudeKumaraswamyActionDist)
        )
        self.assertFalse(
            issubclass(
                GumbelSoftmaxSignMagnitudeKumaraswamyActionDist,
                ReparameterizedSignMagnitudeKumaraswamyActionDist,
            )
        )

    def test_forward_selection_is_hard_and_log_prob_is_exact(self) -> None:
        for dist in _make_distributions():
            with self.subTest(distribution=type(dist).__name__):
                torch.manual_seed(11)
                latent = torch.ones((2, 5, 4))
                dist.update_latent_features(latent)

                _actions, selection, _negative_magnitudes, _positive_magnitudes = dist._rsample_with_selection()
                actions, sampled_log_probs = dist.get_actions_with_log_probs(latent, use_rsample=True)
                exact_log_probs = dist.log_prob(actions)

                self.assertTrue(((selection == 0.0) | (selection == 1.0)).all())
                self.assertTrue((selection.sum(dim=-1) == 1.0).all())
                torch.testing.assert_close(exact_log_probs, sampled_log_probs)

    def test_deterministic_action_is_component_density_mode_not_mixture_mean(self) -> None:
        for dist in _make_distributions():
            with self.subTest(distribution=type(dist).__name__):
                latent = torch.ones((2, 5, 4))

                actions = dist.update_latent_features(latent).mode()

                torch.testing.assert_close(actions, torch.full_like(actions, -0.5))

    def test_negative_branch_preserves_legacy_unit_interval_mapping(self) -> None:
        for dist in _make_distributions():
            with self.subTest(distribution=type(dist).__name__):
                negative_components = torch.full((2, 3), dist._NEGATIVE_INDEX)
                actions = dist._select_actions(
                    negative_components,
                    torch.full((2, 3), 0.2),
                    torch.full((2, 3), 0.7),
                )

                torch.testing.assert_close(actions, torch.full((2, 3), -0.8))

    def test_deterministic_action_stays_finite_when_shape_logits_saturate(self) -> None:
        for dist in _make_distributions():
            with self.subTest(distribution=type(dist).__name__):
                with torch.no_grad():
                    output_bias = dist.output_net.bias.view(dist.action_dim, 6)
                    output_bias[:, 2:].fill_(-100.0)

                actions = dist.update_latent_features(torch.ones((2, 5, 4))).mode()

                self.assertTrue(torch.isfinite(actions).all())

    def test_sac_sample_backpropagates_through_sign_and_both_magnitudes(self) -> None:
        for dist in _make_distributions():
            with self.subTest(distribution=type(dist).__name__):
                torch.manual_seed(17)
                latent = torch.ones((3, 2, 4))

                actions, log_probs = dist.get_actions_with_log_probs(latent, use_rsample=True)
                loss = actions.square().sum() + 0.2 * log_probs.sum()
                loss.backward()

                output_grad = dist.output_net.bias.grad
                self.assertIsNotNone(output_grad)
                assert output_grad is not None
                output_grad = output_grad.view(dist.action_dim, 6)
                self.assertTrue(torch.isfinite(output_grad).all())
                self.assertTrue((output_grad[:, :2].abs().sum(dim=-1) > 0.0).all())
                self.assertTrue((output_grad[:, 2:].abs().sum(dim=-1) > 0.0).all())

    def test_configs_explicitly_report_straight_through_gradients(self) -> None:
        for config in (
                GumbelSoftmaxSignMagnitudeBetaConfig(),
                GumbelSoftmaxSignMagnitudeKumaraswamyConfig(),
        ):
            self.assertIs(
                continuous_action_gradient_estimator(config),
                ActionGradientEstimator.STRAIGHT_THROUGH,
            )

    def test_non_reparameterized_sampling_does_not_track_actions(self) -> None:
        for dist in _make_distributions():
            with self.subTest(distribution=type(dist).__name__):
                latent = torch.ones((2, 3, 4), requires_grad=True)

                with patch(
                        "swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist."
                        "_straight_through_gumbel_softmax",
                        side_effect=AssertionError("ordinary sampling must use the exact categorical distribution"),
                ):
                    actions, log_probs = dist.get_actions_with_log_probs(latent, use_rsample=False)

                self.assertFalse(actions.requires_grad)
                self.assertTrue(log_probs.requires_grad)

    def test_rejects_non_positive_temperature(self) -> None:
        with self.assertRaisesRegex(ValueError, "gumbel_temperature"):
            GumbelSoftmaxSignMagnitudeBetaActionDist(
                latent_dim=4,
                action_dim=3,
                action_net_initialization=_zero_init,
                gumbel_temperature=0.0,
            )

    def test_temperature_update_round_trips_through_state_dict(self) -> None:
        for original, restored in zip(_make_distributions(), _make_distributions(), strict=True):
            with self.subTest(distribution=type(original).__name__):
                original.set_gumbel_temperature(0.25)

                restored.load_state_dict(original.state_dict())

                self.assertEqual(restored.gumbel_temperature, 0.25)


if __name__ == "__main__":
    unittest.main()
