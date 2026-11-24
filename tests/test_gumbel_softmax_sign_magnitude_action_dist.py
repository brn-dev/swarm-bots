import unittest
import math
from unittest.mock import patch

import pytest
import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionGradientEstimator
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaActionDist,
    GumbelSoftmaxSignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.hybrid_action_dist import continuous_action_gradient_estimator
from swarmbots.learn.action_dists.sign_magnitude_action_dist import SignMagnitudeActionDist
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaActionDist


def _zero_init(linear: nn.Linear) -> None:
    nn.init.zeros_(linear.weight)
    nn.init.zeros_(linear.bias)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_sign_magnitude_beta_rollout_and_rescoring_under_cpu_autocast(dtype: torch.dtype) -> None:
    trunk = nn.Linear(3, 4)
    dist = SignMagnitudeBetaActionDist(4, 2, None)
    observations = torch.ones((2, 3, 3))

    with torch.no_grad(), torch.autocast("cpu", dtype):
        actions, rollout_scores = dist.get_on_policy_actions_with_log_probs(
            trunk(observations), storage_dtype=torch.float32,
        )
    with torch.autocast("cpu", dtype):
        scores = dist.update_latent_features(trunk(observations)).log_prob(actions)

    assert actions.dtype == torch.float32
    assert (actions.abs() < 1).all()
    assert torch.isfinite(rollout_scores).all()
    torch.testing.assert_close(scores, rollout_scores, rtol=0, atol=0)
    gradients = torch.autograd.grad(scores.sum(), (*trunk.parameters(), *dist.parameters()))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def _make_distributions() -> list[
        GumbelSoftmaxSignMagnitudeBetaActionDist
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
    ]


class TestGumbelSoftmaxSignMagnitudeActionDist(unittest.TestCase):
    def test_entropy_metrics_describe_disjoint_mixture_without_consuming_rng(self) -> None:
        dist = GumbelSoftmaxSignMagnitudeBetaActionDist(
            latent_dim=4,
            action_dim=3,
            action_net_initialization=_zero_init,
            initial_positive_prob=0.4,
            negative_alpha=2.0,
            negative_beta=5.0,
            positive_alpha=3.0,
            positive_beta=4.0,
        )
        actions = dist.update_latent_features(torch.ones(2, 3, 4)).mode()
        mask = torch.tensor([[True, False, True], [False, True, False]])
        rng_state = torch.random.get_rng_state().clone()

        metrics = dist.get_metrics(actions[mask], agent_mask=mask)

        categorical_entropy = torch.tensor(-0.4 * math.log(0.4) - 0.6 * math.log(0.6))
        magnitude_entropy = (
            0.6 * torch.distributions.Beta(2.0, 5.0).entropy()
            + 0.4 * torch.distributions.Beta(3.0, 4.0).entropy()
        )
        torch.testing.assert_close(metrics["ent_categorical"], categorical_entropy)
        torch.testing.assert_close(metrics["ent_beta"], magnitude_entropy)
        torch.testing.assert_close(metrics["ent"], categorical_entropy + magnitude_entropy)
        torch.testing.assert_close(torch.random.get_rng_state(), rng_state)
        for name in ("ent", "ent_categorical", "ent_beta"):
            self.assertFalse(metrics[name].requires_grad)

    def test_beta_family_uses_sign_magnitude_base(self) -> None:
        self.assertTrue(issubclass(SignMagnitudeBetaActionDist, SignMagnitudeActionDist))

    def test_mode_selects_the_tallest_beta_peak_when_its_location_rounds_to_one(self) -> None:
        for component in (0, 1):
            dist = SignMagnitudeBetaActionDist(4, 1, _zero_init, initial_positive_prob=0.5)
            with torch.no_grad():
                start = 2 + component * 2
                dist.output_net.bias[start:start + 2] = torch.tensor([10.0, -100.0])
            action = dist.update_latent_features(torch.zeros(1, 4)).mode().item()
            if component == 0:
                self.assertTrue(-1e-7 < action < 0)
            else:
                self.assertEqual(action, 1.0)

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

    def test_actor_and_target_score_beta_tails_without_clipping(self) -> None:
        dist = GumbelSoftmaxSignMagnitudeBetaActionDist(
            4, 1, _zero_init, epsilon=0.2, initial_positive_prob=0.5,
            negative_alpha=2, negative_beta=3, positive_alpha=2, positive_beta=3,
        )
        latent = torch.zeros(2048, 4)
        for use_rsample in (True, False):
            with self.subTest(use_rsample=use_rsample):
                torch.manual_seed(43)
                actions, scores = dist.get_actions_with_log_probs(latent, use_rsample=use_rsample)
                magnitude = actions.abs().squeeze(-1)
                self.assertTrue((magnitude < 0.2).any() and (magnitude > 0.8).any())
                expected = torch.where(
                    actions.squeeze(-1) < 0,
                    torch.distributions.Beta(3.0, 2.0).log_prob(magnitude),
                    torch.distributions.Beta(2.0, 3.0).log_prob(magnitude),
                ) - math.log(2)
                torch.testing.assert_close(scores, expected)
                torch.testing.assert_close(dist.log_prob(actions), expected)
                if use_rsample:
                    actions.squeeze(-1)[magnitude < 0.2].sum().backward()
                    self.assertGreater(dist.output_net.bias.grad[2:].abs().sum().item(), 0)

    def test_hard_signs_remain_identical_across_temperatures(self) -> None:
        dist = GumbelSoftmaxSignMagnitudeBetaActionDist(4, 1, _zero_init)
        latent = torch.zeros(256, 4)
        torch.manual_seed(47)
        actions, _ = dist.get_actions_with_log_probs(latent, use_rsample=True)
        self.assertTrue((actions < 0).any() and (actions > 0).any())
        for temperature in (1e9, 1e-30):
            dist.set_gumbel_temperature(temperature)
            torch.manual_seed(47)
            actual, scores = dist.get_actions_with_log_probs(latent, use_rsample=True)
            torch.testing.assert_close(actual, actions, rtol=0, atol=0)
            gradients = torch.autograd.grad(scores.sum(), dist.output_net.bias)[0]
            self.assertTrue(torch.isfinite(gradients).all())

    def test_actor_and_target_preserve_and_score_tiny_negative_draw(self) -> None:
        dist = GumbelSoftmaxSignMagnitudeBetaActionDist(
            4, 1, _zero_init, epsilon=0.2, initial_positive_prob=1e-10, negative_alpha=2, negative_beta=3,
        )
        latent = torch.zeros(1, 4)
        near_zero = torch.tensor([[1e-9]])
        expected = torch.distributions.Beta(3.0, 2.0).log_prob(near_zero[0]) + math.log1p(-1e-10)
        for use_rsample in (True, False):
            torch.manual_seed(43)
            with patch.object(torch.distributions.Beta, "rsample", return_value=near_zero):
                actions, scores = dist.get_actions_with_log_probs(latent, use_rsample=use_rsample)
            torch.testing.assert_close(actions, -near_zero, rtol=0, atol=0)
            torch.testing.assert_close(scores, expected)
            torch.testing.assert_close(dist.log_prob(actions), expected)

    def test_stochastic_negative_tail_stays_inside_support_and_endpoints_can_be_scored(self) -> None:
        for cls in (SignMagnitudeBetaActionDist, GumbelSoftmaxSignMagnitudeBetaActionDist):
            dist = cls(
                4, 1, _zero_init, initial_positive_prob=1e-12, negative_alpha=1.1, negative_beta=1e6,
            )
            latent = torch.zeros(2048, 4)
            torch.manual_seed(53)
            actions, scores = dist.get_on_policy_actions_with_log_probs(latent)
            self.assertTrue(((actions > -1) & (actions < -0.99)).all())
            torch.testing.assert_close(scores, dist.log_prob(actions))
            endpoints = torch.tensor([[-1.0], [0.0], [1.0]])
            dist.update_latent_features(latent[:3])
            self.assertTrue(torch.isfinite(dist.log_prob(endpoints)).all())

    def test_concentrated_actor_and_target_scores_and_stored_action_gradients_match_double_reference(self) -> None:
        dist = GumbelSoftmaxSignMagnitudeBetaActionDist(
            4, 1, _zero_init, initial_positive_prob=0.5,
            negative_alpha=1.1, negative_beta=1e6, positive_alpha=3e5, positive_beta=7e5,
        )
        latent = torch.zeros(256, 4)
        for use_rsample in (False, True):
            torch.manual_seed(67)
            actions, scores = dist.get_actions_with_log_probs(latent, use_rsample=use_rsample)
            magnitudes = actions.detach().abs().double()
            negative, positive = dist.negative_magnitude_dist, dist.positive_magnitude_dist
            assert isinstance(negative, torch.distributions.Beta) and isinstance(positive, torch.distributions.Beta)
            negative_reference = torch.distributions.Beta(negative.concentration1.double(), negative.concentration0.double())
            positive_reference = torch.distributions.Beta(positive.concentration1.double(), positive.concentration0.double())
            assert dist.weight_logits is not None
            log_weights = dist.weight_logits.double().log_softmax(dim=-1)
            expected = torch.where(
                actions < 0,
                negative_reference.log_prob(magnitudes) + log_weights[..., 0],
                positive_reference.log_prob(magnitudes) + log_weights[..., 1],
            ).squeeze(-1)
            torch.testing.assert_close(scores, expected.float())
            stored_scores = dist.log_prob(actions.detach())
            torch.testing.assert_close(stored_scores, expected.float())
            actual_grad = torch.autograd.grad(stored_scores.sum(), dist.output_net.bias, retain_graph=True)[0]
            expected_grad = torch.autograd.grad(expected.sum(), dist.output_net.bias)[0]
            torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-5, atol=1e-6)

    def test_deterministic_action_is_component_density_mode_not_mixture_mean(self) -> None:
        for dist in _make_distributions():
            with self.subTest(distribution=type(dist).__name__):
                latent = torch.ones((2, 5, 4))

                actions = dist.update_latent_features(latent).mode()

                torch.testing.assert_close(actions, torch.full_like(actions, -0.5))

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

    def test_rejects_invalid_temperature_at_construction_and_update(self) -> None:
        dist = GumbelSoftmaxSignMagnitudeBetaActionDist(4, 1, _zero_init)
        for value in (0.0, -1.0, math.nan, math.inf, 1e-100, 1e100):
            with self.subTest(value=value):
                with self.assertRaisesRegex(ValueError, "gumbel_temperature"):
                    GumbelSoftmaxSignMagnitudeBetaActionDist(4, 1, _zero_init, gumbel_temperature=value)
                with self.assertRaisesRegex(ValueError, "gumbel_temperature"):
                    dist.set_gumbel_temperature(value)
                self.assertEqual(dist.gumbel_temperature, 1.0)

    def test_temperature_update_round_trips_through_state_dict(self) -> None:
        for original, restored in zip(_make_distributions(), _make_distributions(), strict=True):
            with self.subTest(distribution=type(original).__name__):
                original.set_gumbel_temperature(0.25)

                restored.load_state_dict(original.state_dict())

                self.assertEqual(restored.gumbel_temperature, 0.25)


if __name__ == "__main__":
    unittest.main()
