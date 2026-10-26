import math
import unittest
from unittest.mock import patch

import pytest
import torch

from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])])
@pytest.mark.parametrize("full_std", [False, True])
@pytest.mark.parametrize("squash_output", [False, True])
def test_gsde_sampling_and_likelihood_gradients_stay_on_device(
    device: str, full_std: bool, squash_output: bool,
) -> None:
    dist = GSDEActionDist(
        latent_dim=4, action_dim=2, base_std=0.5, full_std=full_std, squash_output=squash_output,
    ).to(device)
    # Bounded features and noise keep float32 tanh inversion away from saturation.
    latent_pi = torch.linspace(-0.5, 0.5, 24, device=device).reshape(2, 3, 4).requires_grad_()
    dist.set_temporal_correlation_state(torch.linspace(-1.0, 1.0, 48, device=device).reshape(2, 3, 4, 2))

    with (
        patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("sampling must stay on device")),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("sampling must stay on device")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("sampling must stay on device")),
    ):
        actions, log_probs = dist.get_actions_with_log_probs(latent_pi, use_rsample=True)
        evaluated_log_probs = dist.log_prob(actions)
        (actions.square().mean() + log_probs.mean()).backward()

    torch.testing.assert_close(evaluated_log_probs, log_probs, rtol=1e-4, atol=1e-5)
    assert torch.isfinite(actions).all() and torch.isfinite(log_probs).all()
    for gradient in (latent_pi.grad, dist.action_net.weight.grad, dist.log_stds.grad):
        assert gradient is not None and torch.isfinite(gradient).all()


@pytest.mark.parametrize("full_std", [False, True])
@pytest.mark.parametrize("base_std, expected_std", [(0.01, math.exp(-3)), (0.2, 0.2), (1.0, math.exp(-1))])
def test_gsde_std_metrics_follow_the_sampling_clamp(
    full_std: bool, base_std: float, expected_std: float,
) -> None:
    dist = GSDEActionDist(
        latent_dim=2, action_dim=3, base_std=base_std,
        full_std=full_std, log_std_clamp_range=(-3.0, -1.0),
    )
    actions = dist.update_latent_features(torch.ones(4, 2)).mode()
    metrics = dist.get_metrics(actions)
    assert metrics["std"].mean == pytest.approx(expected_std)
    assert metrics["std"].min_value == pytest.approx(expected_std)
    assert metrics["std"].max_value == pytest.approx(expected_std)


class GSDEActionDistTests(unittest.TestCase):
    def test_squashed_gsde_labels_underlying_normal_entropy(self) -> None:
        unsquashed_dist = GSDEActionDist(latent_dim=2, action_dim=3, base_std=0.75)
        squashed_dist = GSDEActionDist(latent_dim=2, action_dim=3, base_std=0.75, squash_output=True)
        squashed_dist.load_state_dict(unsquashed_dist.state_dict())
        latent_pi = torch.ones((2, 4, 2), dtype=torch.float32)
        unsquashed_actions = unsquashed_dist.update_latent_features(latent_pi).mode()
        squashed_actions = squashed_dist.update_latent_features(latent_pi).mode()

        unsquashed_metrics = unsquashed_dist.get_metrics(unsquashed_actions)
        squashed_metrics = squashed_dist.get_metrics(squashed_actions)

        torch.testing.assert_close(squashed_metrics["ent_normal"], unsquashed_metrics["ent"])
        self.assertNotIn("ent", squashed_metrics)
        self.assertFalse(squashed_metrics["ent_normal"].requires_grad)

    def test_squashed_gsde_entropy_metrics_respect_agent_mask_and_action_splitter(self) -> None:
        dist = GSDEActionDist(latent_dim=2, action_dim=4, base_std=0.75, squash_output=True)
        latent_pi = torch.tensor([[[1., 1.], [100., 100.], [2., 2.]], [[100., 100.], [3., 3.], [4., 4.]]])
        agent_mask = torch.tensor([[True, False, True], [False, True, True]])

        def split_actions(values: torch.Tensor) -> dict[str, torch.Tensor]:
            return {"left": values[..., :2], "right": values[..., 2:]}

        actions = dist.update_latent_features(latent_pi).mode()
        metrics = dist.get_metrics(actions[agent_mask], agent_mask=agent_mask, action_splitter=split_actions)
        entropy = dist.distribution.entropy()

        for name, values in (("ent_normal", entropy), ("ent_normal_left", entropy[..., :2]),
                             ("ent_normal_right", entropy[..., 2:])):
            torch.testing.assert_close(metrics[name], values.mean(dim=-1)[agent_mask].mean())
            self.assertFalse(metrics[name].requires_grad)

    def test_episode_start_reset_expands_env_mask_across_agent_noise(self) -> None:
        torch.manual_seed(1234)
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        dist.reset_noise((2, 3))
        original_noise = dist._exploration_noise.clone()

        dist.reset_on_ep_start(torch.tensor([True, False], dtype=torch.bool))

        self.assertEqual(tuple(dist._exploration_batch_shape), (2, 3))
        self.assertEqual(tuple(dist._exploration_noise.shape), (2, 3, 2, 1))
        self.assertFalse(torch.equal(dist._exploration_noise[0], original_noise[0]))
        self.assertTrue(torch.equal(dist._exploration_noise[1], original_noise[1]))

    def test_masked_reset_only_replaces_selected_agent_noise(self) -> None:
        torch.manual_seed(1234)
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        dist.reset_noise((2, 3))
        original_noise = dist._exploration_noise.clone()
        reset_mask = torch.tensor(
            [
                [False, True, False],
                [True, False, True],
            ],
            dtype=torch.bool,
        )

        dist.reset_noise_masked(reset_mask)

        self.assertTrue(torch.equal(dist._exploration_noise[0, 0], original_noise[0, 0]))
        self.assertFalse(torch.equal(dist._exploration_noise[0, 1], original_noise[0, 1]))
        self.assertTrue(torch.equal(dist._exploration_noise[0, 2], original_noise[0, 2]))
        self.assertFalse(torch.equal(dist._exploration_noise[1, 0], original_noise[1, 0]))
        self.assertTrue(torch.equal(dist._exploration_noise[1, 1], original_noise[1, 1]))
        self.assertFalse(torch.equal(dist._exploration_noise[1, 2], original_noise[1, 2]))

    def test_zero_mask_does_not_replace_noise(self) -> None:
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        dist.reset_noise((2, 3))
        original_noise = dist._exploration_noise.clone()

        dist.reset_noise_masked(torch.zeros((2, 3), dtype=torch.bool))

        self.assertTrue(torch.equal(dist._exploration_noise, original_noise))

    def test_env_mask_initializes_env_only_shape_when_no_batch_shape_exists(self) -> None:
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )

        dist.reset_on_ep_start(torch.tensor([True, False], dtype=torch.bool))

        self.assertEqual(tuple(dist._exploration_batch_shape), (2,))
        self.assertEqual(tuple(dist._exploration_noise.shape), (2, 2, 1))

    def test_agent_mask_reinitializes_noise_to_agent_batch_shape(self) -> None:
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        dist.reset_on_ep_start(torch.tensor([True, False], dtype=torch.bool))

        dist.reset_noise_masked(torch.zeros((2, 3), dtype=torch.bool))

        self.assertEqual(tuple(dist._exploration_batch_shape), (2, 3))
        self.assertEqual(tuple(dist._exploration_noise.shape), (2, 3, 2, 1))

    def test_step_reset_with_batch_shape_replaces_all_noise(self) -> None:
        torch.manual_seed(1234)
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        dist.reset_noise((2, 3))
        original_noise = dist._exploration_noise.clone()

        dist.reset_on_step(batch_shape=(2, 3))

        self.assertEqual(tuple(dist._exploration_batch_shape), (2, 3))
        self.assertFalse(torch.equal(dist._exploration_noise, original_noise))

    def test_sampling_requires_noise_reset(self) -> None:
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        latent_pi = torch.ones((2, 3, 2), dtype=torch.float32)

        with self.assertRaisesRegex(RuntimeError, "reset_noise"):
            dist.get_actions_with_log_probs(latent_pi)

    def test_agent_sampling_matches_full_sampling_slice(self) -> None:
        torch.manual_seed(1234)
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        for batch_shape in ((2,), (2, 1), (2, 4)):
            with self.subTest(batch_shape=batch_shape):
                latent_pi = torch.randn(*batch_shape, 3, 2)
                dist.reset_noise((*batch_shape, 3))
                full_actions = dist.update_latent_features(latent_pi).sample()
                for agent in range(3):
                    agent_actions = dist.update_latent_features(
                        latent_pi[..., agent:agent + 1, :],
                    ).sample(agent=agent)
                    torch.testing.assert_close(agent_actions, full_actions[..., agent:agent + 1, :])

    def test_rsample_tracks_distribution_head_gradient(self) -> None:
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        latent_pi = torch.ones((2, 3, 2), dtype=torch.float32, requires_grad=True)
        dist.reset_noise((2, 3))

        actions = dist.update_latent_features(latent_pi).rsample()
        actions.sum().backward()

        self.assertTrue(actions.requires_grad)
        self.assertIsNotNone(dist.action_net.bias.grad)
        self.assertGreater(dist.action_net.bias.grad.abs().sum().item(), 0.0)

    def test_rsample_tracks_log_std_gradient_through_actions(self) -> None:
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        latent_pi = torch.ones((2, 3, 2), dtype=torch.float32)
        dist.reset_noise((2, 3))
        noise_state = dist.get_temporal_correlation_state()
        assert noise_state is not None
        with torch.no_grad():
            dist.action_net.weight.zero_()
            dist.action_net.bias.zero_()
            noise_state.fill_(1.0)

        actions = dist.update_latent_features(latent_pi).rsample()
        actions.sum().backward()

        self.assertIsNotNone(dist.log_stds.grad)
        self.assertGreater(dist.log_stds.grad.abs().sum().item(), 0.0)

    def test_updated_std_applies_to_existing_noise_state(self) -> None:
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        latent_pi = torch.ones((2, 3, 2), dtype=torch.float32)
        dist.reset_noise((2, 3))
        noise_state = dist.get_temporal_correlation_state()
        assert noise_state is not None
        with torch.no_grad():
            dist.action_net.weight.zero_()
            dist.action_net.bias.zero_()
            noise_state.fill_(1.0)

        original_actions = dist.update_latent_features(latent_pi).sample()
        with torch.no_grad():
            dist.log_stds.fill_(math.log(0.5))
        scaled_actions = dist.update_latent_features(latent_pi).sample()

        self.assertTrue(torch.allclose(scaled_actions, original_actions * 0.5))

    def test_ppo_log_prob_evaluation_does_not_depend_on_current_noise(self) -> None:
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        latent_pi = torch.ones((2, 3, 2), dtype=torch.float32)
        dist.reset_noise((2, 3))
        actions, rollout_log_probs = dist.get_actions_with_log_probs(latent_pi)

        dist.reset_noise((2, 3))
        dist.update_latent_features(latent_pi)
        evaluated_log_probs = dist.log_prob(actions)

        self.assertTrue(torch.allclose(evaluated_log_probs, rollout_log_probs))

    def test_sample_does_not_track_distribution_head_gradient(self) -> None:
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=1,
            base_std=1.0,
        )
        latent_pi = torch.ones((2, 3, 2), dtype=torch.float32, requires_grad=True)
        dist.reset_noise((2, 3))

        actions = dist.update_latent_features(latent_pi).sample()
        log_probs = dist.log_prob(actions)

        self.assertFalse(actions.requires_grad)
        self.assertTrue(log_probs.requires_grad)

    def test_squashed_get_actions_with_log_probs_uses_cached_gaussian_actions(self) -> None:
        torch.manual_seed(1234)
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=2,
            base_std=1.0,
            squash_output=True,
        )
        latent_pi = torch.ones((2, 3, 2), dtype=torch.float32)
        dist.reset_noise((2, 3))

        actions, log_probs = dist.get_actions_with_log_probs(latent_pi)

        self.assertTrue(torch.all(actions < 1.0))
        self.assertTrue(torch.all(actions > -1.0))
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.allclose(log_probs, dist.log_prob(actions, gaussian_actions=dist._last_gaussian_actions)))


if __name__ == "__main__":
    unittest.main()
