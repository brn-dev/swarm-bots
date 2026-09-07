import math
import unittest

import torch

from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist


class GSDEActionDistTests(unittest.TestCase):
    def test_squashed_gsde_entropy_loss_matches_unsquashed_gaussian_proxy(self) -> None:
        unsquashed_dist = GSDEActionDist(
            latent_dim=2,
            action_dim=3,
            base_std=0.75,
            squash_output=False,
            ent_loss_coef=0.1,
        )
        squashed_dist = GSDEActionDist(
            latent_dim=2,
            action_dim=3,
            base_std=0.75,
            squash_output=True,
            ent_loss_coef=0.1,
        )
        squashed_dist.load_state_dict(unsquashed_dist.state_dict())
        latent_pi = torch.ones((2, 4, 2), dtype=torch.float32)

        unsquashed_dist.update_latent_features(latent_pi)
        squashed_dist.update_latent_features(latent_pi)
        unsquashed_losses, unsquashed_metrics = unsquashed_dist.compute_extra_losses()
        squashed_losses, squashed_metrics = squashed_dist.compute_extra_losses()

        self.assertIn("entropy", squashed_losses)
        self.assertEqual(tuple(squashed_losses["entropy"].shape), (2, 4))
        self.assertTrue(torch.allclose(squashed_losses["entropy"], unsquashed_losses["entropy"]))
        self.assertEqual(squashed_metrics["ent"], unsquashed_metrics["ent"])

    def test_squashed_gsde_entropy_metrics_respect_agent_mask_and_action_splitter(self) -> None:
        dist = GSDEActionDist(
            latent_dim=2,
            action_dim=4,
            base_std=0.75,
            squash_output=True,
            ent_loss_coef=0.1,
        )
        latent_pi = torch.ones((2, 3, 2), dtype=torch.float32)
        agent_mask = torch.tensor(
            [
                [True, False, True],
                [False, True, True],
            ],
            dtype=torch.bool,
        )

        def split_actions(values: torch.Tensor) -> dict[str, torch.Tensor]:
            return {
                "left": values[..., :2],
                "right": values[..., 2:],
            }

        dist.update_latent_features(latent_pi)
        losses, metrics = dist.compute_extra_losses(
            agent_mask=agent_mask,
            action_splitter=split_actions,
        )

        self.assertIn("entropy", losses)
        self.assertEqual({"ent", "ent_left", "ent_right"}, set(metrics.keys()))
        self.assertTrue(all(torch.isfinite(torch.tensor(value)) for value in metrics.values()))

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
