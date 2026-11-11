import math
import unittest
from copy import deepcopy
from unittest.mock import patch

import pytest
import torch

from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist
from swarmbots.learn.action_dists.gsde_noise import reset_noise_masked_


CUDA = pytest.param("cuda", marks=[
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("mask_layout", ["agent", "environment", "empty", "strided"])
@pytest.mark.parametrize(("latent_dim", "action_dim"), [(64, 8), (7, 3)])
def test_cuda_masked_noise_resets_preserve_other_agents_without_host_reads(
    mask_layout: str, latent_dim: int, action_dim: int,
) -> None:
    pytest.importorskip("triton")
    dist = GSDEActionDist(latent_dim=latent_dim, action_dim=action_dim, base_std=.5).cuda()
    dist.reset_noise((128, 5))
    if mask_layout == "strided":
        dist.set_temporal_correlation_state(torch.randn(128, 5, latent_dim, action_dim * 2, device="cuda")[..., ::2])
    mask = torch.arange(128 * 5, device="cuda").reshape(128, 5) % 6 == 0
    if mask_layout == "environment":
        mask = mask[:, 0]
    elif mask_layout == "empty":
        mask.zero_()
    with torch.no_grad():
        dist.reset_noise_masked(mask)
    original = dist.get_temporal_correlation_state().clone()
    with (
        torch.no_grad(),
        patch.object(torch.Tensor, "item", side_effect=AssertionError("resets must stay on device")),
        patch.object(torch.Tensor, "cpu", side_effect=AssertionError("resets must stay on device")),
        patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("resets must stay on device")),
    ):
        dist.reset_noise_masked(mask)
    updated = dist.get_temporal_correlation_state()
    expanded_mask = mask[:, None].expand(128, 5) if mask.ndim == 1 else mask
    torch.testing.assert_close(updated[~expanded_mask], original[~expanded_mask], rtol=0, atol=0)
    if mask_layout != "empty":
        assert not torch.equal(updated[expanded_mask], original[expanded_mask])


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_masked_noise_is_seeded_independent_standard_normal() -> None:
    dist = GSDEActionDist(latent_dim=64, action_dim=8, base_std=.5).cuda()
    dist.reset_noise((256, 5))
    mask = torch.ones(256, 5, device="cuda", dtype=torch.bool)
    with torch.no_grad():
        torch.manual_seed(1234)
        dist.reset_noise_masked(mask)
        first = dist.get_temporal_correlation_state().clone()
        torch.manual_seed(1234)
        dist.reset_noise_masked(mask)
    torch.testing.assert_close(dist.get_temporal_correlation_state(), first, rtol=0, atol=0)
    assert first.mean().abs() < .01
    assert (first.std() - 1).abs() < .01
    assert (first[:, 0] * first[:, 1]).mean().abs() < .01


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_masked_noise_follows_the_current_stream() -> None:
    dist = GSDEActionDist(latent_dim=7, action_dim=3, base_std=.5).cuda()
    dist.reset_noise((128, 5))
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), torch.no_grad():
        dist.set_temporal_correlation_state(torch.zeros(128, 5, 7, 3, device="cuda"))
        mask = torch.arange(128 * 5, device="cuda").reshape(128, 5) % 6 == 0
        dist.reset_noise_masked(mask)
        snapshot = dist.get_temporal_correlation_state().clone()
    stream.synchronize()
    assert torch.count_nonzero(snapshot[~mask]) == 0
    assert torch.count_nonzero(snapshot[mask]) > 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_cuda_masked_noise_retains_autograd_mutation_detection() -> None:
    dist = GSDEActionDist(latent_dim=7, action_dim=3, base_std=.5).cuda()
    dist.reset_noise((2, 5))
    features = torch.ones_like(dist.get_temporal_correlation_state(), requires_grad=True)
    loss = (features * dist.get_temporal_correlation_state()).sum()
    with torch.no_grad():
        dist.reset_noise_masked(torch.ones(2, 5, dtype=torch.bool, device="cuda"))
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        loss.backward()


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("reset_kind", ["empty", "partial", "full", "restored"])
@pytest.mark.parametrize("reset_context", ["grad", "no_grad", "inference"])
@pytest.mark.parametrize("std_learnable", [False, True])
def test_rsample_gradients_survive_noise_resets(
    device: str, reset_kind: str, reset_context: str, std_learnable: bool,
) -> None:
    dist = GSDEActionDist(4, 2, .5, std_learnable=std_learnable).to(device)
    dist.reset_noise((2, 3))
    reference = deepcopy(dist)
    features = torch.randn(2, 3, 4, device=device, requires_grad=True)
    reference_features = features.detach().clone().requires_grad_()
    loss = dist.update_latent_features(features).rsample().square().sum()
    reference_loss = reference.update_latent_features(reference_features).rsample().square().sum()
    if reset_kind == "restored":
        state = dist.get_temporal_correlation_state()
        dist.reset_noise((2, 3))
        dist.set_temporal_correlation_state(state.view_as(state))
    mask = torch.arange(6, device=device).reshape(2, 3) % 2 == 0
    if reset_kind == "empty":
        mask.zero_()
    elif reset_kind == "full":
        mask.fill_(True)
    with torch.inference_mode(reset_context == "inference"), torch.set_grad_enabled(reset_context == "grad"):
        dist.reset_noise_masked(mask)
    loss.backward()
    reference_loss.backward()
    torch.testing.assert_close(features.grad, reference_features.grad)
    for parameter, expected_parameter in zip(dist.parameters(), reference.parameters(), strict=True):
        torch.testing.assert_close(parameter.grad, expected_parameter.grad)
    dist.zero_grad()
    dist.update_latent_features(features.detach()).rsample().sum().backward()
    assert dist.action_net.weight.grad is not None


@pytest.mark.parametrize("device", ["cpu", CUDA])
def test_multiple_pending_rsample_generations_survive_later_resets(device: str) -> None:
    dist = GSDEActionDist(4, 2, .5).to(device)
    dist.reset_noise((2, 3))
    losses, expected_losses, references = [], [], []
    for index in range(3):
        reference = GSDEActionDist(4, 2, .5).to(device)
        reference.load_state_dict(dist.state_dict())
        reference.set_temporal_correlation_state(dist.get_temporal_correlation_state().clone())
        features = torch.randn(2, 1, 4, device=device)
        losses.append(dist.update_latent_features(features).rsample(agent=1).square().sum())
        expected_losses.append(reference.update_latent_features(features).rsample(agent=1).square().sum())
        references.append(reference)
        with torch.no_grad():
            dist.reset_noise_masked(torch.tensor([index % 2 == 0, True], device=device))
    sum(losses).backward()
    sum(expected_losses).backward()
    for name, parameter in dist.named_parameters():
        expected = sum(dict(reference.named_parameters())[name].grad for reference in references)
        torch.testing.assert_close(parameter.grad, expected)


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("reset_grad_enabled", [False, True])
def test_inference_noise_cannot_be_mutated_outside_inference_mode(
    device: str, reset_grad_enabled: bool,
) -> None:
    dist = GSDEActionDist(4, 2, .5).to(device)
    with torch.inference_mode():
        dist.reset_noise((2, 3))
    original = dist.get_temporal_correlation_state().clone()
    mask = torch.zeros(2, 3, device=device, dtype=torch.bool)
    with torch.set_grad_enabled(reset_grad_enabled):
        dist.reset_noise_masked(mask)
        torch.testing.assert_close(dist.get_temporal_correlation_state(), original, rtol=0, atol=0)
        mask[0, 1] = True
        with pytest.raises(RuntimeError, match="Inplace update to inference tensor outside InferenceMode"):
            dist.reset_noise_masked(mask)
    torch.testing.assert_close(dist.get_temporal_correlation_state(), original, rtol=0, atol=0)
    with torch.inference_mode():
        dist.reset_noise_masked(mask)
    updated = dist.get_temporal_correlation_state()
    torch.testing.assert_close(updated[~mask], original[~mask], rtol=0, atol=0)
    assert not torch.equal(updated[mask], original[mask])


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("batch_shape", [(), (4,), (2, 3), (2, 3, 4)])
@pytest.mark.parametrize("force_fallback", [False, True])
@pytest.mark.parametrize("contiguous", [False, True])
def test_selective_noise_resets_preserve_unselected_storage(
    device: str, batch_shape: tuple[int, ...], force_fallback: bool, contiguous: bool,
) -> None:
    if contiguous:
        backing = torch.zeros(*batch_shape, 5, 3, device=device)
        noise = backing
    else:
        backing = torch.zeros(*reversed(batch_shape), 6, 5, device=device)
        noise = backing.permute(*reversed(range(len(batch_shape))), len(batch_shape) + 1, len(batch_shape))[..., ::2]
    mask = torch.arange(math.prod(batch_shape), device=device).reshape(batch_shape) % 2 == 0
    with torch.no_grad():
        if force_fallback:
            with patch("swarmbots.learn.action_dists.gsde_noise._masked_normal_kernel", return_value=None):
                reset_noise_masked_(noise, mask)
        else:
            reset_noise_masked_(noise, mask)
    assert torch.count_nonzero(noise[~mask]) == 0
    assert torch.count_nonzero(noise[mask]) == int(mask.sum()) * 15
    if not contiguous:
        assert torch.count_nonzero(backing[..., 1::2, :]) == 0


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("selected_rows", [0, 10, 5120 // 6])
def test_cuda_fallback_temporary_storage_scales_with_selected_rows(selected_rows: int) -> None:
    dist = GSDEActionDist(256, 8, .5).cuda()
    dist.reset_noise((1024, 5))
    noise = dist.get_temporal_correlation_state()
    mask = torch.arange(5120, device="cuda").reshape(1024, 5) < selected_rows
    with torch.no_grad(), patch("swarmbots.learn.action_dists.gsde_noise._masked_normal_kernel", return_value=None):
        dist.reset_noise_masked(mask)
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        dist.reset_noise_masked(mask)
        torch.cuda.synchronize()
        extra = torch.cuda.max_memory_allocated() - baseline
    # Allow allocator rounding and the mask reduction, but exclude full-batch temporaries.
    assert extra <= selected_rows * 256 * 8 * noise.element_size() + 65536
    assert dist.get_temporal_correlation_state() is noise


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
