import math
import unittest
from copy import deepcopy

import pytest
import torch
from torch import nn

from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig, NextObsPredWrapper
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig, NextObsPredMixin
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamples


def test_rodrigues_gradient_is_correct_at_zero_and_matches_matrix_exponential() -> None:
    for values in ([0.0, 0.0, 0.0], [1e-8, -2e-8, 3e-8], [0.3, -0.2, 0.5]):
        rotvec = torch.tensor(values, dtype=torch.float64, requires_grad=True)
        assert torch.autograd.gradcheck(NextObsPredMixin._rotvec_to_matrix, (rotvec,))
        x, y, z = rotvec.unbind()
        zero = torch.zeros_like(x)
        skew = torch.stack((zero, -z, y, z, zero, -x, -y, x, zero)).reshape(3, 3)
        torch.testing.assert_close(NextObsPredMixin._rotvec_to_matrix(rotvec), torch.matrix_exp(skew))
    predictor = nn.Linear(1, 3)
    nn.init.zeros_(predictor.weight)
    nn.init.zeros_(predictor.bias)
    target = NextObsPredMixin._rotvec_to_matrix(torch.tensor([[0.0, 0.0, 0.5]]))
    prediction = NextObsPredMixin._rotvec_to_matrix(predictor(torch.ones(1, 1)))
    (prediction - target).square().mean().backward()
    assert predictor.bias.grad[2].abs() > 0.0


class _IdentityTransition(nn.Module):
    def forward(
            self,
            z_t: torch.Tensor,
            a_t: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = a_t, agent_mask
        return z_t

    def predict_n_steps(
            self,
            z_0: torch.Tensor,
            action_seq: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = agent_mask
        return z_0[:, None].expand(-1, action_seq.shape[1], -1, -1)

    def get_hyper_parameters(self) -> dict[str, object]:
        return {}


class _NOPHarness(nn.Module, NextObsPredMixin):
    pass


class _PerItemMSELoss(nn.Module):
    def forward(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return (predictions - targets).square().mean(dim=-1)


class _LatentPolicy:
    def _evaluate_actions(
            self,
            batch: PPOWMSamples,
            action_splitter: object | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, float], torch.Tensor]:
        _ = action_splitter
        return (
            torch.zeros_like(batch.log_probs),
            torch.zeros_like(batch.values),
            {},
            {},
            torch.ones_like(batch.local_obs),
        )


def _make_harness(
        *,
        config: NextObsPredConfig,
        local_scalars_predictor: nn.Module | None = None,
        local_angles_predictor: nn.Module | None = None,
        local_rot6ds_predictor: nn.Module | None = None,
        local_binaries_predictor: nn.Module | None = None,
        global_scalars_predictor: nn.Module | None = None,
        global_rot6ds_predictor: nn.Module | None = None,
        scalar_loss_fn: nn.Module | None = None,
) -> _NOPHarness:
    harness = _NOPHarness()
    harness.setup_next_obs_pred(
        transition_model=_IdentityTransition(),
        config=config,
        scalar_loss_fn=nn.MSELoss(reduction="none") if scalar_loss_fn is None else scalar_loss_fn,
        local_scalars_predictor=local_scalars_predictor,
        local_angles_predictor=local_angles_predictor,
        local_rot6ds_predictor=local_rot6ds_predictor,
        local_binaries_predictor=local_binaries_predictor,
        global_scalars_predictor=global_scalars_predictor,
        global_rot6ds_predictor=global_rot6ds_predictor,
    )
    return harness


def _angle_observations(angles: torch.Tensor, defined: torch.Tensor) -> torch.Tensor:
    pairs = torch.stack((angles.sin(), angles.cos()), dim=-1)
    return pairs.masked_fill(~defined.unsqueeze(-1), 0.0).flatten(-2)


@pytest.mark.parametrize("compiled", [False, True])
def test_mixed_angle_counts_preserve_per_target_loss_and_learned_head_gradients(
    compiled: bool,
) -> None:
    predictor = nn.Linear(2, 3, dtype=torch.float64)
    with torch.no_grad():
        predictor.weight.copy_(torch.tensor([[0.2, -0.1], [0.3, 0.4], [0.6, -0.2]]))
        predictor.bias.copy_(torch.tensor([0.05, -0.2, 0.3]))
    reference_predictor = deepcopy(predictor)
    harness = _make_harness(
        config=NextObsPredConfig(local_angle_target_indices=[0, 2, 4]),
        local_angles_predictor=predictor,
    )
    features = torch.tensor([[[1., 0.], [0., 2.]], [[-1., 1.], [2., -1.]]], dtype=torch.float64, requires_grad=True)
    reference_features = features.detach().clone().requires_grad_()
    base_angles = torch.tensor([[[0.1, 0.4, 0.], [0.2, 0., 0.]],
                                [[0., 0., 0.], [0.3, -0.5, 0.2]]], dtype=torch.float64)
    target_angles = torch.tensor([[[0.8, -0.3, 0.], [1.1, 0., 0.]],
                                  [[0., 0., 0.], [-0.2, 0.7, -0.6]]], dtype=torch.float64)
    defined = torch.tensor([[[True, True, False], [True, False, False]],
                            [[False, False, False], [True, True, True]]])
    item_mask = torch.tensor([[True, True], [True, False]])
    reference_outputs = reference_predictor(reference_features)
    target_values = (0.7, -0.7, 0.9)
    errors = []
    for (batch, agent, angle), target in zip(((0, 0, 0), (0, 0, 1), (0, 1, 0)), target_values, strict=True):
        prediction = reference_outputs[batch, agent, angle]
        errors.append(1.0 - torch.cos(prediction - target))
    expected_loss = torch.stack(errors).mean()
    per_item_average = ((errors[0] + errors[1]) / 2 + errors[2]) / 2
    assert not torch.isclose(expected_loss, per_item_average)
    if compiled:
        torch.compiler.reset()
    try:
        loss_fn = harness.compute_angle_loss
        if compiled:
            loss_fn = torch.compile(loss_fn, backend="aot_eager", fullgraph=True)
        loss = loss_fn(
            features, _angle_observations(target_angles, defined), item_mask,
            _angle_observations(base_angles, defined),
        )
        torch.testing.assert_close(loss, expected_loss)
        loss.backward()
        expected_loss.backward()
        torch.testing.assert_close(features.grad, reference_features.grad)
        torch.testing.assert_close(features.grad[1], torch.zeros_like(features.grad[1]))
        for actual, expected in zip(predictor.parameters(), reference_predictor.parameters(), strict=True):
            assert torch.isfinite(actual.grad).all()
            torch.testing.assert_close(actual.grad, expected.grad)
        unused_head_start = 2
        unused_weight_grad = predictor.weight.grad[unused_head_start:]
        unused_bias_grad = predictor.bias.grad[unused_head_start:]
        torch.testing.assert_close(unused_weight_grad, torch.zeros_like(unused_weight_grad))
        torch.testing.assert_close(unused_bias_grad, torch.zeros_like(unused_bias_grad))
    finally:
        if compiled:
            torch.compiler.reset()


@pytest.mark.parametrize("compiled", [False, True])
def test_per_step_delta_resumes_scoring_after_reconnection(compiled: bool) -> None:
    harness = _make_harness(
        config=NextObsPredConfig(local_angle_target_indices=[0]),
        local_angles_predictor=nn.Identity(),
    )
    initial_angle = torch.tensor([[[0.3]]], dtype=torch.float64)
    initial_obs = _angle_observations(initial_angle, torch.ones_like(initial_angle, dtype=torch.bool))
    future_angles = torch.tensor([[[[0.7]], [[0.0]], [[1.1]], [[1.7]], [[2.4]]]], dtype=torch.float64)
    targets = _angle_observations(future_angles, torch.tensor([[[[True]], [[False]], [[True]], [[True]], [[True]]]]))
    prediction = torch.tensor([[[0.15]]], dtype=torch.float64, requires_grad=True)
    expected_errors = (0.15 - (0.7 - 0.3), 0.15 - (1.7 - 1.1))
    expected_loss = sum(1.0 - math.cos(error) for error in expected_errors) / 2
    expected_gradient = sum(math.sin(error) for error in expected_errors) / 2
    if compiled:
        torch.compiler.reset()
        harness._compute_next_obs_pred_loss_fn = torch.compile(
            harness._compute_next_obs_pred_loss_impl, backend="aot_eager", fullgraph=True,
        )
    try:
        loss, _ = harness.compute_next_obs_pred_loss(
            local_latents=prediction, local_obs=initial_obs,
            next_local_obs=targets, actions=torch.zeros(1, 5, 1, 1, dtype=torch.float64),
            agent_mask=torch.ones(1, 5, 1, dtype=torch.bool), time_mask=torch.tensor([[True, True, True, True, False]]),
        )
        torch.testing.assert_close(loss, prediction.new_tensor(expected_loss))
        loss.backward()
        torch.testing.assert_close(prediction.grad, torch.full_like(prediction, expected_gradient))
    finally:
        if compiled:
            torch.compiler.reset()


@pytest.mark.parametrize(("base_connected", "target_connected"), [(True, True), (False, False),
                                                                  (False, True), (True, False)])
def test_angle_loss_scores_only_defined_connection_targets(
    base_connected: bool, target_connected: bool,
) -> None:
    harness = _make_harness(
        config=NextObsPredConfig(local_angle_target_indices=[0, 2]),
        local_angles_predictor=nn.Identity(),
    )
    base_obs = torch.tensor([[[math.sin(0.2), math.cos(0.2),
                               math.sin(0.4) if base_connected else 0.0,
                               math.cos(0.4) if base_connected else 0.0]]])
    next_obs = torch.tensor([[[math.sin(0.7), math.cos(0.7),
                               math.sin(1.0) if target_connected else 0.0,
                               math.cos(1.0) if target_connected else 0.0]]])
    connector_valid = target_connected and base_connected
    predictions = torch.tensor([[[0.1, 0.3 if connector_valid else torch.nan]]], requires_grad=True)
    errors = [0.1 - (0.7 - 0.2)]
    if connector_valid:
        errors.append(0.3 - (1.0 - 0.4))
    hinge_width = 1
    loss = harness.compute_angle_loss(predictions, next_obs, None, base_obs)
    expected_loss = sum(1.0 - math.cos(error) for error in errors) / len(errors)
    torch.testing.assert_close(loss, torch.tensor(expected_loss))
    loss.backward()
    assert torch.isfinite(predictions.grad).all()
    assert predictions.grad[..., :hinge_width].abs().sum() > 0
    if not connector_valid:
        assert torch.count_nonzero(predictions.grad[..., hinge_width:]) == 0
        reference = _make_harness(
            config=NextObsPredConfig(local_angle_target_indices=[0]),
            local_angles_predictor=nn.Identity(),
        )
        hinge_predictions = predictions[..., :hinge_width].detach().clone().requires_grad_()
        reference_loss = reference.compute_angle_loss(hinge_predictions, next_obs[..., :2], None, base_obs[..., :2])
        reference_loss.backward()
        torch.testing.assert_close(predictions.grad[..., :hinge_width], hinge_predictions.grad)


def test_zero_angle_remains_a_defined_prediction_target() -> None:
    harness = _make_harness(
        config=NextObsPredConfig(local_angle_target_indices=[0]),
        local_angles_predictor=nn.Identity(),
    )
    predicted_angle = torch.tensor([[[0.2]]], requires_grad=True)
    predictions = predicted_angle
    observations = torch.tensor([[[0.0, 1.0]]])
    loss = harness.compute_angle_loss(predictions, observations, None, observations)
    torch.testing.assert_close(loss, torch.tensor(1.0 - math.cos(0.2)))
    loss.backward()
    torch.testing.assert_close(predicted_angle.grad, torch.full_like(predicted_angle, math.sin(0.2)))


def test_angle_loss_without_defined_targets_is_zero_with_finite_zero_gradients() -> None:
    harness = _make_harness(
        config=NextObsPredConfig(local_angle_target_indices=[0]),
        local_angles_predictor=nn.Identity(),
    )
    predictions = torch.full((2, 3, 2, 1), torch.nan, requires_grad=True)
    observations = torch.zeros(2, 3, 2, 2)
    loss = harness.compute_angle_loss(predictions, observations, torch.ones(2, 3, 2, dtype=torch.bool), observations)
    torch.testing.assert_close(loss, torch.tensor(0.0))
    loss.backward()
    torch.testing.assert_close(predictions.grad, torch.zeros_like(predictions))


def test_multistep_angle_loss_respects_connectivity_and_future_masks() -> None:
    harness = _make_harness(
        config=NextObsPredConfig(local_angle_target_indices=[0]),
        local_angles_predictor=nn.Identity(),
    )
    angles = torch.tensor([0.4, 0.7, 0.0, 1.1])
    targets = torch.stack((angles.sin(), angles.cos()), dim=-1).reshape(1, 4, 1, 2)
    targets[:, 2] = 0.0
    predictions = torch.tensor([[[0.15]]])
    loss, _ = harness.compute_next_obs_pred_loss(
        local_latents=predictions.requires_grad_(), local_obs=torch.zeros(1, 1, 2),
        next_local_obs=targets, actions=torch.zeros(1, 4, 1, 1),
        agent_mask=torch.ones(1, 4, 1, dtype=torch.bool),
        time_mask=torch.tensor([[True, True, True, False]]),
    )
    expected_loss = 1.0 - math.cos(0.15 - (0.7 - 0.4))
    torch.testing.assert_close(loss, torch.tensor(expected_loss))
    loss.backward()
    assert torch.isfinite(predictions.grad).all()


def test_compiled_angle_loss_excludes_undefined_targets() -> None:
    harness = _make_harness(
        config=NextObsPredConfig(local_angle_target_indices=[0, 2]),
        local_angles_predictor=nn.Identity(),
    )
    predictions = torch.tensor([[[0.1, torch.nan]]], requires_grad=True)
    next_obs = torch.tensor([[[math.sin(0.7), math.cos(0.7), 0.0, 0.0]]])
    base_obs = torch.tensor([[[math.sin(0.2), math.cos(0.2), 0.0, 0.0]]])
    torch.compiler.reset()
    try:
        loss_fn = torch.compile(harness.compute_angle_loss, backend="aot_eager", fullgraph=True)
        loss = loss_fn(predictions, next_obs, torch.ones(1, 1, dtype=torch.bool), base_obs)
        expected_loss = 1.0 - math.cos(0.1 - (0.7 - 0.2))
        torch.testing.assert_close(loss, torch.tensor(expected_loss))
        loss.backward()
        assert torch.isfinite(predictions.grad).all()
    finally:
        torch.compiler.reset()


@pytest.mark.parametrize("loss_fn", [_PerItemMSELoss(), nn.MSELoss()])
def test_global_scalar_activity_masks_explain_unreduced_custom_loss_requirement(loss_fn: nn.Module) -> None:
    harness = _make_harness(
        config=NextObsPredConfig(
            global_scalar_target_indices=[1, 3], global_scalar_active_mask_indices=[0, 2],
        ),
        global_scalars_predictor=nn.Identity(), scalar_loss_fn=loss_fn,
    )
    with pytest.raises(ValueError, match="activity masks require scalar_loss_fn to return unreduced per-target losses"):
        observations = torch.tensor([[1., 0., 0., 100.]])
        harness.compute_global_scalar_loss(torch.ones(1, 2), observations, None, observations)


def test_global_scalar_activity_masks_support_unreduced_custom_loss_and_ignore_inactive_gradients() -> None:
    harness = _make_harness(
        config=NextObsPredConfig(
            global_scalar_target_indices=[1, 3], global_scalar_active_mask_indices=[0, 2],
        ),
        global_scalars_predictor=nn.Identity(), scalar_loss_fn=nn.MSELoss(reduction="none"),
    )
    predictions = torch.tensor([[2., torch.nan], [torch.nan, torch.nan]], requires_grad=True)
    targets = torch.tensor([[1., 1., 0., torch.nan], [0., torch.nan, 0., torch.nan]])
    current_obs = torch.tensor([[1., 0., 0., 0.], [0., 0., 0., 0.]])
    loss = harness.compute_global_scalar_loss(predictions, targets, None, current_obs)
    torch.testing.assert_close(loss, torch.tensor(1.0))
    loss.backward()
    torch.testing.assert_close(predictions.grad, torch.tensor([[2., 0.], [0., 0.]]))


class NextObsPredGlobalTests(unittest.TestCase):
    def test_empty_global_target_lists_are_noop(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_scalar_target_indices=[0],
                global_scalar_target_indices=[],
                global_rot6d_target_indices=[],
            ),
            local_scalars_predictor=nn.Identity(),
        )
        local_latents = torch.tensor([[[1.0], [2.0]]])
        next_local_obs = torch.tensor([[[1.0], [2.0]]])
        actions = torch.zeros(1, 2, 1)

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_obs=torch.zeros_like(next_local_obs),
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            actions=actions,
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertFalse(harness.has_global_next_obs_pred_targets)
        self.assertEqual(set(metrics), {"scalar_loss", "scalar_loss_scaled"})

    def test_local_only_path_does_not_require_global_tensors(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_scalar_target_indices=[0],
            ),
            local_scalars_predictor=nn.Identity(),
        )
        local_latents = torch.tensor([[[1.0], [2.0]]])
        next_local_obs = torch.tensor([[[1.5], [1.0]]])
        actions = torch.zeros(1, 2, 1)

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_obs=torch.zeros_like(next_local_obs),
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            actions=actions,
        )

        self.assertAlmostEqual(loss.item(), 0.625)
        self.assertEqual(set(metrics), {"scalar_loss", "scalar_loss_scaled"})
        self.assertIsNone(harness.global_scalars_predictor)
        self.assertIsNone(harness.global_rot6ds_predictor)

    def test_local_scalar_multistep_delta_is_relative_to_previous_target_state(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_scalar_target_indices=[0],
            ),
            local_scalars_predictor=nn.Identity(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_latents=torch.tensor([[[1.0], [2.0]]]),
            local_obs=torch.tensor([[[10.0], [20.0]]]),
            next_local_obs=torch.tensor([[
                [[11.0], [22.0]],
                [[12.0], [24.0]],
                [[13.0], [26.0]],
            ]]),
            actions=torch.zeros(1, 3, 2, 1),
            time_mask=torch.ones(1, 3, dtype=torch.bool),
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertEqual(metrics["scalar_loss"], 0.0)

    def test_delta_targets_retain_unscored_predecessors(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_scalar_target_indices=[0], global_scalar_target_indices=[0],
            ),
            local_scalars_predictor=nn.Identity(), global_scalars_predictor=nn.Identity(),
        )
        origin = 10.0
        loss, metrics = harness.compute_next_obs_pred_loss(
            local_latents=torch.full((1, 2, 1), 4.0),
            local_obs=torch.tensor([[[origin], [origin + 10.0]]]),
            global_obs=torch.tensor([[origin]]),
            next_local_obs=torch.tensor([[[[13.0], [23.0]], [[17.0], [27.0]]]]),
            next_global_obs=torch.tensor([[[13.0], [17.0]]]),
            actions=torch.zeros(1, 2, 2, 1), time_mask=torch.tensor([[False, True]]),
        )
        self.assertEqual(loss.item(), 0.0)
        self.assertEqual(metrics["scalar_loss"], 0.0)
        self.assertEqual(metrics["global_scalar_loss"], 0.0)

    def test_local_loss_uses_next_state_agent_mask(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_scalar_target_indices=[0],
            ),
            local_scalars_predictor=nn.Identity(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_obs=torch.zeros(1, 2, 1),
            local_latents=torch.tensor([[[1.0], [100.0]]]),
            next_local_obs=torch.tensor([[[1.0], [999.0]]]),
            actions=torch.zeros(1, 2, 1),
            agent_mask=torch.tensor([[True, True]]),
            loss_agent_mask=torch.tensor([[True, False]]),
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertEqual(metrics["scalar_loss"], 0.0)

    def test_local_scalar_multistep_accepts_per_item_custom_loss_with_mask(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_scalar_target_indices=[0],
            ),
            local_scalars_predictor=nn.Identity(),
            scalar_loss_fn=_PerItemMSELoss(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_obs=torch.zeros(1, 2, 1),
            local_latents=torch.tensor([[[1.0], [100.0]]]),
            next_local_obs=torch.tensor([[
                [[0.0], [99.0]],
                [[1_000.0], [1_000.0]],
            ]]),
            actions=torch.zeros(1, 2, 2, 1),
            agent_mask=torch.ones(1, 2, 2, dtype=torch.bool),
            loss_agent_mask=torch.ones(1, 2, 2, dtype=torch.bool),
            time_mask=torch.tensor([[True, False]]),
        )

        self.assertAlmostEqual(loss.item(), 1.0)
        torch.testing.assert_close(metrics["scalar_loss"], torch.tensor(1.0))

    def test_local_angle_multistep_delta_wraps_and_uses_previous_target_state(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_angle_target_indices=[0],
            ),
            local_angles_predictor=nn.Identity(),
        )
        base_angle = torch.deg2rad(torch.tensor(170.0))
        target_angles = torch.deg2rad(torch.tensor([-170.0, -150.0]))

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_latents=torch.deg2rad(torch.tensor([[[20.0]]])),
            local_obs=torch.stack((base_angle.sin(), base_angle.cos())).reshape(1, 1, 2),
            next_local_obs=torch.stack((target_angles.sin(), target_angles.cos()), dim=-1).reshape(1, 2, 1, 2),
            actions=torch.zeros(1, 2, 1, 1),
            time_mask=torch.ones(1, 2, dtype=torch.bool),
        )

        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        torch.testing.assert_close(metrics["angle_loss"], torch.tensor(0.0), rtol=0.0, atol=5e-6)

    def test_binary_target_ema_ignores_batches_without_valid_targets(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_binary_target_indices=[0],
                binary_target_ema_decay=0.0,
            ),
            local_binaries_predictor=nn.Identity(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_latents=torch.zeros(1, 2, 1),
            next_local_obs=torch.zeros(1, 2, 1),
            actions=torch.zeros(1, 2, 1),
            loss_agent_mask=torch.zeros(1, 2, dtype=torch.bool),
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        torch.testing.assert_close(metrics["binary_loss"], torch.tensor(0.0))
        torch.testing.assert_close(harness.binary_target_ema, torch.tensor([0.5]))

    def test_local_rot6d_multistep_delta_uses_previous_target_orientation(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_rot6d_target_indices=[0],
            ),
            local_rot6ds_predictor=nn.Identity(),
        )

        def z_rotation_rot6d(degrees: torch.Tensor) -> torch.Tensor:
            radians = torch.deg2rad(degrees)
            zeros = torch.zeros_like(radians)
            return torch.stack(
                (radians.cos(), radians.sin(), zeros, -radians.sin(), radians.cos(), zeros),
                dim=-1,
            )

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_latents=torch.deg2rad(torch.tensor([[[0.0, 0.0, 20.0]]])),
            local_obs=z_rotation_rot6d(torch.tensor(170.0)).reshape(1, 1, 6),
            next_local_obs=z_rotation_rot6d(torch.tensor([-170.0, -150.0])).reshape(1, 2, 1, 6),
            actions=torch.zeros(1, 2, 1, 1),
            time_mask=torch.ones(1, 2, dtype=torch.bool),
        )

        self.assertAlmostEqual(loss.item(), 0.0, places=5)
        torch.testing.assert_close(metrics["rot6d_loss"], torch.tensor(0.0), rtol=0.0, atol=5e-6)

    def test_wrapper_masks_padded_horizons_and_applies_world_model_coefficient_once(self) -> None:
        policy = NextObsPredWrapper(
            policy=_LatentPolicy(),
            world_model_config=NOPWorldModelConfig(
                n_agents=1,
                local_latent_dim=1,
                action_dim=1,
                world_model_loss_coef=0.1,
                d_model_transition_model=1,
                nhead_transition_model=1,
                num_layers_transition_model=1,
                dim_feedforward_transition_model=2,
                scalar_loss_fn="mse",
                next_obs_pred_config=NextObsPredConfig(
                    local_scalar_target_indices=[0],
                ),
            ),
        )
        policy.transition_model = _IdentityTransition()
        policy.pre_transition_transform = nn.Identity()
        policy.pre_predictors_transform = nn.Identity()
        policy.local_scalars_predictor = nn.Identity()
        batch = PPOWMSamples(
            local_obs=torch.tensor([[[10.0]]]),
            global_obs=torch.zeros(1, 1),
            hidden_local_vars=torch.zeros(1, 1, 1),
            hidden_global_vars=torch.zeros(1, 1),
            agent_mask=torch.ones(1, 1, dtype=torch.bool),
            previous_actions=None,
            actions=torch.zeros(1, 1, 1),
            log_probs=torch.zeros(1, 1),
            values=torch.zeros(1),
            returns=torch.zeros(1),
            advantages=torch.zeros(1),
            wm_actions=torch.zeros(1, 3, 1, 1),
            next_local_obs=torch.tensor([[[[12.0]], [[14.0]], [[999.0]]]]),
            wm_target_time_mask=torch.tensor([[True, True, False]]),
            next_global_obs=torch.zeros(1, 3, 1),
            wm_agent_mask=torch.ones(1, 3, 1, dtype=torch.bool),
            wm_loss_agent_mask=torch.ones(1, 3, 1, dtype=torch.bool),
        )

        _log_probs, _values, extra_losses, metrics = policy.evaluate_actions(batch)

        torch.testing.assert_close(metrics["wm_loss"], torch.tensor(1.0))
        torch.testing.assert_close(metrics["wm_loss_scaled"], torch.tensor(0.1))
        self.assertAlmostEqual(extra_losses["world_model"].item(), 0.1)

    def test_global_scalar_loss_uses_masked_agent_pool_and_does_not_require_local_obs(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0, 1],
            ),
            global_scalars_predictor=nn.Identity(),
        )
        local_latents = torch.tensor([[[1.0, 2.0], [100.0, 200.0], [3.0, 4.0]]])
        actions = torch.zeros(1, 3, 1)
        agent_mask = torch.tensor([[True, False, True]])
        global_obs = torch.tensor([[10.0, 20.0]])
        next_global_obs = torch.tensor([[12.0, 23.0]])
        next_local_obs = torch.zeros(1, 3, 1)

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            next_global_obs=next_global_obs,
            global_obs=global_obs,
            actions=actions,
            agent_mask=agent_mask,
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertEqual(metrics["global_scalar_loss"], 0.0)

    def test_global_scalar_loss_accepts_fully_reduced_custom_loss(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0, 1],
            ),
            global_scalars_predictor=nn.Identity(),
            scalar_loss_fn=nn.MSELoss(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
            global_obs=torch.zeros(2, 2),
            local_latents=torch.tensor([
                [[1.0, 3.0], [1.0, 3.0]],
                [[2.0, 4.0], [2.0, 4.0]],
            ]),
            next_local_obs=torch.zeros(2, 2, 1),
            next_global_obs=torch.tensor([[0.0, 1.0], [2.0, 6.0]]),
            actions=torch.zeros(2, 2, 1),
        )

        self.assertAlmostEqual(loss.item(), 2.25)
        torch.testing.assert_close(metrics["global_scalar_loss"], torch.tensor(2.25))

    def test_global_scalar_loss_accepts_per_item_custom_loss_with_mask(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0, 1],
            ),
            global_scalars_predictor=nn.Identity(),
            scalar_loss_fn=_PerItemMSELoss(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
            global_obs=torch.zeros(2, 2),
            local_latents=torch.tensor([
                [[100.0, 100.0], [100.0, 100.0]],
                [[2.0, 4.0], [2.0, 4.0]],
            ]),
            next_local_obs=torch.zeros(2, 2, 1),
            next_global_obs=torch.tensor([[0.0, 0.0], [1.0, 3.0]]),
            actions=torch.zeros(2, 2, 1),
            agent_mask=torch.ones(2, 2, dtype=torch.bool),
            loss_agent_mask=torch.tensor([[False, False], [True, True]]),
        )

        self.assertAlmostEqual(loss.item(), 1.0)
        torch.testing.assert_close(metrics["global_scalar_loss"], torch.tensor(1.0))

    def test_global_pool_uses_next_state_loss_mask(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0],
            ),
            global_scalars_predictor=nn.Identity(),
        )
        local_latents = torch.tensor([[[1.0], [100.0], [3.0]]])
        next_global_obs = torch.tensor([[2.0]])

        loss, metrics = harness.compute_next_obs_pred_loss(
            global_obs=torch.zeros(1, 1),
            local_latents=local_latents,
            next_local_obs=torch.zeros(1, 3, 1),
            next_global_obs=next_global_obs,
            actions=torch.zeros(1, 3, 1),
            agent_mask=torch.tensor([[True, True, True]]),
            loss_agent_mask=torch.tensor([[True, False, True]]),
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertEqual(metrics["global_scalar_loss"], 0.0)

    def test_global_loss_ignores_rows_without_next_active_agents(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0],
            ),
            global_scalars_predictor=nn.Identity(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
            global_obs=torch.zeros(2, 1),
            local_latents=torch.tensor([[[100.0], [200.0]], [[1.0], [3.0]]]),
            next_local_obs=torch.zeros(2, 2, 1),
            next_global_obs=torch.tensor([[999.0], [2.0]]),
            actions=torch.zeros(2, 2, 1),
            agent_mask=torch.ones(2, 2, dtype=torch.bool),
            loss_agent_mask=torch.tensor([[False, False], [True, True]]),
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertEqual(metrics["global_scalar_loss"], 0.0)

    def test_global_delta_targets_require_current_and_successor_observations(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0],
            ),
            global_scalars_predictor=nn.Identity(),
        )

        with self.assertRaisesRegex(ValueError, "global_obs and next_global_obs are required"):
            harness.compute_next_obs_pred_loss(
                local_latents=torch.ones(1, 2, 1),
                next_local_obs=torch.zeros(1, 2, 1),
                actions=torch.zeros(1, 2, 1),
            )

    def test_global_scalar_multistep_delta_uses_previous_predicted_target_as_base(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0, 1],
            ),
            global_scalars_predictor=nn.Identity(),
        )
        local_latents = torch.tensor([[[1.0, 2.0], [1.0, 2.0]]])
        actions = torch.zeros(1, 2, 2, 1)
        global_obs = torch.tensor([[10.0, 20.0]])
        next_global_obs = torch.tensor([[[11.0, 22.0], [99.0, 99.0]]])
        next_local_obs = torch.zeros(1, 2, 2, 1)
        time_mask = torch.tensor([[True, False]])

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            next_global_obs=next_global_obs,
            global_obs=global_obs,
            actions=actions,
            time_mask=time_mask,
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertEqual(metrics["global_scalar_loss"], 0.0)

    def test_global_scalar_multistep_loss_ignores_masked_future_steps(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0],
            ),
            global_scalars_predictor=nn.Identity(),
        )
        local_latents = torch.tensor([[[1.0], [1.0]]])
        actions = torch.zeros(1, 2, 2, 1)
        next_global_obs = torch.tensor([[[1.0], [1000.0]]])
        next_local_obs = torch.zeros(1, 2, 2, 1)
        time_mask = torch.tensor([[True, False]])

        loss, metrics = harness.compute_next_obs_pred_loss(
            global_obs=torch.zeros(1, 1),
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            next_global_obs=next_global_obs,
            actions=actions,
            time_mask=time_mask,
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertEqual(metrics["global_scalar_loss"], 0.0)

    def test_global_rot6d_loss_predicts_from_pooled_agent_latents(self) -> None:
        identity_rot6d = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
        harness = _make_harness(
            config=NextObsPredConfig(
                global_rot6d_target_indices=[0],
            ),
            global_rot6ds_predictor=nn.Identity(),
        )
        local_latents = torch.tensor([[[0.0, 0.0, 0.2], [0.0, 0.0, 0.6]]])
        actions = torch.zeros(1, 2, 1)
        next_global_obs = torch.tensor([[math.cos(0.4), math.sin(0.4), 0.0, -math.sin(0.4), math.cos(0.4), 0.0]])
        next_local_obs = torch.zeros(1, 2, 1)

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            global_obs=identity_rot6d.view(1, 6),
            next_global_obs=next_global_obs,
            actions=actions,
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        torch.testing.assert_close(metrics["global_rot6d_loss"], torch.tensor(0.0))

    def test_wrapper_without_global_targets_does_not_allocate_global_modules(self) -> None:
        policy = NextObsPredWrapper(
            policy=object(),
            world_model_config=NOPWorldModelConfig(
                n_agents=2,
                local_latent_dim=4,
                action_dim=1,
                d_model_transition_model=4,
                nhead_transition_model=1,
                num_layers_transition_model=1,
                dim_feedforward_transition_model=8,
                next_obs_pred_config=NextObsPredConfig(
                    local_scalar_target_indices=[0],
                ),
            ),
        )

        self.assertIsNone(policy.global_pool_encoder)
        self.assertIsNone(policy.global_scalars_predictor)
        self.assertIsNone(policy.global_rot6ds_predictor)

    def test_wrapper_with_global_targets_allocates_global_modules(self) -> None:
        policy = NextObsPredWrapper(
            policy=object(),
            world_model_config=NOPWorldModelConfig(
                n_agents=2,
                local_latent_dim=4,
                action_dim=1,
                d_model_transition_model=4,
                nhead_transition_model=1,
                num_layers_transition_model=1,
                dim_feedforward_transition_model=8,
                wm_global_pool_hidden_dims=[5],
                next_obs_pred_config=NextObsPredConfig(
                    global_scalar_target_indices=[0],
                    global_rot6d_target_indices=[1],
                ),
            ),
        )

        self.assertIsNotNone(policy.global_pool_encoder)
        self.assertIsNotNone(policy.global_scalars_predictor)
        self.assertIsNotNone(policy.global_rot6ds_predictor)


if __name__ == "__main__":
    unittest.main()
