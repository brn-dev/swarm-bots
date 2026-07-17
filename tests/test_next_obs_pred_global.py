import unittest

import torch
from torch import nn

from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig, NextObsPredWrapper
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig, NextObsPredMixin
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamples


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


class NextObsPredGlobalTests(unittest.TestCase):
    def test_empty_global_target_lists_are_noop(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_scalar_target_indices=[0],
                global_scalar_target_indices=[],
                global_rot6d_target_indices=[],
                predict_delta=False,
            ),
            local_scalars_predictor=nn.Identity(),
        )
        local_latents = torch.tensor([[[1.0], [2.0]]])
        next_local_obs = torch.tensor([[[1.0], [2.0]]])
        actions = torch.zeros(1, 2, 1)

        loss, metrics = harness.compute_next_obs_pred_loss(
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
                predict_delta=False,
            ),
            local_scalars_predictor=nn.Identity(),
        )
        local_latents = torch.tensor([[[1.0], [2.0]]])
        next_local_obs = torch.tensor([[[1.5], [1.0]]])
        actions = torch.zeros(1, 2, 1)

        loss, metrics = harness.compute_next_obs_pred_loss(
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
                predict_delta=True,
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

    def test_local_loss_uses_next_state_agent_mask(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_scalar_target_indices=[0],
                predict_delta=False,
            ),
            local_scalars_predictor=nn.Identity(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
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
                predict_delta=False,
            ),
            local_scalars_predictor=nn.Identity(),
            scalar_loss_fn=_PerItemMSELoss(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
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
        self.assertAlmostEqual(metrics["scalar_loss"], 1.0)

    def test_local_angle_multistep_delta_wraps_and_uses_previous_target_state(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_angle_target_indices=[0],
                predict_delta=True,
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
        self.assertAlmostEqual(metrics["angle_loss"], 0.0, places=5)

    def test_binary_target_ema_ignores_batches_without_valid_targets(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_binary_target_indices=[0],
                binary_target_ema_decay=0.0,
                predict_delta=False,
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
        self.assertAlmostEqual(metrics["binary_loss"], 0.0)
        torch.testing.assert_close(harness.binary_target_ema, torch.tensor([0.5]))

    def test_local_rot6d_multistep_delta_uses_previous_target_orientation(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                local_rot6d_target_indices=[0],
                predict_delta=True,
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
        self.assertAlmostEqual(metrics["rot6d_loss"], 0.0, places=5)

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
                    predict_delta=True,
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

        self.assertAlmostEqual(metrics["wm_loss"], 1.0)
        self.assertAlmostEqual(metrics["wm_loss_scaled"], 0.1)
        self.assertAlmostEqual(extra_losses["world_model"].item(), 0.1)

    def test_global_scalar_loss_uses_masked_agent_pool_and_does_not_require_local_obs(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0, 1],
                predict_delta=True,
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
                predict_delta=False,
            ),
            global_scalars_predictor=nn.Identity(),
            scalar_loss_fn=nn.MSELoss(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_latents=torch.tensor([
                [[1.0, 3.0], [1.0, 3.0]],
                [[2.0, 4.0], [2.0, 4.0]],
            ]),
            next_local_obs=torch.zeros(2, 2, 1),
            next_global_obs=torch.tensor([[0.0, 1.0], [2.0, 6.0]]),
            actions=torch.zeros(2, 2, 1),
        )

        self.assertAlmostEqual(loss.item(), 2.25)
        self.assertAlmostEqual(metrics["global_scalar_loss"], 2.25)

    def test_global_scalar_loss_accepts_per_item_custom_loss_with_mask(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0, 1],
                predict_delta=False,
            ),
            global_scalars_predictor=nn.Identity(),
            scalar_loss_fn=_PerItemMSELoss(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
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
        self.assertAlmostEqual(metrics["global_scalar_loss"], 1.0)

    def test_global_pool_uses_next_state_loss_mask(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0],
                predict_delta=False,
            ),
            global_scalars_predictor=nn.Identity(),
        )
        local_latents = torch.tensor([[[1.0], [100.0], [3.0]]])
        next_global_obs = torch.tensor([[2.0]])

        loss, metrics = harness.compute_next_obs_pred_loss(
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
                predict_delta=False,
            ),
            global_scalars_predictor=nn.Identity(),
        )

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_latents=torch.tensor([[[100.0], [200.0]], [[1.0], [3.0]]]),
            next_local_obs=torch.zeros(2, 2, 1),
            next_global_obs=torch.tensor([[999.0], [2.0]]),
            actions=torch.zeros(2, 2, 1),
            agent_mask=torch.ones(2, 2, dtype=torch.bool),
            loss_agent_mask=torch.tensor([[False, False], [True, True]]),
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertEqual(metrics["global_scalar_loss"], 0.0)

    def test_global_targets_require_next_global_obs_only_when_configured(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0],
                predict_delta=False,
            ),
            global_scalars_predictor=nn.Identity(),
        )

        with self.assertRaisesRegex(ValueError, "next_global_obs is required"):
            harness.compute_next_obs_pred_loss(
                local_latents=torch.ones(1, 2, 1),
                next_local_obs=torch.zeros(1, 2, 1),
                actions=torch.zeros(1, 2, 1),
            )

    def test_global_scalar_multistep_delta_uses_previous_predicted_target_as_base(self) -> None:
        harness = _make_harness(
            config=NextObsPredConfig(
                global_scalar_target_indices=[0, 1],
                predict_delta=True,
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
                predict_delta=False,
            ),
            global_scalars_predictor=nn.Identity(),
        )
        local_latents = torch.tensor([[[1.0], [1.0]]])
        actions = torch.zeros(1, 2, 2, 1)
        next_global_obs = torch.tensor([[[1.0], [1000.0]]])
        next_local_obs = torch.zeros(1, 2, 2, 1)
        time_mask = torch.tensor([[True, False]])

        loss, metrics = harness.compute_next_obs_pred_loss(
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
                predict_delta=False,
            ),
            global_rot6ds_predictor=nn.Identity(),
        )
        local_latents = identity_rot6d.view(1, 1, 6).expand(1, 2, 6).clone()
        actions = torch.zeros(1, 2, 1)
        next_global_obs = identity_rot6d.view(1, 6)
        next_local_obs = torch.zeros(1, 2, 1)

        loss, metrics = harness.compute_next_obs_pred_loss(
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            next_global_obs=next_global_obs,
            actions=actions,
        )

        self.assertAlmostEqual(loss.item(), 0.0)
        self.assertAlmostEqual(metrics["global_rot6d_loss"], 0.0)

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
                    predict_delta=False,
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
                    predict_delta=False,
                ),
            ),
        )

        self.assertIsNotNone(policy.global_pool_encoder)
        self.assertIsNotNone(policy.global_scalars_predictor)
        self.assertIsNotNone(policy.global_rot6ds_predictor)


if __name__ == "__main__":
    unittest.main()
