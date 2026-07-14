import unittest

import torch
from torch import nn

from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig, NextObsPredWrapper
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig, NextObsPredMixin


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


def _make_harness(
        *,
        config: NextObsPredConfig,
        local_scalars_predictor: nn.Module | None = None,
        global_scalars_predictor: nn.Module | None = None,
        global_rot6ds_predictor: nn.Module | None = None,
) -> _NOPHarness:
    harness = _NOPHarness()
    harness.setup_next_obs_pred(
        transition_model=_IdentityTransition(),
        config=config,
        scalar_loss_fn=nn.MSELoss(reduction="none"),
        local_scalars_predictor=local_scalars_predictor,
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
