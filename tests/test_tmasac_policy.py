import math
import unittest
from collections.abc import Callable
from dataclasses import replace
from typing import Any
from unittest.mock import Mock, patch

import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.beta_action_dist import BetaConfig
from swarmbots.learn.action_dists.beta_mixture_action_dist import BetaMixtureConfig
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaConfig,
    GumbelSoftmaxSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.action_dists.hybrid_action_dist import ContinuousActionDistConfig
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdConfig
from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyConfig,
)
from swarmbots.learn.action_dists.reparameterized_squashed_gaussian_mixture_action_dist import (
    ReparameterizedSquashedGaussianMixtureConfig,
)
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoderConfig
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch, OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.algos.sac import (
    SACNOPConfig,
    SACNOPLatentSource,
    ScenarioFieldEncoderConfig,
    ScenarioObservationSpec,
    TMASACActorHeadConfig,
    TMASACActorHeadKind,
    TMASACCriticConfig,
    TMASACPolicy,
    TMASACPolicyConfig,
    TMASACScenarioEncoderConfig,
)
from swarmbots.learn.algos.sac.scenario_obs_encoder import TMASACScenarioObservationEncoder
from swarmbots.learn.algos.sac.tmasac_actor_heads import TMASACQCXActorHead
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.nn_components.feed_forward import GLUStackConfig, SwiGLUConfig, feedforward_linear_layers


_REAL_TORCH_COMPILE = torch.compile


class _DummyContinuousEnv:
    n_agents = 3
    local_obs_dim = 5
    global_obs_dim = 2
    hidden_local_vars_dim = 1
    hidden_global_vars_dim = 2
    action_space = HybridActionSpace(
        {
            "actuators": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 2), dtype=float),
            "connectors": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 1), dtype=float),
        }
    )


class _DummyDiscreteEnv(_DummyContinuousEnv):
    action_space = HybridActionSpace(
        {
            "actuators": spaces.Box(low=-1.0, high=1.0, shape=(_DummyContinuousEnv.n_agents, 2), dtype=float),
            "connectors": spaces.MultiBinary((_DummyContinuousEnv.n_agents, 1)),
        }
    )


class _DummyScenarioEnv(_DummyContinuousEnv):
    global_obs_dim = 4
    hidden_local_vars_dim = 3
    hidden_global_vars_dim = 3
    has_scenario_id = True


def _scenario_encoder_config() -> TMASACScenarioEncoderConfig:
    return TMASACScenarioEncoderConfig(
        scenarios=(
            ScenarioObservationSpec(
                scenario_id=0,
                name="wall",
                global_obs_dim=0,
                hidden_local_vars_dim=3,
                hidden_global_vars_dim=1,
            ),
            ScenarioObservationSpec(
                scenario_id=1,
                name="payload",
                global_obs_dim=4,
                hidden_local_vars_dim=0,
                hidden_global_vars_dim=3,
            ),
        ),
        global_obs=ScenarioFieldEncoderConfig(output_dim=6),
        hidden_local_vars=ScenarioFieldEncoderConfig(output_dim=5, hidden_dims=(7,)),
        hidden_global_vars=ScenarioFieldEncoderConfig(output_dim=4),
        scenario_embedding_dim=3,
    )


def _small_encoder_config() -> MATEncoderConfig:
    return MATEncoderConfig(
        d_model=12,
        nhead=3,
        num_layers=1,
        dim_feedforward=24,
    )


def _make_config(
        nop_config: SACNOPConfig | None = None,
        *,
        share_observation_encoder: bool = False,
        shared_encoder_config: MATEncoderConfig | None = None,
        continuous_config: ContinuousActionDistConfig | None = None,
        dropout: float = 0.0,
        independent_critic_encoders: bool = False,
        separate_observation_action_encoders: bool = False,
        action_encoder_dim: int | None = None,
        actor_head_kind: TMASACActorHeadKind | str = TMASACActorHeadKind.INDEPENDENT,
) -> TMASACPolicyConfig:
    return TMASACPolicyConfig(
        actor_encoder_config=_small_encoder_config(),
        critic_encoder_config=_small_encoder_config(),
        shared_encoder_config=shared_encoder_config,
        share_observation_encoder=share_observation_encoder,
        actor_head_config=TMASACActorHeadConfig(
            kind=actor_head_kind,
            hidden_dims=[10],
            qcx_decoder_config=MATQCXDecoderConfig(
                d_model=12,
                nhead=3,
                num_layers=1,
                dim_feedforward=24,
                assume_agent_mask_is_active_prefix=False,
            ),
        ),
        critic_config=TMASACCriticConfig(
            n_local_projection_hidden_layers=1,
            n_value_regressor_hidden_layers=1,
            independent_encoders=independent_critic_encoders,
            separate_observation_action_encoders=separate_observation_action_encoders,
            action_encoder_dim=action_encoder_dim,
        ),
        dropout=dropout,
        continuous_config=PredictedStdConfig(base_std=0.5) if continuous_config is None else continuous_config,
        nop_config=SACNOPConfig() if nop_config is None else nop_config,
    )


def _make_batch(batch_size: int = 4) -> OffPolicyReplayBatch:
    env = _DummyContinuousEnv()
    action_dim = env.action_space.total_agent_action_dim
    agent_mask = torch.tensor(
        [
            [True, True, True],
            [True, True, False],
            [True, False, False],
            [True, True, True],
        ],
        dtype=torch.bool,
    )[:batch_size]
    next_agent_mask = torch.tensor(
        [
            [True, True, True],
            [True, False, False],
            [True, False, False],
            [True, True, False],
        ],
        dtype=torch.bool,
    )[:batch_size]
    return OffPolicyReplayBatch(
        local_obs=torch.randn(batch_size, env.n_agents, env.local_obs_dim),
        global_obs=torch.randn(batch_size, env.global_obs_dim),
        hidden_local_vars=torch.randn(batch_size, env.n_agents, env.hidden_local_vars_dim),
        hidden_global_vars=torch.randn(batch_size, env.hidden_global_vars_dim),
        agent_mask=agent_mask,
        actions=torch.randn(batch_size, env.n_agents, action_dim).clamp(-0.9, 0.9),
        rewards=torch.randn(batch_size),
        terminations=torch.zeros(batch_size, dtype=torch.bool),
        truncations=torch.zeros(batch_size, dtype=torch.bool),
        previous_actions=None,
        next_local_obs=torch.randn(batch_size, env.n_agents, env.local_obs_dim),
        next_global_obs=torch.randn(batch_size, env.global_obs_dim),
        next_hidden_local_vars=torch.randn(batch_size, env.n_agents, env.hidden_local_vars_dim),
        next_hidden_global_vars=torch.randn(batch_size, env.hidden_global_vars_dim),
        next_agent_mask=next_agent_mask,
    )


def _make_segment_batch(batch_size: int = 3, num_next_steps: int = 4) -> OffPolicyReplayEpisodeSegmentBatch:
    env = _DummyContinuousEnv()
    action_dim = env.action_space.total_agent_action_dim
    agent_mask = torch.ones(batch_size, num_next_steps, env.n_agents, dtype=torch.bool)
    if batch_size > 1:
        agent_mask[1, :, -1] = False
    return OffPolicyReplayEpisodeSegmentBatch(
        local_obs=torch.randn(batch_size, num_next_steps, env.n_agents, env.local_obs_dim),
        global_obs=torch.randn(batch_size, num_next_steps, env.global_obs_dim),
        hidden_local_vars=torch.randn(batch_size, num_next_steps, env.n_agents, env.hidden_local_vars_dim),
        hidden_global_vars=torch.randn(batch_size, num_next_steps, env.hidden_global_vars_dim),
        agent_mask=agent_mask,
        actions=torch.randn(batch_size, num_next_steps, env.n_agents, action_dim).clamp(-0.9, 0.9),
        rewards=torch.randn(batch_size, num_next_steps),
        terminations=torch.zeros(batch_size, num_next_steps, dtype=torch.bool),
        truncations=torch.zeros(batch_size, num_next_steps, dtype=torch.bool),
        previous_actions=None,
        next_local_obs=torch.randn(batch_size, num_next_steps, env.n_agents, env.local_obs_dim),
        next_global_obs=torch.randn(batch_size, num_next_steps, env.global_obs_dim),
        next_hidden_local_vars=torch.randn(batch_size, num_next_steps, env.n_agents, env.hidden_local_vars_dim),
        next_hidden_global_vars=torch.randn(batch_size, num_next_steps, env.hidden_global_vars_dim),
        next_agent_mask=agent_mask.clone(),
        episode_start_mask=torch.zeros(batch_size, num_next_steps, dtype=torch.bool),
        train_mask=torch.ones(batch_size, num_next_steps, dtype=torch.bool),
        initial_temporal_state=None,
        burn_in_steps=0,
    )


def _supported_differentiable_configs() -> list[ContinuousActionDistConfig]:
    return [
        BetaConfig(),
        PredictedStdConfig(base_std=0.5),
        SquashedDiagGaussianConfig(std=0.5, std_learnable=True),
        ReparameterizedSignMagnitudeKumaraswamyConfig(),
        ReparameterizedSquashedGaussianMixtureConfig(inverse_cdf_iterations=8),
    ]


def _small_nop_config(
        *,
        latent_source: SACNOPLatentSource = SACNOPLatentSource.CRITIC,
        skip_first_transition_for_critic: bool = True,
) -> SACNOPConfig:
    return SACNOPConfig(
        enabled=True,
        latent_source=latent_source,
        skip_first_transition_for_critic=skip_first_transition_for_critic,
        nop_latent_dim=8,
        transition_model_d_model=8,
        transition_model_nhead=2,
        transition_model_num_layers=1,
        transition_model_dim_feedforward=16,
        next_obs_pred_config=NextObsPredConfig(
            local_scalar_target_indices=[0, 1],
            predict_delta=False,
        ),
    )


class _RecordingTransitionModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.action_inputs: list[torch.Tensor] = []
        self.agent_masks: list[torch.Tensor | None] = []

    def forward(
            self,
            z_t: torch.Tensor,
            a_t: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.action_inputs.append(a_t.detach().clone())
        self.agent_masks.append(None if agent_mask is None else agent_mask.detach().clone())
        return z_t

    def predict_n_steps(
            self,
            z_0: torch.Tensor,
            action_seq: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        self.action_inputs.append(action_seq.detach().clone())
        self.agent_masks.append(None if agent_mask is None else agent_mask.detach().clone())
        return z_0.unsqueeze(1).expand(-1, action_seq.shape[1], -1, -1)


class TMASACPolicyTests(unittest.TestCase):
    def test_critic_swiglu_initializes_every_block_output_with_projection_gain(self) -> None:
        transformer_gain = 1.5
        critic_encoder_config = replace(
            _small_encoder_config(),
            transformer_ff_config=SwiGLUConfig(
                hidden_dim=16,
                stacked=GLUStackConfig(n_layers=2),
            ),
            transformer_ff_init_gain=transformer_gain,
        )
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=replace(_make_config(), critic_encoder_config=critic_encoder_config),
        )
        hidden_layers, output_layers = feedforward_linear_layers(
            policy.critic.encoder.layers[0].feedforward
        )

        for linear in hidden_layers:
            expected_norm = transformer_gain * math.sqrt(min(linear.weight.shape))
            torch.testing.assert_close(
                torch.linalg.vector_norm(linear.weight),
                torch.tensor(expected_norm, dtype=linear.weight.dtype),
            )
        for linear in output_layers:
            expected_norm = math.sqrt(min(linear.weight.shape))
            torch.testing.assert_close(
                torch.linalg.vector_norm(linear.weight),
                torch.tensor(expected_norm, dtype=linear.weight.dtype),
            )

    def test_independent_actor_is_the_default_and_keeps_agent_attention(self) -> None:
        policy = TMASACPolicy(env=_DummyContinuousEnv(), config=_make_config())

        self.assertIs(policy.actor_head_kind, TMASACActorHeadKind.INDEPENDENT)
        self.assertTrue(all(layer.self_attn is not None for layer in policy.actor_encoder.layers))

    def test_policy_delegates_action_generation_to_every_actor_head(self) -> None:
        env = _DummyContinuousEnv()
        batch = _make_batch(batch_size=2)
        expected_actions = torch.randn(
            2,
            env.n_agents,
            env.action_space.total_agent_action_dim,
        )
        expected_log_probs = torch.randn(2, env.n_agents)

        for actor_head_kind in TMASACActorHeadKind:
            with self.subTest(actor_head_kind=actor_head_kind):
                policy = TMASACPolicy(
                    env=env,
                    config=_make_config(actor_head_kind=actor_head_kind),
                )
                generate_actions = Mock(
                    return_value=(expected_actions, expected_log_probs)
                )
                policy.actor_head.actions_and_log_probs = generate_actions

                actions, log_probs = policy.action_log_prob(
                    local_obs=batch.local_obs,
                    global_obs=batch.global_obs,
                    agent_mask=batch.agent_mask,
                    deterministic=True,
                    use_rsample=False,
                )

                self.assertIs(actions, expected_actions)
                self.assertIs(log_probs, expected_log_probs)
                generate_actions.assert_called_once()
                call_kwargs = generate_actions.call_args.kwargs
                self.assertEqual(
                    tuple(call_kwargs["actor_latents"].shape),
                    (2, env.n_agents, policy.actor_encoder_config.d_model),
                )
                self.assertIs(call_kwargs["action_dist"], policy.action_dist)
                self.assertIs(call_kwargs["agent_mask"], batch.agent_mask)
                self.assertIsNone(call_kwargs["previous_actions"])
                self.assertTrue(call_kwargs["deterministic"])
                self.assertFalse(call_kwargs["use_rsample"])

    def test_decentralized_actor_removes_only_agent_attention(self) -> None:
        torch.manual_seed(0)
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(actor_head_kind=TMASACActorHeadKind.DECENTRALIZED),
        )
        batch = _make_batch(batch_size=1)
        changed_local_obs = batch.local_obs.clone()
        changed_local_obs[:, 1:, :] += 100.0

        original_latents = policy.encode_actor(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            agent_mask=batch.agent_mask,
        )
        changed_latents = policy.encode_actor(
            local_obs=changed_local_obs,
            global_obs=batch.global_obs,
            agent_mask=batch.agent_mask,
        )

        self.assertFalse(policy.actor_encoder_config.use_agent_attention)
        self.assertTrue(all(layer.self_attn is None for layer in policy.actor_encoder.layers))
        self.assertTrue(all(len(layer._feedforward_linear_layers()) == 2 for layer in policy.actor_encoder.layers))
        self.assertTrue(all(isinstance(layer.norm2, torch.nn.LayerNorm) for layer in policy.actor_encoder.layers))
        torch.testing.assert_close(original_latents[:, 0], changed_latents[:, 0])
        self.assertFalse(torch.allclose(original_latents[:, 1:], changed_latents[:, 1:]))

    def test_decentralized_actor_rejects_shared_agent_attention(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot share an observation encoder"):
            TMASACPolicy(
                env=_DummyContinuousEnv(),
                config=_make_config(
                    actor_head_kind=TMASACActorHeadKind.DECENTRALIZED,
                    shared_encoder_config=_small_encoder_config(),
                ),
            )

    def test_qcx_actor_latents_depend_only_on_previous_agent_actions(self) -> None:
        torch.manual_seed(0)
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(actor_head_kind=TMASACActorHeadKind.QCX),
        )
        batch = _make_batch(batch_size=1)
        actor_latents = policy.encode_actor(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            agent_mask=batch.agent_mask,
        )
        actor_head = policy.actor_head
        assert isinstance(actor_head, TMASACQCXActorHead)
        base_actions = torch.zeros(
            1,
            policy.n_agents,
            policy.agent_action_dim,
        )

        base_outputs = actor_head(actor_latents, base_actions, batch.agent_mask)
        changed_first_action = base_actions.clone()
        changed_first_action[:, 0, :] = 1.0
        first_changed_outputs = actor_head(actor_latents, changed_first_action, batch.agent_mask)
        changed_second_action = base_actions.clone()
        changed_second_action[:, 1, :] = 1.0
        second_changed_outputs = actor_head(actor_latents, changed_second_action, batch.agent_mask)

        torch.testing.assert_close(base_outputs[:, 0], first_changed_outputs[:, 0])
        self.assertFalse(torch.allclose(base_outputs[:, 1:], first_changed_outputs[:, 1:]))
        torch.testing.assert_close(base_outputs[:, :2], second_changed_outputs[:, :2])
        self.assertFalse(torch.allclose(base_outputs[:, 2], second_changed_outputs[:, 2]))

    def test_qcx_actor_samples_autoregressively_with_arbitrary_agent_masks(self) -> None:
        torch.manual_seed(0)
        env = _DummyContinuousEnv()
        policy = TMASACPolicy(
            env=env,
            config=_make_config(actor_head_kind=TMASACActorHeadKind.QCX),
        )
        batch = _make_batch(batch_size=2)
        agent_mask = torch.tensor(
            [[True, False, True], [False, True, True]],
            dtype=torch.bool,
        )

        actions, log_probs = policy.action_log_prob(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            agent_mask=agent_mask,
            deterministic=False,
        )

        self.assertEqual(tuple(actions.shape), (2, env.n_agents, env.action_space.total_agent_action_dim))
        self.assertEqual(tuple(log_probs.shape), (2, env.n_agents))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(log_probs).all())
        self.assertTrue(torch.equal(actions[~agent_mask], torch.zeros_like(actions[~agent_mask])))
        self.assertTrue(torch.equal(log_probs[~agent_mask], torch.zeros_like(log_probs[~agent_mask])))
        for distribution in policy.action_dist.distributions:
            self.assertEqual(tuple(distribution.distribution.mean.shape[:-1]), (2, env.n_agents))

    def test_qcx_actor_excludes_inactive_agents_from_active_actions(self) -> None:
        torch.manual_seed(0)
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(actor_head_kind=TMASACActorHeadKind.QCX),
        )
        batch = _make_batch(batch_size=1)
        agent_mask = torch.tensor([[True, False, True]], dtype=torch.bool)
        changed_local_obs = batch.local_obs.clone()
        changed_local_obs[:, 1, :] += 100.0
        call_kwargs = {
            "global_obs": batch.global_obs,
            "agent_mask": agent_mask,
            "deterministic": True,
            "use_rsample": False,
        }

        original_actions, original_log_probs = policy.action_log_prob(
            local_obs=batch.local_obs,
            **call_kwargs,
        )
        changed_actions, changed_log_probs = policy.action_log_prob(
            local_obs=changed_local_obs,
            **call_kwargs,
        )

        torch.testing.assert_close(original_actions[:, [0, 2]], changed_actions[:, [0, 2]])
        torch.testing.assert_close(original_log_probs[:, [0, 2]], changed_log_probs[:, [0, 2]])
        self.assertTrue(torch.equal(original_actions[:, 1], torch.zeros_like(original_actions[:, 1])))
        self.assertTrue(torch.equal(original_log_probs[:, 1], torch.zeros_like(original_log_probs[:, 1])))

    def test_qcx_actor_gradients_reach_every_actor_head_stage(self) -> None:
        torch.manual_seed(0)
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(actor_head_kind=TMASACActorHeadKind.QCX),
        )
        batch = _make_batch()

        actions, log_probs = policy.action_log_prob(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            agent_mask=batch.agent_mask,
            deterministic=False,
            use_rsample=True,
        )
        (actions.square().mean() + log_probs.square().mean()).backward()

        actor_head = policy.actor_head
        assert isinstance(actor_head, TMASACQCXActorHead)
        for module_name, module in (
            ("action_encoder", actor_head.action_encoder),
            ("decoder", actor_head.decoder),
            ("output_head", actor_head.output_head),
        ):
            gradients = [
                parameter.grad
                for parameter in module.parameters()
                if parameter.requires_grad
            ]
            self.assertTrue(gradients, msg=f"{module_name} has no trainable parameters")
            self.assertTrue(
                all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients),
                msg=f"{module_name} has missing or non-finite gradients",
            )
            self.assertTrue(
                any(torch.count_nonzero(gradient).item() > 0 for gradient in gradients if gradient is not None),
                msg=f"{module_name} has only zero gradients",
            )

    def test_qcx_actor_restores_full_distribution_state_for_auxiliary_losses(self) -> None:
        torch.manual_seed(0)
        env = _DummyContinuousEnv()
        policy = TMASACPolicy(
            env=env,
            config=_make_config(
                actor_head_kind=TMASACActorHeadKind.QCX,
                continuous_config=PredictedStdConfig(
                    base_std=0.5,
                    ent_loss_coef=0.1,
                ),
            ),
        )
        batch = _make_batch()

        actions, log_probs = policy.action_log_prob(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            agent_mask=batch.agent_mask,
            deterministic=False,
            use_rsample=True,
        )
        extra_losses = policy.action_dist.compute_extra_losses_without_metrics(
            agent_mask=batch.agent_mask,
        )

        self.assertTrue(extra_losses)
        for distribution in policy.action_dist.distributions:
            self.assertEqual(
                tuple(distribution.distribution.mean.shape[:-1]),
                (batch.local_obs.shape[0], env.n_agents),
            )
        total_loss = (
            actions.square().mean()
            + log_probs.square().mean()
            + torch.stack(tuple(extra_losses.values())).sum()
        )
        total_loss.backward()
        self.assertTrue(
            any(
                parameter.grad is not None
                and torch.isfinite(parameter.grad).all()
                and torch.count_nonzero(parameter.grad).item() > 0
                for parameter in policy.actor_head.parameters()
            )
        )

    def test_actor_head_kind_normalizes_strings_and_rejects_unknown_values(self) -> None:
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(actor_head_kind="QCX"),
        )

        self.assertIs(policy.actor_head_kind, TMASACActorHeadKind.QCX)
        with self.assertRaisesRegex(ValueError, "Unknown TMASAC actor head kind"):
            TMASACPolicy(
                env=_DummyContinuousEnv(),
                config=_make_config(actor_head_kind="centralized"),
            )

    def test_scenario_input_normalization_preserves_single_feature_values(self) -> None:
        scenario_config = TMASACScenarioEncoderConfig(
            scenarios=(
                ScenarioObservationSpec(
                    scenario_id=0,
                    name="single_feature",
                    global_obs_dim=1,
                    hidden_local_vars_dim=1,
                    hidden_global_vars_dim=1,
                ),
            ),
            global_obs=ScenarioFieldEncoderConfig(
                output_dim=1,
                normalize_input=True,
            ),
            hidden_local_vars=ScenarioFieldEncoderConfig(output_dim=1),
            hidden_global_vars=ScenarioFieldEncoderConfig(output_dim=1),
            scenario_embedding_dim=0,
        )
        encoder = TMASACScenarioObservationEncoder(
            config=scenario_config,
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            act_fn_cls=_make_config().act_fn_cls,
            include_hidden_fields=False,
        )
        global_layer = encoder.global_obs_encoder.layers[0]
        with torch.no_grad():
            global_layer.weight.fill_(1.0)
            global_layer.bias.zero_()

        scenario_ids = torch.zeros(2, dtype=torch.long)
        encoded = encoder.encode_global_obs(
            torch.tensor([[1.0], [2.0]]),
            scenario_ids,
        )

        torch.testing.assert_close(encoded, torch.tensor([[1.0], [2.0]]))

    def test_scenario_encoders_support_mixed_batches_and_target_updates(self) -> None:
        env = _DummyScenarioEnv()
        scenario_config = _scenario_encoder_config()
        policy = TMASACPolicy(
            env=env,
            config=replace(_make_config(), scenario_encoder_config=scenario_config),
        )
        batch_size = 4
        scenario_ids = torch.tensor([0, 1, 0, 1], dtype=torch.long)
        local_obs = torch.randn(batch_size, env.n_agents, env.local_obs_dim)
        global_obs = torch.randn(batch_size, env.global_obs_dim)
        hidden_local_vars = torch.randn(
            batch_size,
            env.n_agents,
            env.hidden_local_vars_dim,
        )
        hidden_global_vars = torch.randn(batch_size, env.hidden_global_vars_dim)
        agent_mask = torch.ones(batch_size, env.n_agents, dtype=torch.bool)

        actions, log_probs = policy.action_log_prob(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
        )
        q1, q2 = policy.q_values(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            scenario_ids=scenario_ids,
            actions=actions,
        )

        self.assertEqual(
            tuple(actions.shape),
            (batch_size, env.n_agents, env.action_space.total_agent_action_dim),
        )
        self.assertEqual(tuple(log_probs.shape), (batch_size, env.n_agents))
        self.assertEqual(tuple(q1.shape), (batch_size,))
        self.assertEqual(tuple(q2.shape), (batch_size,))
        self.assertFalse(
            {id(parameter) for parameter in policy.actor_parameters()}
            & {id(parameter) for parameter in policy.critic_parameters()}
        )

        assert policy.critic_scenario_encoder is not None
        assert policy.critic_scenario_encoder_target is not None
        assert policy.actor_scenario_encoder is not None
        encoded_zero_dim_global_a = policy.actor_scenario_encoder.encode_global_obs(
            torch.randn(1, env.global_obs_dim),
            torch.zeros(1, dtype=torch.long),
        )
        encoded_zero_dim_global_b = policy.actor_scenario_encoder.encode_global_obs(
            torch.randn(1, env.global_obs_dim),
            torch.zeros(1, dtype=torch.long),
        )
        torch.testing.assert_close(
            encoded_zero_dim_global_a,
            encoded_zero_dim_global_b,
        )
        online_parameter = next(policy.critic_scenario_encoder.parameters())
        target_parameter = next(policy.critic_scenario_encoder_target.parameters())
        with torch.no_grad():
            online_parameter.add_(1.0)
        policy.polyak_update_targets(tau=1.0)
        self.assertTrue(torch.equal(online_parameter, target_parameter))
        self.assertFalse(target_parameter.requires_grad)

    def test_scenario_feature_disabled_preserves_policy_schema(self) -> None:
        policy = TMASACPolicy(env=_DummyContinuousEnv(), config=_make_config())

        self.assertFalse(any("scenario_encoder" in key for key in policy.state_dict()))
        self.assertNotIn(
            "scenario_encoder_config",
            policy.get_hyper_parameters()["tmasac_policy_config"],
        )
        self.assertFalse(any("scenario_encoder" in key for key in policy.get_grad_norms()))

    def test_scenario_environment_requires_scenario_encoder_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "requires TMASACPolicyConfig.scenario_encoder_config"):
            TMASACPolicy(env=_DummyScenarioEnv(), config=_make_config())

    def test_scenario_encoder_config_requires_scenario_environment(self) -> None:
        config = replace(
            _make_config(),
            scenario_encoder_config=_scenario_encoder_config(),
        )

        with self.assertRaisesRegex(ValueError, "requires an environment observation_space"):
            TMASACPolicy(env=_DummyContinuousEnv(), config=config)

    def test_scenario_policy_requires_ids_for_actor_calls(self) -> None:
        env = _DummyScenarioEnv()
        policy = TMASACPolicy(
            env=env,
            config=replace(
                _make_config(),
                scenario_encoder_config=_scenario_encoder_config(),
            ),
        )
        batch = _make_batch()

        with self.assertRaisesRegex(ValueError, "scenario_ids were not provided"):
            policy.action_log_prob(
                local_obs=batch.local_obs,
                global_obs=torch.zeros(batch.local_obs.shape[0], env.global_obs_dim),
                hidden_local_vars=torch.zeros(
                    batch.local_obs.shape[0],
                    env.n_agents,
                    env.hidden_local_vars_dim,
                ),
                hidden_global_vars=torch.zeros(
                    batch.local_obs.shape[0],
                    env.hidden_global_vars_dim,
                ),
                agent_mask=batch.agent_mask,
            )

    def test_scenario_policy_requires_ids_for_critic_calls(self) -> None:
        env = _DummyScenarioEnv()
        policy = TMASACPolicy(
            env=env,
            config=replace(
                _make_config(),
                scenario_encoder_config=_scenario_encoder_config(),
            ),
        )
        batch_size = 2

        with self.assertRaisesRegex(ValueError, "scenario_ids were not provided"):
            policy.q_values(
                local_obs=torch.zeros(batch_size, env.n_agents, env.local_obs_dim),
                global_obs=torch.zeros(batch_size, env.global_obs_dim),
                hidden_local_vars=torch.zeros(
                    batch_size,
                    env.n_agents,
                    env.hidden_local_vars_dim,
                ),
                hidden_global_vars=torch.zeros(batch_size, env.hidden_global_vars_dim),
                agent_mask=torch.ones(batch_size, env.n_agents, dtype=torch.bool),
                actions=torch.zeros(
                    batch_size,
                    env.n_agents,
                    env.action_space.total_agent_action_dim,
                ),
            )

    def test_scenario_critic_requires_both_hidden_fields(self) -> None:
        env = _DummyScenarioEnv()
        policy = TMASACPolicy(
            env=env,
            config=replace(
                _make_config(),
                scenario_encoder_config=_scenario_encoder_config(),
            ),
        )
        batch_size = 2

        with self.assertRaisesRegex(ValueError, "requires hidden_local_vars and hidden_global_vars"):
            policy.q_values(
                local_obs=torch.zeros(batch_size, env.n_agents, env.local_obs_dim),
                global_obs=torch.zeros(batch_size, env.global_obs_dim),
                hidden_local_vars=None,
                hidden_global_vars=torch.zeros(batch_size, env.hidden_global_vars_dim),
                agent_mask=torch.ones(batch_size, env.n_agents, dtype=torch.bool),
                scenario_ids=torch.tensor([0, 1]),
                actions=torch.zeros(
                    batch_size,
                    env.n_agents,
                    env.action_space.total_agent_action_dim,
                ),
            )

    def test_non_scenario_policy_rejects_unexpected_scenario_ids(self) -> None:
        policy = TMASACPolicy(env=_DummyContinuousEnv(), config=_make_config())
        batch = _make_batch()

        with self.assertRaisesRegex(ValueError, "has no scenario encoder configured"):
            policy.action_log_prob(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                scenario_ids=torch.zeros(batch.local_obs.shape[0], dtype=torch.long),
            )

    def test_scenario_policy_rejects_shared_observation_encoding(self) -> None:
        config = replace(
            _make_config(),
            share_observation_encoder=True,
            scenario_encoder_config=_scenario_encoder_config(),
        )

        with self.assertRaisesRegex(ValueError, "does not yet support shared observation encoding"):
            TMASACPolicy(env=_DummyScenarioEnv(), config=config)

    def test_scenario_policy_validates_environment_name_order(self) -> None:
        env = _DummyScenarioEnv()
        env.scenario_names = ("payload", "wall")
        config = replace(
            _make_config(),
            scenario_encoder_config=_scenario_encoder_config(),
        )

        with self.assertRaisesRegex(ValueError, "names/order must match"):
            TMASACPolicy(env=env, config=config)

    def test_scenario_policy_validates_environment_field_dimensions(self) -> None:
        env = _DummyScenarioEnv()
        env.scenario_names = ("wall", "payload")
        env.scenario_observation_dims = {
            "wall": {
                "global_obs": 1,
                "hidden_local_vars": 3,
                "hidden_global_vars": 1,
            },
            "payload": {
                "global_obs": 4,
                "hidden_local_vars": 0,
                "hidden_global_vars": 3,
            },
        }
        config = replace(
            _make_config(),
            scenario_encoder_config=_scenario_encoder_config(),
        )

        with self.assertRaisesRegex(ValueError, "dimensions for 'wall' do not match"):
            TMASACPolicy(env=env, config=config)

    def test_scenario_encoder_state_dict_round_trip_preserves_target_freezing(self) -> None:
        config = replace(
            _make_config(),
            scenario_encoder_config=_scenario_encoder_config(),
        )
        source = TMASACPolicy(env=_DummyScenarioEnv(), config=config)
        restored = TMASACPolicy(env=_DummyScenarioEnv(), config=config)

        restored.load_state_dict(source.state_dict())

        for key, source_value in source.state_dict().items():
            torch.testing.assert_close(restored.state_dict()[key], source_value)
        assert restored.critic_scenario_encoder_target is not None
        self.assertTrue(
            all(
                not parameter.requires_grad
                for parameter in restored.critic_scenario_encoder_target.parameters()
            )
        )

    def test_scenario_target_encoder_stays_in_eval_mode(self) -> None:
        policy = TMASACPolicy(
            env=_DummyScenarioEnv(),
            config=replace(
                _make_config(),
                scenario_encoder_config=_scenario_encoder_config(),
            ),
        )
        policy.train()

        assert policy.actor_scenario_encoder is not None
        assert policy.critic_scenario_encoder is not None
        assert policy.critic_scenario_encoder_target is not None
        self.assertTrue(policy.actor_scenario_encoder.training)
        self.assertTrue(policy.critic_scenario_encoder.training)
        self.assertFalse(policy.critic_scenario_encoder_target.training)

    def test_scenario_hyper_parameters_include_encoder_config(self) -> None:
        policy = TMASACPolicy(
            env=_DummyScenarioEnv(),
            config=replace(
                _make_config(),
                scenario_encoder_config=_scenario_encoder_config(),
            ),
        )

        serialized = policy.get_hyper_parameters()["tmasac_policy_config"]

        self.assertIn("scenario_encoder_config", serialized)
        self.assertEqual(
            serialized["scenario_encoder_config"]["scenario_embedding_dim"],
            3,
        )

    def test_target_modules_stay_in_eval_mode_when_policy_trains(self) -> None:
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(
                shared_encoder_config=_small_encoder_config(),
                dropout=0.5,
            ),
        )

        policy.train()

        self.assertTrue(policy.actor_encoder.training)
        self.assertTrue(policy.critic.training)
        self.assertIsNotNone(policy.shared_observation_encoder)
        assert policy.shared_observation_encoder is not None
        self.assertTrue(policy.shared_observation_encoder.training)
        self.assertFalse(policy.critic_target.training)
        self.assertIsNotNone(policy.shared_observation_encoder_target)
        assert policy.shared_observation_encoder_target is not None
        self.assertFalse(policy.shared_observation_encoder_target.training)

    def test_action_and_q_paths_match_pipeline_shapes(self) -> None:
        torch.manual_seed(0)
        env = _DummyContinuousEnv()
        policy = TMASACPolicy(env=env, config=_make_config())
        batch = _make_batch()

        actions, log_probs = policy.action_log_prob(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            deterministic=False,
        )
        q1, q2 = policy.q_values(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            actions=actions,
        )

        self.assertEqual(tuple(actions.shape), (4, env.n_agents, env.action_space.total_agent_action_dim))
        self.assertEqual(tuple(log_probs.shape), (4, env.n_agents))
        self.assertEqual(tuple(q1.shape), (4,))
        self.assertEqual(tuple(q2.shape), (4,))
        self.assertTrue(torch.equal(actions[~batch.agent_mask], torch.zeros_like(actions[~batch.agent_mask])))
        self.assertTrue(torch.equal(log_probs[~batch.agent_mask], torch.zeros_like(log_probs[~batch.agent_mask])))

    def test_full_graph_compile_friendly_actors_match_eager_forward_and_gradients(self) -> None:
        configs: tuple[ContinuousActionDistConfig, ...] = (
            PredictedStdConfig(base_std=0.5, ent_loss_coef=0.1),
            SquashedDiagGaussianConfig(std=0.5, std_learnable=True, ent_loss_coef=0.1),
            BetaConfig(ent_loss_coef=0.1),
            GumbelSoftmaxSignMagnitudeBetaConfig(ent_loss_coef=0.1),
            GumbelSoftmaxSignMagnitudeKumaraswamyConfig(ent_loss_coef=0.1),
            ReparameterizedSignMagnitudeKumaraswamyConfig(ent_loss_coef=0.1),
        )
        for continuous_config in configs:
            with self.subTest(continuous_config=type(continuous_config).__name__):
                torch._dynamo.reset()
                try:
                    self._assert_full_graph_actor_matches_eager(continuous_config)
                finally:
                    torch._dynamo.reset()

    def test_full_graph_qcx_actor_matches_eager_forward_and_gradients(self) -> None:
        torch._dynamo.reset()
        try:
            self._assert_full_graph_actor_matches_eager(
                PredictedStdConfig(base_std=0.5, ent_loss_coef=0.1),
                actor_head_kind=TMASACActorHeadKind.QCX,
            )
        finally:
            torch._dynamo.reset()

    def test_shared_encoder_qcx_compiles_autoregressive_actor_tail(self) -> None:
        compiled_graphs: list[torch.fx.GraphModule] = []
        compiled_function_names: list[str] = []

        def eager_backend(
                graph_module: torch.fx.GraphModule,
                _example_inputs: list[torch.Tensor],
                **_kwargs: Any,
        ) -> Callable[..., Any]:
            compiled_graphs.append(graph_module)
            return graph_module.forward

        def compile_with_eager_backend(
                function: Callable[..., Any],
                **kwargs: Any,
        ) -> Callable[..., Any]:
            compiled_function_names.append(getattr(function, "__name__", type(function).__name__))
            return _REAL_TORCH_COMPILE(function, backend=eager_backend, **kwargs)

        config = _make_config(
            shared_encoder_config=_small_encoder_config(),
            actor_head_kind=TMASACActorHeadKind.QCX,
        )
        torch.manual_seed(123)
        eager_policy = TMASACPolicy(env=_DummyContinuousEnv(), config=config)
        torch.manual_seed(123)
        with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=compile_with_eager_backend,
        ):
            compiled_policy = TMASACPolicy(
                env=_DummyContinuousEnv(),
                config=replace(config, compile_modules=True),
            )

        self.assertIsNone(compiled_policy._compiled_action_log_prob)
        self.assertIsNotNone(compiled_policy._compiled_actor_actions_and_log_probs)
        self.assertIn("_actor_actions_and_log_probs_impl", compiled_function_names)

        batch = _make_batch()
        call_kwargs = {
            "local_obs": batch.local_obs,
            "global_obs": batch.global_obs,
            "agent_mask": batch.agent_mask,
            "deterministic": True,
            "use_rsample": False,
        }
        eager_outputs = eager_policy.action_log_prob(**call_kwargs)
        compiled_outputs = compiled_policy.action_log_prob(**call_kwargs)

        for compiled_output, eager_output in zip(compiled_outputs, eager_outputs, strict=True):
            torch.testing.assert_close(compiled_output, eager_output)
        graph_count = len(compiled_graphs)
        self.assertGreaterEqual(graph_count, 3)

        compiled_policy.action_log_prob(**call_kwargs)
        self.assertEqual(len(compiled_graphs), graph_count)

        q_call_kwargs = {
            "local_obs": batch.local_obs,
            "global_obs": batch.global_obs,
            "actions": batch.actions,
            "hidden_local_vars": batch.hidden_local_vars,
            "hidden_global_vars": batch.hidden_global_vars,
            "agent_mask": batch.agent_mask,
        }
        for method_name in ("q_values", "q_values_with_nop_latents", "target_q_values"):
            eager_q_values = getattr(eager_policy, method_name)(**q_call_kwargs)
            compiled_q_values = getattr(compiled_policy, method_name)(**q_call_kwargs)
            for compiled_value, eager_value in zip(compiled_q_values, eager_q_values, strict=True):
                if compiled_value is None:
                    self.assertIsNone(eager_value)
                else:
                    torch.testing.assert_close(compiled_value, eager_value)
        self.assertTrue({
            "_q_values_impl",
            "_q_values_with_nop_latents_impl",
            "_target_q_values_impl",
        }.issubset(compiled_function_names))

    def _assert_full_graph_actor_matches_eager(
            self,
            continuous_config: ContinuousActionDistConfig,
            *,
            actor_head_kind: TMASACActorHeadKind = TMASACActorHeadKind.INDEPENDENT,
    ) -> None:
        compiled_graphs: list[torch.fx.GraphModule] = []
        actor_compile_options: list[dict[str, Any]] = []

        def eager_backend(
                graph_module: torch.fx.GraphModule,
                _example_inputs: list[torch.Tensor],
                **_kwargs: Any,
        ) -> Callable[..., Any]:
            compiled_graphs.append(graph_module)
            return graph_module.forward

        def compile_with_eager_backend(
                function: Callable[..., Any],
                **kwargs: Any,
        ) -> Callable[..., Any]:
            if getattr(function, "__name__", "") == "_action_log_prob_impl":
                actor_compile_options.append(kwargs)
            return _REAL_TORCH_COMPILE(function, backend=eager_backend, **kwargs)

        eager_config = _make_config(
            continuous_config=continuous_config,
            actor_head_kind=actor_head_kind,
        )
        eager_policy = TMASACPolicy(env=_DummyContinuousEnv(), config=eager_config)
        with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=compile_with_eager_backend,
        ):
            compiled_policy = TMASACPolicy(
                env=_DummyContinuousEnv(),
                config=replace(eager_config, compile_modules=True),
            )
        self.assertEqual(
            actor_compile_options,
            [{"mode": "default", "fullgraph": True, "dynamic": False}],
        )

        compiled_policy.actor_encoder.load_state_dict(eager_policy.actor_encoder.state_dict())
        compiled_policy.actor_head.load_state_dict(eager_policy.actor_head.state_dict())
        compiled_policy.action_dist.load_state_dict(eager_policy.action_dist.state_dict())
        batch = _make_batch()
        call_kwargs = {
            "local_obs": batch.local_obs,
            "global_obs": batch.global_obs,
            "hidden_local_vars": batch.hidden_local_vars,
            "hidden_global_vars": batch.hidden_global_vars,
            "agent_mask": batch.agent_mask,
            "previous_actions": batch.previous_actions,
        }

        actor_update_outputs: tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ] | None = None
        actor_update_extra_losses: tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]] | None = None
        action_modes = (
            {"deterministic": False, "use_rsample": False},
            {"deterministic": False, "use_rsample": True},
            {"deterministic": True, "use_rsample": False},
        )
        for seed, action_mode in enumerate(action_modes, start=123):
            mode_kwargs = {**call_kwargs, **action_mode}
            torch.manual_seed(seed)
            eager_actions, eager_log_probs = eager_policy.action_log_prob(**mode_kwargs)
            torch.manual_seed(seed)
            compiled_actions, compiled_log_probs = compiled_policy.action_log_prob(**mode_kwargs)
            torch.testing.assert_close(compiled_actions, eager_actions)
            torch.testing.assert_close(compiled_log_probs, eager_log_probs)
            eager_extra_losses = eager_policy.action_dist.compute_extra_losses_without_metrics(
                agent_mask=batch.agent_mask,
            )
            compiled_extra_losses = compiled_policy.action_dist.compute_extra_losses_without_metrics(
                agent_mask=batch.agent_mask,
            )
            self._assert_loss_dict_close(compiled_extra_losses, eager_extra_losses)
            if action_mode["use_rsample"]:
                actor_update_outputs = (
                    eager_actions,
                    eager_log_probs,
                    compiled_actions,
                    compiled_log_probs,
                )
                actor_update_extra_losses = eager_extra_losses, compiled_extra_losses

        assert actor_update_outputs is not None
        assert actor_update_extra_losses is not None
        eager_actions, eager_log_probs, compiled_actions, compiled_log_probs = actor_update_outputs
        eager_extra_losses, compiled_extra_losses = actor_update_extra_losses
        eager_loss = (
            eager_actions.square().mean()
            + eager_log_probs.square().mean()
            + torch.stack(tuple(eager_extra_losses.values())).sum()
        )
        compiled_loss = (
            compiled_actions.square().mean()
            + compiled_log_probs.square().mean()
            + torch.stack(tuple(compiled_extra_losses.values())).sum()
        )
        eager_loss.backward()
        compiled_loss.backward()
        for module_name in ("actor_encoder", "actor_head", "action_dist"):
            eager_module = getattr(eager_policy, module_name)
            compiled_module = getattr(compiled_policy, module_name)
            for (eager_name, eager_parameter), (compiled_name, compiled_parameter) in zip(
                    eager_module.named_parameters(),
                    compiled_module.named_parameters(),
                    strict=True,
            ):
                self.assertEqual(compiled_name, eager_name)
                if eager_parameter.grad is None or compiled_parameter.grad is None:
                    self.assertIsNone(eager_parameter.grad)
                    self.assertIsNone(compiled_parameter.grad)
                    continue
                torch.testing.assert_close(compiled_parameter.grad, eager_parameter.grad)

        self.assertEqual(len(compiled_graphs), len(action_modes))
        for seed, action_mode in enumerate(action_modes, start=456):
            torch.manual_seed(seed)
            eager_policy.action_log_prob(
                **{
                    **call_kwargs,
                    "local_obs": call_kwargs["local_obs"] + 0.25,
                    "global_obs": call_kwargs["global_obs"] - 0.25,
                },
                **action_mode,
            )
            torch.manual_seed(seed)
            compiled_policy.action_log_prob(
                **{
                    **call_kwargs,
                    "local_obs": call_kwargs["local_obs"] + 0.25,
                    "global_obs": call_kwargs["global_obs"] - 0.25,
                },
                **action_mode,
            )
            self._assert_loss_dict_close(
                compiled_policy.action_dist.compute_extra_losses_without_metrics(
                    agent_mask=batch.agent_mask,
                ),
                eager_policy.action_dist.compute_extra_losses_without_metrics(
                    agent_mask=batch.agent_mask,
                ),
            )
        self.assertEqual(len(compiled_graphs), len(action_modes))

    def _assert_loss_dict_close(
            self,
            actual: dict[str, torch.Tensor],
            expected: dict[str, torch.Tensor],
    ) -> None:
        self.assertEqual(actual.keys(), expected.keys())
        self.assertTrue(actual)
        for name in actual:
            torch.testing.assert_close(actual[name], expected[name])

    def test_twin_critics_share_action_conditioned_encoder_by_default(self) -> None:
        policy = TMASACPolicy(env=_DummyContinuousEnv(), config=_make_config())

        self.assertIsNone(policy.critic.encoder2)

    def test_critic_keeps_joint_observation_action_coembedding_by_default(self) -> None:
        env = _DummyContinuousEnv()
        policy = TMASACPolicy(env=env, config=_make_config())
        encoder = policy.critic.encoder
        first_fusion_linear = next(
            module
            for module in encoder.local_action_encoder
            if isinstance(module, torch.nn.Linear)
        )

        self.assertFalse(policy.config.critic_config.separate_observation_action_encoders)
        self.assertIsNone(encoder.observation_action_encoder)
        self.assertFalse(any("observation_action_encoder" in key for key in policy.state_dict()))
        self.assertEqual(
            first_fusion_linear.in_features,
            env.local_obs_dim
            + env.hidden_local_vars_dim
            + env.action_space.total_agent_action_dim,
        )

    def test_critic_can_preprocess_observations_and_actions_separately(self) -> None:
        torch.manual_seed(0)
        env = _DummyContinuousEnv()
        policy = TMASACPolicy(
            env=env,
            config=_make_config(separate_observation_action_encoders=True),
        )
        encoder = policy.critic.encoder
        observation_action_encoder = encoder.observation_action_encoder
        assert observation_action_encoder is not None
        observation_linear = next(
            module
            for module in observation_action_encoder.observation_encoder
            if isinstance(module, torch.nn.Linear)
        )
        action_linear = next(
            module
            for module in observation_action_encoder.action_encoder
            if isinstance(module, torch.nn.Linear)
        )
        first_fusion_linear = next(
            module
            for module in encoder.local_action_encoder
            if isinstance(module, torch.nn.Linear)
        )

        self.assertEqual(
            (observation_linear.in_features, observation_linear.out_features),
            (env.local_obs_dim + env.hidden_local_vars_dim, encoder.d_model),
        )
        self.assertEqual(
            (action_linear.in_features, action_linear.out_features),
            (env.action_space.total_agent_action_dim, encoder.d_model // 2),
        )
        self.assertEqual(
            first_fusion_linear.in_features,
            encoder.d_model + encoder.d_model // 2,
        )

        batch = _make_batch()
        actions = batch.actions.detach().requires_grad_(True)
        q1, q2 = policy.q_values(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            actions=actions,
        )
        (q1 + q2).sum().backward()

        self.assertEqual(q1.shape, (batch.local_obs.shape[0],))
        self.assertEqual(q2.shape, (batch.local_obs.shape[0],))
        self.assertIsNotNone(actions.grad)
        assert actions.grad is not None
        self.assertGreater(actions.grad.abs().sum().item(), 0.0)

    def test_critic_action_encoder_dim_can_be_overridden(self) -> None:
        action_encoder_dim = 7
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(
                separate_observation_action_encoders=True,
                action_encoder_dim=action_encoder_dim,
            ),
        )
        encoder = policy.critic.encoder
        observation_action_encoder = encoder.observation_action_encoder
        assert observation_action_encoder is not None
        action_linear = next(
            module
            for module in observation_action_encoder.action_encoder
            if isinstance(module, torch.nn.Linear)
        )
        first_fusion_linear = next(
            module
            for module in encoder.local_action_encoder
            if isinstance(module, torch.nn.Linear)
        )

        self.assertEqual(action_linear.out_features, action_encoder_dim)
        self.assertEqual(
            first_fusion_linear.in_features,
            encoder.d_model + action_encoder_dim,
        )

    def test_critic_action_encoder_dim_must_be_positive(self) -> None:
        with self.assertRaisesRegex(ValueError, "action_encoder_dim must be positive"):
            TMASACPolicy(
                env=_DummyContinuousEnv(),
                config=_make_config(action_encoder_dim=0),
            )

    def test_twin_critics_can_use_independent_action_conditioned_encoders(self) -> None:
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(independent_critic_encoders=True),
        )

        assert policy.critic.encoder2 is not None
        encoder1_parameter_ids = {id(parameter) for parameter in policy.critic.encoder.parameters()}
        encoder2_parameter_ids = {id(parameter) for parameter in policy.critic.encoder2.parameters()}
        self.assertFalse(encoder1_parameter_ids & encoder2_parameter_ids)

    def test_legacy_shared_critic_encoder_checkpoint_initializes_both_encoders(self) -> None:
        config = _make_config(independent_critic_encoders=True)
        policy = TMASACPolicy(env=_DummyContinuousEnv(), config=config)
        legacy_state_dict = {
            key: value
            for key, value in policy.state_dict().items()
            if ".encoder2." not in key
        }
        restored = TMASACPolicy(env=_DummyContinuousEnv(), config=config)

        restored.load_state_dict(legacy_state_dict, strict=True)

        assert restored.critic.encoder2 is not None
        for primary, secondary in zip(
                restored.critic.encoder.parameters(),
                restored.critic.encoder2.parameters(),
                strict=True,
        ):
            torch.testing.assert_close(primary, secondary)

    def test_critic_actions_condition_transformer_inputs(self) -> None:
        torch.manual_seed(0)
        policy = TMASACPolicy(env=_DummyContinuousEnv(), config=_make_config())
        batch = _make_batch()
        captured_tokens = []

        def capture_transformer_inputs(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            captured_tokens.append(args[0].detach().clone())

        hook = policy.critic.encoder.layers[0].register_forward_pre_hook(capture_transformer_inputs)
        try:
            policy.q_values(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                actions=batch.actions,
            )
            policy.q_values(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                actions=torch.zeros_like(batch.actions),
            )
        finally:
            hook.remove()

        self.assertEqual(len(captured_tokens), 2)
        self.assertFalse(torch.allclose(captured_tokens[0], captured_tokens[1]))

    def test_hidden_global_vars_condition_transformer_inputs(self) -> None:
        torch.manual_seed(0)
        env = _DummyContinuousEnv()
        policy = TMASACPolicy(env=env, config=_make_config())
        batch = _make_batch()
        captured_tokens = []

        def capture_transformer_inputs(_module: torch.nn.Module, args: tuple[torch.Tensor, ...]) -> None:
            captured_tokens.append(args[0].detach().clone())

        hook = policy.critic.encoder.layers[0].register_forward_pre_hook(capture_transformer_inputs)
        try:
            for hidden_global_vars in (batch.hidden_global_vars, torch.zeros_like(batch.hidden_global_vars)):
                policy.q_values(
                    local_obs=batch.local_obs,
                    global_obs=batch.global_obs,
                    hidden_local_vars=batch.hidden_local_vars,
                    hidden_global_vars=hidden_global_vars,
                    agent_mask=batch.agent_mask,
                    actions=batch.actions,
                )
        finally:
            hook.remove()

        self.assertEqual(
            policy.critic.encoder.global_encoder_input_dim,
            env.global_obs_dim + env.hidden_global_vars_dim,
        )
        self.assertEqual(policy.critic.q1.num_global_features, 0)
        self.assertEqual(len(captured_tokens), 2)
        self.assertFalse(torch.allclose(captured_tokens[0], captured_tokens[1]))

    def test_independent_critic_latents_are_concatenated_for_nop(self) -> None:
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(
                SACNOPConfig(
                    enabled=True,
                    latent_source=SACNOPLatentSource.CRITIC,
                    nop_latent_dim=8,
                    transition_model_d_model=8,
                    transition_model_nhead=2,
                    transition_model_num_layers=1,
                    transition_model_dim_feedforward=16,
                    next_obs_pred_config=NextObsPredConfig(
                        local_scalar_target_indices=[0],
                        predict_delta=False,
                    ),
                ),
                independent_critic_encoders=True,
            ),
        )
        batch = _make_batch()

        critic_latents = policy.encode_critic(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            actions=batch.actions,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
        )

        expected_latent_dim = 2 * policy.critic_encoder_config.d_model
        self.assertEqual(critic_latents.shape[-1], expected_latent_dim)
        assert policy.critic_nop is not None
        self.assertEqual(policy.critic_nop.source_latent_dim, expected_latent_dim)
        self.assertEqual(policy.critic_nop.pre_transition_transform.input_dim, expected_latent_dim)

    def test_shared_observation_encoder_is_policy_owned_critic_optimized_and_detached_for_actor(self) -> None:
        torch.manual_seed(0)
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(shared_encoder_config=_small_encoder_config()),
        )
        batch = _make_batch()
        assert policy.shared_observation_encoder is not None
        shared_encoder_param_ids = {id(parameter) for parameter in policy.shared_observation_encoder.parameters()}
        actor_param_ids = {id(parameter) for parameter in policy.actor_parameters()}
        critic_param_ids = {id(parameter) for parameter in policy.critic_parameters()}

        self.assertIsNot(policy.actor_encoder, policy.critic.encoder)
        self.assertFalse(shared_encoder_param_ids & actor_param_ids)
        self.assertTrue(shared_encoder_param_ids <= critic_param_ids)

        policy.zero_grad(set_to_none=True)
        _actions, log_probs = policy.action_log_prob(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            agent_mask=batch.agent_mask,
            deterministic=False,
        )
        log_probs.sum().backward()
        self.assertTrue(all(parameter.grad is None for parameter in policy.shared_observation_encoder.parameters()))

        policy.zero_grad(set_to_none=True)
        q1, q2 = policy.q_values(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            actions=batch.actions,
        )
        (q1 + q2).sum().backward()
        self.assertTrue(any(parameter.grad is not None for parameter in policy.shared_observation_encoder.parameters()))

    def test_legacy_share_observation_encoder_uses_policy_owned_encoder(self) -> None:
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(share_observation_encoder=True),
        )

        self.assertIsNotNone(policy.shared_observation_encoder)
        self.assertIsNot(policy.actor_encoder, policy.critic.encoder)

    def test_nop_config_defaults_to_disabled(self) -> None:
        policy = TMASACPolicy(env=_DummyContinuousEnv(), config=_make_config())

        self.assertFalse(policy.has_nop_loss())
        self.assertIsNone(policy.actor_nop)
        self.assertIsNone(policy.critic_nop)

    def test_nop_config_rejects_non_positive_horizon(self) -> None:
        with self.assertRaisesRegex(ValueError, "num_next_steps"):
            SACNOPConfig(num_next_steps=0)

    def test_critic_nop_skips_first_transition_by_default(self) -> None:
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(_small_nop_config()),
        )
        batch = _make_batch()
        transition_model = _RecordingTransitionModel()
        assert policy.critic_nop is not None
        policy.critic_nop.transition_model = transition_model

        loss, _metrics = policy.compute_critic_nop_loss(batch)

        self.assertIsNotNone(loss)
        self.assertTrue(policy.critic_nop.skip_first_transition)
        self.assertEqual(
            policy.critic_nop.latent_projection_hidden_dims,
            [policy.critic_nop.source_latent_dim],
        )
        self.assertEqual(transition_model.action_inputs, [])

    def test_critic_nop_multistep_transitions_start_with_second_action(self) -> None:
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(_small_nop_config()),
        )
        batch = _make_segment_batch(num_next_steps=4)
        transition_model = _RecordingTransitionModel()
        assert policy.critic_nop is not None
        policy.critic_nop.transition_model = transition_model

        loss, _metrics = policy.compute_critic_nop_loss(batch)

        self.assertIsNotNone(loss)
        self.assertEqual(len(transition_model.action_inputs), 1)
        torch.testing.assert_close(transition_model.action_inputs[0], batch.actions[:, 1:])
        assert transition_model.agent_masks[0] is not None
        torch.testing.assert_close(transition_model.agent_masks[0], batch.agent_mask[:, 1:])

    def test_critic_nop_can_keep_first_transition(self) -> None:
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(_small_nop_config(skip_first_transition_for_critic=False)),
        )
        batch = _make_batch()
        transition_model = _RecordingTransitionModel()
        assert policy.critic_nop is not None
        policy.critic_nop.transition_model = transition_model

        loss, _metrics = policy.compute_critic_nop_loss(batch)

        self.assertIsNotNone(loss)
        self.assertFalse(policy.critic_nop.skip_first_transition)
        self.assertEqual(len(transition_model.action_inputs), 1)
        torch.testing.assert_close(transition_model.action_inputs[0], batch.actions)

    def test_enabled_nop_requires_prediction_targets(self) -> None:
        with self.assertRaisesRegex(ValueError, "prediction target"):
            TMASACPolicy(
                env=_DummyContinuousEnv(),
                config=_make_config(SACNOPConfig(enabled=True)),
            )

    def test_rejects_discrete_action_subspaces(self) -> None:
        with self.assertRaisesRegex(ValueError, "continuous Box action sub-spaces only"):
            TMASACPolicy(env=_DummyDiscreteEnv(), config=_make_config())

    def test_supported_differentiable_actor_paths(self) -> None:
        env = _DummyContinuousEnv()
        batch = _make_batch()

        for idx, continuous_config in enumerate(_supported_differentiable_configs()):
            with self.subTest(config=type(continuous_config).__name__):
                torch.manual_seed(idx)
                policy = TMASACPolicy(
                    env=env,
                    config=_make_config(continuous_config=continuous_config),
                )

                actions, log_probs = policy.action_log_prob(
                    local_obs=batch.local_obs,
                    global_obs=batch.global_obs,
                    hidden_local_vars=batch.hidden_local_vars,
                    hidden_global_vars=batch.hidden_global_vars,
                    agent_mask=batch.agent_mask,
                    deterministic=False,
                )
                q1, q2 = policy.q_values(
                    local_obs=batch.local_obs,
                    global_obs=batch.global_obs,
                    hidden_local_vars=batch.hidden_local_vars,
                    hidden_global_vars=batch.hidden_global_vars,
                    agent_mask=batch.agent_mask,
                    actions=actions,
                )

                self.assertEqual(tuple(actions.shape), (4, env.n_agents, env.action_space.total_agent_action_dim))
                self.assertEqual(tuple(log_probs.shape), (4, env.n_agents))
                self.assertTrue(torch.isfinite(actions).all())
                self.assertTrue(torch.isfinite(log_probs).all())
                self.assertTrue(torch.isfinite(q1).all())
                self.assertTrue(torch.isfinite(q2).all())
                self.assertTrue(torch.all(actions <= 1.0))
                self.assertTrue(torch.all(actions >= -1.0))
                self.assertTrue(torch.equal(actions[~batch.agent_mask], torch.zeros_like(actions[~batch.agent_mask])))
                self.assertTrue(torch.equal(
                    log_probs[~batch.agent_mask],
                    torch.zeros_like(log_probs[~batch.agent_mask]),
                ))

    def test_rejects_non_differentiable_continuous_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "pathwise or straight-through"):
            TMASACPolicy(
                env=_DummyContinuousEnv(),
                config=_make_config(continuous_config=BetaMixtureConfig(
                    num_components=2,
                    alphas=(2.0, 2.0),
                    betas=(2.0, 2.0),
                )),
            )

    def test_gsde_actor_path_works_after_noise_reset(self) -> None:
        torch.manual_seed(0)
        env = _DummyContinuousEnv()
        policy = TMASACPolicy(
            env=env,
            config=_make_config(continuous_config=GSDEConfig(base_std=0.5)),
        )
        batch = _make_batch()

        with self.assertRaisesRegex(RuntimeError, "reset_noise"):
            policy.action_log_prob(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
                deterministic=False,
            )

        policy.action_dist.reset_temporal_correlations_on_step(batch_shape=tuple(batch.local_obs.shape[:-1]))
        actions, log_probs = policy.action_log_prob(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            hidden_local_vars=batch.hidden_local_vars,
            hidden_global_vars=batch.hidden_global_vars,
            agent_mask=batch.agent_mask,
            deterministic=False,
        )

        self.assertTrue(policy.gsde_enabled)
        self.assertEqual(tuple(actions.shape), (4, env.n_agents, env.action_space.total_agent_action_dim))
        self.assertEqual(tuple(log_probs.shape), (4, env.n_agents))
        self.assertTrue(torch.isfinite(actions).all())
        self.assertTrue(torch.isfinite(log_probs).all())

    def test_rejects_popart_until_sac_target_stat_updates_exist(self) -> None:
        config = _make_config()
        config = TMASACPolicyConfig(
            actor_encoder_config=config.actor_encoder_config,
            critic_encoder_config=config.critic_encoder_config,
            actor_head_config=config.actor_head_config,
            critic_config=TMASACCriticConfig(use_popart=True),
            continuous_config=config.continuous_config,
            nop_config=config.nop_config,
        )

        with self.assertRaisesRegex(NotImplementedError, "PopArt critics"):
            TMASACPolicy(env=_DummyContinuousEnv(), config=config)

    def test_nop_source_modes_allocate_expected_modules_and_compute_losses(self) -> None:
        batch = _make_batch()
        source_expectations = {
            SACNOPLatentSource.CRITIC: (False, True, "critic_nop_loss_scaled"),
            SACNOPLatentSource.ACTOR: (True, False, None),
            SACNOPLatentSource.BOTH: (True, True, "critic_nop_loss_scaled"),
            SACNOPLatentSource.SHARED_ENCODER: (False, True, "shared_encoder_nop_loss_scaled"),
        }

        for source, (expect_actor_nop, expect_critic_nop, expected_critic_metric) in source_expectations.items():
            with self.subTest(source=source):
                torch.manual_seed(10)
                policy = TMASACPolicy(
                    env=_DummyContinuousEnv(),
                    config=_make_config(
                        SACNOPConfig(
                            enabled=True,
                            latent_source=source,
                            nop_latent_dim=8,
                            transition_model_d_model=8,
                            transition_model_nhead=2,
                            transition_model_num_layers=1,
                            transition_model_dim_feedforward=16,
                            next_obs_pred_config=NextObsPredConfig(
                                local_scalar_target_indices=[0, 1],
                                predict_delta=False,
                            ),
                        ),
                        shared_encoder_config=(
                            _small_encoder_config()
                            if source is SACNOPLatentSource.SHARED_ENCODER
                            else None
                        ),
                    ),
                )
                actor_loss, actor_metrics = policy.compute_actor_nop_loss(batch)
                critic_loss, critic_metrics = policy.compute_critic_nop_loss(batch)

                self.assertEqual(policy.actor_nop is not None, expect_actor_nop)
                self.assertEqual(policy.critic_nop is not None, expect_critic_nop)
                self.assertEqual(actor_loss is not None, expect_actor_nop)
                self.assertEqual(critic_loss is not None, expect_critic_nop)
                if expect_actor_nop:
                    assert policy.actor_nop is not None
                    self.assertFalse(policy.actor_nop.skip_first_transition)
                    self.assertIn("actor_nop_loss_scaled", actor_metrics)
                    self.assertTrue(torch.isfinite(actor_loss))
                if expect_critic_nop:
                    assert policy.critic_nop is not None
                    self.assertEqual(
                        policy.critic_nop.skip_first_transition,
                        source in (SACNOPLatentSource.CRITIC, SACNOPLatentSource.BOTH),
                    )
                    assert expected_critic_metric is not None
                    self.assertIn(expected_critic_metric, critic_metrics)
                    self.assertTrue(torch.isfinite(critic_loss))
                if source is SACNOPLatentSource.BOTH:
                    assert policy.actor_nop is not None
                    assert policy.critic_nop is not None
                    self.assertIsNot(
                        policy.actor_nop.pre_transition_transform,
                        policy.critic_nop.pre_transition_transform,
                    )
                    self.assertIsNot(policy.actor_nop.transition_model, policy.critic_nop.transition_model)

    def test_nop_losses_support_multi_step_segments(self) -> None:
        torch.manual_seed(0)
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(
                SACNOPConfig(
                    enabled=True,
                    latent_source=SACNOPLatentSource.BOTH,
                    nop_latent_dim=8,
                    transition_model_d_model=8,
                    transition_model_nhead=2,
                    transition_model_num_layers=1,
                    transition_model_dim_feedforward=16,
                    next_obs_pred_config=NextObsPredConfig(
                        local_scalar_target_indices=[0, 1],
                        predict_delta=False,
                    ),
                )
            ),
        )
        batch = _make_segment_batch(num_next_steps=4)

        actor_loss, actor_metrics = policy.compute_actor_nop_loss(batch)
        critic_loss, critic_metrics = policy.compute_critic_nop_loss(batch)

        self.assertIsNotNone(actor_loss)
        self.assertIsNotNone(critic_loss)
        assert actor_loss is not None
        assert critic_loss is not None
        self.assertTrue(torch.isfinite(actor_loss))
        self.assertTrue(torch.isfinite(critic_loss))
        self.assertIn("actor_nop_loss_scaled", actor_metrics)
        self.assertIn("critic_nop_loss_scaled", critic_metrics)

    def test_shared_encoder_nop_source_requires_shared_encoder(self) -> None:
        with self.assertRaisesRegex(ValueError, "SHARED_ENCODER"):
            TMASACPolicy(
                env=_DummyContinuousEnv(),
                config=_make_config(SACNOPConfig(
                    enabled=True,
                    latent_source=SACNOPLatentSource.SHARED_ENCODER,
                )),
            )

    def test_both_nop_modules_do_not_share_parameters(self) -> None:
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(
                SACNOPConfig(
                    enabled=True,
                    latent_source=SACNOPLatentSource.BOTH,
                    nop_latent_dim=8,
                    transition_model_d_model=8,
                    transition_model_nhead=2,
                    transition_model_num_layers=1,
                    transition_model_dim_feedforward=16,
                    next_obs_pred_config=NextObsPredConfig(
                        local_scalar_target_indices=[0, 1],
                        predict_delta=False,
                    ),
                )
            ),
        )

        assert policy.actor_nop is not None
        assert policy.critic_nop is not None
        actor_nop_param_ids = {id(parameter) for parameter in policy.actor_nop.parameters()}
        critic_nop_param_ids = {id(parameter) for parameter in policy.critic_nop.parameters()}

        self.assertFalse(actor_nop_param_ids & critic_nop_param_ids)

    def test_nop_loss_weight_updates_support_global_and_source_specific_aliases(self) -> None:
        policy = TMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_make_config(
                SACNOPConfig(
                    enabled=True,
                    latent_source=SACNOPLatentSource.BOTH,
                    nop_loss_coef=1.0,
                    nop_latent_dim=8,
                    transition_model_d_model=8,
                    transition_model_nhead=2,
                    transition_model_num_layers=1,
                    transition_model_dim_feedforward=16,
                    next_obs_pred_config=NextObsPredConfig(
                        local_scalar_target_indices=[0],
                        predict_delta=False,
                    ),
                )
            ),
        )

        assert policy.actor_nop is not None
        assert policy.critic_nop is not None
        policy.update_loss_weights(nop_loss_coef=0.25)
        self.assertEqual(policy.actor_nop.nop_loss_coef, 0.25)
        self.assertEqual(policy.critic_nop.nop_loss_coef, 0.25)

        policy.update_loss_weights(actor_nop_loss_coef=0.5)
        self.assertEqual(policy.actor_nop.nop_loss_coef, 0.5)
        self.assertEqual(policy.critic_nop.nop_loss_coef, 0.25)

        policy.update_loss_weights(critic_world_model_loss_coef=0.75)
        self.assertEqual(policy.actor_nop.nop_loss_coef, 0.5)
        self.assertEqual(policy.critic_nop.nop_loss_coef, 0.75)

        with self.assertRaisesRegex(ValueError, "Multiple aliases"):
            policy.update_loss_weights(nop_loss_coef=0.1, wm_loss_coef=0.2)


if __name__ == "__main__":
    unittest.main()
