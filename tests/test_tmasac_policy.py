import unittest

import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.beta_action_dist import BetaConfig
from swarmbots.learn.action_dists.beta_mixture_action_dist import BetaMixtureConfig
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
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch, OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.algos.sac import (
    SACNOPConfig,
    SACNOPLatentSource,
    TMASACActorHeadConfig,
    TMASACCriticConfig,
    TMASACPolicy,
    TMASACPolicyConfig,
)
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace


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
) -> TMASACPolicyConfig:
    return TMASACPolicyConfig(
        actor_encoder_config=_small_encoder_config(),
        critic_encoder_config=_small_encoder_config(),
        shared_encoder_config=shared_encoder_config,
        share_observation_encoder=share_observation_encoder,
        actor_head_config=TMASACActorHeadConfig(hidden_dims=[10]),
        critic_config=TMASACCriticConfig(
            n_local_projection_hidden_layers=1,
            n_value_regressor_hidden_layers=1,
        ),
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


def _make_segment_batch(batch_size: int = 3, nop_steps: int = 4) -> OffPolicyReplayEpisodeSegmentBatch:
    env = _DummyContinuousEnv()
    action_dim = env.action_space.total_agent_action_dim
    agent_mask = torch.ones(batch_size, nop_steps, env.n_agents, dtype=torch.bool)
    if batch_size > 1:
        agent_mask[1, :, -1] = False
    return OffPolicyReplayEpisodeSegmentBatch(
        local_obs=torch.randn(batch_size, nop_steps, env.n_agents, env.local_obs_dim),
        global_obs=torch.randn(batch_size, nop_steps, env.global_obs_dim),
        hidden_local_vars=torch.randn(batch_size, nop_steps, env.n_agents, env.hidden_local_vars_dim),
        hidden_global_vars=torch.randn(batch_size, nop_steps, env.hidden_global_vars_dim),
        agent_mask=agent_mask,
        actions=torch.randn(batch_size, nop_steps, env.n_agents, action_dim).clamp(-0.9, 0.9),
        rewards=torch.randn(batch_size, nop_steps),
        terminations=torch.zeros(batch_size, nop_steps, dtype=torch.bool),
        truncations=torch.zeros(batch_size, nop_steps, dtype=torch.bool),
        previous_actions=None,
        next_local_obs=torch.randn(batch_size, nop_steps, env.n_agents, env.local_obs_dim),
        next_global_obs=torch.randn(batch_size, nop_steps, env.global_obs_dim),
        next_hidden_local_vars=torch.randn(batch_size, nop_steps, env.n_agents, env.hidden_local_vars_dim),
        next_hidden_global_vars=torch.randn(batch_size, nop_steps, env.hidden_global_vars_dim),
        next_agent_mask=agent_mask.clone(),
        episode_start_mask=torch.zeros(batch_size, nop_steps, dtype=torch.bool),
        train_mask=torch.ones(batch_size, nop_steps, dtype=torch.bool),
        initial_temporal_state=None,
        burn_in_steps=0,
    )


def _supported_reparameterized_configs() -> list[ContinuousActionDistConfig]:
    return [
        BetaConfig(),
        PredictedStdConfig(base_std=0.5),
        SquashedDiagGaussianConfig(std=0.5, std_learnable=True),
        ReparameterizedSignMagnitudeKumaraswamyConfig(),
        ReparameterizedSquashedGaussianMixtureConfig(inverse_cdf_iterations=8),
    ]


class TMASACPolicyTests(unittest.TestCase):
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

    def test_rejects_discrete_action_subspaces(self) -> None:
        with self.assertRaisesRegex(ValueError, "continuous Box action sub-spaces only"):
            TMASACPolicy(env=_DummyDiscreteEnv(), config=_make_config())

    def test_supported_reparameterized_actor_paths(self) -> None:
        env = _DummyContinuousEnv()
        batch = _make_batch()

        for idx, continuous_config in enumerate(_supported_reparameterized_configs()):
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

    def test_rejects_non_reparameterized_continuous_config(self) -> None:
        with self.assertRaisesRegex(ValueError, "reparameterized"):
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
                    self.assertIn("actor_nop_loss_scaled", actor_metrics)
                    self.assertTrue(torch.isfinite(actor_loss))
                if expect_critic_nop:
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
        batch = _make_segment_batch(nop_steps=4)

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
