from typing import Any
import unittest

import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.beta_action_dist import BetaConfig
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
    BaseSACPolicy,
    SAC,
    SACNOPConfig,
    SACNOPLatentSource,
    TMASACActorHeadConfig,
    TMASACCriticConfig,
    TMASACPolicy,
    TMASACPolicyConfig,
)
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.gsde_reset import GSDEIntervalResetMode
from swarmbots.learn.testing_env import TestingSwarmBotsEnv


def _make_env(*, max_steps: int = 20) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [
            lambda: TestingSwarmBotsEnv(
                n_agents=2,
                n_local_obs=4,
                n_global_obs=2,
                actuators_dim=1,
                connectors_dim=1,
                n_hidden_local_vars=1,
                n_hidden_global_vars=1,
                max_steps=max_steps,
                continuous_connector_actions=True,
            )
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_policy(
        env: SwarmBotsLearnEnvWrapper,
        *,
        continuous_config: ContinuousActionDistConfig | None = None,
        nop_config: SACNOPConfig | None = None,
) -> TMASACPolicy:
    encoder_config = MATEncoderConfig(
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
    )
    return TMASACPolicy(
        env=env,
        config=TMASACPolicyConfig(
            actor_encoder_config=encoder_config,
            critic_encoder_config=encoder_config,
            actor_head_config=TMASACActorHeadConfig(hidden_dims=[8]),
            critic_config=TMASACCriticConfig(
                n_local_projection_hidden_layers=1,
                n_value_regressor_hidden_layers=1,
            ),
            continuous_config=PredictedStdConfig(base_std=0.7) if continuous_config is None else continuous_config,
            nop_config=SACNOPConfig() if nop_config is None else nop_config,
        ),
    )


def _supported_reparameterized_configs() -> list[ContinuousActionDistConfig]:
    return [
        BetaConfig(),
        PredictedStdConfig(base_std=0.7),
        SquashedDiagGaussianConfig(std=0.5, std_learnable=True),
        ReparameterizedSignMagnitudeKumaraswamyConfig(),
        ReparameterizedSquashedGaussianMixtureConfig(inverse_cdf_iterations=8),
    ]


class _ConstantTargetSACPolicy(BaseSACPolicy):
    def __init__(self, *, n_agents: int, action_dim: int, target_q_value: float) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.action_dim = int(action_dim)
        self.target_q_value = float(target_q_value)
        self.actor_scale = torch.nn.Parameter(torch.tensor(0.1))
        self.critic_bias = torch.nn.Parameter(torch.tensor(0.0))

    def action_log_prob(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            deterministic: bool = False,
            use_rsample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (global_obs, hidden_local_vars, hidden_global_vars, deterministic, use_rsample)
        actions = local_obs.new_ones((local_obs.shape[0], self.n_agents, self.action_dim)) * self.actor_scale
        if agent_mask is not None:
            actions = actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        return actions, actions[..., 0] * 0.0

    def q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (local_obs, global_obs, hidden_local_vars, hidden_global_vars, agent_mask)
        q_value = actions.sum(dim=(1, 2)) + self.critic_bias
        return q_value, q_value

    def target_q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (global_obs, actions, hidden_local_vars, hidden_global_vars, agent_mask)
        target_q = local_obs.new_full((local_obs.shape[0],), self.target_q_value)
        return target_q, target_q

    def actor_parameters(self) -> list[torch.nn.Parameter]:
        return [self.actor_scale]

    def critic_parameters(self) -> list[torch.nn.Parameter]:
        return [self.critic_bias]

    def compute_actor_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        _ = batch
        return None, {}

    def compute_critic_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        _ = batch
        return None, {}

    def polyak_update_targets(self, tau: float) -> None:
        _ = tau

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {"target_q_value": self.target_q_value}

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "actor": self._parameter_grad_norm(self.actor_scale),
            "critic": self._parameter_grad_norm(self.critic_bias),
        }

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unknown weights given: {weights}")


def _make_bootstrap_batch(env: SwarmBotsLearnEnvWrapper) -> OffPolicyReplayBatch:
    batch_size = 2
    n_agents = env.n_agents
    action_dim = env.action_space.total_agent_action_dim
    return OffPolicyReplayBatch(
        local_obs=torch.zeros(batch_size, n_agents, env.local_obs_dim),
        global_obs=torch.zeros(batch_size, env.global_obs_dim),
        hidden_local_vars=torch.zeros(batch_size, n_agents, env.hidden_local_vars_dim),
        hidden_global_vars=torch.zeros(batch_size, env.hidden_global_vars_dim),
        agent_mask=None,
        actions=torch.zeros(batch_size, n_agents, action_dim),
        rewards=torch.zeros(batch_size),
        terminations=torch.tensor([True, False]),
        truncations=torch.tensor([False, True]),
        previous_actions=None,
        next_local_obs=torch.ones(batch_size, n_agents, env.local_obs_dim),
        next_global_obs=torch.ones(batch_size, env.global_obs_dim),
        next_hidden_local_vars=torch.ones(batch_size, n_agents, env.hidden_local_vars_dim),
        next_hidden_global_vars=torch.ones(batch_size, env.hidden_global_vars_dim),
        next_agent_mask=None,
    )


class SACTests(unittest.TestCase):
    def test_perform_iteration_collects_replay_and_updates_once(self) -> None:
        env = _make_env()
        try:
            policy = _make_policy(env)
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                rollout_steps_per_iteration=2,
                gradient_steps=1,
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )

            metrics, rollout_steps = algo.perform_iteration(
                ExponentialMovingAverage(alpha=0.1),
                ExponentialMovingAverage(alpha=0.1),
                update_ema=False,
            )

            self.assertEqual(rollout_steps, 2)
            self.assertEqual(algo.n_total_timesteps, 2)
            self.assertEqual(algo.n_total_iterations, 1)
            self.assertEqual(algo.n_total_updates, 1)
            self.assertEqual(len(algo.replay_buffer), 2)
            self.assertEqual(metrics["updates"], 1)
            self.assertFalse(metrics["random_actions"])
            self.assertIn("actor_loss", metrics)
            self.assertIn("critic_loss", metrics)
            self.assertIn("ent_coef", metrics)
        finally:
            env.close()

    def test_learning_starts_collects_random_actions_and_skips_training(self) -> None:
        env = _make_env()
        try:
            policy = _make_policy(env)
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=8,
                learning_starts=100,
                batch_size=2,
                rollout_steps_per_iteration=2,
                gradient_steps=1,
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )

            metrics, rollout_steps = algo.perform_iteration(
                ExponentialMovingAverage(alpha=0.1),
                ExponentialMovingAverage(alpha=0.1),
                update_ema=False,
            )

            self.assertEqual(rollout_steps, 2)
            self.assertEqual(algo.n_total_timesteps, 2)
            self.assertEqual(algo.n_total_updates, 0)
            self.assertEqual(len(algo.replay_buffer), 2)
            self.assertEqual(metrics["updates"], 0)
            self.assertTrue(metrics["random_actions"])
            self.assertTrue(metrics["training_skipped"])
        finally:
            env.close()

    def test_rollout_warmup_advances_state_without_training_counters_or_replay(self) -> None:
        env = _make_env()
        try:
            policy = _make_policy(env)
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                rollout_steps_per_iteration=2,
                rollout_warmup_steps_per_env=3,
                gradient_steps=1,
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )

            algo._before_learn_loop()

            self.assertEqual(algo.n_total_timesteps, 0)
            self.assertEqual(algo.n_total_iterations, 0)
            self.assertEqual(algo.n_total_updates, 0)
            self.assertEqual(len(algo.replay_buffer), 0)
            self.assertIsNotNone(algo._rollout_state)
            assert algo._rollout_state is not None
            self.assertEqual(algo._rollout_state.rollout_step_idx, 3)

            metrics, rollout_steps = algo.perform_iteration(
                ExponentialMovingAverage(alpha=0.1),
                ExponentialMovingAverage(alpha=0.1),
                update_ema=False,
            )

            self.assertEqual(rollout_steps, 2)
            self.assertEqual(algo.n_total_timesteps, 2)
            self.assertEqual(algo.n_total_iterations, 1)
            self.assertEqual(len(algo.replay_buffer), 2)
            self.assertEqual(metrics["updates"], 1)
        finally:
            env.close()

    def test_rollout_warmup_is_skipped_for_started_training_state(self) -> None:
        env = _make_env()
        try:
            policy = _make_policy(env)
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                rollout_steps_per_iteration=2,
                rollout_warmup_steps_per_env=3,
                gradient_steps=1,
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )
            algo.n_total_timesteps = 2

            algo._before_learn_loop()

            self.assertIsNone(algo._rollout_state)
            self.assertTrue(algo._rollout_warmup_done)
            self.assertEqual(len(algo.replay_buffer), 0)
        finally:
            env.close()

    def test_gradient_steps_minus_one_uses_collected_transition_count(self) -> None:
        env = _make_env()
        try:
            policy = _make_policy(env)
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                rollout_steps_per_iteration=2,
                gradient_steps=-1,
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )

            metrics, rollout_steps = algo.perform_iteration(
                ExponentialMovingAverage(alpha=0.1),
                ExponentialMovingAverage(alpha=0.1),
                update_ema=False,
            )

            self.assertEqual(rollout_steps, 2)
            self.assertEqual(metrics["updates"], 2)
            self.assertEqual(algo.n_total_updates, 2)
        finally:
            env.close()

    def test_perform_iteration_with_supported_reparameterized_actor_configs(self) -> None:
        for continuous_config in _supported_reparameterized_configs():
            with self.subTest(config=type(continuous_config).__name__):
                env = _make_env()
                try:
                    policy = _make_policy(env, continuous_config=continuous_config)
                    algo = SAC(
                        policy=policy,
                        env=env,
                        learning_rate=1e-3,
                        buffer_capacity_per_env=8,
                        learning_starts=0,
                        batch_size=2,
                        rollout_steps_per_iteration=2,
                        gradient_steps=1,
                        train_device="cpu",
                        rollout_device="cpu",
                        replay_storage_device="cpu",
                    )

                    metrics, rollout_steps = algo.perform_iteration(
                        ExponentialMovingAverage(alpha=0.1),
                        ExponentialMovingAverage(alpha=0.1),
                        update_ema=False,
                    )

                    self.assertEqual(rollout_steps, 2)
                    self.assertEqual(metrics["updates"], 1)
                    self.assertIn("actor_loss", metrics)
                    self.assertIn("critic_loss", metrics)
                    self.assertIn("ent_coef", metrics)
                finally:
                    env.close()

    def test_perform_iteration_with_gsde_actor_requires_and_uses_reset_mode(self) -> None:
        env = _make_env()
        try:
            policy = _make_policy(env, continuous_config=GSDEConfig(base_std=0.5))

            with self.assertRaisesRegex(ValueError, "gsde_reset_mode"):
                SAC(
                    policy=policy,
                    env=env,
                    learning_rate=1e-3,
                    buffer_capacity_per_env=8,
                    learning_starts=0,
                    batch_size=2,
                    rollout_steps_per_iteration=2,
                    gradient_steps=1,
                    train_device="cpu",
                    rollout_device="cpu",
                    replay_storage_device="cpu",
                )

            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                rollout_steps_per_iteration=2,
                gradient_steps=1,
                gsde_reset_mode=GSDEIntervalResetMode(interval=1),
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )

            metrics, rollout_steps = algo.perform_iteration(
                ExponentialMovingAverage(alpha=0.1),
                ExponentialMovingAverage(alpha=0.1),
                update_ema=False,
            )

            self.assertEqual(rollout_steps, 2)
            self.assertEqual(metrics["updates"], 1)
            self.assertIn("actor_loss", metrics)
            self.assertIn("critic_loss", metrics)
        finally:
            env.close()

    def test_multi_step_nop_samples_episode_segments(self) -> None:
        env = _make_env()
        try:
            policy = _make_policy(
                env,
                nop_config=SACNOPConfig(
                    enabled=True,
                    latent_source=SACNOPLatentSource.CRITIC,
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
            )
            seen_nop_action_shapes: list[tuple[int, ...]] = []
            original_compute_critic_nop_loss = policy.compute_critic_nop_loss

            def capture_critic_nop_loss(
                    batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
            ) -> tuple[torch.Tensor | None, dict[str, Any]]:
                seen_nop_action_shapes.append(tuple(batch.actions.shape))
                return original_compute_critic_nop_loss(batch)

            policy.compute_critic_nop_loss = capture_critic_nop_loss
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=16,
                learning_starts=0,
                batch_size=2,
                rollout_steps_per_iteration=4,
                gradient_steps=1,
                nop_steps=4,
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )

            metrics, rollout_steps = algo.perform_iteration(
                ExponentialMovingAverage(alpha=0.1),
                ExponentialMovingAverage(alpha=0.1),
                update_ema=False,
            )

            self.assertEqual(rollout_steps, 4)
            self.assertEqual(metrics["updates"], 1)
            self.assertIn("critic_nop_loss_scaled", metrics)
            self.assertNotIn("nop_loss_skipped", metrics)
            self.assertEqual(
                seen_nop_action_shapes,
                [(2, 4, env.n_agents, env.action_space.total_agent_action_dim)],
            )
        finally:
            env.close()

    def test_multi_step_nop_skip_does_not_block_main_sac_update(self) -> None:
        env = _make_env(max_steps=1)
        try:
            policy = _make_policy(
                env,
                nop_config=SACNOPConfig(
                    enabled=True,
                    latent_source=SACNOPLatentSource.CRITIC,
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
            )
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=16,
                learning_starts=0,
                batch_size=2,
                rollout_steps_per_iteration=2,
                gradient_steps=1,
                nop_steps=4,
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )

            metrics, rollout_steps = algo.perform_iteration(
                ExponentialMovingAverage(alpha=0.1),
                ExponentialMovingAverage(alpha=0.1),
                update_ema=False,
            )

            self.assertEqual(rollout_steps, 2)
            self.assertEqual(metrics["updates"], 1)
            self.assertEqual(algo.n_total_updates, 1)
            self.assertLess(algo.replay_buffer.size_per_env, algo.nop_steps)
            self.assertIn("actor_loss", metrics)
            self.assertIn("critic_loss", metrics)
            self.assertIn("nop_loss_skipped", metrics)
            self.assertNotIn("critic_nop_loss_scaled", metrics)
        finally:
            env.close()

    def test_target_q_bootstraps_truncations_but_not_terminations(self) -> None:
        env = _make_env()
        try:
            policy = _ConstantTargetSACPolicy(
                n_agents=env.n_agents,
                action_dim=env.action_space.total_agent_action_dim,
                target_q_value=10.0,
            )
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=4,
                learning_starts=0,
                batch_size=2,
                gamma=0.5,
                ent_coef=0.0,
                max_grad_norm=None,
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )

            metrics, _actor_grad_norm, _critic_grad_norm = algo._train_step(
                _make_bootstrap_batch(env),
                global_update_idx=0,
            )

            self.assertAlmostEqual(metrics["target_q"], 2.5)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
