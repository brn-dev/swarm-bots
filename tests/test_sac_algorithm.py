import tempfile
import unittest
from dataclasses import replace
from typing import Any

import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.beta_action_dist import BetaConfig
from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaConfig,
    GumbelSoftmaxSignMagnitudeKumaraswamyConfig,
)
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
from swarmbots.learn.algos.off_policy.replay_buffer import (
    OffPolicyReplayBatch,
    OffPolicyReplayBuffer,
    OffPolicyReplayEpisodeSegmentBatch,
)
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
from swarmbots.learn.temporal_state import clone_temporal_state
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


def _supported_differentiable_configs() -> list[ContinuousActionDistConfig]:
    return [
        BetaConfig(),
        GumbelSoftmaxSignMagnitudeBetaConfig(),
        GumbelSoftmaxSignMagnitudeKumaraswamyConfig(),
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
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            use_rsample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (global_obs, hidden_local_vars, hidden_global_vars, previous_actions, deterministic, use_rsample)
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
            *,
            source_latents: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        _ = (batch, source_latents)
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


class _NonFiniteTerminalTargetSACPolicy(_ConstantTargetSACPolicy):
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
        if not all(torch.isfinite(tensor).all() for tensor in (
                local_obs,
                global_obs,
                hidden_local_vars,
                hidden_global_vars,
        )):
            raise AssertionError("SAC must sanitize terminal next-observation rows before bootstrapping")
        if agent_mask is not None and not bool(agent_mask[0].all()):
            raise AssertionError("SAC must replace an all-inactive terminal mask before attention")
        target_q1, target_q2 = super().target_q_values(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        target_q1[0] = torch.nan
        target_q2[0] = torch.nan
        return target_q1, target_q2


class _ConstantActionDistExtraLoss:
    has_gsde = False

    def compute_extra_losses(
            self,
            *,
            agent_mask: torch.Tensor | None = None,
            action_splitter: Any = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, float]]:
        _ = action_splitter
        if agent_mask is None:
            raise ValueError("agent_mask is required for this test helper.")
        return {
            "entropy": torch.full(agent_mask.shape, 2.0, dtype=torch.float32, device=agent_mask.device),
        }, {
            "ent_categorical": 0.25,
        }


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
                buffer_capacity_per_env=128,
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
            self.assertIs(type(algo.replay_buffer), OffPolicyReplayBuffer)
            self.assertIs(type(algo.replay_buffer.sample(1)), OffPolicyReplayBatch)
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

    def test_load_requires_fresh_replay_fill_before_training_resumes(self) -> None:
        source_env = _make_env()
        restored_env = _make_env()
        try:
            common_kwargs = {
                "learning_rate": 1e-3,
                "buffer_capacity_per_env": 8,
                "learning_starts": 4,
                "batch_size": 2,
                "rollout_steps_per_iteration": 2,
                "gradient_steps": 1,
                "train_device": "cpu",
                "rollout_device": "cpu",
                "replay_storage_device": "cpu",
            }
            source = SAC(policy=_make_policy(source_env), env=source_env, **common_kwargs)
            source.perform_iteration(
                ExponentialMovingAverage(alpha=0.1),
                ExponentialMovingAverage(alpha=0.1),
                update_ema=False,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = f"{temp_dir}/sac.pt"
                source.save(checkpoint_path)
                restored = SAC(policy=_make_policy(restored_env), env=restored_env, **common_kwargs)
                restored.load(checkpoint_path)

                self.assertEqual(restored.n_total_timesteps, 2)
                self.assertEqual(len(restored.replay_buffer), 0)
                self.assertIsNone(restored._rollout_state)

                first_metrics, _ = restored.perform_iteration(
                    ExponentialMovingAverage(alpha=0.1),
                    ExponentialMovingAverage(alpha=0.1),
                    update_ema=False,
                )
                self.assertEqual(restored.n_total_timesteps, 4)
                self.assertEqual(len(restored.replay_buffer), 2)
                self.assertEqual(first_metrics["updates"], 0)

                second_metrics, _ = restored.perform_iteration(
                    ExponentialMovingAverage(alpha=0.1),
                    ExponentialMovingAverage(alpha=0.1),
                    update_ema=False,
                )
                self.assertEqual(len(restored.replay_buffer), 4)
                self.assertEqual(second_metrics["updates"], 1)
        finally:
            source_env.close()
            restored_env.close()

    def test_learning_rate_warmup_preserves_explicit_entropy_coefficient_rate(self) -> None:
        env = _make_env()
        try:
            policy = _ConstantTargetSACPolicy(
                n_agents=env.n_agents,
                action_dim=env.action_space.total_agent_action_dim,
                target_q_value=0.0,
            )
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                learning_rate_warmup_updates=4,
                learning_rate_warmup_start_factor=0.25,
                buffer_capacity_per_env=4,
                learning_starts=0,
                batch_size=2,
                ent_coef="auto_0.1",
                ent_coef_learning_rate=3e-3,
                max_grad_norm=None,
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )
            batch = _make_bootstrap_batch(env)
            assert algo.ent_coef_optimizer is not None

            metrics, _actor_grad_norm, _critic_grad_norm = algo._train_step(
                batch,
                global_update_idx=0,
            )

            self.assertAlmostEqual(metrics["actor_critic_learning_rate"], 2.5e-4)
            self.assertAlmostEqual(metrics["ent_coef_learning_rate"], 3e-3)
            self.assertAlmostEqual(algo.actor_optimizer.param_groups[0]["lr"], 2.5e-4)
            self.assertAlmostEqual(algo.critic_optimizer.param_groups[0]["lr"], 2.5e-4)
            self.assertAlmostEqual(algo.ent_coef_optimizer.param_groups[0]["lr"], 3e-3)

            metrics, _actor_grad_norm, _critic_grad_norm = algo._train_step(
                batch,
                global_update_idx=4,
            )

            self.assertAlmostEqual(metrics["actor_critic_learning_rate"], 1e-3)
            self.assertAlmostEqual(algo.actor_optimizer.param_groups[0]["lr"], 1e-3)
            self.assertAlmostEqual(algo.critic_optimizer.param_groups[0]["lr"], 1e-3)
            self.assertAlmostEqual(algo.ent_coef_optimizer.param_groups[0]["lr"], 3e-3)

            algo.set_learning_rate(2e-3)

            self.assertAlmostEqual(algo.ent_coef_optimizer.param_groups[0]["lr"], 3e-3)
        finally:
            env.close()

    def test_load_reapplies_configured_entropy_coefficient_learning_rate(self) -> None:
        source_env = _make_env()
        restored_env = _make_env()
        try:
            common_kwargs = {
                "learning_rate": 1e-3,
                "learning_rate_warmup_updates": 0,
                "buffer_capacity_per_env": 4,
                "learning_starts": 0,
                "batch_size": 2,
                "ent_coef": "auto_0.1",
                "train_device": "cpu",
                "rollout_device": "cpu",
                "replay_storage_device": "cpu",
            }
            source = SAC(
                policy=_ConstantTargetSACPolicy(
                    n_agents=source_env.n_agents,
                    action_dim=source_env.action_space.total_agent_action_dim,
                    target_q_value=0.0,
                ),
                env=source_env,
                ent_coef_learning_rate=1e-4,
                **common_kwargs,
            )
            restored = SAC(
                policy=_ConstantTargetSACPolicy(
                    n_agents=restored_env.n_agents,
                    action_dim=restored_env.action_space.total_agent_action_dim,
                    target_q_value=0.0,
                ),
                env=restored_env,
                ent_coef_learning_rate=3e-3,
                **common_kwargs,
            )

            with tempfile.TemporaryDirectory() as temp_dir:
                checkpoint_path = f"{temp_dir}/sac.pt"
                source.save(
                    checkpoint_path,
                    optimizer_state_dict=source._get_optimizer_state_dict(),
                )
                restored.load(checkpoint_path)

            assert restored.ent_coef_optimizer is not None
            self.assertAlmostEqual(restored.ent_coef_optimizer.param_groups[0]["lr"], 3e-3)
        finally:
            source_env.close()
            restored_env.close()

    def test_entropy_coefficient_learning_rate_defaults_to_main_rate(self) -> None:
        env = _make_env()
        try:
            policy = _ConstantTargetSACPolicy(
                n_agents=env.n_agents,
                action_dim=env.action_space.total_agent_action_dim,
                target_q_value=0.0,
            )
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                learning_rate_warmup_updates=0,
                buffer_capacity_per_env=4,
                learning_starts=0,
                batch_size=2,
                ent_coef="auto_0.1",
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )
            assert algo.ent_coef_optimizer is not None

            self.assertIsNone(algo.ent_coef_learning_rate)
            self.assertAlmostEqual(algo.ent_coef_optimizer.param_groups[0]["lr"], 1e-3)

            algo.set_learning_rate(2e-3)

            self.assertAlmostEqual(algo.ent_coef_optimizer.param_groups[0]["lr"], 2e-3)
            self.assertAlmostEqual(algo.get_hyper_parameters()["resolved_ent_coef_learning_rate"], 2e-3)
        finally:
            env.close()

    def test_perform_iteration_with_supported_differentiable_actor_configs(self) -> None:
        for continuous_config in _supported_differentiable_configs():
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

    def test_gsde_rollout_noise_persists_across_training_iterations(self) -> None:
        env = _make_env(max_steps=20)
        try:
            policy = _make_policy(env, continuous_config=GSDEConfig(base_std=0.5))
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=1,
                rollout_steps_per_iteration=1,
                gradient_steps=1,
                gsde_reset_mode=GSDEIntervalResetMode(interval=10),
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )
            episode_return_ema = ExponentialMovingAverage(alpha=0.1)
            episode_success_rate_ema = ExponentialMovingAverage(alpha=0.1)

            algo.perform_iteration(
                episode_return_ema,
                episode_success_rate_ema,
                update_ema=False,
            )
            assert algo._rollout_state is not None
            first_noise_state = clone_temporal_state(algo._rollout_state.gsde_noise_state)

            algo.perform_iteration(
                episode_return_ema,
                episode_success_rate_ema,
                update_ema=False,
            )
            assert algo._rollout_state is not None
            second_noise_state = algo._rollout_state.gsde_noise_state

            self.assertIsInstance(first_noise_state, tuple)
            self.assertIsInstance(second_noise_state, tuple)
            for first_noise, second_noise in zip(first_noise_state, second_noise_state, strict=True):
                self.assertIsNotNone(first_noise)
                self.assertIsNotNone(second_noise)
                self.assertTrue(torch.equal(first_noise, second_noise))
        finally:
            env.close()

    def test_shared_origin_nop_reuses_bellman_critic_latents_for_multi_step_windows(self) -> None:
        env = _make_env()
        try:
            policy = _make_policy(
                env,
                nop_config=SACNOPConfig(
                    enabled=True,
                    num_next_steps=3,
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
            encoder_forward_calls = 0
            reused_latents = []
            seen_nop_action_shapes: list[tuple[int, ...]] = []

            def count_encoder_forward(
                    _module: torch.nn.Module,
                    _args: tuple[torch.Tensor, ...],
                    _kwargs: dict[str, Any],
            ) -> None:
                nonlocal encoder_forward_calls
                encoder_forward_calls += 1

            original_compute_critic_nop_loss = policy.compute_critic_nop_loss

            def capture_critic_nop_loss(
                    batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
                    *,
                    source_latents: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor | None, dict[str, Any]]:
                reused_latents.append(source_latents)
                seen_nop_action_shapes.append(tuple(batch.actions.shape))
                return original_compute_critic_nop_loss(batch, source_latents=source_latents)

            hook = policy.critic.encoder.register_forward_pre_hook(count_encoder_forward, with_kwargs=True)
            policy.compute_critic_nop_loss = capture_critic_nop_loss
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
            self.assertFalse(algo.independent_nop_sampling)
            self.assertFalse(algo.get_hyper_parameters()["independent_nop_sampling"])

            try:
                metrics, _rollout_steps = algo.perform_iteration(
                    ExponentialMovingAverage(alpha=0.1),
                    ExponentialMovingAverage(alpha=0.1),
                    update_ema=False,
                )
            finally:
                hook.remove()

            self.assertIn("critic_nop_loss_scaled", metrics)
            self.assertEqual(encoder_forward_calls, 2)
            self.assertEqual(len(reused_latents), 1)
            self.assertIsNotNone(reused_latents[0])
            self.assertEqual(
                seen_nop_action_shapes,
                [(2, 3, env.n_agents, env.action_space.total_agent_action_dim)],
            )
        finally:
            env.close()

    def test_shared_origin_nop_requires_the_bellman_batch_size(self) -> None:
        env = _make_env()
        try:
            policy = _make_policy(
                env,
                nop_config=SACNOPConfig(
                    enabled=True,
                    next_obs_pred_config=NextObsPredConfig(local_scalar_target_indices=[0]),
                ),
            )

            with self.assertRaisesRegex(ValueError, "nop_batch_size must equal batch_size"):
                SAC(
                    policy=policy,
                    env=env,
                    learning_rate=1e-3,
                    batch_size=2,
                    nop_batch_size=3,
                    train_device="cpu",
                    rollout_device="cpu",
                    replay_storage_device="cpu",
                )
        finally:
            env.close()

    def test_independent_multi_step_nop_samples_separate_episode_segments(self) -> None:
        env = _make_env()
        try:
            policy = _make_policy(
                env,
                continuous_config=ReparameterizedSignMagnitudeKumaraswamyConfig(),
                nop_config=SACNOPConfig(
                    enabled=True,
                    num_next_steps=3,
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
            seen_source_latents: list[torch.Tensor | None] = []
            original_compute_critic_nop_loss = policy.compute_critic_nop_loss

            def capture_critic_nop_loss(
                    batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
                    *,
                    source_latents: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor | None, dict[str, Any]]:
                seen_nop_action_shapes.append(tuple(batch.actions.shape))
                seen_source_latents.append(source_latents)
                return original_compute_critic_nop_loss(batch, source_latents=source_latents)

            policy.compute_critic_nop_loss = capture_critic_nop_loss
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=16,
                learning_starts=0,
                batch_size=2,
                independent_nop_sampling=True,
                rollout_steps_per_iteration=4,
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

            self.assertEqual(rollout_steps, 4)
            self.assertEqual(metrics["updates"], 1)
            self.assertIn("critic_nop_loss_scaled", metrics)
            self.assertNotIn("nop_loss_skipped", metrics)
            self.assertEqual(
                seen_nop_action_shapes,
                [(2, 3, env.n_agents, env.action_space.total_agent_action_dim)],
            )
            self.assertEqual(seen_source_latents, [None])
        finally:
            env.close()

    def test_independent_one_step_nop_uses_a_separate_replay_batch(self) -> None:
        env = _make_env()
        try:
            policy = _make_policy(
                env,
                nop_config=SACNOPConfig(
                    enabled=True,
                    num_next_steps=1,
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
            seen_source_latents: list[torch.Tensor | None] = []
            seen_action_shapes: list[tuple[int, ...]] = []
            original_compute_critic_nop_loss = policy.compute_critic_nop_loss

            def capture_critic_nop_loss(
                    batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
                    *,
                    source_latents: torch.Tensor | None = None,
            ) -> tuple[torch.Tensor | None, dict[str, Any]]:
                seen_source_latents.append(source_latents)
                seen_action_shapes.append(tuple(batch.actions.shape))
                return original_compute_critic_nop_loss(batch, source_latents=source_latents)

            policy.compute_critic_nop_loss = capture_critic_nop_loss
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                independent_nop_sampling=True,
                rollout_steps_per_iteration=2,
                gradient_steps=1,
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )

            metrics, _rollout_steps = algo.perform_iteration(
                ExponentialMovingAverage(alpha=0.1),
                ExponentialMovingAverage(alpha=0.1),
                update_ema=False,
            )

            self.assertIn("critic_nop_loss_scaled", metrics)
            self.assertEqual(seen_source_latents, [None])
            self.assertEqual(
                seen_action_shapes,
                [(2, env.n_agents, env.action_space.total_agent_action_dim)],
            )
        finally:
            env.close()

    def test_independent_multi_step_nop_skip_does_not_block_main_sac_update(self) -> None:
        env = _make_env(max_steps=1)
        try:
            policy = _make_policy(
                env,
                nop_config=SACNOPConfig(
                    enabled=True,
                    num_next_steps=4,
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
                independent_nop_sampling=True,
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
            self.assertEqual(algo.n_total_updates, 1)
            self.assertLess(algo.replay_buffer.size_per_env, policy.config.nop_config.num_next_steps)
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

    def test_scaled_auto_target_entropy_uses_active_agent_count(self) -> None:
        env = _make_env()
        try:
            policy = _ConstantTargetSACPolicy(
                n_agents=env.n_agents,
                action_dim=env.action_space.total_agent_action_dim,
                target_q_value=0.0,
            )
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=4,
                learning_starts=0,
                batch_size=2,
                ent_coef="auto*0.1",
                target_entropy="auto*0.25",
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )
            batch = _make_bootstrap_batch(env)
            agent_mask = torch.tensor(
                [
                    [True, True],
                    [True, False],
                ],
                dtype=torch.bool,
            )
            masked_batch = OffPolicyReplayBatch(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=agent_mask,
                actions=batch.actions,
                rewards=batch.rewards,
                terminations=batch.terminations,
                truncations=batch.truncations,
                previous_actions=batch.previous_actions,
                next_local_obs=batch.next_local_obs,
                next_global_obs=batch.next_global_obs,
                next_hidden_local_vars=batch.next_hidden_local_vars,
                next_hidden_global_vars=batch.next_hidden_global_vars,
                next_agent_mask=batch.next_agent_mask,
            )

            target_entropy = algo._target_entropy(
                batch=masked_batch,
                dtype=torch.float32,
                device=torch.device("cpu"),
            )

            expected = torch.tensor([-1.0, -0.5])
            self.assertTrue(torch.allclose(target_entropy, expected))
            assert algo.log_ent_coef is not None
            self.assertAlmostEqual(algo.log_ent_coef.detach().exp().item(), 0.1)
        finally:
            env.close()

    def test_actor_action_dist_extra_losses_are_included_with_agent_mask(self) -> None:
        env = _make_env()
        try:
            policy = _ConstantTargetSACPolicy(
                n_agents=env.n_agents,
                action_dim=env.action_space.total_agent_action_dim,
                target_q_value=0.0,
            )
            policy.action_dist = _ConstantActionDistExtraLoss()
            algo = SAC(
                policy=policy,
                env=env,
                learning_rate=1e-3,
                buffer_capacity_per_env=4,
                learning_starts=0,
                batch_size=2,
                ent_coef=0.0,
                max_grad_norm=None,
                train_device="cpu",
                rollout_device="cpu",
                replay_storage_device="cpu",
            )
            batch = _make_bootstrap_batch(env)
            agent_mask = torch.tensor(
                [
                    [True, True],
                    [True, False],
                ],
                dtype=torch.bool,
            )
            masked_batch = OffPolicyReplayBatch(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=agent_mask,
                actions=batch.actions,
                rewards=batch.rewards,
                terminations=batch.terminations,
                truncations=batch.truncations,
                previous_actions=batch.previous_actions,
                next_local_obs=batch.next_local_obs,
                next_global_obs=batch.next_global_obs,
                next_hidden_local_vars=batch.next_hidden_local_vars,
                next_hidden_global_vars=batch.next_hidden_global_vars,
                next_agent_mask=batch.next_agent_mask,
            )

            metrics, _actor_grad_norm, _critic_grad_norm = algo._train_step(
                masked_batch,
                global_update_idx=0,
            )

            self.assertAlmostEqual(metrics["actor_action_dist_entropy_loss_scaled"], 3.0)
            self.assertAlmostEqual(metrics["actor_action_dist_ent_categorical"], 0.25)
        finally:
            env.close()

    def test_non_finite_terminal_next_q_does_not_poison_target(self) -> None:
        env = _make_env()
        try:
            policy = _NonFiniteTerminalTargetSACPolicy(
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

            batch = _make_bootstrap_batch(env)
            next_local_obs = batch.next_local_obs.clone()
            next_global_obs = batch.next_global_obs.clone()
            next_hidden_local_vars = batch.next_hidden_local_vars.clone()
            next_hidden_global_vars = batch.next_hidden_global_vars.clone()
            next_local_obs[0] = torch.nan
            next_global_obs[0] = torch.nan
            next_hidden_local_vars[0] = torch.nan
            next_hidden_global_vars[0] = torch.nan
            next_agent_mask = torch.ones(
                batch.actions.shape[:2],
                dtype=torch.bool,
                device=batch.actions.device,
            )
            next_agent_mask[0] = False
            batch = replace(
                batch,
                next_local_obs=next_local_obs,
                next_global_obs=next_global_obs,
                next_hidden_local_vars=next_hidden_local_vars,
                next_hidden_global_vars=next_hidden_global_vars,
                next_agent_mask=next_agent_mask,
            )

            metrics, _actor_grad_norm, _critic_grad_norm = algo._train_step(
                batch,
                global_update_idx=0,
            )

            self.assertAlmostEqual(metrics["target_q"], 2.5)
            self.assertTrue(torch.isfinite(torch.tensor(metrics["critic_loss"])))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
