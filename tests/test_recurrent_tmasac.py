import unittest
from dataclasses import replace
from unittest.mock import Mock, patch

import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.r_mat.temporal_sequence_model import (
    LSTMTemporalSequenceModel,
    LSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.sac import (
    RecurrentSAC,
    RecurrentTMASACPolicy,
    RecurrentTMASACPolicyConfig,
    SAC,
    SACNOPConfig,
    SACNOPLatentSource,
    TMASACActorHeadConfig,
    TMASACCriticConfig,
    TMASACPolicy,
    TMASACPolicyConfig,
)
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import RecurrentTMASACTwinCritic
from swarmbots.learn.algos.sac.tmasac_policy import TMASACTwinCritic
from swarmbots.learn.algos.xlstm.mlstm import (
    MLSTMTemporalSequenceModel,
    MLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.xlstm.slstm import (
    SLSTMTemporalSequenceModel,
    SLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.testing_env import TestingSwarmBotsEnv


class _DummyContinuousEnv:
    n_agents = 3
    local_obs_dim = 5
    global_obs_dim = 2
    hidden_local_vars_dim = 1
    hidden_global_vars_dim = 2
    action_space = HybridActionSpace({
        "actions": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 2), dtype=float),
    })


def _encoder_config(
        temporal_model_cls: type,
        temporal_model_config: object,
) -> RMATEncoderConfig:
    return RMATEncoderConfig(
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
        temporal_model_cls=temporal_model_cls,
        temporal_model_config=temporal_model_config,
    )


def _policy_config(
        encoder_config: RMATEncoderConfig,
        *,
        recurrent_critic: bool = False,
        critic_encoder_config: MATEncoderConfig | None = None,
        nop_config: SACNOPConfig | None = None,
) -> RecurrentTMASACPolicyConfig:
    if critic_encoder_config is None:
        critic_encoder_config = (
            encoder_config
            if recurrent_critic
            else MATEncoderConfig(
                d_model=8,
                nhead=2,
                num_layers=1,
                dim_feedforward=16,
            )
        )
    return RecurrentTMASACPolicyConfig(
        actor_encoder_config=encoder_config,
        critic_encoder_config=critic_encoder_config,
        actor_head_config=TMASACActorHeadConfig(hidden_dims=[8]),
        critic_config=TMASACCriticConfig(
            n_local_projection_hidden_layers=1,
            n_value_regressor_hidden_layers=1,
        ),
        continuous_config=PredictedStdConfig(base_std=0.5),
        recurrent_critic=recurrent_critic,
        nop_config=SACNOPConfig() if nop_config is None else nop_config,
    )


def _small_nop_config(
        *,
        num_next_steps: int = 2,
        latent_source: SACNOPLatentSource = SACNOPLatentSource.BOTH,
        global_scalar_target_indices: list[int] | None = None,
) -> SACNOPConfig:
    return SACNOPConfig(
        enabled=True,
        num_next_steps=num_next_steps,
        latent_source=latent_source,
        nop_latent_dim=8,
        transition_model_d_model=8,
        transition_model_nhead=2,
        transition_model_num_layers=1,
        transition_model_dim_feedforward=16,
        next_obs_pred_config=NextObsPredConfig(
            local_scalar_target_indices=[0],
            global_scalar_target_indices=global_scalar_target_indices,
            predict_delta=False,
        ),
    )


def _make_segment_batch(
        *,
        batch_size: int,
        sequence_length: int,
        n_agents: int,
        local_obs_dim: int,
        global_obs_dim: int,
        hidden_local_vars_dim: int,
        hidden_global_vars_dim: int,
        action_dim: int,
) -> OffPolicyReplayEpisodeSegmentBatch:
    time_values = torch.arange(sequence_length, dtype=torch.float32).view(1, sequence_length, 1, 1)
    local_obs = time_values.expand(batch_size, sequence_length, n_agents, local_obs_dim).clone()
    global_obs = time_values[..., 0, :].expand(batch_size, sequence_length, global_obs_dim).clone()
    hidden_local_vars = time_values.expand(
        batch_size,
        sequence_length,
        n_agents,
        hidden_local_vars_dim,
    ).clone()
    hidden_global_vars = time_values[..., 0, :].expand(
        batch_size,
        sequence_length,
        hidden_global_vars_dim,
    ).clone()
    actions = time_values.expand(batch_size, sequence_length, n_agents, action_dim).clone()
    agent_mask = torch.ones(batch_size, sequence_length, n_agents, dtype=torch.bool)
    return OffPolicyReplayEpisodeSegmentBatch(
        local_obs=local_obs,
        global_obs=global_obs,
        hidden_local_vars=hidden_local_vars,
        hidden_global_vars=hidden_global_vars,
        agent_mask=agent_mask,
        actions=actions,
        rewards=torch.zeros(batch_size, sequence_length),
        terminations=torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        truncations=torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        previous_actions=actions.clone(),
        next_local_obs=local_obs + 1.0,
        next_global_obs=global_obs + 1.0,
        next_hidden_local_vars=hidden_local_vars + 1.0,
        next_hidden_global_vars=hidden_global_vars + 1.0,
        next_agent_mask=agent_mask.clone(),
        episode_start_mask=torch.zeros(batch_size, sequence_length, dtype=torch.bool),
        train_mask=torch.ones(batch_size, sequence_length, dtype=torch.bool),
        initial_temporal_state=None,
        burn_in_steps=0,
    )


def _make_env(*, max_steps: int = 200) -> SwarmBotsLearnEnvWrapper:
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


def _perform_short_recurrent_update(
        *,
        recurrent_critic: bool = False,
        nop_config: SACNOPConfig | None = None,
        max_steps: int = 20,
) -> tuple[dict[str, object], int]:
    env = _make_env(max_steps=max_steps)
    try:
        policy = RecurrentTMASACPolicy(
            env=env,
            config=_policy_config(
                _encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                ),
                recurrent_critic=recurrent_critic,
                nop_config=nop_config,
            ),
        )
        algorithm = RecurrentSAC(
            policy=policy,
            env=env,
            burn_in_steps=2,
            learning_steps=3,
            temporal_state_store_interval=1,
            buffer_capacity_per_env=16,
            learning_starts=5,
            batch_size=2,
            rollout_steps_per_iteration=1,
            gradient_steps=1,
            replay_storage_device="cpu",
            train_device="cpu",
            rollout_device="cpu",
        )
        episode_return_ema = ExponentialMovingAverage(alpha=0.1)
        episode_success_rate_ema = ExponentialMovingAverage(alpha=0.1)

        metrics: dict[str, object] = {}
        for _ in range(5):
            metrics, _steps = algorithm.perform_iteration(
                episode_return_ema,
                episode_success_rate_ema,
                update_ema=True,
            )
        return metrics, algorithm.n_total_updates
    finally:
        env.close()


class RecurrentTMASACTests(unittest.TestCase):
    def test_plain_sac_rejects_recurrent_policy(self) -> None:
        env = _make_env()
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(
                        LSTMTemporalSequenceModel,
                        LSTMTemporalSequenceModelConfig(),
                    ),
                ),
            )

            with self.assertRaisesRegex(TypeError, "requires a recurrent SAC algorithm"):
                SAC(
                    policy=policy,
                    env=env,
                    buffer_capacity_per_env=128,
                    learning_starts=0,
                    batch_size=2,
                    replay_storage_device="cpu",
                    train_device="cpu",
                )
        finally:
            env.close()

    def test_default_only_replaces_actor_encoder_with_rmat(self) -> None:
        config = _policy_config(
            _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig())
        )
        with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.MATEncoder",
                side_effect=AssertionError("Recurrent actor must be built directly as R-MAT"),
        ):
            policy = RecurrentTMASACPolicy(env=_DummyContinuousEnv(), config=config)

        self.assertIsInstance(policy.actor_encoder, RMATEncoder)
        self.assertIsInstance(policy.critic, TMASACTwinCritic)
        self.assertNotIsInstance(policy.critic, RecurrentTMASACTwinCritic)
        self.assertIs(type(policy.critic_encoder_config), MATEncoderConfig)
        self.assertFalse(policy.recurrent_critic)

    def test_recurrent_critic_is_explicit(self) -> None:
        config = _policy_config(
            _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig()),
            recurrent_critic=True,
        )
        policy = RecurrentTMASACPolicy(env=_DummyContinuousEnv(), config=config)

        self.assertIsInstance(policy.critic, RecurrentTMASACTwinCritic)
        self.assertIsInstance(policy.critic_target, RecurrentTMASACTwinCritic)

    def test_recurrent_critic_requires_an_rmat_config(self) -> None:
        with self.assertRaisesRegex(TypeError, "requires an RMATEncoderConfig"):
            RecurrentTMASACPolicy(
                env=_DummyContinuousEnv(),
                config=_policy_config(
                    _encoder_config(
                        LSTMTemporalSequenceModel,
                        LSTMTemporalSequenceModelConfig(),
                    ),
                    recurrent_critic=True,
                    critic_encoder_config=MATEncoderConfig(
                        d_model=8,
                        nhead=2,
                        num_layers=1,
                        dim_feedforward=16,
                    ),
                ),
            )

    def test_recurrent_actor_requires_an_rmat_config(self) -> None:
        recurrent_config = _policy_config(
            _encoder_config(
                LSTMTemporalSequenceModel,
                LSTMTemporalSequenceModelConfig(),
            ),
        )
        with self.assertRaisesRegex(TypeError, "actor_encoder_config"):
            RecurrentTMASACPolicy(
                env=_DummyContinuousEnv(),
                config=replace(
                    recurrent_config,
                    actor_encoder_config=MATEncoderConfig(
                        d_model=8,
                        nhead=2,
                        num_layers=1,
                        dim_feedforward=16,
                    ),
                ),
            )

    def test_feedforward_critic_preserves_single_step_output_shapes(self) -> None:
        env = _DummyContinuousEnv()
        policy = RecurrentTMASACPolicy(
            env=env,
            config=_policy_config(
                _encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                ),
            ),
        )
        batch_size = 2
        local_obs = torch.randn(batch_size, env.n_agents, env.local_obs_dim)
        global_obs = torch.randn(batch_size, env.global_obs_dim)
        hidden_local_vars = torch.randn(
            batch_size,
            env.n_agents,
            env.hidden_local_vars_dim,
        )
        hidden_global_vars = torch.randn(batch_size, env.hidden_global_vars_dim)
        actions = torch.randn(
            batch_size,
            env.n_agents,
            env.action_space.total_agent_action_dim,
        )
        agent_mask = torch.ones(batch_size, env.n_agents, dtype=torch.bool)

        q1, q2, nop_latents, next_state = policy.q_values_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            target=False,
        )
        target_q1, target_q2, target_nop_latents, target_next_state = policy.q_values_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            target=True,
        )

        self.assertEqual(q1.shape, (batch_size,))
        self.assertEqual(q2.shape, (batch_size,))
        self.assertEqual(target_q1.shape, (batch_size,))
        self.assertEqual(target_q2.shape, (batch_size,))
        self.assertIsNone(nop_latents)
        self.assertIsNone(next_state)
        self.assertIsNone(target_nop_latents)
        self.assertIsNone(target_next_state)

    def test_actor_sequence_matches_step_flow_for_all_temporal_cores(self) -> None:
        temporal_configs = (
            (LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig()),
            (SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2)),
            (MLSTMTemporalSequenceModel, MLSTMTemporalSequenceModelConfig(num_heads=2)),
        )
        batch_size = 2
        sequence_length = 4
        env = _DummyContinuousEnv()
        local_obs = torch.randn(batch_size, sequence_length, env.n_agents, env.local_obs_dim)
        global_obs = torch.randn(batch_size, sequence_length, env.global_obs_dim)
        agent_mask = torch.ones(batch_size, sequence_length, env.n_agents, dtype=torch.bool)
        reset_mask = torch.tensor(
            [[True, False, True, False], [True, False, False, True]],
            dtype=torch.bool,
        )

        for temporal_model_cls, temporal_model_config in temporal_configs:
            with self.subTest(temporal_model=temporal_model_cls.__name__):
                torch.manual_seed(0)
                policy = RecurrentTMASACPolicy(
                    env=env,
                    config=_policy_config(_encoder_config(temporal_model_cls, temporal_model_config)),
                )
                initial_state = policy.initial_temporal_state(
                    batch_size=batch_size,
                    n_agents=env.n_agents,
                    device=local_obs.device,
                    dtype=local_obs.dtype,
                )
                sequence_actions, sequence_log_probs, _latents, _state = policy.action_log_prob_sequence(
                    local_obs=local_obs,
                    global_obs=global_obs,
                    agent_mask=agent_mask,
                    previous_actions=None,
                    deterministic=True,
                    use_rsample=False,
                    initial_state=initial_state,
                    reset_mask=reset_mask,
                )

                state = initial_state
                step_actions = []
                step_log_probs = []
                for time_idx in range(sequence_length):
                    actions, log_probs, _latents, state = policy.action_log_prob_sequence(
                        local_obs=local_obs[:, time_idx],
                        global_obs=global_obs[:, time_idx],
                        agent_mask=agent_mask[:, time_idx],
                        previous_actions=None,
                        deterministic=True,
                        use_rsample=False,
                        initial_state=state,
                        reset_mask=reset_mask[:, time_idx],
                    )
                    step_actions.append(actions)
                    step_log_probs.append(log_probs)

                torch.testing.assert_close(
                    sequence_actions,
                    torch.stack(step_actions, dim=1),
                    atol=1e-5,
                    rtol=1e-5,
                )
                torch.testing.assert_close(
                    sequence_log_probs,
                    torch.stack(step_log_probs, dim=1),
                    atol=1e-5,
                    rtol=1e-5,
                )

    def test_recurrent_critic_sequence_matches_step_flow(self) -> None:
        torch.manual_seed(0)
        env = _DummyContinuousEnv()
        policy = RecurrentTMASACPolicy(
            env=env,
            config=_policy_config(
                _encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                ),
                recurrent_critic=True,
            ),
        )
        batch_size = 2
        sequence_length = 4
        local_obs = torch.randn(batch_size, sequence_length, env.n_agents, env.local_obs_dim)
        global_obs = torch.randn(batch_size, sequence_length, env.global_obs_dim)
        hidden_local_vars = torch.randn(
            batch_size,
            sequence_length,
            env.n_agents,
            env.hidden_local_vars_dim,
        )
        hidden_global_vars = torch.randn(
            batch_size,
            sequence_length,
            env.hidden_global_vars_dim,
        )
        actions = torch.randn(
            batch_size,
            sequence_length,
            env.n_agents,
            env.action_space.total_agent_action_dim,
        )
        agent_mask = torch.ones(batch_size, sequence_length, env.n_agents, dtype=torch.bool)
        reset_mask = torch.tensor(
            [[True, False, False, True], [True, False, True, False]],
            dtype=torch.bool,
        )
        initial_state = policy.initial_critic_state(
            batch_size=batch_size,
            n_agents=env.n_agents,
            device=local_obs.device,
            dtype=local_obs.dtype,
            target=False,
        )
        sequence_q1, sequence_q2, _sequence_latents, _state = policy.q_values_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            target=False,
            initial_state=initial_state,
            reset_mask=reset_mask,
        )

        state = initial_state
        step_q1 = []
        step_q2 = []
        for time_idx in range(sequence_length):
            q1, q2, _latents, state = policy.q_values_sequence(
                local_obs=local_obs[:, time_idx],
                global_obs=global_obs[:, time_idx],
                actions=actions[:, time_idx],
                hidden_local_vars=hidden_local_vars[:, time_idx],
                hidden_global_vars=hidden_global_vars[:, time_idx],
                agent_mask=agent_mask[:, time_idx],
                target=False,
                initial_state=state,
                reset_mask=reset_mask[:, time_idx],
            )
            step_q1.append(q1)
            step_q2.append(q2)

        torch.testing.assert_close(sequence_q1, torch.stack(step_q1, dim=1))
        torch.testing.assert_close(sequence_q2, torch.stack(step_q2, dim=1))

    def test_nop_processes_every_aligned_learning_window_in_parallel(self) -> None:
        env = _make_env()
        try:
            encoder_config = _encoder_config(
                LSTMTemporalSequenceModel,
                LSTMTemporalSequenceModelConfig(),
            )
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    encoder_config,
                    nop_config=_small_nop_config(num_next_steps=2),
                ),
            )
            algorithm = RecurrentSAC(
                policy=policy,
                env=env,
                burn_in_steps=1,
                learning_steps=5,
                temporal_state_store_interval=1,
                buffer_capacity_per_env=16,
                learning_starts=0,
                batch_size=3,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            segment = _make_segment_batch(
                batch_size=3,
                sequence_length=5,
                n_agents=env.n_agents,
                local_obs_dim=env.local_obs_dim,
                global_obs_dim=env.global_obs_dim,
                hidden_local_vars_dim=env.hidden_local_vars_dim,
                hidden_global_vars_dim=env.hidden_global_vars_dim,
                action_dim=env.action_space.total_agent_action_dim,
            )
            agent_mask = segment.agent_mask.clone()
            next_agent_mask = segment.next_agent_mask.clone()
            agent_pattern = torch.tensor([True, False, True, True, False])
            next_agent_pattern = torch.tensor([False, True, True, False, True])
            agent_mask[:, :, 0] = agent_pattern
            next_agent_mask[:, :, 0] = next_agent_pattern
            segment = replace(
                segment,
                agent_mask=agent_mask,
                next_agent_mask=next_agent_mask,
            )
            latent_times = torch.arange(5, dtype=torch.float32).view(1, 5, 1, 1)
            actor_latents = latent_times.expand(3, 5, env.n_agents, 8).clone()
            critic_latents = actor_latents + 100.0

            segment_sampler = Mock(return_value=segment)
            algorithm.replay_buffer.sample_episode_segments = segment_sampler
            sampled_batch, sampled_nop_batch, reuse_nop_latents = algorithm._sample_training_batches()
            self.assertIs(sampled_batch, segment)
            self.assertIsNone(sampled_nop_batch)
            self.assertFalse(reuse_nop_latents)
            segment_sampler.assert_called_once_with(
                algorithm.batch_size,
                segment_length=5,
                burn_in_steps=1,
                require_initial_temporal_state=True,
                allow_episode_boundaries=True,
            )

            nop_training_batch = algorithm._build_nop_training_batch(
                batch=segment,
                actor_latents=actor_latents,
                critic_latents=critic_latents,
            )

            assert nop_training_batch is not None
            origins = torch.arange(4).view(1, 4).expand(3, 4)
            expected_times = origins.unsqueeze(2) + torch.arange(2).view(1, 1, 2)
            self.assertEqual(nop_training_batch.batch.actions.shape[:3], (3, 4, 2))
            torch.testing.assert_close(
                nop_training_batch.batch.local_obs[..., 0, 0],
                origins.to(dtype=torch.float32),
            )
            torch.testing.assert_close(
                nop_training_batch.batch.next_local_obs[..., 0, 0],
                expected_times.to(dtype=torch.float32) + 1.0,
            )
            torch.testing.assert_close(
                nop_training_batch.batch.actions[..., 0, 0],
                expected_times.to(dtype=torch.float32),
            )
            assert nop_training_batch.batch.agent_mask is not None
            assert nop_training_batch.batch.next_agent_mask is not None
            torch.testing.assert_close(
                nop_training_batch.batch.agent_mask[..., 0],
                agent_pattern[expected_times],
            )
            torch.testing.assert_close(
                nop_training_batch.batch.next_agent_mask[..., 0],
                next_agent_pattern[expected_times],
            )
            self.assertIsNone(nop_training_batch.batch.global_obs)
            self.assertIsNone(nop_training_batch.batch.next_global_obs)
            assert nop_training_batch.actor_source_latents is not None
            torch.testing.assert_close(
                nop_training_batch.actor_source_latents[..., 0, 0],
                origins.to(dtype=torch.float32),
            )
            assert nop_training_batch.critic_source_latents is not None
            torch.testing.assert_close(
                nop_training_batch.critic_source_latents[..., 0, 0],
                origins.to(dtype=torch.float32) + 100.0,
            )
            self.assertTrue(nop_training_batch.batch.train_mask.all())
        finally:
            env.close()

    def test_all_origin_nop_only_materializes_global_windows_when_needed(self) -> None:
        env = _make_env()
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(
                        LSTMTemporalSequenceModel,
                        LSTMTemporalSequenceModelConfig(),
                    ),
                    nop_config=_small_nop_config(
                        num_next_steps=2,
                        latent_source=SACNOPLatentSource.ACTOR,
                        global_scalar_target_indices=[0],
                    ),
                ),
            )
            algorithm = RecurrentSAC(
                policy=policy,
                env=env,
                burn_in_steps=1,
                learning_steps=3,
                temporal_state_store_interval=1,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            segment = _make_segment_batch(
                batch_size=2,
                sequence_length=3,
                n_agents=env.n_agents,
                local_obs_dim=env.local_obs_dim,
                global_obs_dim=env.global_obs_dim,
                hidden_local_vars_dim=env.hidden_local_vars_dim,
                hidden_global_vars_dim=env.hidden_global_vars_dim,
                action_dim=env.action_space.total_agent_action_dim,
            )
            nop_training_batch = algorithm._build_nop_training_batch(
                batch=segment,
                actor_latents=torch.zeros(2, 3, env.n_agents, 8),
                critic_latents=None,
            )

            assert nop_training_batch is not None
            assert nop_training_batch.batch.global_obs is not None
            assert nop_training_batch.batch.next_global_obs is not None
            expected_origins = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
            expected_windows = torch.tensor([
                [[0.0, 1.0], [1.0, 2.0]],
                [[0.0, 1.0], [1.0, 2.0]],
            ])
            torch.testing.assert_close(
                nop_training_batch.batch.global_obs[..., 0],
                expected_origins,
            )
            torch.testing.assert_close(
                nop_training_batch.batch.next_global_obs[..., 0],
                expected_windows + 1.0,
            )
            assert nop_training_batch.actor_source_latents is not None
            actor_loss, _metrics = policy.compute_actor_nop_loss_from_latents(
                source_latents=nop_training_batch.actor_source_latents,
                batch=nop_training_batch.batch,
            )
            assert actor_loss is not None
            self.assertTrue(torch.isfinite(actor_loss))
        finally:
            env.close()

    def test_all_origin_nop_masks_targets_after_an_episode_boundary(self) -> None:
        env = _make_env()
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(
                        LSTMTemporalSequenceModel,
                        LSTMTemporalSequenceModelConfig(),
                    ),
                    nop_config=_small_nop_config(num_next_steps=2),
                ),
            )
            algorithm = RecurrentSAC(
                policy=policy,
                env=env,
                burn_in_steps=1,
                learning_steps=5,
                temporal_state_store_interval=1,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            segment = _make_segment_batch(
                batch_size=2,
                sequence_length=5,
                n_agents=env.n_agents,
                local_obs_dim=env.local_obs_dim,
                global_obs_dim=env.global_obs_dim,
                hidden_local_vars_dim=env.hidden_local_vars_dim,
                hidden_global_vars_dim=env.hidden_global_vars_dim,
                action_dim=env.action_space.total_agent_action_dim,
            )
            segment = replace(
                segment,
                truncations=torch.tensor([
                    [False, True, False, False, False],
                    [False, True, False, False, False],
                ]),
                episode_start_mask=torch.tensor([
                    [False, False, True, False, False],
                    [False, False, True, False, False],
                ]),
            )

            nop_training_batch = algorithm._build_nop_training_batch(
                batch=segment,
                actor_latents=torch.zeros(2, 5, env.n_agents, 8),
                critic_latents=torch.zeros(2, 5, env.n_agents, 8),
            )

            assert nop_training_batch is not None
            self.assertEqual(nop_training_batch.batch.train_mask.tolist(), [
                [[True, True], [True, False], [True, True], [True, True]],
                [[True, True], [True, False], [True, True], [True, True]],
            ])
        finally:
            env.close()

    def test_nop_only_indexes_enabled_latent_sources(self) -> None:
        env = _make_env()
        try:
            segment = _make_segment_batch(
                batch_size=2,
                sequence_length=3,
                n_agents=env.n_agents,
                local_obs_dim=env.local_obs_dim,
                global_obs_dim=env.global_obs_dim,
                hidden_local_vars_dim=env.hidden_local_vars_dim,
                hidden_global_vars_dim=env.hidden_global_vars_dim,
                action_dim=env.action_space.total_agent_action_dim,
            )
            for latent_source in (SACNOPLatentSource.ACTOR, SACNOPLatentSource.CRITIC):
                with self.subTest(latent_source=latent_source.value):
                    policy = RecurrentTMASACPolicy(
                        env=env,
                        config=_policy_config(
                            _encoder_config(
                                LSTMTemporalSequenceModel,
                                LSTMTemporalSequenceModelConfig(),
                            ),
                            nop_config=_small_nop_config(
                                num_next_steps=2,
                                latent_source=latent_source,
                            ),
                        ),
                    )
                    algorithm = RecurrentSAC(
                        policy=policy,
                        env=env,
                        burn_in_steps=1,
                        learning_steps=3,
                        temporal_state_store_interval=1,
                        buffer_capacity_per_env=8,
                        learning_starts=0,
                        batch_size=2,
                        replay_storage_device="cpu",
                        train_device="cpu",
                    )
                    nop_training_batch = algorithm._build_nop_training_batch(
                        batch=segment,
                        actor_latents=torch.zeros(2, 3, env.n_agents, 8),
                        critic_latents=(
                            None
                            if latent_source is SACNOPLatentSource.ACTOR
                            else torch.zeros(2, 3, env.n_agents, 8)
                        ),
                    )

                    assert nop_training_batch is not None
                    self.assertEqual(
                        nop_training_batch.actor_source_latents is not None,
                        latent_source is SACNOPLatentSource.ACTOR,
                    )
                    self.assertEqual(
                        nop_training_batch.critic_source_latents is not None,
                        latent_source is SACNOPLatentSource.CRITIC,
                    )
        finally:
            env.close()

    def test_all_origin_nop_supports_horizon_boundaries_and_gradients(self) -> None:
        env = _make_env()
        try:
            segment = _make_segment_batch(
                batch_size=2,
                sequence_length=3,
                n_agents=env.n_agents,
                local_obs_dim=env.local_obs_dim,
                global_obs_dim=env.global_obs_dim,
                hidden_local_vars_dim=env.hidden_local_vars_dim,
                hidden_global_vars_dim=env.hidden_global_vars_dim,
                action_dim=env.action_space.total_agent_action_dim,
            )
            for horizon, expected_num_origins in ((1, 3), (3, 1)):
                with self.subTest(horizon=horizon):
                    policy = RecurrentTMASACPolicy(
                        env=env,
                        config=_policy_config(
                            _encoder_config(
                                LSTMTemporalSequenceModel,
                                LSTMTemporalSequenceModelConfig(),
                            ),
                            nop_config=_small_nop_config(num_next_steps=horizon),
                        ),
                    )
                    algorithm = RecurrentSAC(
                        policy=policy,
                        env=env,
                        burn_in_steps=1,
                        learning_steps=3,
                        temporal_state_store_interval=1,
                        buffer_capacity_per_env=8,
                        learning_starts=0,
                        batch_size=2,
                        replay_storage_device="cpu",
                        train_device="cpu",
                    )
                    actor_latents = torch.randn(
                        2,
                        3,
                        env.n_agents,
                        8,
                        requires_grad=True,
                    )
                    critic_latents = torch.randn_like(actor_latents, requires_grad=True)
                    nop_training_batch = algorithm._build_nop_training_batch(
                        batch=segment,
                        actor_latents=actor_latents,
                        critic_latents=critic_latents,
                    )

                    assert nop_training_batch is not None
                    self.assertEqual(
                        nop_training_batch.batch.actions.shape[:3],
                        (2, expected_num_origins, horizon),
                    )
                    assert nop_training_batch.actor_source_latents is not None
                    assert nop_training_batch.critic_source_latents is not None
                    actor_loss, _actor_metrics = policy.compute_actor_nop_loss_from_latents(
                        source_latents=nop_training_batch.actor_source_latents,
                        batch=nop_training_batch.batch,
                    )
                    critic_loss, _critic_metrics = policy.compute_critic_nop_loss_from_latents(
                        source_latents=nop_training_batch.critic_source_latents,
                        batch=nop_training_batch.batch,
                    )
                    assert actor_loss is not None
                    assert critic_loss is not None
                    self.assertTrue(torch.isfinite(actor_loss))
                    self.assertTrue(torch.isfinite(critic_loss))

                    (actor_loss + critic_loss).backward()
                    assert actor_latents.grad is not None
                    assert critic_latents.grad is not None
                    self.assertTrue(torch.isfinite(actor_latents.grad).all())
                    self.assertTrue(torch.isfinite(critic_latents.grad).all())
                    if expected_num_origins < segment.sequence_length:
                        self.assertTrue(
                            torch.count_nonzero(actor_latents.grad[:, expected_num_origins:]) == 0
                        )
                        self.assertTrue(
                            torch.count_nonzero(critic_latents.grad[:, expected_num_origins:]) == 0
                        )
        finally:
            env.close()

    def test_recurrent_sac_defaults_and_plain_sac_storage_regression(self) -> None:
        env = _make_env()
        try:
            encoder_config = _encoder_config(
                LSTMTemporalSequenceModel,
                LSTMTemporalSequenceModelConfig(),
            )
            recurrent_policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(encoder_config),
            )
            recurrent_sac = RecurrentSAC(
                policy=recurrent_policy,
                env=env,
                buffer_capacity_per_env=128,
                learning_starts=0,
                batch_size=2,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            self.assertEqual(recurrent_sac.burn_in_steps, 32)
            self.assertEqual(recurrent_sac.learning_steps, 64)
            self.assertEqual(recurrent_sac.replay_buffer.temporal_state_store_interval, 32)
            self.assertIsNotNone(recurrent_sac.replay_buffer._temporal_state_available)
            sampled_segment = object()
            segment_sampler = Mock(return_value=sampled_segment)
            recurrent_sac.replay_buffer.sample_episode_segments = segment_sampler
            sampled_batch, sampled_nop_batch, reuse_nop_latents = recurrent_sac._sample_training_batches()
            self.assertIs(sampled_batch, sampled_segment)
            self.assertIsNone(sampled_nop_batch)
            self.assertFalse(reuse_nop_latents)
            segment_sampler.assert_called_once_with(
                recurrent_sac.batch_size,
                segment_length=64,
                burn_in_steps=32,
                require_initial_temporal_state=True,
                allow_episode_boundaries=True,
            )
            no_nop_segment = _make_segment_batch(
                batch_size=2,
                sequence_length=2,
                n_agents=env.n_agents,
                local_obs_dim=env.local_obs_dim,
                global_obs_dim=env.global_obs_dim,
                hidden_local_vars_dim=env.hidden_local_vars_dim,
                hidden_global_vars_dim=env.hidden_global_vars_dim,
                action_dim=env.action_space.total_agent_action_dim,
            )
            with patch(
                    "swarmbots.learn.algos.sac.recurrent_sac._parallel_nop_windows",
                    side_effect=AssertionError("NOP-disabled path must not construct windows"),
            ):
                self.assertIsNone(recurrent_sac._build_nop_training_batch(
                    batch=no_nop_segment,
                    actor_latents=torch.zeros(2, 2, env.n_agents, 8),
                    critic_latents=None,
                ))

            plain_encoder_config = encoder_config
            plain_policy = TMASACPolicy(
                env=env,
                config=TMASACPolicyConfig(
                    actor_encoder_config=plain_encoder_config,
                    critic_encoder_config=plain_encoder_config,
                    actor_head_config=TMASACActorHeadConfig(hidden_dims=[8]),
                    continuous_config=PredictedStdConfig(base_std=0.5),
                ),
            )
            plain_sac = SAC(
                policy=plain_policy,
                env=env,
                buffer_capacity_per_env=128,
                learning_starts=0,
                batch_size=2,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            self.assertIsInstance(plain_policy.actor_encoder, MATEncoder)
            self.assertIsNone(plain_sac.replay_buffer._temporal_state_available)
            self.assertIsNone(plain_sac.replay_buffer.temporal_states)
        finally:
            env.close()

    def test_actor_burn_in_starts_from_the_saved_checkpoint(self) -> None:
        env = _make_env()
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(
                        LSTMTemporalSequenceModel,
                        LSTMTemporalSequenceModelConfig(),
                    ),
                ),
            )
            algorithm = RecurrentSAC(
                policy=policy,
                env=env,
                burn_in_steps=2,
                learning_steps=2,
                temporal_state_store_interval=1,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            initial_state = policy.initial_temporal_state(
                batch_size=2,
                n_agents=env.n_agents,
                device=torch.device("cpu"),
                dtype=torch.float32,
            )
            nonzero_initial_state = [
                (torch.ones_like(hidden), torch.ones_like(cell))
                for hidden, cell in initial_state
            ]
            segment = replace(
                _make_segment_batch(
                    batch_size=2,
                    sequence_length=4,
                    n_agents=env.n_agents,
                    local_obs_dim=env.local_obs_dim,
                    global_obs_dim=env.global_obs_dim,
                    hidden_local_vars_dim=env.hidden_local_vars_dim,
                    hidden_global_vars_dim=env.hidden_global_vars_dim,
                    action_dim=env.action_space.total_agent_action_dim,
                ),
                initial_temporal_state=nonzero_initial_state,
            )
            _latents, expected_state = policy.encode_actor_sequence(
                local_obs=segment.local_obs[:, :2],
                global_obs=segment.global_obs[:, :2],
                agent_mask=segment.agent_mask[:, :2],
                initial_state=nonzero_initial_state,
                time_mask=torch.ones_like(segment.train_mask[:, :2]),
                reset_mask=segment.episode_start_mask[:, :2],
            )

            actor_state, critic_state, target_critic_state = algorithm._burn_in_states(segment)

            for actual_layer, expected_layer in zip(actor_state, expected_state, strict=True):
                torch.testing.assert_close(actual_layer[0], expected_layer[0])
                torch.testing.assert_close(actual_layer[1], expected_layer[1])
            self.assertIsNone(critic_state)
            self.assertIsNone(target_critic_state)
        finally:
            env.close()

    def test_truncated_bootstrap_action_branches_from_pre_reset_actor_history(self) -> None:
        env = _make_env()
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(
                        LSTMTemporalSequenceModel,
                        LSTMTemporalSequenceModelConfig(),
                    ),
                ),
            )
            algorithm = RecurrentSAC(
                policy=policy,
                env=env,
                burn_in_steps=1,
                learning_steps=3,
                temporal_state_store_interval=1,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            batch = replace(
                _make_segment_batch(
                    batch_size=2,
                    sequence_length=3,
                    n_agents=env.n_agents,
                    local_obs_dim=env.local_obs_dim,
                    global_obs_dim=env.global_obs_dim,
                    hidden_local_vars_dim=env.hidden_local_vars_dim,
                    hidden_global_vars_dim=env.hidden_global_vars_dim,
                    action_dim=env.action_space.total_agent_action_dim,
                ),
                truncations=torch.tensor([
                    [False, True, False],
                    [False, False, False],
                ]),
                episode_start_mask=torch.tensor([
                    [False, False, True],
                    [False, False, False],
                ]),
            )
            actions_pi = torch.arange(
                2 * 3 * env.n_agents * env.action_space.total_agent_action_dim,
                dtype=torch.float32,
            ).reshape(2, 3, env.n_agents, env.action_space.total_agent_action_dim)
            log_prob_pi = torch.arange(
                2 * 3 * env.n_agents,
                dtype=torch.float32,
            ).reshape(2, 3, env.n_agents)
            last_actions = torch.full_like(actions_pi[:, 0], 90.0)
            last_log_probs = torch.full_like(log_prob_pi[:, 0], -90.0)
            branch_actions = torch.full_like(actions_pi[:1, 0], 77.0)
            branch_log_probs = torch.full_like(log_prob_pi[:1, 0], -77.0)
            initial_actor_state = torch.tensor([[10.0], [20.0]])
            next_actor_state = torch.tensor([[30.0], [40.0]])
            truncated_history_state = torch.tensor([[123.0]])

            action_log_prob_sequence = Mock(side_effect=[
                (last_actions, last_log_probs, torch.empty(0), torch.empty(0)),
                (branch_actions, branch_log_probs, torch.empty(0), torch.empty(0)),
            ])
            encode_actor_sequence = Mock(return_value=(torch.empty(0), truncated_history_state))
            with (
                    patch.object(policy, "action_log_prob_sequence", action_log_prob_sequence),
                    patch.object(policy, "encode_actor_sequence", encode_actor_sequence),
            ):
                next_actions, next_log_probs = algorithm._next_policy_actions(
                    batch=batch,
                    actions_pi=actions_pi,
                    log_prob_pi=log_prob_pi,
                    initial_actor_state=initial_actor_state,
                    next_actor_state=next_actor_state,
                )

            torch.testing.assert_close(next_actions[:, 0], actions_pi[:, 1])
            torch.testing.assert_close(next_actions[0, 1], branch_actions[0])
            torch.testing.assert_close(next_actions[1, 1], actions_pi[1, 2])
            torch.testing.assert_close(next_actions[:, 2], last_actions)
            torch.testing.assert_close(next_log_probs[:, 0], log_prob_pi[:, 1])
            torch.testing.assert_close(next_log_probs[0, 1], branch_log_probs[0])
            torch.testing.assert_close(next_log_probs[1, 1], log_prob_pi[1, 2])
            torch.testing.assert_close(next_log_probs[:, 2], last_log_probs)

            history_call = encode_actor_sequence.call_args.kwargs
            torch.testing.assert_close(history_call["local_obs"], batch.local_obs[:1, :2])
            torch.testing.assert_close(history_call["initial_state"], initial_actor_state[:1])
            torch.testing.assert_close(
                history_call["reset_mask"],
                batch.episode_start_mask[:1, :2],
            )
            branch_call = action_log_prob_sequence.call_args_list[1].kwargs
            torch.testing.assert_close(branch_call["local_obs"], batch.next_local_obs[:1, 1])
            torch.testing.assert_close(branch_call["initial_state"], truncated_history_state)
            torch.testing.assert_close(branch_call["previous_actions"], batch.actions[:1, 1])
        finally:
            env.close()

    def test_recurrent_sac_rejects_independent_nop_sampling(self) -> None:
        env = _make_env()
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(
                        LSTMTemporalSequenceModel,
                        LSTMTemporalSequenceModelConfig(),
                    )
                ),
            )
            with self.assertRaisesRegex(ValueError, "independent_nop_sampling"):
                RecurrentSAC(
                    policy=policy,
                    env=env,
                    buffer_capacity_per_env=128,
                    learning_starts=0,
                    batch_size=2,
                    independent_nop_sampling=True,
                    replay_storage_device="cpu",
                    train_device="cpu",
                )
        finally:
            env.close()

    def test_recurrent_sac_requires_checkpoint_slack_in_replay_capacity(self) -> None:
        env = _make_env()
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(
                        LSTMTemporalSequenceModel,
                        LSTMTemporalSequenceModelConfig(),
                    )
                ),
            )
            with self.assertRaisesRegex(
                    ValueError,
                    "checkpoint-anchored training segment",
            ):
                RecurrentSAC(
                    policy=policy,
                    env=env,
                    burn_in_steps=2,
                    learning_steps=3,
                    temporal_state_store_interval=4,
                    buffer_capacity_per_env=7,
                    learning_starts=0,
                    batch_size=2,
                    replay_storage_device="cpu",
                    train_device="cpu",
                )
        finally:
            env.close()

    def test_recurrent_sac_rejects_nop_horizon_longer_than_learning_sequence(self) -> None:
        env = _make_env()
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(
                        LSTMTemporalSequenceModel,
                        LSTMTemporalSequenceModelConfig(),
                    ),
                    nop_config=_small_nop_config(num_next_steps=4),
                ),
            )
            with self.assertRaisesRegex(ValueError, "num_next_steps cannot exceed learning_steps"):
                RecurrentSAC(
                    policy=policy,
                    env=env,
                    burn_in_steps=2,
                    learning_steps=3,
                    temporal_state_store_interval=1,
                    buffer_capacity_per_env=16,
                    learning_starts=0,
                    batch_size=2,
                    replay_storage_device="cpu",
                    train_device="cpu",
                )
        finally:
            env.close()

    def test_short_recurrent_sac_update_smoke(self) -> None:
        metrics, total_updates = _perform_short_recurrent_update()

        self.assertEqual(metrics["updates"], 1)
        self.assertEqual(total_updates, 1)
        self.assertIn("actor_loss", metrics)
        self.assertIn("critic_loss", metrics)

    def test_short_recurrent_sac_nop_update_smoke(self) -> None:
        metrics, _total_updates = _perform_short_recurrent_update(
            nop_config=_small_nop_config(num_next_steps=2),
        )

        self.assertEqual(metrics["updates"], 1)
        self.assertIn("actor_nop_loss", metrics)
        self.assertIn("critic_nop_loss", metrics)

    def test_short_recurrent_critic_update_smoke(self) -> None:
        metrics, total_updates = _perform_short_recurrent_update(recurrent_critic=True)

        self.assertEqual(metrics["updates"], 1)
        self.assertEqual(total_updates, 1)
        self.assertIn("actor_loss", metrics)
        self.assertIn("critic_loss", metrics)

    def test_short_recurrent_update_crosses_truncation_boundaries(self) -> None:
        metrics, total_updates = _perform_short_recurrent_update(
            recurrent_critic=True,
            nop_config=_small_nop_config(num_next_steps=2),
            max_steps=2,
        )

        self.assertEqual(metrics["updates"], 1)
        self.assertEqual(total_updates, 1)
        self.assertIn("actor_loss", metrics)
        self.assertIn("critic_loss", metrics)
        self.assertIn("actor_nop_loss", metrics)
        self.assertIn("critic_nop_loss", metrics)


if __name__ == "__main__":
    unittest.main()
