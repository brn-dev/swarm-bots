import math
import unittest
from collections.abc import Callable
from contextlib import nullcontext
from dataclasses import replace
from typing import Any
from unittest.mock import Mock, patch

import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaConfig,
)
from swarmbots.learn.action_dists.hybrid_action_dist import (
    ContinuousActionDistConfigInput,
)
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdConfig
from swarmbots.learn.action_dists.reparameterized_squashed_gaussian_mixture_action_dist import (
    ReparameterizedSquashedGaussianMixtureConfig,
)
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoderConfig
from swarmbots.learn.algos.off_policy import collect_off_policy_steps
from swarmbots.learn.algos.off_policy.replay_buffer import (
    OffPolicyReplayEpisodeSegmentBatch,
)
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.r_mat.temporal_sequence_model import (
    LSTMTemporalSequenceModel,
    LSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.sac import (
    SAC,
    ActorStateCriticInputConfig,
    RecurrentSAC,
    RecurrentTMASACPolicy,
    RecurrentTMASACPolicyConfig,
    SACNOPConfig,
    SACNOPLatentSource,
    ScenarioFieldEncoderConfig,
    ScenarioObservationSpec,
    SegmentTMASACPolicy,
    TMASACActorHeadConfig,
    TMASACActorHeadKind,
    TMASACCriticConfig,
    TMASACPolicy,
    TMASACPolicyConfig,
    TMASACScenarioEncoderConfig,
)
from swarmbots.learn.algos.sac.recurrent_sac import _flatten_segment, _slice_segment
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import RecurrentTMASACTwinCritic
from swarmbots.learn.algos.sac.tmasac_policy import TMASACTwinCritic
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.xlstm.mlstm import (
    MLSTMTemporalSequenceModel,
    MLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.xlstm.slstm import (
    SLSTMTemporalSequenceModel,
    SLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import (
    SwarmBotsLearnEnvWrapper,
)
from swarmbots.learn.env_wrappers.multi_scenario_vector_env import (
    MultiScenarioVectorEnv,
)
from swarmbots.learn.env_wrappers.torch_record_episode_statistics_wrapper import (
    TorchRecordEpisodeStatisticsWrapper,
)
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.summary_statistics import SummaryStatistics
from swarmbots.learn.temporal_state import (
    index_temporal_state_batch_time,
    stack_temporal_states,
)
from swarmbots.learn.testing_env import TestingSwarmBotsEnv

_REAL_TORCH_COMPILE = torch.compile


def _make_recording_eager_compile(
        compiled_graphs: list[torch.fx.GraphModule],
) -> Callable[..., Any]:
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
        return _REAL_TORCH_COMPILE(function, backend=eager_backend, **kwargs)

    return compile_with_eager_backend


def _summary_mean(value: object) -> float:
    assert isinstance(value, SummaryStatistics)
    assert isinstance(value.mean, float)
    return value.mean


class _DummyContinuousEnv:
    n_agents = 3
    local_obs_dim = 5
    global_obs_dim = 2
    hidden_local_vars_dim = 1
    hidden_global_vars_dim = 2
    action_space = HybridActionSpace({
        "actions": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 2), dtype=float),
    })


class _DummyScenarioEnv(_DummyContinuousEnv):
    global_obs_dim = 3
    hidden_local_vars_dim = 2
    hidden_global_vars_dim = 2
    has_scenario_id = True
    scenario_names = ("wall", "payload")
    scenario_observation_dims = {
        "wall": {
            "global_obs": 0,
            "hidden_local_vars": 1,
            "hidden_global_vars": 2,
        },
        "payload": {
            "global_obs": 3,
            "hidden_local_vars": 2,
            "hidden_global_vars": 1,
        },
    }


def _scenario_encoder_config() -> TMASACScenarioEncoderConfig:
    return TMASACScenarioEncoderConfig(
        scenarios=(
            ScenarioObservationSpec(
                scenario_id=0,
                name="wall",
                global_obs_dim=0,
                hidden_local_vars_dim=1,
                hidden_global_vars_dim=2,
            ),
            ScenarioObservationSpec(
                scenario_id=1,
                name="payload",
                global_obs_dim=3,
                hidden_local_vars_dim=2,
                hidden_global_vars_dim=1,
            ),
        ),
        global_obs=ScenarioFieldEncoderConfig(output_dim=4),
        hidden_local_vars=ScenarioFieldEncoderConfig(output_dim=3),
        hidden_global_vars=ScenarioFieldEncoderConfig(output_dim=3),
        scenario_embedding_dim=2,
    )


def _scenario_inputs(
        *,
        sequence_length: int | None,
) -> dict[str, torch.Tensor]:
    env = _DummyScenarioEnv()
    batch_size = 2
    local_prefix = (
        (batch_size, env.n_agents)
        if sequence_length is None
        else (batch_size, sequence_length, env.n_agents)
    )
    global_prefix = (
        (batch_size,)
        if sequence_length is None
        else (batch_size, sequence_length)
    )
    scenario_ids = (
        torch.tensor([0, 1])
        if sequence_length is None
        else torch.tensor([
            [step % 2 for step in range(sequence_length)],
            [(step + 1) % 2 for step in range(sequence_length)],
        ])
    )
    return {
        "local_obs": torch.randn(*local_prefix, env.local_obs_dim),
        "global_obs": torch.randn(*global_prefix, env.global_obs_dim),
        "actions": torch.randn(*local_prefix, 2),
        "hidden_local_vars": torch.randn(
            *local_prefix,
            env.hidden_local_vars_dim,
        ),
        "hidden_global_vars": torch.randn(
            *global_prefix,
            env.hidden_global_vars_dim,
        ),
        "agent_mask": torch.ones(*local_prefix, dtype=torch.bool),
        "scenario_ids": scenario_ids,
    }


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


def _actor_head_config(
        kind: TMASACActorHeadKind = TMASACActorHeadKind.INDEPENDENT,
) -> TMASACActorHeadConfig:
    return TMASACActorHeadConfig(
        kind=kind,
        hidden_dims=[8],
        qcx_decoder_config=MATQCXDecoderConfig(
            d_model=8,
            nhead=2,
            num_layers=1,
            dim_feedforward=16,
            assume_agent_mask_is_active_prefix=False,
        ),
    )


def _policy_config(
        encoder_config: RMATEncoderConfig,
        *,
        continuous_config: ContinuousActionDistConfigInput | None = None,
        recurrent_critic: bool = False,
        critic_encoder_config: MATEncoderConfig | None = None,
        nop_config: SACNOPConfig | None = None,
        separate_observation_action_encoders: bool = False,
        action_encoder_dim: int | None = None,
        scenario_encoder_config: TMASACScenarioEncoderConfig | None = None,
        actor_head_kind: TMASACActorHeadKind = TMASACActorHeadKind.INDEPENDENT,
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
        actor_head_config=_actor_head_config(actor_head_kind),
        critic_config=TMASACCriticConfig(
            n_local_projection_hidden_layers=1,
            n_value_regressor_hidden_layers=1,
            separate_observation_action_encoders=separate_observation_action_encoders,
            action_encoder_dim=action_encoder_dim,
        ),
        continuous_config=(
            PredictedStdConfig(base_std=0.5)
            if continuous_config is None
            else continuous_config
        ),
        recurrent_critic=recurrent_critic,
        nop_config=SACNOPConfig() if nop_config is None else nop_config,
        scenario_encoder_config=scenario_encoder_config,
    )


def _actor_state_policy(
        temporal_model_cls: type,
        temporal_model_config: object,
        *,
        actor_state_config: ActorStateCriticInputConfig = ActorStateCriticInputConfig(projection_dim=4),
        compile_modules: bool = False,
) -> RecurrentTMASACPolicy:
    return RecurrentTMASACPolicy(
        env=_DummyContinuousEnv(),
        config=replace(
            _policy_config(_encoder_config(temporal_model_cls, temporal_model_config)),
            actor_state_critic_input_config=actor_state_config,
            compile_modules=compile_modules,
        ),
    )


def _actor_state_critic_inputs(
        *,
        sequence_length: int | None = None,
        agent_mask: torch.Tensor | None = None,
) -> dict[str, torch.Tensor]:
    env = _DummyContinuousEnv()
    batch_size = 2
    local_prefix = (
        (batch_size, env.n_agents)
        if sequence_length is None
        else (batch_size, sequence_length, env.n_agents)
    )
    global_prefix = (batch_size,) if sequence_length is None else (batch_size, sequence_length)
    if agent_mask is None:
        agent_mask = torch.ones(*local_prefix, dtype=torch.bool)
    return {
        "local_obs": torch.randn(*local_prefix, env.local_obs_dim),
        "global_obs": torch.randn(*global_prefix, env.global_obs_dim),
        "actions": torch.randn(*local_prefix, 2),
        "hidden_local_vars": torch.randn(*local_prefix, env.hidden_local_vars_dim),
        "hidden_global_vars": torch.randn(*global_prefix, env.hidden_global_vars_dim),
        "agent_mask": agent_mask,
    }


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


def _segment_policy_config(
        *,
        nop_config: SACNOPConfig | None = None,
        ent_loss_coef: float = 0.0,
        scenario_encoder_config: TMASACScenarioEncoderConfig | None = None,
        actor_head_kind: TMASACActorHeadKind = TMASACActorHeadKind.INDEPENDENT,
) -> TMASACPolicyConfig:
    encoder_config = MATEncoderConfig(
        d_model=8,
        nhead=2,
        num_layers=1,
        dim_feedforward=16,
    )
    return TMASACPolicyConfig(
        actor_encoder_config=encoder_config,
        critic_encoder_config=encoder_config,
        actor_head_config=_actor_head_config(actor_head_kind),
        critic_config=TMASACCriticConfig(
            n_local_projection_hidden_layers=1,
            n_value_regressor_hidden_layers=1,
        ),
        continuous_config=PredictedStdConfig(
            base_std=0.5,
            ent_loss_coef=ent_loss_coef,
        ),
        nop_config=SACNOPConfig() if nop_config is None else nop_config,
        scenario_encoder_config=scenario_encoder_config,
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
        scenario_ids: torch.Tensor | None = None,
        next_scenario_ids: torch.Tensor | None = None,
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
        scenario_ids=scenario_ids,
        next_scenario_ids=next_scenario_ids,
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


def _make_recurrent_critic_algorithm(
        env: SwarmBotsLearnEnvWrapper,
) -> tuple[RecurrentTMASACPolicy, RecurrentSAC]:
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
    return policy, algorithm


def _make_recurrent_critic_batch(
        env: SwarmBotsLearnEnvWrapper,
) -> OffPolicyReplayEpisodeSegmentBatch:
    return replace(
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
        episode_start_mask=torch.tensor([
            [True, False, False],
            [False, True, False],
        ]),
    )


class _RecurrentCriticCallRecorder:
    def __init__(
            self,
            batch: OffPolicyReplayEpisodeSegmentBatch,
            *,
            history_call_first: bool,
    ) -> None:
        self.batch = batch
        self.history_call_first = history_call_first
        self.calls: list[dict[str, Any]] = []
        self.history_states = [object() for _ in range(batch.sequence_length)]
        self.branch_states = [object() for _ in range(batch.sequence_length)]

    def __call__(
            self,
            **kwargs: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, None, object]:
        call_index = len(self.calls)
        self.calls.append(kwargs)
        time_index = call_index // 2
        is_first_call = call_index % 2 == 0
        is_history_call = is_first_call == self.history_call_first
        q_value = torch.full((self.batch.actions.shape[0],), float(call_index))
        state = (
            self.history_states[time_index]
            if is_history_call
            else self.branch_states[time_index]
        )
        return q_value, q_value + 0.5, None, state


def _make_scenario_env(*, max_steps: int = 200) -> TorchRecordEpisodeStatisticsWrapper:
    def make_vector_env(
            *,
            global_obs_dim: int,
            hidden_local_vars_dim: int,
            hidden_global_vars_dim: int,
    ) -> SyncVectorEnv:
        return SyncVectorEnv(
            [
                lambda: TestingSwarmBotsEnv(
                    n_agents=2,
                    n_local_obs=4,
                    n_global_obs=global_obs_dim,
                    actuators_dim=1,
                    connectors_dim=1,
                    n_hidden_local_vars=hidden_local_vars_dim,
                    n_hidden_global_vars=hidden_global_vars_dim,
                    max_steps=max_steps,
                    continuous_connector_actions=True,
                )
            ],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )

    return TorchRecordEpisodeStatisticsWrapper(
        SwarmBotsLearnEnvWrapper(MultiScenarioVectorEnv({
            "wall": make_vector_env(
                global_obs_dim=0,
                hidden_local_vars_dim=1,
                hidden_global_vars_dim=2,
            ),
            "payload": make_vector_env(
                global_obs_dim=3,
                hidden_local_vars_dim=2,
                hidden_global_vars_dim=1,
            ),
        }))
    )


def _perform_short_recurrent_update(
        *,
        recurrent_critic: bool = False,
        nop_config: SACNOPConfig | None = None,
        max_steps: int = 20,
        compile_modules: bool = False,
        continuous_config: ContinuousActionDistConfigInput | None = None,
        use_slstm: bool = False,
        actor_state_critic_input_config: ActorStateCriticInputConfig | None = None,
        selected_state_capacities: list[int] | None = None,
        nop_parameter_updates: dict[str, bool] | None = None,
        parameter_updates: dict[str, bool] | None = None,
) -> tuple[dict[str, object], int]:
    env = _make_env(max_steps=max_steps)
    try:
        temporal_model_class = SLSTMTemporalSequenceModel if use_slstm else LSTMTemporalSequenceModel
        temporal_model_config = (
            SLSTMTemporalSequenceModelConfig(num_heads=2)
            if use_slstm
            else LSTMTemporalSequenceModelConfig()
        )
        compile_context = (
            patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=_make_recording_eager_compile([]),
            )
            if compile_modules
            else nullcontext()
        )
        with compile_context:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=replace(
                    _policy_config(
                        _encoder_config(temporal_model_class, temporal_model_config),
                        recurrent_critic=recurrent_critic,
                        nop_config=nop_config,
                        continuous_config=continuous_config,
                    ),
                    compile_modules=compile_modules,
                    actor_state_critic_input_config=actor_state_critic_input_config,
                ),
            )
            algorithm = RecurrentSAC(
                policy=policy,
                env=env,
                burn_in_steps=2,
                learning_steps=3,
                temporal_state_store_interval=1,
                max_truncations_per_segment=1,
                buffer_capacity_per_env=16,
                learning_starts=5,
                batch_size=2,
                rollout_steps_per_iteration=1,
                gradient_steps=1,
                replay_storage_device="cpu",
                train_device="cpu",
                rollout_device="cpu",
            )
            initial_nop_parameters = {}
            if nop_parameter_updates is not None:
                initial_nop_parameters = {
                    module_name: [
                        parameter.detach().clone()
                        for parameter in module.parameters()
                    ]
                    for module_name, module in (
                        ("actor", policy.actor_nop),
                        ("critic", policy.critic_nop),
                    )
                    if module is not None
                }
            initial_training_parameters = {}
            if parameter_updates is not None:
                initial_training_parameters = {
                    group_name: [parameter.detach().clone() for parameter in parameters]
                    for group_name, parameters in (
                        ("actor", policy.actor_parameters()),
                        ("critic", policy.critic_parameters()),
                    )
                }
            if selected_state_capacities is not None:
                action_with_selected_states = policy.action_log_prob_sequence_with_selected_states

                def record_selected_state_capacity(**kwargs: object) -> object:
                    state_output_indices = kwargs["state_output_indices"]
                    assert isinstance(state_output_indices, torch.Tensor)
                    selected_state_capacities.append(state_output_indices.shape[0])
                    return action_with_selected_states(**kwargs)

                policy.action_log_prob_sequence_with_selected_states = record_selected_state_capacity
            episode_return_ema = ExponentialMovingAverage(alpha=0.1)
            episode_success_rate_ema = ExponentialMovingAverage(alpha=0.1)

            metrics: dict[str, object] = {}
            for _ in range(5):
                metrics, _steps = algorithm.perform_iteration(
                    episode_return_ema,
                    episode_success_rate_ema,
                    update_ema=True,
                )
            if nop_parameter_updates is not None:
                for module_name, module in (
                    ("actor", policy.actor_nop),
                    ("critic", policy.critic_nop),
                ):
                    if module is None:
                        continue
                    nop_parameter_updates[module_name] = any(
                        not torch.equal(before, after)
                        for before, after in zip(
                            initial_nop_parameters[module_name],
                            module.parameters(),
                            strict=True,
                        )
                    )
            if parameter_updates is not None:
                for group_name, parameters in (
                    ("actor", policy.actor_parameters()),
                    ("critic", policy.critic_parameters()),
                ):
                    parameter_updates[group_name] = any(
                        not torch.equal(before, after)
                        for before, after in zip(
                            initial_training_parameters[group_name],
                            parameters,
                            strict=True,
                        )
                    )
            return metrics, algorithm.n_total_updates
    finally:
        env.close()


def _perform_short_segment_update(
        *,
        nop_config: SACNOPConfig | None = None,
        ent_loss_coef: float = 0.0,
) -> tuple[dict[str, object], int]:
    env = _make_env(max_steps=20)
    try:
        policy = SegmentTMASACPolicy(
            env=env,
            config=_segment_policy_config(
                nop_config=nop_config,
                ent_loss_coef=ent_loss_coef,
            ),
        )
        algorithm = RecurrentSAC(
            policy=policy,
            env=env,
            burn_in_steps=2,
            learning_steps=3,
            temporal_state_store_interval=1,
            max_truncations_per_segment=1,
            buffer_capacity_per_env=16,
            learning_starts=5,
            batch_size=2,
            rollout_steps_per_iteration=1,
            gradient_steps=1,
            replay_storage_device="cpu",
            train_device="cpu",
            rollout_device="cpu",
        )
        policy.encode_actor_sequence = Mock(
            side_effect=AssertionError("Feed-forward MAT must not execute burn-in."),
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


def _perform_short_scenario_update(
        *,
        recurrent_critic: bool,
        segment_policy: bool,
) -> tuple[dict[str, object], int]:
    env = _make_scenario_env(max_steps=3)
    try:
        scenario_config = _scenario_encoder_config()
        if segment_policy:
            policy = SegmentTMASACPolicy(
                env=env,
                config=_segment_policy_config(
                    scenario_encoder_config=scenario_config,
                ),
            )
        else:
            encoder_config = _encoder_config(
                LSTMTemporalSequenceModel,
                LSTMTemporalSequenceModelConfig(),
            )
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    encoder_config,
                    recurrent_critic=recurrent_critic,
                    scenario_encoder_config=scenario_config,
                ),
            )
        algorithm = RecurrentSAC(
            policy=policy,
            env=env,
            burn_in_steps=2,
            learning_steps=3,
            temporal_state_store_interval=1,
            max_truncations_per_segment=1,
            buffer_capacity_per_env=16,
            learning_starts=10,
            batch_size=2,
            rollout_steps_per_iteration=2,
            gradient_steps=1,
            replay_storage_device="cpu",
            train_device="cpu",
            rollout_device="cpu",
        )
        episode_return_ema = ExponentialMovingAverage(alpha=0.1)
        episode_success_rate_ema = ExponentialMovingAverage(alpha=0.1)
        metrics: dict[str, object] = {}
        for _ in range(8):
            iteration_metrics, _steps = algorithm.perform_iteration(
                episode_return_ema,
                episode_success_rate_ema,
                update_ema=True,
            )
            metrics.update(iteration_metrics)
        return metrics, algorithm.n_total_updates
    finally:
        env.close()


class RecurrentTMASACTests(unittest.TestCase):
    def test_recurrent_actor_supports_qcx_and_decentralized_variants(self) -> None:
        env = _DummyContinuousEnv()
        encoder_config = _encoder_config(
            LSTMTemporalSequenceModel,
            LSTMTemporalSequenceModelConfig(),
        )
        batch_size = 2
        sequence_length = 3
        local_obs = torch.randn(batch_size, sequence_length, env.n_agents, env.local_obs_dim)
        global_obs = torch.randn(batch_size, sequence_length, env.global_obs_dim)
        agent_mask = torch.tensor(
            [
                [[True, False, True], [True, True, False], [False, True, True]],
                [[True, True, True], [False, True, True], [True, False, True]],
            ],
            dtype=torch.bool,
        )
        previous_actions = torch.zeros(
            batch_size,
            sequence_length,
            env.n_agents,
            env.action_space.total_agent_action_dim,
        )

        for actor_head_kind in (TMASACActorHeadKind.QCX, TMASACActorHeadKind.DECENTRALIZED):
            with self.subTest(actor_head_kind=actor_head_kind):
                base_config = _policy_config(encoder_config)
                policy = RecurrentTMASACPolicy(
                    env=env,
                    config=replace(
                        base_config,
                        actor_head_config=_actor_head_config(actor_head_kind),
                    ),
                )

                actions, log_probs, actor_latents, _state = policy.action_log_prob_sequence(
                    local_obs=local_obs,
                    global_obs=global_obs,
                    agent_mask=agent_mask,
                    previous_actions=previous_actions,
                    deterministic=False,
                    use_rsample=True,
                    initial_state=None,
                )

                self.assertEqual(tuple(actions.shape), (batch_size, sequence_length, env.n_agents, 2))
                self.assertEqual(tuple(log_probs.shape), (batch_size, sequence_length, env.n_agents))
                self.assertEqual(tuple(actor_latents.shape), (batch_size, sequence_length, env.n_agents, 8))
                self.assertTrue(torch.isfinite(actions).all())
                self.assertTrue(torch.isfinite(log_probs).all())
                self.assertTrue(torch.equal(actions[~agent_mask], torch.zeros_like(actions[~agent_mask])))
                self.assertTrue(torch.equal(log_probs[~agent_mask], torch.zeros_like(log_probs[~agent_mask])))
                (actions.square().mean() + log_probs.square().mean()).backward()
                self.assertTrue(
                    any(
                        parameter.grad is not None
                        and torch.isfinite(parameter.grad).all()
                        and torch.count_nonzero(parameter.grad).item() > 0
                        for parameter in policy.actor_head.parameters()
                    )
                )
                if actor_head_kind is TMASACActorHeadKind.DECENTRALIZED:
                    self.assertTrue(all(layer.self_attn is None for layer in policy.actor_encoder.layers))

    def test_decentralized_recurrent_actor_does_not_mix_agent_histories(self) -> None:
        torch.manual_seed(0)
        env = _DummyContinuousEnv()
        policy = RecurrentTMASACPolicy(
            env=env,
            config=_policy_config(
                _encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                ),
                actor_head_kind=TMASACActorHeadKind.DECENTRALIZED,
            ),
        )
        batch_size = 2
        sequence_length = 4
        local_obs = torch.randn(batch_size, sequence_length, env.n_agents, env.local_obs_dim)
        changed_local_obs = local_obs.clone()
        changed_local_obs[:, :, 1:, :] += 100.0
        global_obs = torch.randn(batch_size, sequence_length, env.global_obs_dim)
        agent_mask = torch.ones(batch_size, sequence_length, env.n_agents, dtype=torch.bool)

        original_latents, _state = policy.encode_actor_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            initial_state=None,
        )
        changed_latents, _state = policy.encode_actor_sequence(
            local_obs=changed_local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            initial_state=None,
        )

        torch.testing.assert_close(original_latents[:, :, 0], changed_latents[:, :, 0])
        self.assertFalse(torch.allclose(original_latents[:, :, 1:], changed_latents[:, :, 1:]))

    def test_recurrent_actor_scenario_encoder_supports_mixed_sequences_and_gradients(self) -> None:
        policy = RecurrentTMASACPolicy(
            env=_DummyScenarioEnv(),
            config=_policy_config(
                _encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                ),
                scenario_encoder_config=_scenario_encoder_config(),
            ),
        )
        inputs = _scenario_inputs(sequence_length=4)

        actions, log_probs, actor_latents, next_state = policy.action_log_prob_sequence(
            local_obs=inputs["local_obs"],
            global_obs=inputs["global_obs"],
            agent_mask=inputs["agent_mask"],
            scenario_ids=inputs["scenario_ids"],
            previous_actions=inputs["actions"],
            deterministic=False,
            use_rsample=True,
            initial_state=None,
        )

        self.assertEqual(tuple(actions.shape), (2, 4, 3, 2))
        self.assertEqual(tuple(log_probs.shape), (2, 4, 3))
        self.assertEqual(tuple(actor_latents.shape), (2, 4, 3, 8))
        self.assertEqual(len(next_state), 1)
        actor_latents.sum().backward()
        assert policy.actor_scenario_encoder is not None
        self.assertTrue(
            any(parameter.grad is not None for parameter in policy.actor_scenario_encoder.parameters())
        )
        assert policy.critic_scenario_encoder is not None
        self.assertTrue(
            all(parameter.grad is None for parameter in policy.critic_scenario_encoder.parameters())
        )

    def test_recurrent_actor_scenario_reset_matches_fresh_state(self) -> None:
        torch.manual_seed(3)
        policy = RecurrentTMASACPolicy(
            env=_DummyScenarioEnv(),
            config=_policy_config(
                _encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                ),
                scenario_encoder_config=_scenario_encoder_config(),
            ),
        ).eval()
        inputs = _scenario_inputs(sequence_length=3)
        reset_mask = torch.zeros(2, 3, dtype=torch.bool)
        reset_mask[:, 1] = True

        sequence_latents, _state = policy.encode_actor_sequence(
            local_obs=inputs["local_obs"],
            global_obs=inputs["global_obs"],
            agent_mask=inputs["agent_mask"],
            scenario_ids=inputs["scenario_ids"],
            initial_state=None,
            reset_mask=reset_mask,
        )
        fresh_latents, _fresh_state = policy.encode_actor_sequence(
            local_obs=inputs["local_obs"][:, 1],
            global_obs=inputs["global_obs"][:, 1],
            agent_mask=inputs["agent_mask"][:, 1],
            scenario_ids=inputs["scenario_ids"][:, 1],
            initial_state=None,
        )

        torch.testing.assert_close(sequence_latents[:, 1], fresh_latents)

    def test_recurrent_critic_scenario_encoder_supports_online_and_target_sequences(self) -> None:
        policy = RecurrentTMASACPolicy(
            env=_DummyScenarioEnv(),
            config=_policy_config(
                _encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                ),
                recurrent_critic=True,
                scenario_encoder_config=_scenario_encoder_config(),
            ),
        )
        inputs = _scenario_inputs(sequence_length=4)

        q1, q2, _latents, next_state = policy.q_values_sequence(
            **inputs,
            target=False,
        )
        target_q1, target_q2, _target_latents, target_next_state = policy.q_values_sequence(
            **inputs,
            target=True,
        )

        self.assertEqual(tuple(q1.shape), (2, 4))
        self.assertEqual(tuple(q2.shape), (2, 4))
        self.assertEqual(tuple(target_q1.shape), (2, 4))
        self.assertEqual(tuple(target_q2.shape), (2, 4))
        self.assertIsNotNone(next_state)
        self.assertIsNotNone(target_next_state)
        (q1 + q2).sum().backward()
        assert policy.critic_scenario_encoder is not None
        self.assertTrue(
            any(parameter.grad is not None for parameter in policy.critic_scenario_encoder.parameters())
        )
        assert policy.critic_scenario_encoder_target is not None
        self.assertTrue(
            all(
                parameter.grad is None and not parameter.requires_grad
                for parameter in policy.critic_scenario_encoder_target.parameters()
            )
        )

    def test_actor_state_critic_supports_scenario_encoders(self) -> None:
        policy = RecurrentTMASACPolicy(
            env=_DummyScenarioEnv(),
            config=replace(
                _policy_config(
                    _encoder_config(
                        LSTMTemporalSequenceModel,
                        LSTMTemporalSequenceModelConfig(),
                    ),
                    scenario_encoder_config=_scenario_encoder_config(),
                ),
                actor_state_critic_input_config=ActorStateCriticInputConfig(projection_dim=4),
            ),
        )
        inputs = _scenario_inputs(sequence_length=None)

        q1, q2 = policy.q_values(**inputs)
        target_q1, target_q2 = policy.target_q_values(**inputs)

        self.assertEqual(tuple(q1.shape), (2,))
        self.assertEqual(tuple(q2.shape), (2,))
        self.assertEqual(tuple(target_q1.shape), (2,))
        self.assertEqual(tuple(target_q2.shape), (2,))

    def test_segment_tmasac_supports_mixed_scenario_sequences(self) -> None:
        policy = SegmentTMASACPolicy(
            env=_DummyScenarioEnv(),
            config=_segment_policy_config(
                scenario_encoder_config=_scenario_encoder_config(),
            ),
        )
        inputs = _scenario_inputs(sequence_length=4)

        actions, log_probs, actor_latents, next_state = policy.action_log_prob_sequence(
            local_obs=inputs["local_obs"],
            global_obs=inputs["global_obs"],
            agent_mask=inputs["agent_mask"],
            scenario_ids=inputs["scenario_ids"],
            previous_actions=inputs["actions"],
            deterministic=False,
            use_rsample=True,
            initial_state=None,
        )
        q1, q2, _latents, critic_state = policy.q_values_sequence(
            **inputs,
            target=False,
        )

        self.assertEqual(tuple(actions.shape), (2, 4, 3, 2))
        self.assertEqual(tuple(log_probs.shape), (2, 4, 3))
        self.assertEqual(tuple(actor_latents.shape), (2, 4, 3, 8))
        self.assertEqual(tuple(next_state.shape), (2, 1))
        self.assertEqual(tuple(q1.shape), (2, 4))
        self.assertEqual(tuple(q2.shape), (2, 4))
        self.assertIsNone(critic_state)

    def test_recurrent_batch_slice_and_flatten_preserve_scenario_ids(self) -> None:
        scenario_ids = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]])
        next_scenario_ids = 1 - scenario_ids
        batch = _make_segment_batch(
            batch_size=2,
            sequence_length=4,
            n_agents=3,
            local_obs_dim=5,
            global_obs_dim=3,
            hidden_local_vars_dim=2,
            hidden_global_vars_dim=2,
            action_dim=2,
            scenario_ids=scenario_ids,
            next_scenario_ids=next_scenario_ids,
        )

        sliced = _slice_segment(batch, 1, 4)
        flattened = _flatten_segment(sliced)

        assert sliced.scenario_ids is not None
        assert sliced.next_scenario_ids is not None
        assert flattened.scenario_ids is not None
        assert flattened.next_scenario_ids is not None
        torch.testing.assert_close(sliced.scenario_ids, scenario_ids[:, 1:])
        torch.testing.assert_close(sliced.next_scenario_ids, next_scenario_ids[:, 1:])
        torch.testing.assert_close(flattened.scenario_ids, scenario_ids[:, 1:].reshape(-1))
        torch.testing.assert_close(
            flattened.next_scenario_ids,
            next_scenario_ids[:, 1:].reshape(-1),
        )

    def test_recurrent_feature_off_preserves_direct_actor_encoder_behavior_and_schema(self) -> None:
        policy = RecurrentTMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_policy_config(
                _encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                )
            ),
        ).eval()
        batch_size, sequence_length = 2, 3
        local_obs = torch.randn(batch_size, sequence_length, 3, 5)
        global_obs = torch.randn(batch_size, sequence_length, 2)
        agent_mask = torch.ones(batch_size, sequence_length, 3, dtype=torch.bool)

        public_latents, public_state = policy.encode_actor_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            initial_state=None,
        )
        direct_latents, direct_state = policy.actor_encoder(
            local_obs,
            global_obs,
            agent_mask=agent_mask,
        )

        torch.testing.assert_close(public_latents, direct_latents)
        for public_layer_state, direct_layer_state in zip(
                public_state,
                direct_state,
                strict=True,
        ):
            torch.testing.assert_close(public_layer_state, direct_layer_state)
        self.assertFalse(any("scenario_encoder" in key for key in policy.state_dict()))
        self.assertNotIn(
            "scenario_encoder_config",
            policy.get_hyper_parameters()["tmasac_policy_config"],
        )
        self.assertFalse(any("scenario_encoder" in key for key in policy.get_grad_norms()))

    def test_recurrent_sac_updates_with_actor_recurrence_and_scenarios(self) -> None:
        metrics, updates = _perform_short_scenario_update(
            recurrent_critic=False,
            segment_policy=False,
        )

        self.assertGreater(updates, 0)
        self.assertTrue(math.isfinite(_summary_mean(metrics["actor_loss"])))
        self.assertTrue(math.isfinite(_summary_mean(metrics["critic_loss"])))
        self.assertIn("scenario/wall/ep_rew", metrics)
        self.assertIn("scenario/payload/ep_rew", metrics)

    def test_recurrent_sac_updates_with_recurrent_critic_and_scenarios(self) -> None:
        metrics, updates = _perform_short_scenario_update(
            recurrent_critic=True,
            segment_policy=False,
        )

        self.assertGreater(updates, 0)
        self.assertTrue(math.isfinite(_summary_mean(metrics["actor_loss"])))
        self.assertTrue(math.isfinite(_summary_mean(metrics["critic_loss"])))

    def test_recurrent_sac_updates_segment_tmasac_with_scenarios(self) -> None:
        metrics, updates = _perform_short_scenario_update(
            recurrent_critic=False,
            segment_policy=True,
        )

        self.assertGreater(updates, 0)
        self.assertTrue(math.isfinite(_summary_mean(metrics["actor_loss"])))
        self.assertTrue(math.isfinite(_summary_mean(metrics["critic_loss"])))

    def test_scenario_encoder_supports_feedforward_critic_sequences(self) -> None:
        policy = RecurrentTMASACPolicy(
            env=_DummyScenarioEnv(),
            config=replace(
                _policy_config(_encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                )),
                scenario_encoder_config=_scenario_encoder_config(),
            ),
        )
        inputs = _scenario_inputs(sequence_length=3)

        q1, q2, _latents, next_state = policy.q_values_sequence(
            **inputs,
            target=False,
        )

        self.assertEqual(q1.shape, (2, 3))
        self.assertEqual(q2.shape, (2, 3))
        self.assertIsNone(next_state)

    def test_recurrent_scenario_policy_requires_scenario_ids(self) -> None:
        policy = RecurrentTMASACPolicy(
            env=_DummyScenarioEnv(),
            config=replace(
                _policy_config(_encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                )),
                scenario_encoder_config=_scenario_encoder_config(),
            ),
        )
        inputs = _scenario_inputs(sequence_length=2)

        with self.assertRaisesRegex(ValueError, "scenario_ids were not provided"):
            policy.action_log_prob_sequence(
                local_obs=inputs["local_obs"],
                global_obs=inputs["global_obs"],
                agent_mask=inputs["agent_mask"],
                scenario_ids=None,
                previous_actions=None,
                deterministic=False,
                use_rsample=True,
                initial_state=None,
            )

    def test_segment_tmasac_treats_time_as_independent_batch_rows(self) -> None:
        batch_size = 2
        sequence_length = 4
        local_obs = torch.randn(batch_size, sequence_length, 3, 5)
        global_obs = torch.randn(batch_size, sequence_length, 2)
        agent_mask = torch.tensor(
            [
                [[True, True, True], [True, False, True], [False, True, True], [True, True, False]],
                [[True, False, True], [True, True, True], [True, True, False], [False, True, True]],
            ],
            dtype=torch.bool,
        )

        for actor_head_kind in TMASACActorHeadKind:
            with self.subTest(actor_head_kind=actor_head_kind):
                policy = SegmentTMASACPolicy(
                    env=_DummyContinuousEnv(),
                    config=_segment_policy_config(actor_head_kind=actor_head_kind),
                )
                sequence_actions, sequence_log_probs, _latents, next_state = (
                    policy.action_log_prob_sequence(
                        local_obs=local_obs,
                        global_obs=global_obs,
                        agent_mask=agent_mask,
                        previous_actions=None,
                        deterministic=True,
                        use_rsample=False,
                        initial_state=torch.randn(batch_size, 1),
                    )
                )
                flat_actions, flat_log_probs = policy.action_log_prob(
                    local_obs=local_obs.flatten(0, 1),
                    global_obs=global_obs.flatten(0, 1),
                    agent_mask=agent_mask.flatten(0, 1),
                    deterministic=True,
                    use_rsample=False,
                )

                torch.testing.assert_close(sequence_actions.flatten(0, 1), flat_actions)
                torch.testing.assert_close(sequence_log_probs.flatten(0, 1), flat_log_probs)
                self.assertTrue(
                    torch.equal(
                        sequence_actions[~agent_mask],
                        torch.zeros_like(sequence_actions[~agent_mask]),
                    )
                )
                self.assertTrue(
                    torch.equal(
                        sequence_log_probs[~agent_mask],
                        torch.zeros_like(sequence_log_probs[~agent_mask]),
                    )
                )
                self.assertEqual(next_state.shape, (batch_size, 1))
                if actor_head_kind is TMASACActorHeadKind.DECENTRALIZED:
                    self.assertTrue(
                        all(layer.self_attn is None for layer in policy.actor_encoder.layers)
                    )

    def test_segment_tmasac_uses_recurrent_sac_without_temporal_actor_state(self) -> None:
        metrics, total_updates = _perform_short_segment_update(ent_loss_coef=0.2)

        self.assertEqual(metrics["updates"], 1)
        self.assertEqual(total_updates, 1)
        self.assertIn("actor_loss", metrics)
        self.assertIn("critic_loss", metrics)
        self.assertIn("actor_action_dist_act0_entropy_loss_scaled", metrics)

    def test_segment_tmasac_supports_sequence_nop(self) -> None:
        metrics, total_updates = _perform_short_segment_update(
            nop_config=_small_nop_config(num_next_steps=2),
        )

        self.assertEqual(metrics["updates"], 1)
        self.assertEqual(total_updates, 1)
        self.assertIn("actor_nop_loss", metrics)
        self.assertIn("critic_nop_loss", metrics)

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

    def test_state_free_actor_and_nop_apis_are_unsupported(self) -> None:
        policy = RecurrentTMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_policy_config(
                _encoder_config(SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2))
            ),
        )

        with self.assertRaisesRegex(NotImplementedError, "state-free actor encoding"):
            policy.encode_actor(
                local_obs=torch.randn(
                    2,
                    _DummyContinuousEnv.n_agents,
                    _DummyContinuousEnv.local_obs_dim,
                ),
                global_obs=torch.randn(2, _DummyContinuousEnv.global_obs_dim),
                agent_mask=torch.ones(2, _DummyContinuousEnv.n_agents, dtype=torch.bool),
            )
        with self.assertRaisesRegex(NotImplementedError, "sequence-aware actor latents"):
            policy.compute_actor_nop_loss(Mock())

    def test_compile_modules_creates_static_actor_entry_points(self) -> None:
        config = replace(
            _policy_config(
                _encoder_config(SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2)),
                continuous_config=GumbelSoftmaxSignMagnitudeBetaConfig(),
            ),
            compile_modules=True,
        )

        with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=lambda module, **_kwargs: module,
        ) as compile_mock:
            policy = RecurrentTMASACPolicy(env=_DummyContinuousEnv(), config=config)
            policy.configure_actor_compilation(
                encoder_only_sequence_lengths=(2,),
                action_sequence_lengths=(1,),
                action_sequence_with_selected_states_lengths=(3,),
            )

        compiled_modules = {call.args[0] for call in compile_mock.call_args_list}
        self.assertIn(policy.actor_encoder, compiled_modules)
        self.assertNotIn(policy.actor_head, compiled_modules)
        self.assertIn(policy.critic, compiled_modules)
        self.assertIn(policy.critic_target, compiled_modules)
        self.assertTrue(policy.actor_end_to_end_compilation_enabled)
        self.assertEqual(policy.compiled_actor_encoder_sequence_lengths, frozenset({2}))
        self.assertEqual(policy.compiled_actor_sequence_lengths, frozenset({1}))
        self.assertEqual(policy.compiled_actor_selected_state_sequence_lengths, frozenset({3}))
        actor_compile_calls = [
            call for call in compile_mock.call_args_list
            if (
                call.args[0] is policy.actor_encoder
                or getattr(call.args[0], "__name__", "") in {
                    "_action_log_prob_sequence_impl",
                    "_action_log_prob_sequence_with_selected_states_impl",
                }
            )
        ]
        self.assertEqual(len(actor_compile_calls), 3)
        encoder_compile_call = next(
            call for call in actor_compile_calls
            if call.args[0] is policy.actor_encoder
        )
        self.assertEqual(
            encoder_compile_call.kwargs,
            {"mode": "default", "fullgraph": True, "dynamic": False},
        )
        for actor_compile_call in actor_compile_calls:
            if actor_compile_call is encoder_compile_call:
                continue
            self.assertEqual(
                actor_compile_call.kwargs,
                {"mode": "default", "fullgraph": True, "dynamic": False},
            )

    def test_full_graph_slstm_encoder_matches_eager_forward_and_gradients(self) -> None:
        compiled_graphs: list[torch.fx.GraphModule] = []

        encoder_config = _encoder_config(
            SLSTMTemporalSequenceModel,
            SLSTMTemporalSequenceModelConfig(num_heads=2),
        )
        eager_policy = RecurrentTMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_policy_config(encoder_config),
        )
        compiled_config = replace(
            _policy_config(encoder_config),
            compile_modules=True,
        )
        with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=_make_recording_eager_compile(compiled_graphs),
        ):
            compiled_policy = RecurrentTMASACPolicy(
                env=_DummyContinuousEnv(),
                config=compiled_config,
            )
            compiled_policy.configure_actor_compilation(
                encoder_only_sequence_lengths=(3,),
                action_sequence_lengths=(),
                action_sequence_with_selected_states_lengths=(),
            )

        compiled_policy.actor_encoder.load_state_dict(eager_policy.actor_encoder.state_dict())
        local_obs = torch.randn(
            2,
            3,
            _DummyContinuousEnv.n_agents,
            _DummyContinuousEnv.local_obs_dim,
        )
        global_obs = torch.randn(2, 3, _DummyContinuousEnv.global_obs_dim)
        agent_mask = torch.tensor([
            [[True, True, True], [True, False, True], [True, True, False]],
            [[True, True, False], [True, True, True], [False, True, True]],
        ])
        reset_mask = torch.tensor([
            [True, False, False],
            [False, True, False],
        ])
        state_output_indices = torch.tensor([[0, 1], [1, 2]])

        eager_outputs = eager_policy.encode_actor_sequence_with_selected_states(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            initial_state=None,
            state_output_indices=state_output_indices,
            reset_mask=reset_mask,
        )
        compiled_outputs = compiled_policy.encode_actor_sequence_with_selected_states(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            initial_state=None,
            state_output_indices=state_output_indices,
            reset_mask=reset_mask,
        )
        compiled_policy.encode_actor_sequence_with_selected_states(
            local_obs=local_obs + 0.5,
            global_obs=global_obs - 0.5,
            agent_mask=agent_mask,
            initial_state=None,
            state_output_indices=state_output_indices,
            reset_mask=reset_mask,
        )
        self.assertEqual(len(compiled_graphs), 1)

        eager_latents, eager_final_state, eager_selected_states = eager_outputs
        compiled_latents, compiled_final_state, compiled_selected_states = compiled_outputs
        torch.testing.assert_close(compiled_latents, eager_latents)
        for compiled_state, eager_state in (
                (compiled_final_state, eager_final_state),
                (compiled_selected_states, eager_selected_states),
        ):
            for compiled_layer, eager_layer in zip(compiled_state, eager_state, strict=True):
                for compiled_tensor, eager_tensor in zip(compiled_layer, eager_layer, strict=True):
                    torch.testing.assert_close(compiled_tensor, eager_tensor)

        eager_loss = eager_latents.square().mean() + sum(
            tensor.square().mean()
            for state in (eager_final_state, eager_selected_states)
            for layer in state
            for tensor in layer
        )
        compiled_loss = compiled_latents.square().mean() + sum(
            tensor.square().mean()
            for state in (compiled_final_state, compiled_selected_states)
            for layer in state
            for tensor in layer
        )
        eager_loss.backward()
        compiled_loss.backward()
        for (eager_name, eager_parameter), (compiled_name, compiled_parameter) in zip(
                eager_policy.actor_encoder.named_parameters(),
                compiled_policy.actor_encoder.named_parameters(),
                strict=True,
        ):
            self.assertEqual(compiled_name, eager_name)
            if eager_parameter.grad is None or compiled_parameter.grad is None:
                self.assertIsNone(eager_parameter.grad)
                self.assertIsNone(compiled_parameter.grad)
                continue
            torch.testing.assert_close(compiled_parameter.grad, eager_parameter.grad)

    def test_full_graph_slstm_gumbel_beta_actor_matches_eager_forward_and_gradients(self) -> None:
        compiled_graphs: list[torch.fx.GraphModule] = []
        encoder_config = _encoder_config(
            SLSTMTemporalSequenceModel,
            SLSTMTemporalSequenceModelConfig(num_heads=2),
        )
        eager_config = _policy_config(
            encoder_config,
            continuous_config=GumbelSoftmaxSignMagnitudeBetaConfig(ent_loss_coef=0.1),
        )
        eager_policy = RecurrentTMASACPolicy(
            env=_DummyContinuousEnv(),
            config=eager_config,
        )
        with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=_make_recording_eager_compile(compiled_graphs),
        ):
            compiled_policy = RecurrentTMASACPolicy(
                env=_DummyContinuousEnv(),
                config=replace(eager_config, compile_modules=True),
            )
            compiled_policy.configure_actor_compilation(
                encoder_only_sequence_lengths=(),
                action_sequence_lengths=(),
                action_sequence_with_selected_states_lengths=(3,),
            )

        compiled_policy.actor_encoder.load_state_dict(eager_policy.actor_encoder.state_dict())
        compiled_policy.actor_head.load_state_dict(eager_policy.actor_head.state_dict())
        compiled_policy.action_dist.load_state_dict(eager_policy.action_dist.state_dict())
        local_obs = torch.randn(
            2,
            3,
            _DummyContinuousEnv.n_agents,
            _DummyContinuousEnv.local_obs_dim,
        )
        global_obs = torch.randn(2, 3, _DummyContinuousEnv.global_obs_dim)
        agent_mask = torch.tensor([
            [[True, True, True], [True, False, True], [True, True, False]],
            [[True, True, False], [True, True, True], [False, True, True]],
        ])
        previous_actions = torch.randn(
            2,
            3,
            _DummyContinuousEnv.n_agents,
            _DummyContinuousEnv.action_space.total_agent_action_dim,
        ).clamp(-0.9, 0.9)
        reset_mask = torch.tensor([
            [True, False, False],
            [False, True, False],
        ])
        state_output_indices = torch.tensor([[0, 1], [1, 2]])

        call_kwargs = {
            "local_obs": local_obs,
            "global_obs": global_obs,
            "agent_mask": agent_mask,
            "previous_actions": previous_actions,
            "deterministic": False,
            "use_rsample": True,
            "initial_state": None,
            "state_output_indices": state_output_indices,
            "reset_mask": reset_mask,
        }
        torch.manual_seed(123)
        eager_outputs = eager_policy.action_log_prob_sequence_with_selected_states(**call_kwargs)
        torch.manual_seed(123)
        compiled_outputs = compiled_policy.action_log_prob_sequence_with_selected_states(**call_kwargs)

        for compiled_tensor, eager_tensor in zip(compiled_outputs[:3], eager_outputs[:3], strict=True):
            torch.testing.assert_close(compiled_tensor, eager_tensor)
        for compiled_state, eager_state in (
                (compiled_outputs[3], eager_outputs[3]),
                (compiled_outputs[4], eager_outputs[4]),
        ):
            for compiled_layer, eager_layer in zip(compiled_state, eager_state, strict=True):
                for compiled_tensor, eager_tensor in zip(compiled_layer, eager_layer, strict=True):
                    torch.testing.assert_close(compiled_tensor, eager_tensor)

        eager_extra_losses = eager_policy.action_dist.compute_extra_losses_without_metrics(
            agent_mask=agent_mask,
        )
        compiled_extra_losses = compiled_policy.action_dist.compute_extra_losses_without_metrics(
            agent_mask=agent_mask,
        )
        self.assertEqual(compiled_extra_losses.keys(), eager_extra_losses.keys())
        self.assertTrue(compiled_extra_losses)
        for name in compiled_extra_losses:
            torch.testing.assert_close(compiled_extra_losses[name], eager_extra_losses[name])

        eager_loss = (
            sum(tensor.square().mean() for tensor in eager_outputs[:3])
            + torch.stack(tuple(eager_extra_losses.values())).sum()
        )
        compiled_loss = (
            sum(tensor.square().mean() for tensor in compiled_outputs[:3])
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

        updated_call_kwargs = {
            **call_kwargs,
            "local_obs": local_obs + 0.25,
            "global_obs": global_obs - 0.25,
        }
        torch.manual_seed(456)
        eager_policy.action_log_prob_sequence_with_selected_states(**updated_call_kwargs)
        torch.manual_seed(456)
        compiled_policy.action_log_prob_sequence_with_selected_states(**updated_call_kwargs)
        updated_eager_losses = eager_policy.action_dist.compute_extra_losses_without_metrics(
            agent_mask=agent_mask,
        )
        updated_compiled_losses = compiled_policy.action_dist.compute_extra_losses_without_metrics(
            agent_mask=agent_mask,
        )
        self.assertEqual(updated_compiled_losses.keys(), updated_eager_losses.keys())
        for name in updated_compiled_losses:
            torch.testing.assert_close(updated_compiled_losses[name], updated_eager_losses[name])
        self.assertEqual(len(compiled_graphs), 1)

    def test_lstm_actor_encoder_compilation_is_opt_in(self) -> None:
        config = replace(
            _policy_config(
                _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig()),
                continuous_config=GumbelSoftmaxSignMagnitudeBetaConfig(),
            ),
            compile_modules=True,
        )

        with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=lambda module, **_kwargs: module,
        ) as compile_mock:
            policy = RecurrentTMASACPolicy(env=_DummyContinuousEnv(), config=config)

        compiled_modules = {call.args[0] for call in compile_mock.call_args_list}
        self.assertNotIn(policy.actor_encoder, compiled_modules)
        self.assertNotIn(policy.actor_head, compiled_modules)
        self.assertFalse(policy.actor_encoder_compilation_enabled)
        self.assertFalse(policy.actor_end_to_end_compilation_enabled)
        self.assertEqual(policy.compiled_actor_encoder_sequence_lengths, frozenset())
        self.assertEqual(policy.compiled_actor_sequence_lengths, frozenset())
        actor_tail_compile = next(
            call for call in compile_mock.call_args_list
            if getattr(call.args[0], "__name__", "") == "_actor_actions_and_log_probs_impl"
        )
        self.assertEqual(
            actor_tail_compile.kwargs,
            {"mode": "default", "fullgraph": True, "dynamic": False},
        )
        actor_latents, _state = policy.encode_actor_sequence(
            local_obs=torch.randn(
                2,
                3,
                _DummyContinuousEnv.n_agents,
                _DummyContinuousEnv.local_obs_dim,
            ),
            global_obs=torch.randn(2, 3, _DummyContinuousEnv.global_obs_dim),
            agent_mask=torch.ones(2, 3, _DummyContinuousEnv.n_agents, dtype=torch.bool),
            initial_state=None,
        )
        self.assertEqual(actor_latents.shape[:3], (2, 3, _DummyContinuousEnv.n_agents))

    def test_compiled_recurrent_qcx_actor_tail_matches_eager(self) -> None:
        compiled_graphs: list[torch.fx.GraphModule] = []
        encoder_config = _encoder_config(
            LSTMTemporalSequenceModel,
            LSTMTemporalSequenceModelConfig(),
        )
        eager_config = _policy_config(
            encoder_config,
            continuous_config=PredictedStdConfig(base_std=0.5, ent_loss_coef=0.1),
            actor_head_kind=TMASACActorHeadKind.QCX,
        )
        eager_policy = RecurrentTMASACPolicy(
            env=_DummyContinuousEnv(),
            config=eager_config,
        )
        with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=_make_recording_eager_compile(compiled_graphs),
        ):
            compiled_policy = RecurrentTMASACPolicy(
                env=_DummyContinuousEnv(),
                config=replace(eager_config, compile_modules=True),
            )

        compiled_policy.actor_encoder.load_state_dict(eager_policy.actor_encoder.state_dict())
        compiled_policy.actor_head.load_state_dict(eager_policy.actor_head.state_dict())
        compiled_policy.action_dist.load_state_dict(eager_policy.action_dist.state_dict())
        batch_size = 2
        sequence_length = 3
        agent_mask = torch.tensor(
            [
                [[True, False, True], [True, True, False], [False, True, True]],
                [[True, True, True], [False, True, True], [True, False, True]],
            ],
            dtype=torch.bool,
        )
        call_kwargs = {
            "local_obs": torch.randn(
                batch_size,
                sequence_length,
                _DummyContinuousEnv.n_agents,
                _DummyContinuousEnv.local_obs_dim,
            ),
            "global_obs": torch.randn(
                batch_size,
                sequence_length,
                _DummyContinuousEnv.global_obs_dim,
            ),
            "agent_mask": agent_mask,
            "previous_actions": torch.randn(
                batch_size,
                sequence_length,
                _DummyContinuousEnv.n_agents,
                _DummyContinuousEnv.action_space.total_agent_action_dim,
            ).clamp(-0.9, 0.9),
            "deterministic": False,
            "use_rsample": True,
            "initial_state": None,
        }

        torch.manual_seed(123)
        eager_outputs = eager_policy.action_log_prob_sequence(**call_kwargs)
        torch.manual_seed(123)
        compiled_outputs = compiled_policy.action_log_prob_sequence(**call_kwargs)

        for compiled_tensor, eager_tensor in zip(compiled_outputs[:3], eager_outputs[:3], strict=True):
            torch.testing.assert_close(compiled_tensor, eager_tensor)
        eager_extra_losses = eager_policy.action_dist.compute_extra_losses_without_metrics(
            agent_mask=agent_mask,
        )
        compiled_extra_losses = compiled_policy.action_dist.compute_extra_losses_without_metrics(
            agent_mask=agent_mask,
        )
        self.assertEqual(compiled_extra_losses.keys(), eager_extra_losses.keys())
        for name in compiled_extra_losses:
            torch.testing.assert_close(compiled_extra_losses[name], eager_extra_losses[name])
        self.assertEqual(len(compiled_graphs), 1)

    def test_compiled_segment_qcx_sequence_uses_autoregressive_actor_tail(self) -> None:
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

        config = _segment_policy_config(actor_head_kind=TMASACActorHeadKind.QCX)
        torch.manual_seed(123)
        eager_policy = SegmentTMASACPolicy(env=_DummyContinuousEnv(), config=config)
        torch.manual_seed(123)
        with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=compile_with_eager_backend,
        ):
            compiled_policy = SegmentTMASACPolicy(
                env=_DummyContinuousEnv(),
                config=replace(config, compile_modules=True),
            )

        self.assertIsNotNone(compiled_policy._compiled_action_log_prob)
        self.assertIsNotNone(compiled_policy._compiled_actor_actions_and_log_probs)
        self.assertIn("_action_log_prob_impl", compiled_function_names)
        self.assertIn("_actor_actions_and_log_probs_impl", compiled_function_names)

        batch_size = 2
        sequence_length = 3
        call_kwargs = {
            "local_obs": torch.randn(
                batch_size,
                sequence_length,
                _DummyContinuousEnv.n_agents,
                _DummyContinuousEnv.local_obs_dim,
            ),
            "global_obs": torch.randn(
                batch_size,
                sequence_length,
                _DummyContinuousEnv.global_obs_dim,
            ),
            "agent_mask": torch.tensor([
                [[True, True, True], [True, False, True], [True, True, False]],
                [[True, True, False], [True, True, True], [False, True, True]],
            ]),
            "previous_actions": None,
            "deterministic": True,
            "use_rsample": False,
            "initial_state": None,
        }
        eager_outputs = eager_policy.action_log_prob_sequence(**call_kwargs)
        compiled_outputs = compiled_policy.action_log_prob_sequence(**call_kwargs)

        for compiled_output, eager_output in zip(compiled_outputs, eager_outputs, strict=True):
            torch.testing.assert_close(compiled_output, eager_output)
        graph_count = len(compiled_graphs)
        self.assertGreaterEqual(graph_count, 2)

        compiled_policy.action_log_prob_sequence(**call_kwargs)
        self.assertEqual(len(compiled_graphs), graph_count)

    def test_non_compile_friendly_action_dist_keeps_distribution_outside_compiled_graph(self) -> None:
        config = replace(
            _policy_config(
                _encoder_config(SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2)),
                continuous_config=ReparameterizedSquashedGaussianMixtureConfig(),
            ),
            compile_modules=True,
        )

        with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=lambda module, **_kwargs: module,
        ) as compile_mock:
            policy = RecurrentTMASACPolicy(env=_DummyContinuousEnv(), config=config)

        compiled_modules = {call.args[0] for call in compile_mock.call_args_list}
        self.assertIn(policy.actor_encoder, compiled_modules)
        self.assertIn(policy.actor_head, compiled_modules)
        self.assertFalse(policy.actor_end_to_end_compilation_enabled)
        self.assertEqual(policy.compiled_actor_encoder_sequence_lengths, frozenset({1}))
        self.assertEqual(policy.compiled_actor_sequence_lengths, frozenset())
        self.assertEqual(policy.compiled_actor_selected_state_sequence_lengths, frozenset())

    def test_experimental_lstm_compilation_enables_allow_rnn_while_compiling_and_running_encoder(self) -> None:
        config = replace(
            _policy_config(
                _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig()),
                continuous_config=GumbelSoftmaxSignMagnitudeBetaConfig(),
            ),
            compile_modules=True,
            experimental_compile_lstm=True,
        )
        allow_rnn_values: list[bool] = []
        compile_allow_rnn_values: list[bool] = []

        def fake_compile(module: object, **_kwargs: object) -> object:
            is_recurrent_actor_callable = getattr(module, "__name__", "") in {
                "_action_log_prob_sequence_impl",
                "_action_log_prob_sequence_with_selected_states_impl",
            }
            if not isinstance(module, RMATEncoder) and not is_recurrent_actor_callable:
                return module
            compile_allow_rnn_values.append(bool(torch._dynamo.config.allow_rnn))

            def compiled_actor_callable(*args: object, **kwargs: object) -> object:
                allow_rnn_values.append(bool(torch._dynamo.config.allow_rnn))
                return module(*args, **kwargs)

            return Mock(wraps=compiled_actor_callable)

        with (
                torch._dynamo.config.patch("allow_rnn", False),
                patch(
                    "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                    side_effect=fake_compile,
                ),
        ):
            policy = RecurrentTMASACPolicy(env=_DummyContinuousEnv(), config=config)
            policy.configure_actor_compilation(
                encoder_only_sequence_lengths=(1,),
                action_sequence_lengths=(1,),
                action_sequence_with_selected_states_lengths=(3,),
            )
            policy.encode_actor_sequence(
                local_obs=torch.randn(
                    2,
                    _DummyContinuousEnv.n_agents,
                    _DummyContinuousEnv.local_obs_dim,
                ),
                global_obs=torch.randn(2, _DummyContinuousEnv.global_obs_dim),
                agent_mask=torch.ones(2, _DummyContinuousEnv.n_agents, dtype=torch.bool),
                initial_state=None,
            )
            policy.action_log_prob_sequence(
                local_obs=torch.randn(
                    2,
                    _DummyContinuousEnv.n_agents,
                    _DummyContinuousEnv.local_obs_dim,
                ),
                global_obs=torch.randn(2, _DummyContinuousEnv.global_obs_dim),
                agent_mask=torch.ones(2, _DummyContinuousEnv.n_agents, dtype=torch.bool),
                previous_actions=None,
                deterministic=True,
                use_rsample=False,
                initial_state=None,
            )
            self.assertFalse(torch._dynamo.config.allow_rnn)

        self.assertEqual(compile_allow_rnn_values, [True, True, True])
        self.assertEqual(allow_rnn_values, [True, True])

    @unittest.skipUnless(torch.cuda.is_available(), "experimental nn.LSTM compilation requires CUDA")
    def test_experimental_lstm_compiled_encoder_matches_eager_forward_and_gradients(self) -> None:
        torch._dynamo.reset()
        self.addCleanup(torch._dynamo.reset)
        # Inductor decomposes the LSTM instead of matching cuDNN's float32
        # reduction order, so compiled and eager CUDA results differ slightly.
        compiled_atol = 5e-4
        compiled_rtol = 1e-3
        eager_config = _policy_config(
            _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig())
        )
        compiled_config = replace(
            eager_config,
            compile_modules=True,
            experimental_compile_lstm=True,
        )
        eager_policy = RecurrentTMASACPolicy(
            env=_DummyContinuousEnv(),
            config=eager_config,
        ).to("cuda")
        compiled_policy = RecurrentTMASACPolicy(
            env=_DummyContinuousEnv(),
            config=compiled_config,
        ).to("cuda")
        compiled_policy.actor_encoder.load_state_dict(eager_policy.actor_encoder.state_dict())
        compiled_policy.configure_actor_compilation(
            encoder_only_sequence_lengths=(1, 3),
            action_sequence_lengths=(),
            action_sequence_with_selected_states_lengths=(),
        )

        local_obs = torch.randn(
            2,
            3,
            _DummyContinuousEnv.n_agents,
            _DummyContinuousEnv.local_obs_dim,
            device="cuda",
        )
        global_obs = torch.randn(2, 3, _DummyContinuousEnv.global_obs_dim, device="cuda")
        agent_mask = torch.ones(2, 3, _DummyContinuousEnv.n_agents, dtype=torch.bool, device="cuda")
        reset_mask = torch.tensor([[True, False, False], [False, True, False]], device="cuda")
        state_output_indices = torch.tensor([[0, 1], [1, 2]], device="cuda")

        eager_outputs = eager_policy.encode_actor_sequence_with_selected_states(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            initial_state=None,
            state_output_indices=state_output_indices,
            reset_mask=reset_mask,
        )
        compiled_outputs = compiled_policy.encode_actor_sequence_with_selected_states(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            initial_state=None,
            state_output_indices=state_output_indices,
            reset_mask=reset_mask,
        )

        eager_latents, eager_final_state, eager_selected_states = eager_outputs
        compiled_latents, compiled_final_state, compiled_selected_states = compiled_outputs
        torch.testing.assert_close(
            compiled_latents,
            eager_latents,
            atol=compiled_atol,
            rtol=compiled_rtol,
        )
        for compiled_state, eager_state in (
                (compiled_final_state, eager_final_state),
                (compiled_selected_states, eager_selected_states),
        ):
            for compiled_layer, eager_layer in zip(compiled_state, eager_state, strict=True):
                for compiled_tensor, eager_tensor in zip(compiled_layer, eager_layer, strict=True):
                    torch.testing.assert_close(
                        compiled_tensor,
                        eager_tensor,
                        atol=compiled_atol,
                        rtol=compiled_rtol,
                    )

        eager_loss = eager_latents.square().mean() + sum(
            tensor.square().mean()
            for state in (eager_final_state, eager_selected_states)
            for layer in state
            for tensor in layer
        )
        compiled_loss = compiled_latents.square().mean() + sum(
            tensor.square().mean()
            for state in (compiled_final_state, compiled_selected_states)
            for layer in state
            for tensor in layer
        )
        eager_loss.backward()
        compiled_loss.backward()
        for (eager_name, eager_parameter), (compiled_name, compiled_parameter) in zip(
                eager_policy.actor_encoder.named_parameters(),
                compiled_policy.actor_encoder.named_parameters(),
                strict=True,
        ):
            self.assertEqual(compiled_name, eager_name)
            torch.testing.assert_close(
                compiled_parameter.grad,
                eager_parameter.grad,
                atol=compiled_atol,
                rtol=compiled_rtol,
            )

    def test_actor_compilation_rejects_unconfigured_sequence_lengths(self) -> None:
        config = replace(
            _policy_config(
                _encoder_config(SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2)),
                continuous_config=GumbelSoftmaxSignMagnitudeBetaConfig(),
            ),
            compile_modules=True,
        )
        compiled_actor_encoders: list[Mock] = []
        compiled_actor_action_sequences: list[Mock] = []
        compiled_actor_action_sequences_with_selected_states: list[Mock] = []

        def fake_compile(module: object, **_kwargs: object) -> object:
            if isinstance(module, RMATEncoder):
                compiled_module = Mock(wraps=module)
                compiled_actor_encoders.append(compiled_module)
                return compiled_module
            callable_name = getattr(module, "__name__", "")
            if callable_name == "_action_log_prob_sequence_impl":
                compiled_callable = Mock(wraps=module)
                compiled_actor_action_sequences.append(compiled_callable)
                return compiled_callable
            if callable_name == "_action_log_prob_sequence_with_selected_states_impl":
                compiled_callable = Mock(wraps=module)
                compiled_actor_action_sequences_with_selected_states.append(compiled_callable)
                return compiled_callable
            return module

        with patch(
                "swarmbots.learn.algos.sac.tmasac_policy.torch.compile",
                side_effect=fake_compile,
        ):
            policy = RecurrentTMASACPolicy(env=_DummyContinuousEnv(), config=config)
            policy.configure_actor_compilation(
                encoder_only_sequence_lengths=(1,),
                action_sequence_lengths=(2,),
                action_sequence_with_selected_states_lengths=(3,),
            )

        self.assertEqual(policy.compiled_actor_sequence_lengths, frozenset({2}))
        self.assertEqual(policy.compiled_actor_encoder_sequence_lengths, frozenset({1}))
        self.assertEqual(policy.compiled_actor_selected_state_sequence_lengths, frozenset({3}))
        self.assertEqual(len(compiled_actor_encoders), 1)
        self.assertEqual(len(compiled_actor_action_sequences), 2)
        self.assertEqual(len(compiled_actor_action_sequences_with_selected_states), 1)

        batch_size = 2
        sequence_inputs = {
            sequence_length: (
                torch.randn(
                    batch_size,
                    sequence_length,
                    _DummyContinuousEnv.n_agents,
                    _DummyContinuousEnv.local_obs_dim,
                ),
                torch.randn(batch_size, sequence_length, _DummyContinuousEnv.global_obs_dim),
                torch.ones(batch_size, sequence_length, _DummyContinuousEnv.n_agents, dtype=torch.bool),
            )
            for sequence_length in (2, 4)
        }
        local_obs, global_obs, agent_mask = sequence_inputs[2]
        policy.action_log_prob_sequence(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            previous_actions=None,
            deterministic=True,
            use_rsample=False,
            initial_state=None,
        )
        local_obs, global_obs, agent_mask = (
            torch.randn(
                batch_size,
                3,
                _DummyContinuousEnv.n_agents,
                _DummyContinuousEnv.local_obs_dim,
            ),
            torch.randn(batch_size, 3, _DummyContinuousEnv.global_obs_dim),
            torch.ones(batch_size, 3, _DummyContinuousEnv.n_agents, dtype=torch.bool),
        )
        policy.action_log_prob_sequence_with_selected_states(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            previous_actions=None,
            deterministic=True,
            use_rsample=False,
            initial_state=None,
            state_output_indices=torch.tensor([[0, 1], [1, 2]]),
        )
        local_obs, global_obs, agent_mask = sequence_inputs[4]
        with self.assertRaisesRegex(RuntimeError, "No compiled actor action entry point"):
            policy.action_log_prob_sequence(
                local_obs=local_obs,
                global_obs=global_obs,
                agent_mask=agent_mask,
                previous_actions=None,
                deterministic=True,
                use_rsample=False,
                initial_state=None,
            )

        self.assertEqual(policy._compiled_actor_encoders[1].call_count, 0)
        self.assertEqual(policy._compiled_actor_action_sequences[2].call_count, 1)
        self.assertEqual(policy._compiled_actor_action_sequences_with_selected_states[3].call_count, 1)

    def test_recurrent_sac_configures_rollout_burn_in_and_learning_compile_lengths(self) -> None:
        env = _make_env()
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig())
                ),
            )
            with patch.object(
                    policy,
                    "configure_actor_compilation",
                    wraps=policy.configure_actor_compilation,
            ) as configure_mock:
                RecurrentSAC(
                    policy=policy,
                    env=env,
                    burn_in_steps=2,
                    learning_steps=3,
                    temporal_state_store_interval=1,
                    buffer_capacity_per_env=8,
                    learning_starts=0,
                    batch_size=2,
                    replay_storage_device="cpu",
                    train_device="cpu",
                    rollout_device="cpu",
                )

            configure_mock.assert_called_once_with(
                encoder_only_sequence_lengths=(2,),
                action_sequence_lengths=(1,),
                action_sequence_with_selected_states_lengths=(3,),
            )
        finally:
            env.close()

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

    def test_recurrent_critic_can_preprocess_observations_and_actions_separately(self) -> None:
        env = _DummyContinuousEnv()
        policy = RecurrentTMASACPolicy(
            env=env,
            config=_policy_config(
                _encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                ),
                recurrent_critic=True,
                separate_observation_action_encoders=True,
            ),
        )
        critic = policy.critic
        assert isinstance(critic, RecurrentTMASACTwinCritic)
        observation_action_encoder = critic.observation_action_encoder
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

        self.assertEqual(
            (observation_linear.in_features, observation_linear.out_features),
            (env.local_obs_dim + env.hidden_local_vars_dim, critic.d_model),
        )
        self.assertEqual(
            (action_linear.in_features, action_linear.out_features),
            (env.action_space.total_agent_action_dim, critic.d_model // 2),
        )
        self.assertEqual(
            critic.encoder.local_obs_dim,
            critic.d_model + critic.d_model // 2,
        )

        inputs = _actor_state_critic_inputs(sequence_length=3)
        q1, q2, _nop_latents, next_state = policy.q_values_sequence(
            **inputs,
            target=False,
        )

        self.assertEqual(q1.shape, (2, 3))
        self.assertEqual(q2.shape, (2, 3))
        self.assertIsNotNone(next_state)

    def test_independent_recurrent_critics_match_step_flow_and_concatenate_nop_latents(self) -> None:
        env = _DummyContinuousEnv()
        encoder_config = _encoder_config(
            LSTMTemporalSequenceModel,
            LSTMTemporalSequenceModelConfig(),
        )
        config = _policy_config(
            encoder_config,
            recurrent_critic=True,
            nop_config=_small_nop_config(
                latent_source=SACNOPLatentSource.CRITIC,
            ),
            separate_observation_action_encoders=True,
        )
        config = replace(
            config,
            critic_config=replace(config.critic_config, independent_encoders=True),
        )
        policy = RecurrentTMASACPolicy(env=env, config=config)
        critic = policy.critic
        assert isinstance(critic, RecurrentTMASACTwinCritic)
        self.assertIsNotNone(critic.encoder2)
        self.assertIsNotNone(critic.observation_action_encoder2)

        inputs = _actor_state_critic_inputs(sequence_length=4)
        reset_mask = torch.tensor([
            [True, False, False, True],
            [False, False, True, False],
        ])
        initial_state = policy.initial_critic_state(
            batch_size=2,
            n_agents=env.n_agents,
            device=torch.device("cpu"),
            dtype=torch.float32,
            target=False,
        )
        sequence_q1, sequence_q2, sequence_latents, sequence_state = policy.q_values_sequence(
            **inputs,
            target=False,
            initial_state=initial_state,
            reset_mask=reset_mask,
        )

        step_state = initial_state
        step_q1 = []
        step_q2 = []
        step_latents = []
        for time_idx in range(4):
            q1, q2, latents, step_state = policy.q_values_sequence(
                local_obs=inputs["local_obs"][:, time_idx],
                global_obs=inputs["global_obs"][:, time_idx],
                actions=inputs["actions"][:, time_idx],
                hidden_local_vars=inputs["hidden_local_vars"][:, time_idx],
                hidden_global_vars=inputs["hidden_global_vars"][:, time_idx],
                agent_mask=inputs["agent_mask"][:, time_idx],
                target=False,
                initial_state=step_state,
                reset_mask=reset_mask[:, time_idx],
            )
            step_q1.append(q1)
            step_q2.append(q2)
            assert latents is not None
            step_latents.append(latents)

        self.assertIsNotNone(sequence_latents)
        assert sequence_latents is not None
        self.assertEqual(sequence_latents.shape, (2, 4, env.n_agents, 16))
        torch.testing.assert_close(sequence_q1, torch.stack(step_q1, dim=1))
        torch.testing.assert_close(sequence_q2, torch.stack(step_q2, dim=1))
        torch.testing.assert_close(sequence_latents, torch.stack(step_latents, dim=1))
        self.assertIsInstance(sequence_state, tuple)
        self.assertIsInstance(step_state, tuple)

    def test_recurrent_critic_public_single_step_paths_match_the_sequence_path(self) -> None:
        env = _DummyContinuousEnv()
        encoder_config = _encoder_config(
            LSTMTemporalSequenceModel,
            LSTMTemporalSequenceModelConfig(),
        )
        for independent_encoders in (False, True):
            with self.subTest(independent_encoders=independent_encoders):
                config = _policy_config(
                    encoder_config,
                    recurrent_critic=True,
                    nop_config=_small_nop_config(latent_source=SACNOPLatentSource.CRITIC),
                    separate_observation_action_encoders=independent_encoders,
                )
                config = replace(
                    config,
                    critic_config=replace(
                        config.critic_config,
                        independent_encoders=independent_encoders,
                    ),
                )
                policy = RecurrentTMASACPolicy(env=env, config=config)
                policy.eval()
                inputs = _actor_state_critic_inputs()

                sequence_q1, sequence_q2, sequence_latents, sequence_state = policy.q_values_sequence(
                    **inputs,
                    target=False,
                )
                q1, q2 = policy.q_values(**inputs)
                nop_q1, nop_q2, nop_latents = policy.q_values_with_nop_latents(**inputs)
                encoded_latents = policy.encode_critic(**inputs)

                self.assertIsNotNone(sequence_state)
                self.assertIsNotNone(sequence_latents)
                self.assertIsNotNone(nop_latents)
                assert sequence_latents is not None
                assert nop_latents is not None
                expected_latent_dim = encoder_config.d_model * (2 if independent_encoders else 1)
                self.assertEqual(sequence_latents.shape, (2, env.n_agents, expected_latent_dim))
                torch.testing.assert_close(q1, sequence_q1)
                torch.testing.assert_close(q2, sequence_q2)
                torch.testing.assert_close(nop_q1, sequence_q1)
                torch.testing.assert_close(nop_q2, sequence_q2)
                torch.testing.assert_close(nop_latents, sequence_latents)
                torch.testing.assert_close(encoded_latents, sequence_latents)

                target_q1, target_q2 = policy.target_q_values(**inputs)
                sequence_target_q1, sequence_target_q2, target_latents, target_state = (
                    policy.q_values_sequence(**inputs, target=True)
                )
                self.assertIsNone(target_latents)
                self.assertIsNotNone(target_state)
                torch.testing.assert_close(target_q1, sequence_target_q1)
                torch.testing.assert_close(target_q2, sequence_target_q2)

    def test_single_step_action_log_prob_matches_zero_state_sequence_evaluation(self) -> None:
        policy = RecurrentTMASACPolicy(
            env=_DummyContinuousEnv(),
            config=_policy_config(
                _encoder_config(
                    LSTMTemporalSequenceModel,
                    LSTMTemporalSequenceModelConfig(),
                )
            ),
        )
        policy.eval()
        inputs = _actor_state_critic_inputs()
        actor_inputs = {
            "local_obs": inputs["local_obs"],
            "global_obs": inputs["global_obs"],
            "agent_mask": inputs["agent_mask"],
            "deterministic": True,
            "use_rsample": False,
        }

        actions, log_probs = policy.action_log_prob(**actor_inputs)
        sequence_actions, sequence_log_probs, _latents, sequence_state = policy.action_log_prob_sequence(
            **actor_inputs,
            previous_actions=None,
            initial_state=None,
        )

        self.assertIsNotNone(sequence_state)
        torch.testing.assert_close(actions, sequence_actions)
        torch.testing.assert_close(log_probs, sequence_log_probs)

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
        agent_mask = torch.tensor([
            [
                [True, True, True],
                [True, False, True],
                [True, True, False],
                [True, True, True],
            ],
            [
                [True, True, False],
                [True, True, True],
                [False, True, True],
                [True, False, True],
            ],
        ])
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
                state_output_indices = torch.tensor([[0, 0], [1, 2], [0, 3]])
                (
                    sequence_actions,
                    sequence_log_probs,
                    _latents,
                    sequence_final_state,
                    selected_states,
                    last_layer_state_sequence,
                ) = policy.action_log_prob_sequence_with_selected_states(
                    local_obs=local_obs,
                    global_obs=global_obs,
                    agent_mask=agent_mask,
                    previous_actions=None,
                    deterministic=True,
                    use_rsample=False,
                    initial_state=initial_state,
                    state_output_indices=state_output_indices,
                    reset_mask=reset_mask,
                )
                self.assertIsNone(last_layer_state_sequence)

                state = initial_state
                step_actions = []
                step_log_probs = []
                step_states = []
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
                    step_states.append(state)

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
                expected_state_sequence = stack_temporal_states(step_states, dim=1)
                expected_selected_states = index_temporal_state_batch_time(
                    expected_state_sequence,
                    state_output_indices,
                )
                for actual_layer, expected_layer in zip(
                        selected_states,
                        expected_selected_states,
                        strict=True,
                ):
                    for actual_tensor, expected_tensor in zip(actual_layer, expected_layer, strict=True):
                        self.assertEqual(
                            actual_tensor.shape[:2],
                            (state_output_indices.shape[0], env.n_agents),
                        )
                        torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-5, rtol=1e-5)
                for actual_layer, expected_layer in zip(sequence_final_state, state, strict=True):
                    for actual_tensor, expected_tensor in zip(actual_layer, expected_layer, strict=True):
                        torch.testing.assert_close(actual_tensor, expected_tensor, atol=1e-5, rtol=1e-5)

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

    def test_target_recurrent_critic_branches_from_replay_history_without_committing_branch_state(
            self,
    ) -> None:
        env = _make_env()
        try:
            policy, algorithm = _make_recurrent_critic_algorithm(env)
            batch = _make_recurrent_critic_batch(env)
            next_actions = batch.actions + 50.0
            initial_state = object()
            call_recorder = _RecurrentCriticCallRecorder(
                batch,
                history_call_first=True,
            )

            with patch.object(policy, "q_values_sequence", side_effect=call_recorder):
                q1, q2 = algorithm._target_next_q_values(
                    batch=batch,
                    next_actions=next_actions,
                    initial_state=initial_state,
                    actor_state=None,
                )

            self.assertEqual(len(call_recorder.calls), 2 * batch.sequence_length)
            expected_history_input_state = initial_state
            for time_index in range(batch.sequence_length):
                history_call = call_recorder.calls[2 * time_index]
                branch_call = call_recorder.calls[2 * time_index + 1]
                self.assertIs(history_call["initial_state"], expected_history_input_state)
                self.assertIs(
                    branch_call["initial_state"],
                    call_recorder.history_states[time_index],
                )
                torch.testing.assert_close(
                    history_call["actions"],
                    batch.actions[:, time_index],
                )
                torch.testing.assert_close(
                    branch_call["actions"],
                    next_actions[:, time_index],
                )
                torch.testing.assert_close(
                    history_call["local_obs"],
                    batch.local_obs[:, time_index],
                )
                torch.testing.assert_close(
                    branch_call["local_obs"],
                    batch.next_local_obs[:, time_index],
                )
                torch.testing.assert_close(
                    history_call["reset_mask"],
                    batch.episode_start_mask[:, time_index],
                )
                self.assertNotIn("reset_mask", branch_call)
                self.assertTrue(history_call["target"])
                self.assertTrue(branch_call["target"])
                expected_history_input_state = call_recorder.history_states[time_index]

            torch.testing.assert_close(
                q1,
                torch.tensor([[1.0, 3.0, 5.0], [1.0, 3.0, 5.0]]),
            )
            torch.testing.assert_close(
                q2,
                torch.tensor([[1.5, 3.5, 5.5], [1.5, 3.5, 5.5]]),
            )
        finally:
            env.close()

    def test_actor_recurrent_critic_evaluates_policy_actions_without_advancing_history_with_them(
            self,
    ) -> None:
        env = _make_env()
        try:
            policy, algorithm = _make_recurrent_critic_algorithm(env)
            batch = _make_recurrent_critic_batch(env)
            policy_actions = batch.actions + 50.0
            initial_state = object()
            call_recorder = _RecurrentCriticCallRecorder(
                batch,
                history_call_first=False,
            )

            with patch.object(policy, "q_values_sequence", side_effect=call_recorder):
                q1, q2 = algorithm._actor_q_values(
                    batch=batch,
                    actions_pi=policy_actions,
                    initial_state=initial_state,
                    actor_state=None,
                )

            self.assertEqual(len(call_recorder.calls), 2 * batch.sequence_length)
            expected_history_input_state = initial_state
            for time_index in range(batch.sequence_length):
                branch_call = call_recorder.calls[2 * time_index]
                history_call = call_recorder.calls[2 * time_index + 1]
                self.assertIs(branch_call["initial_state"], expected_history_input_state)
                self.assertIs(history_call["initial_state"], expected_history_input_state)
                torch.testing.assert_close(
                    branch_call["actions"],
                    policy_actions[:, time_index],
                )
                torch.testing.assert_close(
                    history_call["actions"],
                    batch.actions[:, time_index],
                )
                torch.testing.assert_close(
                    branch_call["reset_mask"],
                    batch.episode_start_mask[:, time_index],
                )
                torch.testing.assert_close(
                    history_call["reset_mask"],
                    batch.episode_start_mask[:, time_index],
                )
                self.assertFalse(branch_call["target"])
                self.assertFalse(history_call["target"])
                expected_history_input_state = call_recorder.history_states[time_index]

            torch.testing.assert_close(
                q1,
                torch.tensor([[0.0, 2.0, 4.0], [0.0, 2.0, 4.0]]),
            )
            torch.testing.assert_close(
                q2,
                torch.tensor([[0.5, 2.5, 4.5], [0.5, 2.5, 4.5]]),
            )
        finally:
            env.close()

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
                max_train_truncations=1,
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
                train_mask=torch.tensor([
                    [True, False, True, False, True],
                    [False, True, True, True, False],
                ]),
            )

            nop_training_batch = algorithm._build_nop_training_batch(
                batch=segment,
                actor_latents=torch.zeros(2, 5, env.n_agents, 8),
                critic_latents=torch.zeros(2, 5, env.n_agents, 8),
            )

            assert nop_training_batch is not None
            self.assertEqual(nop_training_batch.batch.train_mask.tolist(), [
                [[True, False], [False, False], [True, False], [False, True]],
                [[False, True], [True, False], [True, True], [True, False]],
            ])
        finally:
            env.close()

    def test_replay_terminal_observations_flow_into_cross_episode_nop_windows(self) -> None:
        env = _make_env(max_steps=2)
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(
                        LSTMTemporalSequenceModel,
                        LSTMTemporalSequenceModelConfig(),
                    ),
                    nop_config=_small_nop_config(num_next_steps=3),
                ),
            )
            algorithm = RecurrentSAC(
                policy=policy,
                env=env,
                burn_in_steps=0,
                learning_steps=3,
                temporal_state_store_interval=1,
                max_truncations_per_segment=1,
                buffer_capacity_per_env=3,
                learning_starts=0,
                batch_size=1,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            collect_off_policy_steps(
                env=env,
                replay_buffer=algorithm.replay_buffer,
                n_steps=3,
                policy=policy,
            )

            segment, nop_segment, reuse_critic_latents = algorithm._sample_training_batches()
            self.assertIsNone(nop_segment)
            self.assertFalse(reuse_critic_latents)
            self.assertEqual(
                segment.local_obs[0, :, 0, 0].tolist(),
                [0.0, 1.0, 0.0],
            )
            self.assertEqual(
                segment.next_local_obs[0, :, 0, 0].tolist(),
                [1.0, 2.0, 1.0],
            )
            self.assertEqual(segment.episode_ends.tolist(), [[False, True, False]])

            nop_training_batch = algorithm._build_nop_training_batch(
                batch=segment,
                actor_latents=torch.zeros(1, 3, env.n_agents, 8),
                critic_latents=torch.zeros(1, 3, env.n_agents, 8),
            )

            assert nop_training_batch is not None
            self.assertEqual(
                nop_training_batch.batch.next_local_obs[0, :, :, 0, 0].tolist(),
                [
                    [1.0, 2.0, 1.0],
                ],
            )
            self.assertEqual(
                nop_training_batch.batch.train_mask.tolist(),
                [[[True, True, False]]],
            )
        finally:
            env.close()

    def test_recurrent_sampling_excludes_training_segments_with_multiple_truncations(self) -> None:
        env = _make_env(max_steps=2)
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
            algorithm = RecurrentSAC(
                policy=policy,
                env=env,
                burn_in_steps=0,
                learning_steps=3,
                temporal_state_store_interval=1,
                max_truncations_per_segment=1,
                buffer_capacity_per_env=5,
                learning_starts=0,
                batch_size=2,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            collect_off_policy_steps(
                env=env,
                replay_buffer=algorithm.replay_buffer,
                n_steps=5,
                policy=policy,
            )

            segment, _nop_segment, _reuse_critic_latents = algorithm._sample_training_batches()

            self.assertTrue((segment.truncations.sum(dim=1) <= 1).all())
            self.assertEqual(
                set(segment.local_obs[:, 0, 0, 0].tolist()),
                {0.0},
            )
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
            with patch(
                    "swarmbots.learn.algos.off_policy.replay_buffer_tensor_ops.torch.compile",
                    side_effect=lambda function, **_kwargs: function,
            ) as replay_compile_mock:
                recurrent_sac = RecurrentSAC(
                    policy=recurrent_policy,
                    env=env,
                    buffer_capacity_per_env=128,
                    learning_starts=0,
                    batch_size=2,
                    replay_storage_device="cpu",
                    train_device="cpu",
                    replay_compile_tensor_operations=True,
                )
            self.assertEqual(recurrent_sac.burn_in_steps, 32)
            self.assertEqual(recurrent_sac.learning_steps, 64)
            self.assertTrue(recurrent_sac.replay_buffer.compile_tensor_operations)
            self.assertEqual(replay_compile_mock.call_count, 6)
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
                max_train_truncations=1,
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

    def test_recurrent_critic_burn_in_uses_replay_prefix_for_online_and_target_states(self) -> None:
        env = _make_env()
        try:
            policy, algorithm = _make_recurrent_critic_algorithm(env)
            segment = _make_recurrent_critic_batch(env)
            initial_state = object()
            burned_state = torch.full((2, env.n_agents, 1), 7.0)

            for target in (False, True):
                with self.subTest(target=target):
                    with (
                        patch.object(
                            policy,
                            "initial_critic_state",
                            return_value=initial_state,
                        ) as initial_state_mock,
                        patch.object(
                            policy,
                            "q_values_sequence",
                            return_value=(
                                torch.zeros(2),
                                torch.zeros(2),
                                None,
                                burned_state,
                            ),
                        ) as critic_forward_mock,
                    ):
                        actual_state = algorithm._burn_in_critic_state(
                            segment,
                            target=target,
                        )

                    initial_state_mock.assert_called_once_with(
                        batch_size=2,
                        n_agents=env.n_agents,
                        device=segment.actions.device,
                        dtype=segment.actions.dtype,
                        target=target,
                    )
                    call = critic_forward_mock.call_args.kwargs
                    torch.testing.assert_close(call["local_obs"], segment.local_obs[:, :1])
                    torch.testing.assert_close(call["global_obs"], segment.global_obs[:, :1])
                    torch.testing.assert_close(call["actions"], segment.actions[:, :1])
                    torch.testing.assert_close(
                        call["hidden_local_vars"],
                        segment.hidden_local_vars[:, :1],
                    )
                    torch.testing.assert_close(
                        call["hidden_global_vars"],
                        segment.hidden_global_vars[:, :1],
                    )
                    torch.testing.assert_close(call["agent_mask"], segment.agent_mask[:, :1])
                    torch.testing.assert_close(
                        call["reset_mask"],
                        segment.episode_start_mask[:, :1],
                    )
                    self.assertTrue(call["time_mask"].all())
                    self.assertIs(call["initial_state"], initial_state)
                    self.assertEqual(call["target"], target)
                    torch.testing.assert_close(actual_state, burned_state)
                    self.assertFalse(actual_state.requires_grad)
        finally:
            env.close()

    def test_next_policy_actions_shift_regular_steps_and_patch_sparse_truncations(self) -> None:
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
            next_hidden_state = torch.arange(
                2 * env.n_agents * 2,
                dtype=torch.float32,
            ).reshape(2, env.n_agents, 1, 2)
            next_cell_state = next_hidden_state + 100.0
            truncation_hidden_state = next_hidden_state + 200.0
            truncation_cell_state = next_hidden_state + 300.0
            next_actor_state = [(next_hidden_state, next_cell_state)]
            truncation_actor_states = [(truncation_hidden_state, truncation_cell_state)]
            truncation_indices = torch.tensor([[0, 1], [0, 0]])
            truncation_mask = torch.tensor([True, False])
            target_actions = torch.full(
                (4, env.n_agents, env.action_space.total_agent_action_dim),
                90.0,
            )
            target_log_probs = torch.full((4, env.n_agents), -90.0)
            target_actions[2] = 77.0
            target_log_probs[2] = -77.0
            current_actor_state = torch.arange(
                2 * 3 * env.n_agents * 2,
                dtype=torch.float32,
            ).reshape(2, 3, env.n_agents, 2)
            target_actor_state = torch.arange(
                4 * env.n_agents * 2,
                dtype=torch.float32,
            ).reshape(4, env.n_agents, 2) + 500.0
            target_next_actor_state = object()

            action_log_prob_sequence = Mock(return_value=(
                target_actions,
                target_log_probs,
                torch.empty(0),
                target_next_actor_state,
            ))
            actor_state_critic_input = Mock(return_value=target_actor_state)
            with (
                patch.object(policy, "action_log_prob_sequence", action_log_prob_sequence),
                patch.object(policy, "actor_state_critic_input", actor_state_critic_input),
            ):
                next_actions, next_log_probs, next_actor_state_input = algorithm._next_policy_actions(
                    batch=batch,
                    actions_pi=actions_pi,
                    log_prob_pi=log_prob_pi,
                    next_actor_state=next_actor_state,
                    truncation_actor_states=truncation_actor_states,
                    truncation_indices=truncation_indices,
                    truncation_mask=truncation_mask,
                    current_actor_state=current_actor_state,
                    return_actor_state=True,
                )

            torch.testing.assert_close(next_actions[:, 0], actions_pi[:, 1])
            torch.testing.assert_close(next_actions[0, 1], target_actions[2])
            torch.testing.assert_close(next_actions[1, 1], actions_pi[1, 2])
            torch.testing.assert_close(next_actions[:, 2], target_actions[:2])
            torch.testing.assert_close(next_log_probs[:, 0], log_prob_pi[:, 1])
            torch.testing.assert_close(next_log_probs[0, 1], target_log_probs[2])
            torch.testing.assert_close(next_log_probs[1, 1], log_prob_pi[1, 2])
            torch.testing.assert_close(next_log_probs[:, 2], target_log_probs[:2])
            assert next_actor_state_input is not None
            expected_next_actor_state_input = torch.cat((
                current_actor_state[:, 1:],
                target_actor_state[:2].unsqueeze(1),
            ), dim=1)
            expected_next_actor_state_input[0, 1] = target_actor_state[2]
            torch.testing.assert_close(
                next_actor_state_input,
                expected_next_actor_state_input,
            )
            actor_state_critic_input.assert_called_once_with(target_next_actor_state)
            action_log_prob_sequence.assert_called_once()
            target_call = action_log_prob_sequence.call_args.kwargs
            torch.testing.assert_close(
                target_call["local_obs"],
                torch.cat((
                    batch.next_local_obs[:, -1],
                    batch.next_local_obs[truncation_indices[:, 0], truncation_indices[:, 1]],
                )),
            )
            torch.testing.assert_close(
                target_call["global_obs"],
                torch.cat((
                    batch.next_global_obs[:, -1],
                    batch.next_global_obs[truncation_indices[:, 0], truncation_indices[:, 1]],
                )),
            )
            torch.testing.assert_close(
                target_call["previous_actions"],
                torch.cat((
                    batch.actions[:, -1],
                    batch.actions[truncation_indices[:, 0], truncation_indices[:, 1]],
                )),
            )
            combined_initial_state = target_call["initial_state"]
            torch.testing.assert_close(
                combined_initial_state[0][0],
                torch.cat((next_hidden_state, truncation_hidden_state)),
            )
            torch.testing.assert_close(
                combined_initial_state[0][1],
                torch.cat((next_cell_state, truncation_cell_state)),
            )
        finally:
            env.close()

    def test_recurrent_bellman_targets_bootstrap_truncations_not_terminations(self) -> None:
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
                burn_in_steps=0,
                learning_steps=2,
                temporal_state_store_interval=1,
                buffer_capacity_per_env=2,
                learning_starts=0,
                batch_size=2,
                gamma=0.5,
                ent_coef=0.0,
                max_grad_norm=None,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            batch = _make_segment_batch(
                batch_size=2,
                sequence_length=2,
                n_agents=env.n_agents,
                local_obs_dim=env.local_obs_dim,
                global_obs_dim=env.global_obs_dim,
                hidden_local_vars_dim=env.hidden_local_vars_dim,
                hidden_global_vars_dim=env.hidden_global_vars_dim,
                action_dim=env.action_space.total_agent_action_dim,
            )
            terminal_position = (0, 0)
            truncation_position = (1, 0)
            next_local_obs = batch.next_local_obs.clone()
            next_global_obs = batch.next_global_obs.clone()
            next_hidden_local_vars = batch.next_hidden_local_vars.clone()
            next_hidden_global_vars = batch.next_hidden_global_vars.clone()
            next_agent_mask = batch.next_agent_mask.clone()
            for tensor in (
                    next_local_obs,
                    next_global_obs,
                    next_hidden_local_vars,
                    next_hidden_global_vars,
            ):
                tensor[terminal_position] = torch.nan
                tensor[truncation_position] = 7.0
            next_agent_mask[terminal_position] = False
            next_agent_mask[truncation_position] = torch.tensor([True, False])
            batch = replace(
                batch,
                terminations=torch.tensor([[True, False], [False, False]]),
                truncations=torch.tensor([[False, False], [True, False]]),
                next_local_obs=next_local_obs,
                next_global_obs=next_global_obs,
                next_hidden_local_vars=next_hidden_local_vars,
                next_hidden_global_vars=next_hidden_global_vars,
                next_agent_mask=next_agent_mask,
            )

            def constant_target_q_values(
                    *,
                    batch: OffPolicyReplayEpisodeSegmentBatch,
                    **_kwargs: Any,
            ) -> tuple[torch.Tensor, torch.Tensor]:
                for tensor in (
                        batch.next_local_obs,
                        batch.next_global_obs,
                        batch.next_hidden_local_vars,
                        batch.next_hidden_global_vars,
                ):
                    self.assertTrue(torch.count_nonzero(tensor[terminal_position]) == 0)
                    self.assertTrue(torch.all(tensor[truncation_position] == 7.0))
                assert batch.next_agent_mask is not None
                self.assertTrue(batch.next_agent_mask[terminal_position].all())
                self.assertEqual(
                    batch.next_agent_mask[truncation_position].tolist(),
                    [True, False],
                )
                target_q = torch.full_like(batch.rewards, 10.0)
                return target_q, target_q

            with patch.object(
                    algorithm,
                    "_target_next_q_values",
                    side_effect=constant_target_q_values,
            ):
                metrics, _actor_grad_norm, _critic_grad_norm = algorithm._train_step(
                    batch,
                    global_update_idx=0,
                )

            self.assertAlmostEqual(metrics["target_q"], 3.75)
            self.assertTrue(math.isfinite(metrics["actor_loss"]))
            self.assertTrue(math.isfinite(metrics["critic_loss"]))
        finally:
            env.close()

    def test_truncation_state_indices_have_one_static_entry_per_batch_row(self) -> None:
        env = _make_env()
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig())
                ),
            )
            algorithm = RecurrentSAC(
                policy=policy,
                env=env,
                burn_in_steps=1,
                learning_steps=3,
                temporal_state_store_interval=1,
                max_truncations_per_segment=1,
                buffer_capacity_per_env=8,
                learning_starts=0,
                batch_size=2,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            padded_indices, valid_mask = algorithm._padded_truncation_indices(torch.tensor([
                [False, True, False],
                [False, False, False],
            ]))

            self.assertEqual(padded_indices.tolist(), [[0, 1], [1, 0]])
            self.assertEqual(valid_mask.tolist(), [True, False])
        finally:
            env.close()

    def test_recurrent_sac_rejects_non_static_truncation_capacity(self) -> None:
        env = _make_env()
        try:
            policy = RecurrentTMASACPolicy(
                env=env,
                config=_policy_config(
                    _encoder_config(LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig())
                ),
            )
            with self.assertRaisesRegex(ValueError, "must be 1"):
                RecurrentSAC(
                    policy=policy,
                    env=env,
                    burn_in_steps=1,
                    learning_steps=3,
                    temporal_state_store_interval=1,
                    max_truncations_per_segment=2,
                    buffer_capacity_per_env=8,
                    learning_starts=0,
                    batch_size=2,
                    replay_storage_device="cpu",
                    train_device="cpu",
                )
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
                    temporal_state_store_interval=3,
                    buffer_capacity_per_env=6,
                    learning_starts=0,
                    batch_size=2,
                    replay_storage_device="cpu",
                    train_device="cpu",
                )
        finally:
            env.close()

    def test_recurrent_sac_rejects_checkpoint_interval_longer_than_learning_sequence(self) -> None:
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
                    "learning_steps must be >= temporal_state_store_interval",
            ):
                RecurrentSAC(
                    policy=policy,
                    env=env,
                    burn_in_steps=2,
                    learning_steps=3,
                    temporal_state_store_interval=4,
                    buffer_capacity_per_env=16,
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
        parameter_updates: dict[str, bool] = {}
        metrics, total_updates = _perform_short_recurrent_update(
            parameter_updates=parameter_updates,
        )

        self.assertEqual(metrics["updates"], 1)
        self.assertEqual(total_updates, 1)
        self.assertIn("actor_loss", metrics)
        self.assertIn("critic_loss", metrics)
        self.assertEqual(parameter_updates, {"actor": True, "critic": True})

    def test_recurrent_update_masks_non_finite_terminal_successor_observations(self) -> None:
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
                burn_in_steps=0,
                learning_steps=3,
                temporal_state_store_interval=1,
                buffer_capacity_per_env=4,
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
            terminations = segment.terminations.clone()
            terminations[:, -1] = True
            next_local_obs = segment.next_local_obs.clone()
            next_global_obs = segment.next_global_obs.clone()
            next_hidden_local_vars = segment.next_hidden_local_vars.clone()
            next_hidden_global_vars = segment.next_hidden_global_vars.clone()
            for tensor in (
                    next_local_obs,
                    next_global_obs,
                    next_hidden_local_vars,
                    next_hidden_global_vars,
            ):
                tensor[:, -1] = torch.nan
            segment = replace(
                segment,
                terminations=terminations,
                next_local_obs=next_local_obs,
                next_global_obs=next_global_obs,
                next_hidden_local_vars=next_hidden_local_vars,
                next_hidden_global_vars=next_hidden_global_vars,
            )

            metrics, actor_grad_norm, critic_grad_norm = algorithm._train_step(
                segment,
                global_update_idx=0,
            )

            for value in (*metrics.values(), actor_grad_norm, critic_grad_norm):
                self.assertTrue(math.isfinite(value))
            for parameter in policy.parameters():
                self.assertTrue(torch.isfinite(parameter).all())
        finally:
            env.close()

    def test_recurrent_training_skips_cleanly_when_replay_has_no_state_anchored_segment(self) -> None:
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
                learning_steps=2,
                temporal_state_store_interval=1,
                buffer_capacity_per_env=3,
                learning_starts=0,
                batch_size=1,
                replay_storage_device="cpu",
                train_device="cpu",
            )
            obs, _info = env.reset()
            actions = torch.zeros(
                1,
                env.n_agents,
                env.action_space.total_agent_action_dim,
            )
            for _ in range(3):
                algorithm.replay_buffer.add(
                    obs=obs,
                    actions=actions,
                    rewards=torch.zeros(1),
                    terminations=torch.zeros(1, dtype=torch.bool),
                    truncations=torch.zeros(1, dtype=torch.bool),
                    next_obs=obs,
                )

            metrics = algorithm.train(gradient_steps=1)

            self.assertEqual(metrics["updates"], 0)
            self.assertEqual(metrics["total_updates"], 0)
            self.assertTrue(metrics["training_skipped"])
            self.assertEqual(metrics["replay_size"], 3)
        finally:
            env.close()

    def test_actor_state_critic_input_uses_last_lstm_layers_and_detaches(self) -> None:
        hidden_dim = 8
        encoder_config = replace(
            _encoder_config(
                LSTMTemporalSequenceModel,
                LSTMTemporalSequenceModelConfig(num_layers=2),
            ),
            num_layers=2,
        )
        policy = RecurrentTMASACPolicy(
            env=_DummyContinuousEnv(),
            config=replace(
                _policy_config(encoder_config),
                actor_state_critic_input_config=ActorStateCriticInputConfig(projection_dim=4),
            ),
        )
        state = policy.initial_temporal_state(
            batch_size=2,
            n_agents=policy.n_agents,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        state[0] = tuple(tensor.fill_(99.0).requires_grad_() for tensor in state[0])
        last_hidden = torch.randn(2, policy.n_agents, 2, hidden_dim, requires_grad=True)
        last_cell = torch.randn(2, policy.n_agents, 2, hidden_dim, requires_grad=True)
        state[1] = last_hidden, last_cell

        critic_input = policy.actor_state_critic_input(state)

        torch.testing.assert_close(
            critic_input,
            torch.cat((last_hidden[..., -1, :], last_cell[..., -1, :]), dim=-1),
        )
        self.assertFalse(critic_input.requires_grad)

    def test_actor_state_critic_input_builds_stable_slstm_representation(self) -> None:
        encoder_config = replace(
            _encoder_config(
                SLSTMTemporalSequenceModel,
                SLSTMTemporalSequenceModelConfig(num_heads=2),
            ),
            num_layers=2,
        )
        policy = RecurrentTMASACPolicy(
            env=_DummyContinuousEnv(),
            config=replace(
                _policy_config(encoder_config),
                actor_state_critic_input_config=ActorStateCriticInputConfig(),
            ),
        )
        state = policy.initial_temporal_state(
            batch_size=2,
            n_agents=policy.n_agents,
            device=torch.device("cpu"),
            dtype=torch.float32,
        )
        hidden = torch.randn_like(state[-1][0], requires_grad=True)
        cell = torch.randn_like(state[-1][1], requires_grad=True)
        normalizer = torch.rand_like(state[-1][2]).add_(1.0).requires_grad_()
        stabilizer = torch.randn_like(state[-1][3], requires_grad=True)
        state[-1] = hidden, cell, normalizer, stabilizer

        critic_input = policy.actor_state_critic_input(state)
        expected = torch.cat((
            hidden,
            cell / normalizer,
            torch.nn.functional.softsign(torch.log(normalizer) + stabilizer),
        ), dim=-1)

        torch.testing.assert_close(critic_input, expected)
        self.assertFalse(critic_input.requires_grad)

    def test_actor_state_critic_input_keeps_inactive_slstm_agents_and_public_critic_api_finite(self) -> None:
        agent_mask = torch.tensor([
            [True, False, True],
            [True, True, False],
        ])
        critic_inputs = _actor_state_critic_inputs(agent_mask=agent_mask)
        for include_memory_strength in (False, True):
            with self.subTest(include_memory_strength=include_memory_strength):
                policy = _actor_state_policy(
                    SLSTMTemporalSequenceModel,
                    SLSTMTemporalSequenceModelConfig(num_heads=2),
                    actor_state_config=ActorStateCriticInputConfig(
                        projection_dim=4,
                        include_slstm_memory_strength=include_memory_strength,
                    ),
                )
                initial_state = policy.initial_temporal_state(
                    batch_size=2,
                    n_agents=policy.n_agents,
                    device=torch.device("cpu"),
                    dtype=torch.float32,
                )
                initial_critic_input = policy.actor_state_critic_input(initial_state)
                self.assertTrue(torch.isfinite(initial_critic_input).all())
                torch.testing.assert_close(initial_critic_input, torch.zeros_like(initial_critic_input))

                q1, q2 = policy.q_values(**critic_inputs)
                nop_q1, nop_q2, _latents = policy.q_values_with_nop_latents(**critic_inputs)
                target_q1, target_q2 = policy.target_q_values(**critic_inputs)
                critic_latents = policy.encode_critic(**critic_inputs)

                for tensor in (q1, q2, nop_q1, nop_q2, target_q1, target_q2, critic_latents):
                    self.assertTrue(torch.isfinite(tensor).all())

    def test_state_free_actor_state_critic_api_matches_explicit_zero_state(self) -> None:
        critic_inputs = _actor_state_critic_inputs(agent_mask=torch.tensor([
            [True, False, True],
            [True, True, True],
        ]))
        temporal_configs = (
            (LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig()),
            (SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=2)),
        )
        for temporal_model_cls, temporal_model_config in temporal_configs:
            with self.subTest(temporal_model=temporal_model_cls.__name__):
                policy = _actor_state_policy(temporal_model_cls, temporal_model_config)
                _actor_latents, actor_state = policy.encode_actor_sequence(
                    local_obs=critic_inputs["local_obs"],
                    global_obs=critic_inputs["global_obs"],
                    agent_mask=critic_inputs["agent_mask"],
                    initial_state=None,
                )
                actor_state_input = policy.actor_state_critic_input(actor_state)
                explicit_q1, explicit_q2, _latents, _state = policy.q_values_sequence(
                    **critic_inputs,
                    target=False,
                    actor_state=actor_state_input,
                )
                explicit_target_q1, explicit_target_q2, _latents, _state = policy.q_values_sequence(
                    **critic_inputs,
                    target=True,
                    actor_state=actor_state_input,
                )

                q1, q2 = policy.q_values(**critic_inputs)
                target_q1, target_q2 = policy.target_q_values(**critic_inputs)

                torch.testing.assert_close(q1, explicit_q1)
                torch.testing.assert_close(q2, explicit_q2)
                torch.testing.assert_close(target_q1, explicit_target_q1)
                torch.testing.assert_close(target_q2, explicit_target_q2)

    def test_state_free_actor_state_critic_api_rejects_sequences(self) -> None:
        policy = _actor_state_policy(
            LSTMTemporalSequenceModel,
            LSTMTemporalSequenceModelConfig(),
        )
        sequence_inputs = _actor_state_critic_inputs(sequence_length=3)

        for method_name in ("q_values", "q_values_with_nop_latents", "target_q_values", "encode_critic"):
            with self.subTest(method=method_name):
                with self.assertRaisesRegex(ValueError, "use q_values_sequence with explicit actor_state"):
                    getattr(policy, method_name)(**sequence_inputs)

    def test_actor_state_critic_sequence_stays_finite_across_inactive_agents_and_resets(self) -> None:
        policy = _actor_state_policy(
            SLSTMTemporalSequenceModel,
            SLSTMTemporalSequenceModelConfig(num_heads=2),
        )
        batch_size = 2
        sequence_length = 4
        sequence_inputs = _actor_state_critic_inputs(
            sequence_length=sequence_length,
            agent_mask=torch.tensor([
                [[True, False, True], [True, True, True], [True, False, True], [True, True, True]],
                [[True, True, False], [True, False, False], [True, True, False], [True, True, True]],
            ]),
        )
        reset_mask = torch.tensor([
            [False, False, True, False],
            [False, True, False, False],
        ])
        state_output_indices = torch.tensor([[0, 1], [1, 3]])
        (
            _actions,
            _log_probs,
            _latents,
            _next_state,
            selected_states,
            last_layer_state_sequence,
        ) = (
            policy.action_log_prob_sequence_with_selected_states(
                local_obs=sequence_inputs["local_obs"],
                global_obs=sequence_inputs["global_obs"],
                agent_mask=sequence_inputs["agent_mask"],
                previous_actions=None,
                deterministic=False,
                use_rsample=True,
                initial_state=None,
                state_output_indices=state_output_indices,
                reset_mask=reset_mask,
            )
        )
        actor_state = policy.actor_last_layer_state_critic_input(
            last_layer_state_sequence,
        )
        expected_selected_last_layer_state = index_temporal_state_batch_time(
            last_layer_state_sequence,
            state_output_indices,
        )
        for actual, expected in zip(selected_states[-1], expected_selected_last_layer_state, strict=True):
            torch.testing.assert_close(actual, expected)

        q1, q2, _latents, _state = policy.q_values_sequence(
            **sequence_inputs,
            target=False,
            actor_state=actor_state,
        )
        target_q1, target_q2, _latents, _state = policy.q_values_sequence(
            **sequence_inputs,
            target=True,
            actor_state=actor_state,
        )

        for tensor in (actor_state, q1, q2, target_q1, target_q2):
            self.assertTrue(torch.isfinite(tensor).all())

    def test_compiled_actor_state_critic_public_api_handles_inactive_slstm_agents(self) -> None:
        compiled_graphs: list[torch.fx.GraphModule] = []
        with patch(
                "torch.compile",
                side_effect=_make_recording_eager_compile(compiled_graphs),
        ):
            policy = _actor_state_policy(
                SLSTMTemporalSequenceModel,
                SLSTMTemporalSequenceModelConfig(num_heads=2),
                compile_modules=True,
            )
            critic_inputs = _actor_state_critic_inputs(agent_mask=torch.tensor([
                [True, False, True],
                [True, True, False],
            ]))

            q1, q2 = policy.q_values(**critic_inputs)
            target_q1, target_q2 = policy.target_q_values(**critic_inputs)
            critic_latents = policy.encode_critic(**critic_inputs)

        self.assertGreaterEqual(len(compiled_graphs), 2)
        for tensor in (q1, q2, target_q1, target_q2, critic_latents):
            self.assertTrue(torch.isfinite(tensor).all())

    def test_actor_state_critic_input_rejects_mlstm(self) -> None:
        with self.assertRaisesRegex(TypeError, "supports only LSTMTemporalSequenceModel"):
            RecurrentTMASACPolicy(
                env=_DummyContinuousEnv(),
                config=replace(
                    _policy_config(_encoder_config(
                        MLSTMTemporalSequenceModel,
                        MLSTMTemporalSequenceModelConfig(num_heads=2),
                    )),
                    actor_state_critic_input_config=ActorStateCriticInputConfig(),
                ),
            )

    def test_short_recurrent_sac_actor_state_critic_input_updates_lstm_and_slstm(self) -> None:
        for use_slstm in (False, True):
            with self.subTest(temporal_model="sLSTM" if use_slstm else "LSTM"):
                metrics, total_updates = _perform_short_recurrent_update(
                    use_slstm=use_slstm,
                    actor_state_critic_input_config=ActorStateCriticInputConfig(
                        projection_dim=4,
                        projection_hidden_dims=(6,),
                    ),
                )

                self.assertEqual(metrics["updates"], 1)
                self.assertEqual(total_updates, 1)
                self.assertTrue(math.isfinite(_summary_mean(metrics["actor_loss"])))
                self.assertTrue(math.isfinite(_summary_mean(metrics["critic_loss"])))

    def test_actor_state_critic_input_keeps_sparse_selection_at_truncation_capacity(self) -> None:
        selected_state_capacities: list[int] = []

        _perform_short_recurrent_update(
            actor_state_critic_input_config=ActorStateCriticInputConfig(projection_dim=4),
            selected_state_capacities=selected_state_capacities,
        )

        self.assertEqual(selected_state_capacities, [2])

    def test_compiled_recurrent_sac_actor_state_critic_input_update(self) -> None:
        torch._dynamo.reset()
        try:
            metrics, total_updates = _perform_short_recurrent_update(
                compile_modules=True,
                use_slstm=True,
                actor_state_critic_input_config=ActorStateCriticInputConfig(projection_dim=4),
            )

            self.assertEqual(metrics["updates"], 1)
            self.assertEqual(total_updates, 1)
        finally:
            torch._dynamo.reset()

    def test_compiled_recurrent_sac_update_uses_post_actor_distribution_state(self) -> None:
        configurations = (
            (
                PredictedStdConfig(base_std=0.5, ent_loss_coef=0.2),
                True,
            ),
            (
                GumbelSoftmaxSignMagnitudeBetaConfig(ent_loss_coef=0.2),
                False,
            ),
        )
        for continuous_config, use_slstm in configurations:
            with self.subTest(
                    config=type(continuous_config).__name__,
                    temporal_model="sLSTM" if use_slstm else "LSTM",
            ):
                torch._dynamo.reset()
                try:
                    metrics, total_updates = _perform_short_recurrent_update(
                        compile_modules=True,
                        continuous_config=continuous_config,
                        use_slstm=use_slstm,
                    )

                    self.assertEqual(metrics["updates"], 1)
                    self.assertEqual(total_updates, 1)
                    self.assertTrue(math.isfinite(_summary_mean(metrics["actor_total_loss"])))
                    for action_idx in range(2):
                        metric_name = f"actor_action_dist_act{action_idx}_entropy_loss_scaled"
                        self.assertIn(metric_name, metrics)
                        self.assertTrue(math.isfinite(_summary_mean(metrics[metric_name])))
                finally:
                    torch._dynamo.reset()

    def test_short_recurrent_sac_nop_update_smoke(self) -> None:
        nop_parameter_updates: dict[str, bool] = {}
        metrics, _total_updates = _perform_short_recurrent_update(
            nop_config=_small_nop_config(num_next_steps=2),
            nop_parameter_updates=nop_parameter_updates,
        )

        self.assertEqual(metrics["updates"], 1)
        self.assertIn("actor_nop_loss", metrics)
        self.assertIn("critic_nop_loss", metrics)
        self.assertEqual(nop_parameter_updates, {
            "actor": True,
            "critic": True,
        })

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
