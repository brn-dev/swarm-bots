import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.algos.ppo.ppo_rollout import collect_steps
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import (
    PPOEpisodeAccumulator,
    PPOEpisodeSegment,
    PPORolloutBuffer,
)
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamplerConfig
from swarmbots.learn.algos.r_mat.r_mat_dec_policy import RMATDecPolicy, RMATDecPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.r_mat.r_mat_policy_mixin import RMATPolicyMixin
from swarmbots.learn.algos.r_mat.r_mat_qcc_policy import RMATQCCPolicy, RMATQCCPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_qcx_policy import RMATQCXPolicy, RMATQCXPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_qcs_policy import RMATQCSPolicy, RMATQCSPolicyConfig
from swarmbots.learn.algos.r_mat.r_ppo_wm_sampler import RPPOWMSamplerConfig
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as _TestingSwarmBotsEnv


def test_rmat_sequence_matches_explicit_step_state_flow() -> None:
    torch.manual_seed(0)
    encoder = RMATEncoder(
        RMATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            dropout=0.0,
        ),
        max_agents=3,
        local_obs_dim=4,
        global_obs_dim=2,
    )
    encoder.eval()

    batch_size = 2
    sequence_length = 4
    n_agents = 3
    local_obs = torch.randn(batch_size, sequence_length, n_agents, 4)
    global_obs = torch.randn(batch_size, sequence_length, 2)
    agent_mask = torch.ones(batch_size, sequence_length, n_agents, dtype=torch.bool)
    episode_start_mask = torch.tensor([
        [True, False, False, False],
        [True, False, True, False],
    ])
    initial_state = encoder.initial_state(
        batch_size=batch_size,
        n_agents=n_agents,
        device=local_obs.device,
        dtype=local_obs.dtype,
    )

    sequence_output, sequence_state = encoder(
        local_obs,
        global_obs,
        agent_mask=agent_mask,
        time_mask=torch.ones(batch_size, sequence_length, dtype=torch.bool),
        initial_state=initial_state,
        reset_mask=episode_start_mask,
    )

    step_state = initial_state
    step_outputs = []
    for time_idx in range(sequence_length):
        step_output, step_state = encoder(
            local_obs[:, time_idx],
            global_obs[:, time_idx],
            agent_mask=agent_mask[:, time_idx],
            initial_state=step_state,
            reset_mask=episode_start_mask[:, time_idx],
        )
        step_outputs.append(step_output)

    torch.testing.assert_close(sequence_output, torch.stack(step_outputs, dim=1))
    for sequence_layer_state, step_layer_state in zip(sequence_state, step_state, strict=True):
        for sequence_tensor, step_tensor in zip(sequence_layer_state, step_layer_state, strict=True):
            torch.testing.assert_close(sequence_tensor, step_tensor)


def test_rmat_policy_mixin_disables_flat_rollout_batch_sampler() -> None:
    assert not RMATPolicyMixin().supports_rollout_batch_sampler(PPOSamplerConfig(batch_size=1))


def test_rollout_accumulator_stores_segment_initial_state_and_position() -> None:
    accumulator = PPOEpisodeAccumulator(
        n_envs=2,
        max_episode_length=4,
        n_agents=1,
        agent_obs_shape=(1,),
        global_obs_shape=(1,),
        hidden_local_vars_shape=(0,),
        hidden_global_vars_shape=(0,),
        n_agent_actions=1,
        has_agent_mask=False,
        storage_device="cpu",
        storage_dtype=torch.float32,
    )

    def add_step(
            *,
            rollout_step_idx: int,
            temporal_state: torch.Tensor,
            episode_start_mask: torch.Tensor,
            dones: torch.Tensor,
    ) -> list[PPOEpisodeSegment]:
        return list(accumulator.add(
            local_obs=torch.zeros(2, 1, 1),
            global_obs=torch.zeros(2, 1),
            hidden_local_vars=torch.empty(2, 1, 0),
            hidden_global_vars=torch.empty(2, 0),
            agent_mask=None,
            actions=torch.zeros(2, 1, 1),
            rewards=torch.zeros(2),
            log_probs=torch.zeros(2, 1),
            values=torch.zeros(2),
            previous_actions=None,
            episode_start_mask=episode_start_mask,
            temporal_state=temporal_state,
            rollout_step_idx=rollout_step_idx,
            bootstrap_obs={
                "local_obs": torch.zeros(2, 1, 1),
                "global_obs": torch.zeros(2, 1),
                "hidden_local_vars": torch.empty(2, 1, 0),
                "hidden_global_vars": torch.empty(2, 0),
            },
            bootstrap_values=torch.zeros(2),
            dones=dones,
        ))

    completed = add_step(
        rollout_step_idx=5,
        temporal_state=torch.tensor([[10.0], [20.0]]),
        episode_start_mask=torch.tensor([True, False]),
        dones=torch.tensor([True, False]),
    )
    assert len(completed) == 1
    assert completed[0].rollout_env_idx == 0
    assert completed[0].rollout_start_step == 5
    torch.testing.assert_close(completed[0].initial_temporal_state, torch.tensor([[10.0]]))

    add_step(
        rollout_step_idx=6,
        temporal_state=torch.tensor([[11.0], [21.0]]),
        episode_start_mask=torch.tensor([True, False]),
        dones=torch.tensor([False, False]),
    )
    env_zero_segment = accumulator.construct_episode(
        env=0,
        final_local_obs=torch.zeros(1, 1),
        final_global_obs=torch.zeros(1),
        final_hidden_local_vars=torch.empty(1, 0),
        final_hidden_global_vars=torch.empty(0),
        final_agent_mask=None,
        final_value=torch.zeros(()),
    )
    env_one_segment = accumulator.construct_episode(
        env=1,
        final_local_obs=torch.zeros(1, 1),
        final_global_obs=torch.zeros(1),
        final_hidden_local_vars=torch.empty(1, 0),
        final_hidden_global_vars=torch.empty(0),
        final_agent_mask=None,
        final_value=torch.zeros(()),
    )

    assert env_zero_segment.rollout_start_step == 6
    torch.testing.assert_close(env_zero_segment.initial_temporal_state, torch.tensor([[11.0]]))
    assert env_one_segment.rollout_start_step == 5
    torch.testing.assert_close(env_one_segment.initial_temporal_state, torch.tensor([[20.0]]))


@pytest.mark.parametrize(
    ("policy_cls", "policy_config_cls"),
    [
        (RMATQCSPolicy, RMATQCSPolicyConfig),
        (RMATQCCPolicy, RMATQCCPolicyConfig),
        (RMATQCXPolicy, RMATQCXPolicyConfig),
        (RMATDecPolicy, RMATDecPolicyConfig),
    ],
)
def test_rmat_rollout_and_training_use_env_major_tbptt_rows(
        policy_cls: type[RMATQCSPolicy | RMATQCCPolicy | RMATQCXPolicy | RMATDecPolicy],
        policy_config_cls: type[RMATQCSPolicyConfig | RMATQCCPolicyConfig | RMATQCXPolicyConfig | RMATDecPolicyConfig],
) -> None:
    vector_env = SyncVectorEnv(
        [lambda: _TestingSwarmBotsEnv(2, 3, 2, 1, 1, max_steps=3) for _ in range(2)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    env = SwarmBotsLearnEnvWrapper(vector_env)
    try:
        policy = policy_cls(
            env,
            policy_config_cls(
                encoder_config=RMATEncoderConfig(
                    d_model=16,
                    nhead=2,
                    num_layers=1,
                    dim_feedforward=32,
                ),
                continuous_config=SquashedDiagGaussianConfig(std=0.5, std_learnable=False),
            ),
        )
        buffer = PPORolloutBuffer(
            max_episode_length=4,
            observation_space=env.observation_space,
            action_space=env.action_space,
            gamma=0.99,
            gae_lambda=0.95,
            rollout_device="cpu",
            train_device="cpu",
        )

        episodes, _, _, _ = collect_steps(
            env=env,
            policy=policy,
            buffer=buffer,
            n_steps=8,
        )
        sampler = policy.make_sampler(
            episodes,
            RPPOWMSamplerConfig(
                batch_size=2,
                sequence_length=4,
                num_next_steps=1,
            ),
        )
        batch = next(sampler.sample())
        log_probs, values, _, _ = policy.evaluate_actions(batch)

        assert batch.local_obs.shape[:2] == (2, 4)
        assert batch.initial_temporal_state[0][0].shape == (2, 2, 1, 16)
        assert torch.equal(
            batch.episode_start_mask,
            torch.tensor([
                [True, False, False, True],
                [True, False, False, True],
            ]),
        )
        assert log_probs.shape == (2, 4, 2)
        assert values.shape == (2, 4)
    finally:
        env.close()
