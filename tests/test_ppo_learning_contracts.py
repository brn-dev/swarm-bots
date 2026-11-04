from copy import deepcopy
from typing import Any

import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.algos.mat_qc_base_policy import MATCriticConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoderConfig
from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.algos.ppo.ppo_rollout import collect_step_rollout_batch, collect_steps
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPORolloutBuffer
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoderConfig
from swarmbots.learn.algos.r_mat.r_mat_ind_policy import RMATIndPolicy, RMATIndPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_qcx_policy import RMATQCXPolicy, RMATQCXPolicyConfig
from swarmbots.learn.algos.r_mat.r_ppo_wm_sampler import RPPOWMSamplerConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NextObsPredWrapper, NOPWorldModelConfig
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from tests.test_same_step_pipeline import _AgentMaskRolloutEnv, _ConstantValuePolicy, _make_scripted_env


@pytest.mark.parametrize("segmented", [False, True])
def test_collectors_match_discounted_finite_sum_gae_at_mixed_boundaries(segmented: bool) -> None:
    env = _make_scripted_env((0, (2,), "truncate"), (1, (3,), "terminate"), (2, (), "none"))
    gamma, gae_lambda = 0.995, 0.95
    rewards = torch.tensor([[1., 2., 1., 2.], [11., 12., 13., 11.], [21., 22., 23., 24.]])
    values = torch.tensor([[100., 101., 200., 201.], [1100., 1101., 1102., 1200.], [2100., 2101., 2102., 2103.]])
    successors = torch.tensor([[101., 102., 201., 202.], [1101., 1102., 0., 1201.], [2101., 2102., 2103., 2104.]])
    episode_end_steps = ((1, 3), (2,), ())
    deltas = rewards + gamma * successors - values
    expected_advantages = torch.empty_like(rewards)
    for lane, episode_ends in enumerate(episode_end_steps):
        for start in range(4):
            last = next((end for end in episode_ends if end >= start), 3)
            expected_advantages[lane, start] = sum(
                (gamma * gae_lambda) ** (step - start) * deltas[lane, step]
                for step in range(start, last + 1)
            )
    try:
        policy = _ConstantValuePolicy(action_dim=env.action_space.total_agent_action_dim)
        buffer = PPORolloutBuffer(
            max_episode_length=4, observation_space=env.observation_space, action_space=env.action_space,
            gamma=gamma, gae_lambda=gae_lambda, rollout_device="cpu", train_device="cpu",
        )
        collector = collect_steps if segmented else collect_step_rollout_batch
        rollout, _, _, _ = collector(env, policy, buffer, n_steps=12)
        if segmented:
            actual_advantages = torch.empty_like(rewards)
            actual_returns = torch.empty_like(rewards)
            for episode in rollout:
                start = episode.rollout_start_step
                end = start + len(episode.rewards)
                actual_advantages[episode.rollout_env_idx, start:end] = episode.advantages
                actual_returns[episode.rollout_env_idx, start:end] = episode.returns
        else:
            actual_advantages, actual_returns = rollout.advantages, rollout.returns
        torch.testing.assert_close(actual_advantages, expected_advantages)
        torch.testing.assert_close(actual_returns, expected_advantages + values)
    finally:
        env.close()


def _make_recurrent_nop_policy(
    env: SwarmBotsLearnEnvWrapper, *, autoregressive: bool,
) -> NextObsPredWrapper:
    options = dict(
        encoder_config=RMATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16),
        critic_encoder_config=RMATEncoderConfig(d_model=12, nhead=2, num_layers=1, dim_feedforward=24),
        critic_config=MATCriticConfig(use_popart=True, popart_config=PopArtConfig(beta=1.0)),
        continuous_config=SignMagnitudeBetaConfig(),
    )
    if autoregressive:
        policy = RMATQCXPolicy(env, RMATQCXPolicyConfig(
            **options, decoder_config=MATQCXDecoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16),
        ))
    else:
        policy = RMATIndPolicy(env, RMATIndPolicyConfig(**options))
    return NextObsPredWrapper(policy, world_model_config=NOPWorldModelConfig(
        n_agents=env.n_agents, local_latent_dim=policy.nop_latent_dim,
        action_dim=env.action_space.total_agent_action_dim, world_model_loss_coef=0.1,
        d_model_transition_model=8, nhead_transition_model=2, num_layers_transition_model=1,
        dim_feedforward_transition_model=16,
        next_obs_pred_config=NextObsPredConfig(local_scalar_target_indices=[0, 1]),
    ))


def _make_recurrent_algorithm(
    env: SwarmBotsLearnEnvWrapper, policy: BasePPOPolicy[Any, Any], *, sequence_length: int,
) -> PPO:
    return PPO(
        policy, env, learning_rate=1e-4, rollout_mode=StepsRolloutMode(12), max_episode_length=4,
        sampler_config=RPPOWMSamplerConfig(batch_size=3, sequence_length=sequence_length, num_next_steps=3),
        n_epochs=1, gamma=0.995, use_popart=True, train_device="cpu", rollout_device="cpu",
    )


@pytest.mark.parametrize("autoregressive", [False, True], ids=["independent", "qcx"])
def test_recurrent_nop_training_preserves_rollout_likelihoods_and_ignores_tbptt_padding(
    autoregressive: bool,
) -> None:
    torch.manual_seed(423)
    env = SwarmBotsLearnEnvWrapper(SyncVectorEnv([
        lambda: _AgentMaskRolloutEnv(env_id=0, done_steps=(2,), done_mode="truncate"),
        lambda: _AgentMaskRolloutEnv(env_id=1, done_steps=(3,), done_mode="terminate"),
        lambda: _AgentMaskRolloutEnv(env_id=2, done_steps=(), done_mode="none"),
    ], autoreset_mode=AutoresetMode.SAME_STEP))
    try:
        policy = _make_recurrent_nop_policy(env, autoregressive=autoregressive)
        padded_algorithm = _make_recurrent_algorithm(env, policy, sequence_length=6)
        unpadded_algorithm = _make_recurrent_algorithm(env, deepcopy(policy), sequence_length=4)
        parameter_groups = {
            "actor": list(policy.policy.actor_encoder.parameters()),
            "critic": list(policy.policy.critic_encoder.parameters()),
            "nop": list(policy.transition_model.parameters()),
        }
        state = None
        for _ in range(2):
            episodes, _, _, state = collect_steps(env, policy, padded_algorithm.rollout_buffer, 12, state)
            batch = next(policy.make_sampler(episodes, padded_algorithm.sampler_config).sample())
            assert batch.time_mask[:, :4].all() and not batch.time_mask[:, 4:].any()
            assert batch.agent_mask.any(dim=-1).all() and not batch.agent_mask.all()
            policy.train()
            log_probs, values, _, _ = policy.evaluate_actions(batch)
            valid_agents = batch.agent_mask & batch.time_mask.unsqueeze(-1)
            torch.testing.assert_close(log_probs[valid_agents], batch.log_probs[valid_agents], atol=1e-6, rtol=1e-5)
            torch.testing.assert_close(values[batch.time_mask], batch.values[batch.time_mask], atol=1e-5, rtol=1e-5)
            assert torch.count_nonzero(log_probs[~valid_agents]) == 0
            assert torch.count_nonzero(values[~batch.time_mask]) == 0

            expected_mean = batch.returns[batch.time_mask].mean().item()
            parameters_before = {
                name: [parameter.detach().clone() for parameter in parameters]
                for name, parameters in parameter_groups.items()
            }
            padded_metrics = padded_algorithm.train(episodes.copy())
            unpadded_metrics = unpadded_algorithm.train(episodes.copy())
            assert padded_metrics["updates"] == unpadded_metrics["updates"] == 1
            assert padded_metrics["popart_mu"] == pytest.approx(expected_mean)
            torch.testing.assert_close(
                padded_algorithm.policy.state_dict(), unpadded_algorithm.policy.state_dict(), atol=2e-6, rtol=1e-4,
            )
            torch.testing.assert_close(
                padded_algorithm.optimizer.state_dict()["state"], unpadded_algorithm.optimizer.state_dict()["state"],
                atol=1e-8, rtol=1e-4,
            )
            for name, parameters in parameter_groups.items():
                assert any(
                    not torch.equal(before, after)
                    for before, after in zip(parameters_before[name], parameters, strict=True)
                ), name
    finally:
        env.close()
