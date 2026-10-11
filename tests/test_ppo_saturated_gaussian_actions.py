import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.action_dists.predicted_std_gaussian_action_dist import PredictedStdGaussianConfig
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_policy import PPOActorConfig, PPOPolicy, PPOPolicyConfig
from swarmbots.learn.algos.ppo.ppo_rollout import collect_step_rollout_batch
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamplerConfig
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEIntervalResetMode
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as _TestingSwarmBotsEnv


_GAUSSIAN_CONFIGS = [
    PredictedStdGaussianConfig(base_std=0.1),
    SquashedDiagGaussianConfig(std=0.1, std_learnable=True),
    GSDEConfig(base_std=0.1),
]


def _set_gaussian_means(action_dist: HybridActionDistribution, mean: float) -> None:
    with torch.no_grad():
        for dist in action_dist.distributions:
            dist.action_net.weight.zero_()
            dist.action_net.bias.fill_(mean)
            if hasattr(dist, "log_std_net"):
                dist.log_std_net.weight.zero_()
                dist.log_std_net.bias.zero_()


def _make_env() -> SwarmBotsLearnEnvWrapper:
    return SwarmBotsLearnEnvWrapper(
        SyncVectorEnv(
            [
                lambda: _TestingSwarmBotsEnv(
                    n_agents=2,
                    n_local_obs=2,
                    n_global_obs=1,
                    actuators_dim=1,
                    connectors_dim=1,
                    continuous_connector_actions=True,
                )
            ],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
    )


@pytest.mark.parametrize("config", _GAUSSIAN_CONFIGS, ids=lambda config: type(config).__name__)
@pytest.mark.parametrize("mean", [-10.0, 10.0])
def test_saturated_gaussian_ppo_rollout_has_unit_ratio_and_trains(config, mean: float) -> None:
    torch.manual_seed(123)
    env = _make_env()
    try:
        policy = PPOPolicy(
            env,
            PPOPolicyConfig(
                actor_config=PPOActorConfig(latent_pi_dim_per_agent=8),
                continuous_config=config,
            ),
        )
        _set_gaussian_means(policy.action_dist, mean)
        algorithm = PPO(
            policy,
            env,
            rollout_mode=StepsRolloutMode(2),
            sampler_config=PPOSamplerConfig(batch_size=2),
            n_epochs=1,
            target_kl=0.01,
            gsde_reset_mode=GSDEIntervalResetMode(interval=1),
            train_device="cpu",
            rollout_device="cpu",
        )
        rollout, _infos, _metrics, _state = collect_step_rollout_batch(
            env, policy, algorithm.rollout_buffer, n_steps=2,
            gsde_reset_mode=algorithm.gsde_reset_mode,
        )
        assert torch.equal(rollout.actions, torch.full_like(rollout.actions, 1.0 if mean > 0 else -1.0))
        policy.train()
        sampler = policy.make_rollout_batch_sampler(rollout, algorithm.sampler_config)
        batch = next(iter(sampler.sample()))
        log_probs, values, _losses, _metrics = policy.evaluate_actions(batch)

        assert torch.isfinite(log_probs).all()
        torch.testing.assert_close(log_probs, batch.log_probs, rtol=0, atol=0)
        torch.testing.assert_close(torch.exp(log_probs - batch.log_probs), torch.ones_like(log_probs))
        _loss, approximate_kl, metrics = algorithm.compute_ppo_loss(batch, log_probs, values)
        assert approximate_kl == 0.0
        assert metrics["clip_frac"] == 0.0
        assert algorithm.train(rollout)["updates"] == 1
    finally:
        env.close()


@pytest.mark.parametrize("config", _GAUSSIAN_CONFIGS, ids=lambda config: type(config).__name__)
def test_sac_gaussian_sampling_retains_cached_likelihood_and_gradients(config) -> None:
    env = _make_env()
    try:
        action_dist = HybridActionDistribution(4, env.action_space, config)
        _set_gaussian_means(action_dist, 10.0)
        action_dist.reset_temporal_correlations_on_step(batch_shape=(1, 2))
        actions, log_probs = action_dist.get_actions_with_log_probs(torch.ones(1, 2, 4), use_rsample=True)

        assert torch.equal(actions, torch.ones_like(actions))
        cached_log_probs = torch.stack(
            [
                dist.log_prob(action, gaussian_actions=dist._last_gaussian_actions)
                for dist, action in zip(action_dist.distributions, actions.split(action_dist.action_dims, dim=-1), strict=True)
            ],
            dim=-1,
        ).sum(dim=-1)
        torch.testing.assert_close(log_probs, cached_log_probs)
        assert not torch.allclose(log_probs, action_dist.log_prob(actions))
        log_probs.sum().backward()
        for dist in action_dist.distributions:
            std_parameter = dist.log_std_net.bias if hasattr(dist, "log_std_net") else dist.log_stds
            assert std_parameter.grad is not None
            assert torch.isfinite(std_parameter.grad).all()
            assert std_parameter.grad.abs().sum() > 0
    finally:
        env.close()
