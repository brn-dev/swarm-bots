import math
from unittest.mock import patch

import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.rational_quadratic_spline_quantile_action_dist import (
    RationalQuadraticSplineQuantileActionDist,
    RationalQuadraticSplineQuantileConfig,
)
from swarmbots.learn.action_dists.sticky_sign_magnitude_beta_action_dist import StickySignMagnitudeBetaConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActorConfig
from swarmbots.learn.algos.mappo.mappo_policy import MAPPOCriticConfig, MAPPOPolicy, MAPPOPolicyConfig
from swarmbots.learn.algos.ppo.ppo_policy import PPOActorConfig, PPOCriticConfig, PPOPolicy, PPOPolicyConfig
from swarmbots.learn.nn_components.deep_set import DeepSetCriticConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace


class _DummyLearnEnv:
    n_agents = 4
    local_obs_dim = 6
    global_obs_dim = 3
    hidden_local_vars_dim = 2
    hidden_global_vars_dim = 5
    action_space = HybridActionSpace(
        {
            "cont": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 2), dtype=float),
            "disc": spaces.MultiBinary((n_agents, 1)),
        }
    )


def _linear_modules(module: nn.Module) -> list[nn.Linear]:
    return [submodule for submodule in module.modules() if isinstance(submodule, nn.Linear)]


def _assert_orthogonal_gain(linear: nn.Linear, gain: float) -> None:
    expected_norm = gain * math.sqrt(min(linear.weight.shape))
    torch.testing.assert_close(
        torch.linalg.vector_norm(linear.weight),
        torch.tensor(expected_norm, device=linear.weight.device, dtype=linear.weight.dtype),
        rtol=1e-5,
        atol=1e-6,
    )


def test_ppo_policy_initializes_internal_layers_and_heads_with_expected_gains() -> None:
    torch.manual_seed(0)
    policy = PPOPolicy(
        env=_DummyLearnEnv(),
        config=PPOPolicyConfig(
            actor_config=PPOActorConfig(
                hidden_dims=[12],
                shared_encoder_latent_dim_per_agent=8,
                actor_head_hidden_dims=[6],
                latent_pi_dim_per_agent=4,
            ),
            critic_config=PPOCriticConfig(
                hidden_dims=[10],
            ),
            continuous_config=StickySignMagnitudeBetaConfig(stickiness=0.25),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
        ),
    )

    actor_linears = _linear_modules(policy.actor.mlp)
    shared_encoder_linears = _linear_modules(policy.shared_encoder.mlp)
    critic_trunk_linears = _linear_modules(policy.critic.value_features)

    _assert_orthogonal_gain(shared_encoder_linears[0], 1.0)
    _assert_orthogonal_gain(actor_linears[0], 1.0)
    _assert_orthogonal_gain(actor_linears[-1], 0.01)
    _assert_orthogonal_gain(critic_trunk_linears[0], 1.0)
    _assert_orthogonal_gain(policy.critic.value_head, 0.01)


def test_ppo_policy_exposes_popart_when_enabled_in_critic_config() -> None:
    policy = PPOPolicy(
        env=_DummyLearnEnv(),
        config=PPOPolicyConfig(
            critic_config=PPOCriticConfig(
                use_popart=True,
            ),
            continuous_config=StickySignMagnitudeBetaConfig(stickiness=0.25),
        ),
    )

    assert policy.has_popart


def test_ppo_rollout_log_probs_round_trip_through_float32_rqs_actions() -> None:
    policy = PPOPolicy(
        env=_DummyLearnEnv(),
        config=PPOPolicyConfig(
            continuous_config=RationalQuadraticSplineQuantileConfig(),
        ),
    )
    quantile_distribution = policy.action_dist.distributions[0]
    assert isinstance(quantile_distribution, RationalQuadraticSplineQuantileActionDist)
    torch.manual_seed(0)
    with torch.no_grad():
        quantile_distribution.action_net.weight.normal_(std=5.0)
        quantile_distribution.action_net.bias.normal_(std=5.0)

    batch_size = 32
    local_obs = torch.randn(batch_size, _DummyLearnEnv.n_agents, _DummyLearnEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyLearnEnv.global_obs_dim)
    hidden_local_vars = torch.randn(
        batch_size,
        _DummyLearnEnv.n_agents,
        _DummyLearnEnv.hidden_local_vars_dim,
    )
    hidden_global_vars = torch.randn(batch_size, _DummyLearnEnv.hidden_global_vars_dim)
    u = torch.rand(batch_size, _DummyLearnEnv.n_agents, 2)
    with torch.no_grad(), patch(
            "swarmbots.learn.action_dists.bounded_quantile_action_dist.torch.rand",
            return_value=u,
    ):
        actions, rollout_log_probs, _values = policy(
            local_obs,
            global_obs,
            hidden_local_vars,
            hidden_global_vars,
        )

    with torch.no_grad():
        latent_pi = policy.actor(policy.shared_encoder(local_obs, global_obs))
        policy.action_dist.update_latent_features(latent_pi)
        recomputed_log_probs = policy.action_dist.log_prob(actions)
        _continuous_actions, continuous_log_det = quantile_distribution._transform_forward_and_log_det(u)
        known_sample_log_probs = (
            -continuous_log_det.sum(dim=-1)
            + policy.action_dist.distributions[1].log_prob(actions[..., 2:])
        )

    torch.testing.assert_close(rollout_log_probs, recomputed_log_probs)
    assert (known_sample_log_probs - recomputed_log_probs).abs().max().item() > 0.2


def test_mappo_policy_initializes_internal_layers_and_heads_with_expected_gains() -> None:
    torch.manual_seed(0)
    policy = MAPPOPolicy(
        env=_DummyLearnEnv(),
        config=MAPPOPolicyConfig(
            actor_config=MAPPOActorConfig(
                hidden_dims=[12],
                shared_encoder_latent_dim=8,
                actor_head_hidden_dims=[6],
                latent_pi_dim=4,
            ),
            critic_config=MAPPOCriticConfig(
                deep_set_config=DeepSetCriticConfig(
                    local_projection_hidden_dims=[10],
                    value_regressor_hidden_dims=[10],
                ),
            ),
            continuous_config=StickySignMagnitudeBetaConfig(stickiness=0.25),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
        ),
    )

    actor_linears = _linear_modules(policy.actor.mlp)
    shared_encoder_linears = _linear_modules(policy.shared_encoder.mlp)
    critic_local_projection_linears = _linear_modules(policy.critic.deepset.element_encoder)
    critic_value_regressor_linears = _linear_modules(policy.critic.deepset.set_decoder[0])

    _assert_orthogonal_gain(shared_encoder_linears[0], 1.0)
    _assert_orthogonal_gain(actor_linears[0], 1.0)
    _assert_orthogonal_gain(actor_linears[-1], 0.01)
    _assert_orthogonal_gain(critic_local_projection_linears[0], 1.0)
    _assert_orthogonal_gain(critic_value_regressor_linears[0], 1.0)
    _assert_orthogonal_gain(policy.critic.deepset.set_decoder[1], 0.01)


def test_mappo_policy_exposes_popart_when_enabled_in_critic_config() -> None:
    policy = MAPPOPolicy(
        env=_DummyLearnEnv(),
        config=MAPPOPolicyConfig(
            critic_config=MAPPOCriticConfig(
                deep_set_config=DeepSetCriticConfig(
                    local_projection_hidden_dims=[10],
                    value_regressor_hidden_dims=[10],
                ),
                use_popart=True,
            ),
            continuous_config=StickySignMagnitudeBetaConfig(stickiness=0.25),
        ),
    )

    assert policy.has_popart
