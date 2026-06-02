import math

import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
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
