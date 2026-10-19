import math
from dataclasses import replace

import pytest
import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActor, MAPPOActorConfig
from swarmbots.learn.algos.mappo.mappo_policy import MAPPOPolicy, MAPPOPolicyConfig
from swarmbots.learn.algos.mat_qc_base_policy import MATCriticConfig
from swarmbots.learn.algos.ppo.ppo_policy import (
    PPOActor, PPOActorConfig, PPOCritic, PPOCriticConfig, PPOPolicy, PPOPolicyConfig, PPOSharedEncoder,
)
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.nn_components.popart import PopArtLinear
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.presets.model_scale import DEFAULT_MODEL_SCALE
from swarmbots.learn.presets.transformer import MATInitGains
from test_model_scale import _make_env, make_policy


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


def _assert_orthogonal_gain(linear: nn.Linear | PopArtLinear, gain: float) -> None:
    weights = linear.weight.to(dtype=torch.float64)
    expected_norm = gain * math.sqrt(min(linear.weight.shape))
    torch.testing.assert_close(
        torch.linalg.vector_norm(weights),
        torch.tensor(expected_norm, device=weights.device, dtype=weights.dtype),
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
            continuous_config=SignMagnitudeBetaConfig(),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
        ),
    )

    actor_linears = _linear_modules(policy.actor.mlp)
    shared_encoder_linears = _linear_modules(policy.shared_encoder.mlp)
    critic_trunk_linears = _linear_modules(policy.critic.value_features)

    for linear in shared_encoder_linears:
        _assert_orthogonal_gain(linear, 1.0)
    for linear in actor_linears[:-1]:
        _assert_orthogonal_gain(linear, 1.0)
    _assert_orthogonal_gain(actor_linears[-1], 0.01)
    for linear in critic_trunk_linears:
        _assert_orthogonal_gain(linear, 1.0)
    _assert_orthogonal_gain(policy.critic.value_head, 0.01)


def test_ppo_policy_exposes_popart_when_enabled_in_critic_config() -> None:
    policy = PPOPolicy(
        env=_DummyLearnEnv(),
        config=PPOPolicyConfig(
            critic_config=PPOCriticConfig(
                use_popart=True,
            ),
            continuous_config=SignMagnitudeBetaConfig(),
        ),
    )

    assert policy.has_popart


def test_mappo_policy_initializes_internal_layers_and_heads_with_expected_gains() -> None:
    torch.manual_seed(0)
    policy = MAPPOPolicy(
        env=_DummyLearnEnv(),
        config=MAPPOPolicyConfig(
            actor_config=MAPPOActorConfig(
                hidden_dims=[12],
                latent_dim=8,
                actor_head_hidden_dims=[6],
                latent_pi_dim=4,
            ),
            critic_config=MATCriticConfig(
                local_projection_hidden_dims=[10],
                value_regressor_hidden_dims=[10],
            ),
            continuous_config=SignMagnitudeBetaConfig(),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
        ),
    )

    actor_linears = _linear_modules(policy.actor.mlp)
    actor_encoder_linears = _linear_modules(policy.actor_encoder.mlp)
    critic_encoder_linears = _linear_modules(policy.critic_encoder.local_obs_encoder)
    critic_local_projection_linears = _linear_modules(policy.critic.deepset.element_encoder)
    critic_value_regressor_linears = _linear_modules(policy.critic.deepset.set_decoder[0])

    _assert_orthogonal_gain(actor_encoder_linears[0], 1.0)
    _assert_orthogonal_gain(critic_encoder_linears[0], 1.0)
    _assert_orthogonal_gain(actor_linears[0], 1.0)
    _assert_orthogonal_gain(actor_linears[-1], 1.0)
    for linear in _linear_modules(policy.action_dist):
        _assert_orthogonal_gain(linear, 0.01)
    _assert_orthogonal_gain(critic_local_projection_linears[0], 1.0)
    _assert_orthogonal_gain(critic_value_regressor_linears[0], 1.0)
    _assert_orthogonal_gain(policy.critic.deepset.set_decoder[1], 0.01)


def test_mappo_policy_exposes_popart_when_enabled_in_critic_config() -> None:
    policy = MAPPOPolicy(
        env=_DummyLearnEnv(),
        config=MAPPOPolicyConfig(
            critic_config=MATCriticConfig(
                local_projection_hidden_dims=[10],
                value_regressor_hidden_dims=[10],
                use_popart=True,
            ),
            continuous_config=SignMagnitudeBetaConfig(),
        ),
    )

    assert policy.has_popart


@pytest.mark.parametrize("hidden_gain", [None, 0.4])
def test_mappo_feature_head_gain_applies_to_all_layers_unless_explicitly_overridden(hidden_gain: float | None) -> None:
    actor = MAPPOActor(
        local_latent_dim=8,
        config=MAPPOActorConfig(actor_head_hidden_dims=[12, 10], latent_pi_dim=6, actor_head_init_gain=0.7),
        linear_init=None if hidden_gain is None else make_init_linear_orthogonal(hidden_gain),
    )
    linears = _linear_modules(actor.mlp)
    for linear in linears[:-1]:
        _assert_orthogonal_gain(linear, 0.7 if hidden_gain is None else hidden_gain)
    _assert_orthogonal_gain(linears[-1], 0.7)


def _last_linear(module: nn.Module) -> nn.Linear:
    return _linear_modules(module)[-1]


@pytest.mark.parametrize("variant", ["ppo", "mappo__deepset_critic"])
@pytest.mark.parametrize("scale", [None, DEFAULT_MODEL_SCALE])
@pytest.mark.parametrize("gains", [MATInitGains(), MATInitGains(actor_head=0.7, action_net=0.03, critic_value_head=0.2)])
def test_on_policy_presets_share_feature_and_action_readout_initialization(
    variant: str, scale: str | None, gains: MATInitGains,
) -> None:
    policy = make_policy(variant, scale, use_nop=False, mat_init_gains=gains)
    for linear in _linear_modules(policy.actor):
        _assert_orthogonal_gain(linear, gains.actor_head)
    for distribution in policy.action_dist.distributions:
        _assert_orthogonal_gain(_last_linear(distribution), gains.action_net)
    if variant == "ppo":
        _assert_orthogonal_gain(policy.critic.value_head, gains.critic_value_head)


@pytest.mark.parametrize("scale", [None, DEFAULT_MODEL_SCALE])
@pytest.mark.parametrize("projection_gain", [None, 0.8])
def test_ppo_presets_honor_shared_encoder_and_critic_feature_gains(
    scale: str | None, projection_gain: float | None,
) -> None:
    gains = MATInitGains(
        obs_encoder=0.6, obs_encoder_projection=projection_gain, actor_head=0.7,
        critic_value_regressor=0.5, critic_value_head=0.2,
    )
    policy = make_policy("ppo", scale, use_nop=False, mat_init_gains=gains)
    shared_linears = _linear_modules(policy.shared_encoder)
    for linear in shared_linears[:-1]:
        _assert_orthogonal_gain(linear, gains.obs_encoder)
    _assert_orthogonal_gain(shared_linears[-1], gains.obs_encoder if projection_gain is None else projection_gain)
    for linear in _linear_modules(policy.actor):
        _assert_orthogonal_gain(linear, gains.actor_head)
    for linear in _linear_modules(policy.critic.value_features):
        _assert_orthogonal_gain(linear, gains.critic_value_regressor)
    _assert_orthogonal_gain(policy.critic.value_head, gains.critic_value_head)
    settings = policy.get_hyper_parameters()["ppo_policy_config"]
    assert settings["actor_config"]["shared_encoder_init_gain"] == gains.obs_encoder
    assert settings["actor_config"]["shared_encoder_projection_init_gain"] == projection_gain
    assert settings["actor_config"]["actor_head_hidden_init_gain"] == gains.actor_head
    assert settings["critic_config"]["value_features_init_gain"] == gains.critic_value_regressor


@pytest.mark.parametrize("projection_gain", [None, 0.9])
def test_ppo_components_preserve_explicit_hidden_initializers(projection_gain: float | None) -> None:
    initialization = make_init_linear_orthogonal(0.4)
    actor_config = PPOActorConfig(
        hidden_dims=[12], shared_encoder_latent_dim_per_agent=8,
        shared_encoder_init_gain=0.6, shared_encoder_projection_init_gain=projection_gain,
        actor_head_hidden_dims=[10], latent_pi_dim_per_agent=6,
        actor_head_hidden_init_gain=0.6, actor_head_init_gain=0.7,
    )
    encoder = PPOSharedEncoder(3, 5, 2, actor_config, linear_init=initialization)
    encoder_linears = _linear_modules(encoder)
    _assert_orthogonal_gain(encoder_linears[0], 0.4)
    _assert_orthogonal_gain(encoder_linears[-1], 0.4 if projection_gain is None else projection_gain)
    actor = PPOActor(8, actor_config, linear_init=initialization)
    actor_linears = _linear_modules(actor)
    _assert_orthogonal_gain(actor_linears[0], 0.4)
    _assert_orthogonal_gain(actor_linears[-1], 0.7)
    critic = PPOCritic(
        3, 8, 2,
        PPOCriticConfig(hidden_dims=[10, 6], value_features_init_gain=0.6, value_head_init_gain=0.2),
        linear_init=initialization,
    )
    for linear in _linear_modules(critic.value_features):
        _assert_orthogonal_gain(linear, 0.4)
    _assert_orthogonal_gain(critic.value_head, 0.2)


@pytest.mark.parametrize("scale", [None, DEFAULT_MODEL_SCALE])
def test_ppo_honors_shared_popart_settings_and_output_initialization(scale: str | None) -> None:
    policy = make_policy(
        "ppo", scale, use_nop=False, use_popart=True, popart_beta=0.02, popart_init_sigma=0.7,
        mat_init_gains=MATInitGains(critic_value_head=0.2),
    )
    head = policy.critic.value_head
    assert isinstance(head, PopArtLinear)
    assert head.beta == 0.02
    torch.testing.assert_close(head.sigma(), torch.tensor([0.7]))
    _assert_orthogonal_gain(head, 0.2)


def test_low_level_ppo_keeps_default_gains_and_records_custom_readout_gain() -> None:
    config = PPOPolicyConfig(continuous_config=SquashedDiagGaussianConfig(std=0.2, std_learnable=False))
    policy = PPOPolicy(_make_env(), config)
    _assert_orthogonal_gain(_last_linear(policy.actor), 0.01)
    for distribution in policy.action_dist.distributions:
        _assert_orthogonal_gain(_last_linear(distribution), 1.0)

    custom = PPOPolicy(_make_env(), replace(config, action_net_init_gain=0.03))
    assert custom.get_hyper_parameters()["ppo_policy_config"]["action_net_init_gain"] == 0.03
    for distribution in custom.action_dist.distributions:
        _assert_orthogonal_gain(_last_linear(distribution), 0.03)
