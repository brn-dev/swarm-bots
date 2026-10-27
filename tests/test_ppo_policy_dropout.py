from collections.abc import Callable
from types import SimpleNamespace

import pytest
import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSamplerConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig, NextObsPredWrapper
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.nn_components.activations import ActivationFactory


def _make_env() -> SimpleNamespace:
    return SimpleNamespace(
        n_agents=2,
        local_obs_dim=3,
        global_obs_dim=1,
        hidden_local_vars_dim=0,
        hidden_global_vars_dim=0,
        observation_space=spaces.Dict({
            "local_obs": spaces.Box(-1.0, 1.0, (2, 2, 3)),
            "global_obs": spaces.Box(-1.0, 1.0, (2, 1)),
            "hidden_local_vars": spaces.Box(-1.0, 1.0, (2, 2, 0)),
            "hidden_global_vars": spaces.Box(-1.0, 1.0, (2, 0)),
        }),
        action_space=VectorHybridActionSpace({"cont": spaces.Box(-1.0, 1.0, (2, 2, 1))}),
    )


def _make_policy(
    env: SimpleNamespace, *, dropout: float = 0.0, act_fn_cls: ActivationFactory = nn.GELU,
) -> MATIndPolicy:
    return MATIndPolicy(env, MATIndPolicyConfig(
        encoder_config=MATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16, act_fn_cls=act_fn_cls),
        continuous_config=SignMagnitudeBetaConfig(),
        dropout=dropout,
        act_fn_cls=act_fn_cls,
    ))


def _make_ppo(env: SimpleNamespace, policy: MATIndPolicy | NextObsPredWrapper) -> PPO:
    return PPO(
        policy, env, rollout_mode=StepsRolloutMode(4), sampler_config=PPOSamplerConfig(batch_size=2),
        train_device="cpu", rollout_device="cpu",
    )


def test_ppo_rejects_nonzero_policy_dropout() -> None:
    env = _make_env()
    with pytest.raises(ValueError, match="zero policy dropout"):
        _make_ppo(env, _make_policy(env, dropout=0.25))


@pytest.mark.parametrize("dropout_module", [
    lambda: nn.Dropout(0.25),
    lambda: nn.MultiheadAttention(8, 2, dropout=0.25, batch_first=True),
    lambda: nn.LSTM(8, 8, num_layers=2, dropout=0.25, batch_first=True),
])
def test_ppo_rechecks_actual_dropout_modules_before_training(dropout_module: Callable[[], nn.Module]) -> None:
    env = _make_env()
    policy = _make_policy(env)
    algorithm = _make_ppo(env, policy)
    policy.actor_head = dropout_module()

    with pytest.raises(ValueError, match="zero policy dropout"):
        algorithm.train([])


@pytest.mark.parametrize("activation", [nn.RReLU, lambda: nn.Sequential(nn.Identity(), nn.RReLU())])
def test_ppo_rejects_randomized_policy_activations(activation: ActivationFactory) -> None:
    env = _make_env()
    with pytest.raises(ValueError, match="RReLU"):
        _make_ppo(env, _make_policy(env, act_fn_cls=activation))


def test_ppo_rechecks_randomized_activations_before_training() -> None:
    env = _make_env()
    policy = _make_policy(env)
    algorithm = _make_ppo(env, policy)
    policy.actor_head = nn.Sequential(nn.RReLU())
    with pytest.raises(ValueError, match="RReLU"):
        algorithm.train([])


@pytest.mark.parametrize("activation", [nn.GELU, lambda: nn.RReLU(lower=0.2, upper=0.2)])
def test_zero_dropout_keeps_rollout_likelihoods_when_training_mode_changes(activation: ActivationFactory) -> None:
    env = _make_env()
    policy = _make_policy(env, act_fn_cls=activation)
    _make_ppo(env, policy)
    observations = {
        "local_obs": torch.randn(2, 2, 3),
        "global_obs": torch.randn(2, 1),
        "hidden_local_vars": torch.empty(2, 2, 0),
        "hidden_global_vars": torch.empty(2, 0),
        "agent_mask": None,
    }
    policy.eval()
    with torch.no_grad():
        actions, log_probs, values = policy(**observations)
    batch = PPOSamples(
        **observations, previous_actions=None, actions=actions, log_probs=log_probs, values=values,
        returns=values, advantages=torch.ones(2),
    )
    policy.train()
    reevaluated_log_probs, reevaluated_values, _, _ = policy.evaluate_actions(batch)

    torch.testing.assert_close(reevaluated_log_probs, log_probs)
    torch.testing.assert_close(reevaluated_values, values)


def test_nop_dropout_is_allowed_without_bypassing_core_policy_guard() -> None:
    env = _make_env()
    policy = _make_policy(env)
    wrapper = NextObsPredWrapper(policy, world_model_config=NOPWorldModelConfig(
        n_agents=2, local_latent_dim=8, action_dim=1, d_model_transition_model=8,
        nhead_transition_model=2, num_layers_transition_model=1, dim_feedforward_transition_model=16,
        transition_model_dropout=0.25, next_obs_pred_config=NextObsPredConfig(local_scalar_target_indices=[0]),
    ))
    algorithm = _make_ppo(env, wrapper)
    algorithm.policy.train()
    assert wrapper.transition_model.encoder.layers[0].dropout.p == 0.25

    policy.actor_encoder.layers[0].self_attn.dropout = 0.25
    with pytest.raises(ValueError, match="zero policy dropout"):
        _make_ppo(env, wrapper)
