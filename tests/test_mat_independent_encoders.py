from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
from gymnasium import spaces

from swarmbots.learn.algos.mat.mat_dec_policy import MATDecPolicy, MATDecPolicyConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.algos.mat_orig.mat_orig_decoder import MATOrigDecoderConfig
from swarmbots.learn.algos.mat_orig.mat_orig_policy import MATOrigPolicy, MATOrigPolicyConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_policy import MATQCXPolicy, MATQCXPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoderConfig
from swarmbots.learn.algos.r_mat.r_mat_ind_policy import RMATIndPolicy, RMATIndPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_qcx_policy import RMATQCXPolicy, RMATQCXPolicyConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.parameter_counts import count_policy_parameters
from swarmbots.learn.temporal_state import clone_temporal_state
from swarmbots.learn.training import _variant_options

POLICIES = (
    (MATIndPolicy, MATIndPolicyConfig), (MATDecPolicy, MATDecPolicyConfig),
    (MATOrigPolicy, MATOrigPolicyConfig), (MATQCXPolicy, MATQCXPolicyConfig),
    (RMATIndPolicy, RMATIndPolicyConfig), (RMATQCXPolicy, RMATQCXPolicyConfig),
)
RECURRENT_POLICIES = POLICIES[-2:]


def make_policy(policy_class: type, config_class: type) -> torch.nn.Module:
    env = SimpleNamespace(
        n_agents=3, local_obs_dim=5, global_obs_dim=2,
        hidden_local_vars_dim=2, hidden_global_vars_dim=1,
        action_space=HybridActionSpace({"connectors": spaces.MultiBinary((3, 2))}),
    )
    encoder_class = RMATEncoderConfig if policy_class in (RMATIndPolicy, RMATQCXPolicy) else MATEncoderConfig
    settings = dict(
        encoder_config=encoder_class(d_model=16, nhead=4, num_layers=1, dim_feedforward=32),
        critic_encoder_config=encoder_class(d_model=12, nhead=4, num_layers=1, dim_feedforward=24),
    )
    if policy_class is MATOrigPolicy:
        settings["decoder_config"] = MATOrigDecoderConfig(d_model=16, nhead=4, num_layers=1)
    elif policy_class in (MATQCXPolicy, RMATQCXPolicy):
        settings["decoder_config"] = MATQCXDecoderConfig(d_model=16, nhead=4, num_layers=1, dim_feedforward=32)
    return policy_class(env=env, config=config_class(**settings))


def observations() -> dict[str, torch.Tensor]:
    return dict(
        local_obs=torch.randn(2, 3, 5), global_obs=torch.randn(2, 2),
        hidden_local_vars=torch.randn(2, 3, 2), hidden_global_vars=torch.randn(2, 1),
        agent_mask=torch.tensor([[True, True, False], [True, True, True]]),
    )


@pytest.mark.parametrize("policy_class,config_class", POLICIES)
def test_value_updates_leave_actor_parameters_and_actions_unchanged(policy_class: type, config_class: type) -> None:
    torch.manual_seed(31)
    policy = make_policy(policy_class, config_class)
    obs = observations()
    actor_weights = {name: parameter.detach().clone() for name, parameter in policy.named_parameters() if not name.startswith("critic")}
    actions_before = policy.act(**obs, deterministic=True)
    values_before = policy.predict_values(**obs)
    optimizer = torch.optim.Adam(policy.parameters(), lr=0.01)
    (values_before - 10).square().mean().backward()
    assert all(parameter.grad is None for name, parameter in policy.named_parameters() if name in actor_weights)
    assert any(parameter.grad is not None for parameter in policy.critic_encoder.parameters())
    optimizer.step()
    for name, parameter in policy.named_parameters():
        if name in actor_weights:
            torch.testing.assert_close(parameter, actor_weights[name], rtol=0, atol=0)
    torch.testing.assert_close(policy.act(**obs, deterministic=True), actions_before, rtol=0, atol=0)
    assert not torch.allclose(policy.predict_values(**obs), values_before)
    assert count_policy_parameters(policy)["roles"]["shared_encoder"]["total"] == 0


@pytest.mark.parametrize("policy_class,config_class", POLICIES)
def test_actor_loss_leaves_critic_gradients_empty(policy_class: type, config_class: type) -> None:
    policy = make_policy(policy_class, config_class)
    obs = observations()
    with torch.no_grad():
        actions, _, _ = policy(**obs)
    if policy_class in (RMATIndPolicy, RMATQCXPolicy):
        batch = SimpleNamespace(
            **{key: value.unsqueeze(1) for key, value in obs.items()},
            actions=actions.unsqueeze(1), previous_actions=None,
            time_mask=torch.ones(2, 1, dtype=torch.bool),
            episode_start_mask=torch.ones(2, 1, dtype=torch.bool), initial_temporal_state=None,
        )
    else:
        batch = SimpleNamespace(**obs, actions=actions, previous_actions=None)
    log_probs, _, _, _, _ = policy._evaluate_actions(batch)
    (-log_probs.mean()).backward()
    assert any(parameter.grad is not None for parameter in policy.actor_encoder.parameters())
    assert all(parameter.grad is None for name, parameter in policy.named_parameters() if name.startswith("critic"))


@pytest.mark.parametrize("policy_class,config_class", POLICIES)
def test_inference_runs_only_the_required_encoder(policy_class: type, config_class: type) -> None:
    policy = make_policy(policy_class, config_class)
    obs = observations()
    with patch.object(policy.critic_encoder, "forward", side_effect=AssertionError("critic ran during action inference")):
        policy.act(**obs, deterministic=True)
    with patch.object(policy.actor_encoder, "forward", side_effect=AssertionError("actor ran during value prediction")):
        policy.predict_values(**obs)


@pytest.mark.parametrize("policy_class,config_class", RECURRENT_POLICIES)
def test_recurrent_actor_and_critic_use_independent_histories(policy_class: type, config_class: type) -> None:
    policy = make_policy(policy_class, config_class)
    obs = observations()
    _, _, _, state = policy.forward_with_temporal_state(**obs)
    state_before = clone_temporal_state(state)
    _, log_probs, values, _ = policy.forward_with_temporal_state(**obs, temporal_state=state, deterministic=True)
    changed_actor = clone_temporal_state(state)
    changed_actor["actor"] = [(hidden + 5, cell + 5) for hidden, cell in changed_actor["actor"]]
    _, _, changed_values, _ = policy.forward_with_temporal_state(**obs, temporal_state=changed_actor, deterministic=True)
    torch.testing.assert_close(values, changed_values, rtol=0, atol=0)
    changed_critic = clone_temporal_state(state)
    changed_critic["critic"] = [(hidden + 5, cell + 5) for hidden, cell in changed_critic["critic"]]
    _, changed_log_probs, _, _ = policy.forward_with_temporal_state(**obs, temporal_state=changed_critic, deterministic=True)
    torch.testing.assert_close(log_probs, changed_log_probs, rtol=0, atol=0)
    _, action_state = policy.act_with_temporal_state(**obs, temporal_state=state, deterministic=True)
    torch.testing.assert_close(action_state["critic"], state["critic"], rtol=0, atol=0)
    _, value_state = policy.predict_values_with_temporal_state(**obs, temporal_state=state)
    torch.testing.assert_close(value_state["actor"], state["actor"], rtol=0, atol=0)
    torch.testing.assert_close(state, state_before, rtol=0, atol=0)
    starts = torch.tensor([True, False])
    _, reset_log_probs, reset_values, _ = policy.forward_with_temporal_state(
        **obs, temporal_state=state, episode_start_mask=starts, deterministic=True,
    )
    _, fresh_log_probs, fresh_values, _ = policy.forward_with_temporal_state(**obs, deterministic=True)
    torch.testing.assert_close(reset_log_probs[0], fresh_log_probs[0])
    torch.testing.assert_close(reset_values[0], fresh_values[0])


@pytest.mark.parametrize("policy_class,config_class", RECURRENT_POLICIES)
def test_sequence_training_reproduces_both_rollout_histories(policy_class: type, config_class: type) -> None:
    policy = make_policy(policy_class, config_class)
    initial_state = policy.initial_temporal_state(2, 3, device=torch.device("cpu"), dtype=torch.float32)
    initial_state = {role: [(hidden + 0.5, cell - 0.5) for hidden, cell in layers] for role, layers in initial_state.items()}
    state = clone_temporal_state(initial_state)
    rows = [observations() for _ in range(4)]
    starts = torch.tensor([[False, False, True, False], [False, False, False, False]])
    collected = []
    for step, obs in enumerate(rows):
        actions, log_probs, values, state = policy.forward_with_temporal_state(
            **obs, temporal_state=state, episode_start_mask=starts[:, step], deterministic=True,
        )
        collected.append((actions.detach().clone(), log_probs.detach().clone(), values.detach().clone()))
    batch = SimpleNamespace(
        **{key: torch.stack([obs[key] for obs in rows], dim=1) for key in rows[0]},
        actions=torch.stack([row[0] for row in collected], dim=1), previous_actions=None,
        time_mask=torch.ones(2, 4, dtype=torch.bool), episode_start_mask=starts,
        initial_temporal_state=initial_state,
    )
    log_probs, values, _, _, _ = policy._evaluate_actions(batch)
    expected_log_probs = torch.stack([row[1] for row in collected], dim=1).masked_fill(~batch.agent_mask, 0.0)
    torch.testing.assert_close(log_probs, expected_log_probs, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(values, torch.stack([row[2] for row in collected], dim=1), atol=1e-6, rtol=1e-5)
    values.mean().backward()
    assert all(parameter.grad is None for parameter in policy.actor_encoder.parameters())


@pytest.mark.parametrize("variant", [
    "mappo__mlp_critic", "maddpg__mlp_critic", "matd3__mlp_critic", "masac__mlp_critic", "tmasac__ac_shared_observation_encoder",
    "tmasac__swiglu", "tmasac__slstm__swiglu", "tmasac__slstm__no_residual",
    "tmasac__slstm__ac_shared_observation_encoder", "tmasac__slstm__no_nop", "tmasac__lstm__no_actor_state", "tmasac__segment",
])
def test_removed_presets_are_rejected(variant: str) -> None:
    with pytest.raises(ValueError, match="Unknown learning variant"):
        _variant_options(variant)
