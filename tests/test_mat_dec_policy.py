from unittest.mock import patch

import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.algos.mat.mat_dec_policy import MATDecPolicy, MATDecPolicyConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace


class _DummyMATDecEnv:
    n_agents = 4
    local_obs_dim = 6
    global_obs_dim = 3
    hidden_local_vars_dim = 2
    hidden_global_vars_dim = 5
    action_space = HybridActionSpace(
        {"disc": spaces.MultiBinary((n_agents, 2))}
    )


def _make_policy(*, compile_modules: bool = False) -> MATDecPolicy:
    return MATDecPolicy(
        env=_DummyMATDecEnv(),
        config=MATDecPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
            ),
            actor_head_hidden_dims=[12, 12],
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
            max_agents=8,
            compile_modules=compile_modules,
        ),
    )


def test_default_config_constructs_policy() -> None:
    policy = MATDecPolicy(env=_DummyMATDecEnv())

    assert isinstance(policy.config, MATDecPolicyConfig)
    assert policy.action_dist.latent_dim == policy.actor_encoder_config.d_model
    assert policy.actor_encoder.global_obs_dim == _DummyMATDecEnv.global_obs_dim
    assert all(layer.self_attn is None for layer in policy.actor_encoder.layers)


def test_forward_and_evaluation_use_local_actor_and_mat_critic_paths() -> None:
    torch.manual_seed(0)
    policy = _make_policy()
    batch_size = 3
    local_obs = torch.randn(batch_size, _DummyMATDecEnv.n_agents, _DummyMATDecEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATDecEnv.global_obs_dim)
    hidden_local_vars = torch.randn(
        batch_size,
        _DummyMATDecEnv.n_agents,
        _DummyMATDecEnv.hidden_local_vars_dim,
    )
    hidden_global_vars = torch.randn(batch_size, _DummyMATDecEnv.hidden_global_vars_dim)
    agent_mask = torch.tensor([
        [True, True, True, False],
        [True, True, False, False],
        [True, True, True, True],
    ])

    actions, log_probs, values = policy(
        local_obs=local_obs,
        global_obs=global_obs,
        hidden_local_vars=hidden_local_vars,
        hidden_global_vars=hidden_global_vars,
        agent_mask=agent_mask,
        deterministic=False,
    )

    class _Batch:
        pass

    batch = _Batch()
    batch.actions = actions
    batch.local_obs = local_obs
    batch.global_obs = global_obs
    batch.hidden_local_vars = hidden_local_vars
    batch.hidden_global_vars = hidden_global_vars
    batch.agent_mask = agent_mask
    batch.previous_actions = None
    evaluated_log_probs, evaluated_values, _, _, encoder_rep = policy._evaluate_actions(batch)

    assert actions.shape == (batch_size, _DummyMATDecEnv.n_agents, 2)
    assert values.shape == (batch_size,)
    assert encoder_rep.shape == (batch_size, _DummyMATDecEnv.n_agents, 16)
    torch.testing.assert_close(evaluated_log_probs, log_probs)
    torch.testing.assert_close(evaluated_values, values)


def test_act_is_decentralized_and_does_not_run_the_mat_encoder() -> None:
    torch.manual_seed(1)
    policy = _make_policy()
    local_obs = torch.randn(2, _DummyMATDecEnv.n_agents, _DummyMATDecEnv.local_obs_dim)
    changed_local_obs = local_obs.clone()
    changed_local_obs[:, 1:, :] += 100.0
    global_obs = torch.randn(2, _DummyMATDecEnv.global_obs_dim)

    with patch.object(policy.encoder, "forward", side_effect=AssertionError("actor used MAT encoder")):
        actions = policy.act(local_obs, global_obs, deterministic=True)
        changed_actions = policy.act(
            changed_local_obs,
            global_obs,
            deterministic=True,
        )

    torch.testing.assert_close(changed_actions[:, 0], actions[:, 0])


def test_actor_encoder_uses_global_observations_without_mixing_local_tokens() -> None:
    policy = _make_policy()
    local_obs = torch.randn(2, _DummyMATDecEnv.n_agents, _DummyMATDecEnv.local_obs_dim)
    changed_local_obs = local_obs.clone()
    changed_local_obs[:, 1:, :] += 100.0
    global_obs = torch.randn(2, _DummyMATDecEnv.global_obs_dim)

    actor_features = policy._encode_actor_observations(local_obs, global_obs)
    changed_local_features = policy._encode_actor_observations(changed_local_obs, global_obs)
    changed_global_features = policy._encode_actor_observations(local_obs, global_obs + 1.0)

    torch.testing.assert_close(changed_local_features[:, 0], actor_features[:, 0])
    assert not torch.allclose(changed_global_features[:, 0], actor_features[:, 0])


def test_actor_objective_does_not_backpropagate_through_mat_encoder() -> None:
    policy = _make_policy()
    local_obs = torch.randn(2, _DummyMATDecEnv.n_agents, _DummyMATDecEnv.local_obs_dim)
    global_obs = torch.randn(2, _DummyMATDecEnv.global_obs_dim)
    hidden_local_vars = torch.randn(
        2,
        _DummyMATDecEnv.n_agents,
        _DummyMATDecEnv.hidden_local_vars_dim,
    )
    hidden_global_vars = torch.randn(2, _DummyMATDecEnv.hidden_global_vars_dim)

    _, log_probs, _ = policy(
        local_obs=local_obs,
        global_obs=global_obs,
        hidden_local_vars=hidden_local_vars,
        hidden_global_vars=hidden_global_vars,
    )
    (-log_probs.mean()).backward()

    assert all(parameter.grad is None for parameter in policy.encoder.parameters())
    assert any(parameter.grad is not None for parameter in policy.actor_encoder.parameters())
    assert any(parameter.grad is not None for parameter in policy.actor_head.parameters())


def test_hyper_parameters_use_mat_dec_name() -> None:
    hyper_parameters = _make_policy().get_hyper_parameters()

    assert hyper_parameters.keys() == {"mat_dec_policy_config"}
    actor_encoder_config = hyper_parameters["mat_dec_policy_config"]["actor_encoder_config"]
    assert actor_encoder_config["global_obs_encoder_config"] is not None
    assert actor_encoder_config["use_agent_attention"] is False


def test_optional_compilation_includes_actor_encoder() -> None:
    compiled_modules: list[object] = []

    def record_compile(module: object, **_: object) -> object:
        compiled_modules.append(module)
        return module

    with (
        patch("swarmbots.learn.algos.mat.mat_ind_policy._ensure_torch_compile_available"),
        patch("swarmbots.learn.algos.mat.mat_ind_policy.torch.compile", side_effect=record_compile),
    ):
        policy = _make_policy(compile_modules=True)

    assert policy.actor_encoder in compiled_modules
