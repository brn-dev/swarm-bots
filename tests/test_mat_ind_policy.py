from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.sticky_sign_magnitude_beta_action_dist import (
    StickySignMagnitudeBetaConfig,
)
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace

_REAL_TORCH_COMPILE = torch.compile


def _compile_with_eager_backend(
        function: Callable[..., Any],
        **kwargs: Any,
) -> Callable[..., Any]:
    return _REAL_TORCH_COMPILE(function, backend="eager", **kwargs)


class _DummyMATIndEnv:
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


class _DummyDefaultMATIndEnv:
    n_agents = 4
    local_obs_dim = 6
    global_obs_dim = 3
    hidden_local_vars_dim = 2
    hidden_global_vars_dim = 5
    action_space = HybridActionSpace(
        {"disc": spaces.MultiBinary((n_agents, 1))}
    )


def _make_policy(
        *,
        compile_modules: bool = False,
        ent_loss_coef: float = 0.0,
) -> MATIndPolicy:
    return MATIndPolicy(
        env=_DummyMATIndEnv(),
        config=MATIndPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
            ),
            actor_head_hidden_dims=[16],
            continuous_config=StickySignMagnitudeBetaConfig(
                stickiness=0.25,
                ent_loss_coef=ent_loss_coef,
            ),
            bernoulli_config=BernoulliConfig(
                initial_prob=0.5,
                ent_loss_coef=ent_loss_coef,
            ),
            max_agents=8,
            compile_modules=compile_modules,
        ),
    )


def test_default_config_constructs_policy() -> None:
    policy = MATIndPolicy(env=_DummyDefaultMATIndEnv())

    assert isinstance(policy.config, MATIndPolicyConfig)
    assert policy.max_agents == _DummyDefaultMATIndEnv.n_agents


def test_forward_and_value_paths_match_pipeline_shapes() -> None:
    policy = _make_policy()
    torch.manual_seed(0)

    batch_size = 3
    local_obs = torch.randn(batch_size, _DummyMATIndEnv.n_agents, _DummyMATIndEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATIndEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATIndEnv.n_agents, _DummyMATIndEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATIndEnv.hidden_global_vars_dim)
    agent_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    previous_actions = torch.zeros(
        batch_size,
        _DummyMATIndEnv.n_agents,
        _DummyMATIndEnv.action_space.total_agent_action_dim,
    )

    actions, log_probs, values = policy(
        local_obs=local_obs,
        global_obs=global_obs,
        hidden_local_vars=hidden_local_vars,
        hidden_global_vars=hidden_global_vars,
        agent_mask=agent_mask,
        previous_actions=previous_actions,
        deterministic=True,
    )
    predicted_values = policy.predict_values(
        local_obs=local_obs,
        global_obs=global_obs,
        hidden_local_vars=hidden_local_vars,
        hidden_global_vars=hidden_global_vars,
        agent_mask=agent_mask,
    )

    assert actions.shape == (
        batch_size,
        _DummyMATIndEnv.n_agents,
        _DummyMATIndEnv.action_space.total_agent_action_dim,
    )
    assert log_probs.shape == (batch_size, _DummyMATIndEnv.n_agents)
    assert values.shape == (batch_size,)
    assert predicted_values.shape == (batch_size,)
    assert torch.allclose(values, predicted_values, rtol=0.0, atol=1e-6)
    assert torch.equal(actions[~agent_mask], torch.zeros_like(actions[~agent_mask]))
    assert torch.equal(log_probs[~agent_mask], torch.zeros_like(log_probs[~agent_mask]))


def test_evaluate_actions_matches_rollout_log_probs_for_masked_agents() -> None:
    policy = _make_policy()
    torch.manual_seed(1)

    batch_size = 4
    action_dim = _DummyMATIndEnv.action_space.total_agent_action_dim
    local_obs = torch.randn(batch_size, _DummyMATIndEnv.n_agents, _DummyMATIndEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATIndEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATIndEnv.n_agents, _DummyMATIndEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATIndEnv.hidden_global_vars_dim)
    agent_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, False, False, False],
            [True, True, True, True],
        ]
    )
    previous_actions = torch.randn(batch_size, _DummyMATIndEnv.n_agents, action_dim)
    previous_actions = previous_actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

    with torch.no_grad():
        rollout_actions, rollout_log_probs, rollout_values = policy(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=False,
        )

        class _Batch:
            pass

        batch = _Batch()
        batch.actions = rollout_actions
        batch.local_obs = local_obs
        batch.global_obs = global_obs
        batch.hidden_local_vars = hidden_local_vars
        batch.hidden_global_vars = hidden_global_vars
        batch.agent_mask = agent_mask
        batch.previous_actions = previous_actions

        eval_log_probs, eval_values, *_ = policy.evaluate_actions(batch)

    assert torch.allclose(eval_log_probs, rollout_log_probs, rtol=0.0, atol=1e-6)
    assert torch.allclose(eval_values, rollout_values, rtol=0.0, atol=1e-6)
    assert torch.equal(eval_log_probs[~agent_mask], torch.zeros_like(eval_log_probs[~agent_mask]))


def test_compiled_evaluation_refreshes_distribution_state_for_extra_losses() -> None:
    torch.manual_seed(43)
    eager_policy = _make_policy(ent_loss_coef=0.2)
    torch.manual_seed(43)
    with (
        patch(
            "swarmbots.learn.algos.mat.mat_ind_policy._ensure_torch_compile_available",
        ),
        patch(
            "swarmbots.learn.algos.mat.mat_ind_policy.torch.compile",
            side_effect=_compile_with_eager_backend,
        ),
    ):
        compiled_policy = _make_policy(compile_modules=True, ent_loss_coef=0.2)

        batch_size = 3
        local_obs = torch.randn(batch_size, _DummyMATIndEnv.n_agents, _DummyMATIndEnv.local_obs_dim)
        global_obs = torch.randn(batch_size, _DummyMATIndEnv.global_obs_dim)
        hidden_local_vars = torch.randn(
            batch_size,
            _DummyMATIndEnv.n_agents,
            _DummyMATIndEnv.hidden_local_vars_dim,
        )
        hidden_global_vars = torch.randn(batch_size, _DummyMATIndEnv.hidden_global_vars_dim)
        agent_mask = torch.tensor([
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ])
        actions = torch.empty(
            batch_size,
            _DummyMATIndEnv.n_agents,
            _DummyMATIndEnv.action_space.total_agent_action_dim,
        )
        actions[..., :2].uniform_(-0.8, 0.8)
        actions[..., 2:] = torch.randint(0, 2, actions[..., 2:].shape, dtype=actions.dtype)

        class _Batch:
            pass

        batch = _Batch()
        batch.hidden_local_vars = hidden_local_vars
        batch.hidden_global_vars = hidden_global_vars
        batch.agent_mask = agent_mask
        batch.previous_actions = torch.zeros_like(actions)
        batch.actions = actions

        for observation_offset in (0.0, 0.25):
            batch.local_obs = local_obs + observation_offset
            batch.global_obs = global_obs - observation_offset
            eager_outputs = eager_policy.evaluate_actions(batch)
            compiled_outputs = compiled_policy.evaluate_actions(batch)
            torch.testing.assert_close(compiled_outputs[0], eager_outputs[0])
            torch.testing.assert_close(compiled_outputs[1], eager_outputs[1])
            compiled_losses = compiled_outputs[2]
            eager_losses = eager_outputs[2]
            assert compiled_losses.keys() == eager_losses.keys()
            assert compiled_losses
            for name in compiled_losses:
                torch.testing.assert_close(compiled_losses[name], eager_losses[name])

        compiled_total_loss = (
            compiled_outputs[0].mean()
            + compiled_outputs[1].mean()
            + torch.stack(tuple(compiled_outputs[2].values())).sum()
        )
        compiled_total_loss.backward()
        assert any(
            parameter.grad is not None
            for parameter in compiled_policy.action_dist.parameters()
            if parameter.requires_grad
        )
