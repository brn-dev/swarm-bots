from collections.abc import Callable
from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch
import pytest
from gymnasium import spaces

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import (
    SignMagnitudeBetaConfig,
)
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.algos.mat.mat_dec_policy import MATDecPolicy, MATDecPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_ind_policy import RMATIndPolicy, RMATIndPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoderConfig
from swarmbots.learn.temporal_state import clone_temporal_state
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
        recurrent: bool = False,
        decentralized: bool = False,
) -> MATIndPolicy:
    if recurrent:
        policy_cls, config_cls = RMATIndPolicy, RMATIndPolicyConfig
    elif decentralized:
        policy_cls, config_cls = MATDecPolicy, MATDecPolicyConfig
    else:
        policy_cls, config_cls = MATIndPolicy, MATIndPolicyConfig
    encoder_config_cls = RMATEncoderConfig if recurrent else MATEncoderConfig
    return policy_cls(
        env=_DummyMATIndEnv(),
        config=config_cls(
            encoder_config=encoder_config_cls(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
            ),
            actor_head_hidden_dims=[16],
            continuous_config=SignMagnitudeBetaConfig(),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
            max_agents=8,
            compile_modules=compile_modules,
        ),
    )


def test_default_config_constructs_policy() -> None:
    policy = MATIndPolicy(env=_DummyDefaultMATIndEnv())

    assert isinstance(policy.config, MATIndPolicyConfig)
    assert policy.max_agents == _DummyDefaultMATIndEnv.n_agents


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
@pytest.mark.parametrize("variant", ["ind", "recurrent", "dec"])
def test_cuda_graph_policy_calls_preserve_rollout_outputs_across_bootstrap_and_actions(variant: str) -> None:
    recurrent = variant == "recurrent"
    torch.manual_seed(43)
    eager = _make_policy(recurrent=recurrent, decentralized=variant == "dec").cuda().eval()
    torch.manual_seed(43)
    compiled = type(eager)(
        _DummyMATIndEnv(), replace(eager.config, compile_modules=True, compile_mode="reduce-overhead"),
    ).cuda().eval()
    observations = {
        "local_obs": torch.randn(2, 4, 6, device="cuda"),
        "global_obs": torch.randn(2, 3, device="cuda"),
        "hidden_local_vars": torch.randn(2, 4, 2, device="cuda"),
        "hidden_global_vars": torch.randn(2, 5, device="cuda"),
        "agent_mask": torch.tensor([[True, True, True, False], [True, True, False, False]], device="cuda"),
        "previous_actions": torch.zeros(2, 4, 3, device="cuda"),
    }
    with torch.no_grad():
        retained = []
        for offset in (0., .25, .5):
            current = {**observations, "local_obs": observations["local_obs"] + offset}
            expected = eager(**current, deterministic=True)
            outputs = compiled(**current, deterministic=True)
            torch.testing.assert_close(outputs, expected, rtol=1e-4, atol=1e-5)
            retained.append((outputs, tuple(value.clone() for value in outputs)))
            torch.testing.assert_close(compiled.predict_values(**current), eager.predict_values(**current),
                                       rtol=1e-4, atol=1e-5)
            torch.testing.assert_close(compiled.act(**current, deterministic=True), expected[0],
                                       rtol=1e-4, atol=1e-5)
        for outputs, snapshot in retained:
            torch.testing.assert_close(outputs, snapshot, rtol=0, atol=0)
        if recurrent:
            eager_state = eager.initial_temporal_state(2, 4, device=torch.device("cuda"), dtype=torch.float32)
            compiled_state = clone_temporal_state(eager_state)
            for _ in range(3):
                expected = eager.forward_with_temporal_state(
                    **observations, temporal_state=eager_state, deterministic=True,
                )
                outputs = compiled.forward_with_temporal_state(
                    **observations, temporal_state=compiled_state, deterministic=True,
                )
                torch.testing.assert_close(outputs, expected, rtol=1e-4, atol=1e-5)
                snapshot = clone_temporal_state(outputs)
                bootstrap, _ = compiled.predict_values_with_temporal_state(
                    **observations, temporal_state=outputs[-1],
                )
                eager_bootstrap, _ = eager.predict_values_with_temporal_state(
                    **observations, temporal_state=expected[-1],
                )
                torch.testing.assert_close(bootstrap, eager_bootstrap, rtol=1e-4, atol=1e-5)
                compiled.act_with_temporal_state(**observations, temporal_state=outputs[-1], deterministic=True)
                torch.testing.assert_close(outputs, snapshot, rtol=0, atol=0)
                eager_state, compiled_state = expected[-1], outputs[-1]
    if not recurrent:
        batch = SimpleNamespace(**observations, actions=retained[0][0][0])
        eager_log_probs, eager_values, *_ = eager.evaluate_actions(batch)
        compiled_log_probs, compiled_values, *_ = compiled.evaluate_actions(batch)
        torch.testing.assert_close(compiled_log_probs, eager_log_probs, rtol=1e-4, atol=1e-5)
        torch.testing.assert_close(compiled_values, eager_values, rtol=1e-4, atol=1e-5)
        (-eager_log_probs.mean() + eager_values.square().mean()).backward()
        (-compiled_log_probs.mean() + compiled_values.square().mean()).backward()
        torch.testing.assert_close(
            {".".join(part for part in name.split(".") if part != "_orig_mod"): parameter.grad
             for name, parameter in compiled.named_parameters()},
            {name: parameter.grad for name, parameter in eager.named_parameters()},
            rtol=1e-4, atol=1e-5,
        )


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


def test_compiled_evaluation_matches_eager_log_probs_values_and_gradients() -> None:
    torch.manual_seed(43)
    eager_policy = _make_policy()
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
        compiled_policy = _make_policy(compile_modules=True)

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

        eager_total_loss = eager_outputs[0].mean() + eager_outputs[1].mean()
        compiled_total_loss = compiled_outputs[0].mean() + compiled_outputs[1].mean()
        eager_total_loss.backward()
        compiled_total_loss.backward()
        torch.testing.assert_close(
            {".".join(part for part in name.split(".") if part != "_orig_mod"): parameter.grad
             for name, parameter in compiled_policy.named_parameters()},
            {name: parameter.grad for name, parameter in eager_policy.named_parameters()},
        )
        assert any(
            parameter.grad is not None
            for parameter in compiled_policy.action_dist.parameters()
            if parameter.requires_grad
        )
