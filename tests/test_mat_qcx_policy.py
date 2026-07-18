import shutil
import sys
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import pytest
import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.sticky_sign_magnitude_beta_action_dist import StickySignMagnitudeBetaConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_policy import MATQCXPolicy, MATQCXPolicyConfig
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples
from swarmbots.learn.hybrid_action_space import HybridActionSpace


_REAL_TORCH_COMPILE = torch.compile


def _compile_with_eager_backend(
        function: Callable[..., Any],
        **kwargs: Any,
) -> Callable[..., Any]:
    return _REAL_TORCH_COMPILE(function, backend="eager", **kwargs)


class _DummyMATQCXEnv:
    n_agents = 5
    local_obs_dim = 6
    global_obs_dim = 3
    hidden_local_vars_dim = 2
    hidden_global_vars_dim = 4
    action_space = HybridActionSpace(
        {
            "cont": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 2), dtype=float),
            "disc": spaces.MultiBinary((n_agents, 1)),
        }
    )


def _make_policy(
        *,
        assume_agent_mask_is_active_prefix: bool = False,
        add_agent_embeddings: bool = False,
        compile_modules: bool = False,
        ent_loss_coef: float = 0.0,
) -> MATQCXPolicy:
    return MATQCXPolicy(
        env=_DummyMATQCXEnv(),
        config=MATQCXPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
            ),
            decoder_config=MATQCXDecoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                context_encoder_hidden_dims=[16],
                action_encoder_dims=[16, 16],
                memory_dims=None,
                add_agent_embeddings=add_agent_embeddings,
                assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
            ),
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


def _random_valid_actions(batch_size: int) -> torch.Tensor:
    actions = torch.empty(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.action_space.total_agent_action_dim)
    actions[..., :2] = torch.empty_like(actions[..., :2]).uniform_(-0.8, 0.8)
    actions[..., 2:] = torch.randint(0, 2, actions[..., 2:].shape, dtype=actions.dtype)
    return actions


def _make_samples(
        *,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        hidden_local_vars: torch.Tensor,
        hidden_global_vars: torch.Tensor,
        agent_mask: torch.Tensor,
        actions: torch.Tensor,
        previous_actions: torch.Tensor | None = None,
) -> PPOSamples:
    batch_size = int(local_obs.shape[0])
    return PPOSamples(
        local_obs=local_obs,
        global_obs=global_obs,
        hidden_local_vars=hidden_local_vars,
        hidden_global_vars=hidden_global_vars,
        agent_mask=agent_mask,
        previous_actions=torch.zeros_like(actions) if previous_actions is None else previous_actions,
        actions=actions,
        log_probs=torch.zeros(actions.shape[:2], dtype=actions.dtype),
        values=torch.zeros(batch_size, dtype=actions.dtype),
        returns=torch.zeros(batch_size, dtype=actions.dtype),
        advantages=torch.zeros(batch_size, dtype=actions.dtype),
    )


def _evaluate_log_probs_and_values(
        policy: MATQCXPolicy,
        *,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        hidden_local_vars: torch.Tensor,
        hidden_global_vars: torch.Tensor,
        agent_mask: torch.Tensor,
        actions: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    log_probs, values, *_ = policy.evaluate_actions(
        _make_samples(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )
    )
    return log_probs, values


def test_qcx_policy_forward_and_evaluate_actions_handle_arbitrary_masks() -> None:
    torch.manual_seed(0)
    policy = _make_policy()

    batch_size = 4
    action_dim = _DummyMATQCXEnv.action_space.total_agent_action_dim
    local_obs = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATQCXEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATQCXEnv.hidden_global_vars_dim)
    previous_actions = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, action_dim)
    agent_mask = torch.tensor(
        [
            [False, True, True, False, True],
            [False, False, False, True, False],
            [True, False, True, False, False],
            [False, False, True, True, False],
        ]
    )
    previous_actions = previous_actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

    with torch.no_grad():
        actions, log_probs, values = policy(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=False,
        )

        eval_log_probs, eval_values, *_ = policy.evaluate_actions(
            _make_samples(
                local_obs=local_obs,
                global_obs=global_obs,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                previous_actions=previous_actions,
                actions=actions,
            )
        )

    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(eval_log_probs).all()
    assert torch.isfinite(eval_values).all()
    assert torch.allclose(eval_log_probs[agent_mask], log_probs[agent_mask], rtol=0.0, atol=1e-5)
    assert torch.allclose(eval_values, values, rtol=0.0, atol=1e-6)


def test_qcx_policy_uses_qcx_hyperparameter_key() -> None:
    policy = _make_policy()

    hyper_parameters = policy.get_hyper_parameters()

    assert "mat_qcx_policy_config" in hyper_parameters
    assert "mat_qcs_policy_config" not in hyper_parameters


def test_qcx_policy_rejects_decoder_agent_embeddings() -> None:
    with pytest.raises(ValueError, match="does not support add_agent_embeddings"):
        _make_policy(add_agent_embeddings=True)


def test_qcx_policy_action_encoder_configuration_constructs_and_runs() -> None:
    torch.manual_seed(1)
    policy = _make_policy()
    policy.eval()

    with torch.no_grad():
        actions, log_probs, values = policy(
            local_obs=torch.randn(2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim),
            global_obs=torch.randn(2, _DummyMATQCXEnv.global_obs_dim),
            hidden_local_vars=torch.randn(2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.hidden_local_vars_dim),
            hidden_global_vars=torch.randn(2, _DummyMATQCXEnv.hidden_global_vars_dim),
            agent_mask=torch.ones(2, _DummyMATQCXEnv.n_agents, dtype=torch.bool),
            previous_actions=torch.zeros(2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.action_space.total_agent_action_dim),
            deterministic=True,
        )

    assert actions.shape == (2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.action_space.total_agent_action_dim)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()


def test_qcx_policy_memory_encoder_configuration_constructs_and_runs() -> None:
    torch.manual_seed(2)
    policy = MATQCXPolicy(
        env=_DummyMATQCXEnv(),
        config=MATQCXPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
            ),
            decoder_config=MATQCXDecoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                memory_dims=[16],
                memory_encoder_end_with_act_fn=True,
                assume_agent_mask_is_active_prefix=False,
            ),
            continuous_config=StickySignMagnitudeBetaConfig(stickiness=0.25),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
            max_agents=8,
        ),
    )
    policy.eval()

    with torch.no_grad():
        actions, log_probs, values = policy(
            local_obs=torch.randn(2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim),
            global_obs=torch.randn(2, _DummyMATQCXEnv.global_obs_dim),
            hidden_local_vars=torch.randn(2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.hidden_local_vars_dim),
            hidden_global_vars=torch.randn(2, _DummyMATQCXEnv.hidden_global_vars_dim),
            agent_mask=torch.ones(2, _DummyMATQCXEnv.n_agents, dtype=torch.bool),
            previous_actions=torch.zeros(2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.action_space.total_agent_action_dim),
            deterministic=True,
        )

    assert actions.shape == (2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.action_space.total_agent_action_dim)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()


def test_qcx_policy_evaluate_ignores_inactive_context_actions() -> None:
    torch.manual_seed(3)
    policy = _make_policy()
    policy.eval()

    batch_size = 3
    local_obs = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATQCXEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATQCXEnv.hidden_global_vars_dim)
    actions = _random_valid_actions(batch_size)
    agent_mask = torch.tensor(
        [
            [False, True, True, False, True],
            [True, False, True, False, True],
            [False, False, True, True, False],
        ]
    )
    modified_actions = actions.clone()
    modified_actions[~agent_mask] = torch.randn_like(modified_actions[~agent_mask])
    modified_actions[..., :2].clamp_(-0.8, 0.8)
    modified_actions[..., 2:] = (modified_actions[..., 2:] > 0.0).to(modified_actions.dtype)

    with torch.no_grad():
        log_probs, values = _evaluate_log_probs_and_values(
            policy,
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )
        modified_log_probs, modified_values = _evaluate_log_probs_and_values(
            policy,
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=modified_actions,
        )

    torch.testing.assert_close(modified_log_probs[agent_mask], log_probs[agent_mask], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(modified_values, values, rtol=0.0, atol=0.0)


def test_qcx_policy_evaluate_ignores_inactive_observations() -> None:
    torch.manual_seed(4)
    policy = _make_policy()
    policy.eval()

    batch_size = 3
    local_obs = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATQCXEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATQCXEnv.hidden_global_vars_dim)
    actions = _random_valid_actions(batch_size)
    agent_mask = torch.tensor(
        [
            [False, True, True, False, True],
            [True, False, True, False, True],
            [False, False, True, True, False],
        ]
    )
    modified_local_obs = local_obs.clone()
    modified_hidden_local_vars = hidden_local_vars.clone()
    modified_local_obs[~agent_mask] = torch.randn_like(modified_local_obs[~agent_mask])
    modified_hidden_local_vars[~agent_mask] = torch.randn_like(modified_hidden_local_vars[~agent_mask])

    with torch.no_grad():
        log_probs, values = _evaluate_log_probs_and_values(
            policy,
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )
        modified_log_probs, modified_values = _evaluate_log_probs_and_values(
            policy,
            local_obs=modified_local_obs,
            global_obs=global_obs,
            hidden_local_vars=modified_hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )

    torch.testing.assert_close(modified_log_probs[agent_mask], log_probs[agent_mask], rtol=0.0, atol=1e-5)
    torch.testing.assert_close(modified_values, values, rtol=0.0, atol=1e-6)


def test_qcx_policy_evaluate_does_not_depend_on_future_actions() -> None:
    torch.manual_seed(5)
    policy = _make_policy()
    policy.eval()

    batch_size = 3
    local_obs = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATQCXEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATQCXEnv.hidden_global_vars_dim)
    actions = _random_valid_actions(batch_size)
    agent_mask = torch.ones(batch_size, _DummyMATQCXEnv.n_agents, dtype=torch.bool)
    modified_actions = actions.clone()
    modified_actions[:, 2:, :] = torch.randn_like(modified_actions[:, 2:, :])
    modified_actions[..., :2].clamp_(-0.8, 0.8)
    modified_actions[..., 2:] = (modified_actions[..., 2:] > 0.0).to(modified_actions.dtype)

    with torch.no_grad():
        log_probs, values = _evaluate_log_probs_and_values(
            policy,
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )
        modified_log_probs, modified_values = _evaluate_log_probs_and_values(
            policy,
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=modified_actions,
        )

    torch.testing.assert_close(modified_log_probs[:, :2], log_probs[:, :2], rtol=0.0, atol=1e-5)
    torch.testing.assert_close(modified_values, values, rtol=0.0, atol=0.0)


def test_qcx_policy_compile_modules_constructs_when_supported() -> None:
    if sys.platform == "win32" and shutil.which("cl") is None:
        pytest.skip("torch.compile requires cl.exe on this Windows setup")

    policy = _make_policy(compile_modules=True)

    assert policy.config.compile_modules


def test_compiled_qc_evaluation_refreshes_distribution_state_for_extra_losses() -> None:
    torch.manual_seed(41)
    eager_policy = _make_policy(ent_loss_coef=0.2)
    torch.manual_seed(41)
    with (
        patch(
            "swarmbots.learn.algos.mat_qc_base_policy._ensure_torch_compile_available",
        ),
        patch(
            "swarmbots.learn.algos.mat_qc_base_policy.torch.compile",
            side_effect=_compile_with_eager_backend,
        ),
    ):
        compiled_policy = _make_policy(compile_modules=True, ent_loss_coef=0.2)

        batch_size = 3
        local_obs = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim)
        global_obs = torch.randn(batch_size, _DummyMATQCXEnv.global_obs_dim)
        hidden_local_vars = torch.randn(
            batch_size,
            _DummyMATQCXEnv.n_agents,
            _DummyMATQCXEnv.hidden_local_vars_dim,
        )
        hidden_global_vars = torch.randn(batch_size, _DummyMATQCXEnv.hidden_global_vars_dim)
        agent_mask = torch.tensor([
            [False, True, True, False, True],
            [True, True, False, False, True],
            [True, False, True, True, False],
        ])
        actions = _random_valid_actions(batch_size)

        for observation_offset in (0.0, 0.25):
            samples = _make_samples(
                local_obs=local_obs + observation_offset,
                global_obs=global_obs - observation_offset,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
                actions=actions,
            )
            eager_outputs = eager_policy.evaluate_actions(samples)
            compiled_outputs = compiled_policy.evaluate_actions(samples)
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
        action_dist_gradients = [
            parameter.grad
            for parameter in compiled_policy.action_dist.parameters()
            if parameter.requires_grad
        ]
        assert any(gradient is not None for gradient in action_dist_gradients)
        assert all(
            gradient is None or torch.isfinite(gradient).all()
            for gradient in action_dist_gradients
        )


def test_qcx_policy_can_use_identity_action_encoder() -> None:
    torch.manual_seed(6)
    action_dim = _DummyMATQCXEnv.action_space.total_agent_action_dim
    policy = MATQCXPolicy(
        env=_DummyMATQCXEnv(),
        config=MATQCXPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=8,
                nhead=1,
                num_layers=1,
                dim_feedforward=16,
            ),
            decoder_config=MATQCXDecoderConfig(
                d_model=action_dim + 1,
                nhead=1,
                num_layers=1,
                dim_feedforward=(action_dim + 1) * 2,
                action_encoder_dims=[],
                memory_dims=None,
                assume_agent_mask_is_active_prefix=False,
            ),
            continuous_config=StickySignMagnitudeBetaConfig(stickiness=0.25),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
            max_agents=8,
            compile_modules=False,
        ),
    )
    policy.eval()

    with torch.no_grad():
        actions, log_probs, values = policy(
            local_obs=torch.randn(2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim),
            global_obs=torch.randn(2, _DummyMATQCXEnv.global_obs_dim),
            hidden_local_vars=torch.randn(2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.hidden_local_vars_dim),
            hidden_global_vars=torch.randn(2, _DummyMATQCXEnv.hidden_global_vars_dim),
            agent_mask=torch.ones(2, _DummyMATQCXEnv.n_agents, dtype=torch.bool),
            previous_actions=torch.zeros(2, _DummyMATQCXEnv.n_agents, action_dim),
            deterministic=True,
        )

    assert actions.shape == (2, _DummyMATQCXEnv.n_agents, action_dim)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()
    assert policy.decoder.action_d_model == action_dim


def test_qcx_policy_accepts_action_encoder_dim_different_from_decoder_model_dim() -> None:
    torch.manual_seed(7)
    action_dim = _DummyMATQCXEnv.action_space.total_agent_action_dim
    action_token_dim = action_dim + 2
    policy = MATQCXPolicy(
        env=_DummyMATQCXEnv(),
        config=MATQCXPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=8,
                nhead=1,
                num_layers=1,
                dim_feedforward=16,
            ),
            decoder_config=MATQCXDecoderConfig(
                d_model=action_dim + 1,
                nhead=1,
                num_layers=1,
                dim_feedforward=(action_dim + 1) * 2,
                action_encoder_dims=[action_token_dim],
                memory_dims=None,
                assume_agent_mask_is_active_prefix=False,
            ),
            continuous_config=StickySignMagnitudeBetaConfig(stickiness=0.25),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
            max_agents=8,
            compile_modules=False,
        ),
    )
    policy.eval()

    with torch.no_grad():
        actions, log_probs, values = policy(
            local_obs=torch.randn(2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim),
            global_obs=torch.randn(2, _DummyMATQCXEnv.global_obs_dim),
            hidden_local_vars=torch.randn(2, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.hidden_local_vars_dim),
            hidden_global_vars=torch.randn(2, _DummyMATQCXEnv.hidden_global_vars_dim),
            agent_mask=torch.ones(2, _DummyMATQCXEnv.n_agents, dtype=torch.bool),
            previous_actions=torch.zeros(2, _DummyMATQCXEnv.n_agents, action_dim),
            deterministic=True,
        )

    assert actions.shape == (2, _DummyMATQCXEnv.n_agents, action_dim)
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()
    assert policy.decoder.action_d_model == action_token_dim
