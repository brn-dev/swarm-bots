import shutil
import sys

import pytest
import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.sticky_sign_magnitude_beta_action_dist import StickySignMagnitudeBetaConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_policy import MATQCXPolicy, MATQCXPolicyConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace


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
                action_encoder_hidden_dims=[16],
                memory_dims=None,
                add_agent_embeddings=add_agent_embeddings,
                assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
            ),
            continuous_config=StickySignMagnitudeBetaConfig(stickiness=0.25),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
            max_agents=8,
            compile_modules=compile_modules,
        ),
    )


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

        class _Batch:
            pass

        batch = _Batch()
        batch.actions = actions
        batch.local_obs = local_obs
        batch.global_obs = global_obs
        batch.hidden_local_vars = hidden_local_vars
        batch.hidden_global_vars = hidden_global_vars
        batch.agent_mask = agent_mask
        batch.previous_actions = previous_actions

        eval_log_probs, eval_values, *_ = policy.evaluate_actions(batch)

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


def test_qcx_policy_action_encoder_ends_with_activation() -> None:
    policy = _make_policy()

    assert isinstance(list(policy.action_encoder.children())[-1], nn.GELU)


def test_qcx_policy_memory_encoder_uses_explicit_memory_activation_flag() -> None:
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

    assert isinstance(list(policy.memory_encoder.children())[-1], nn.GELU)


def test_qcx_policy_evaluate_ignores_inactive_context_actions() -> None:
    torch.manual_seed(1)
    policy = _make_policy()
    policy.eval()

    batch_size = 3
    action_dim = _DummyMATQCXEnv.action_space.total_agent_action_dim
    local_obs = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATQCXEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATQCXEnv.hidden_global_vars_dim)
    actions = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, action_dim)
    agent_mask = torch.tensor(
        [
            [False, True, True, False, True],
            [True, False, True, False, True],
            [False, False, True, True, False],
        ]
    )
    modified_actions = actions.clone()
    modified_actions[~agent_mask] = torch.randn_like(modified_actions[~agent_mask])

    with torch.no_grad():
        latent_pi, values, _ = policy._evaluate_latent_and_values(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )
        modified_latent_pi, modified_values, _ = policy._evaluate_latent_and_values(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=modified_actions,
        )

    torch.testing.assert_close(modified_latent_pi, latent_pi, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(modified_values, values, rtol=0.0, atol=0.0)


def test_qcx_policy_evaluate_ignores_inactive_observations() -> None:
    torch.manual_seed(2)
    policy = _make_policy()
    policy.eval()

    batch_size = 3
    action_dim = _DummyMATQCXEnv.action_space.total_agent_action_dim
    local_obs = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATQCXEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATQCXEnv.hidden_global_vars_dim)
    actions = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, action_dim)
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
        latent_pi, values, _ = policy._evaluate_latent_and_values(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )
        modified_latent_pi, modified_values, _ = policy._evaluate_latent_and_values(
            local_obs=modified_local_obs,
            global_obs=global_obs,
            hidden_local_vars=modified_hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )

    torch.testing.assert_close(modified_latent_pi[agent_mask], latent_pi[agent_mask], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(modified_values, values, rtol=0.0, atol=1e-6)


def test_qcx_policy_evaluate_does_not_depend_on_future_actions() -> None:
    torch.manual_seed(3)
    policy = _make_policy()
    policy.eval()

    batch_size = 3
    action_dim = _DummyMATQCXEnv.action_space.total_agent_action_dim
    local_obs = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATQCXEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, _DummyMATQCXEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATQCXEnv.hidden_global_vars_dim)
    actions = torch.randn(batch_size, _DummyMATQCXEnv.n_agents, action_dim)
    agent_mask = torch.ones(batch_size, _DummyMATQCXEnv.n_agents, dtype=torch.bool)
    modified_actions = actions.clone()
    modified_actions[:, 2:, :] = torch.randn_like(modified_actions[:, 2:, :])

    with torch.no_grad():
        latent_pi, values, _ = policy._evaluate_latent_and_values(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
        )
        modified_latent_pi, modified_values, _ = policy._evaluate_latent_and_values(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=modified_actions,
        )

    torch.testing.assert_close(modified_latent_pi[:, :3, :], latent_pi[:, :3, :], rtol=0.0, atol=1e-6)
    torch.testing.assert_close(modified_values, values, rtol=0.0, atol=0.0)


def test_qcx_policy_compile_modules_constructs_when_supported() -> None:
    if sys.platform == "win32" and shutil.which("cl") is None:
        pytest.skip("torch.compile requires cl.exe on this Windows setup")

    policy = _make_policy(compile_modules=True)

    assert policy.config.compile_modules


def test_qcx_policy_can_skip_action_projection_when_dims_match() -> None:
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
                d_model=action_dim,
                nhead=1,
                num_layers=1,
                dim_feedforward=action_dim * 2,
                project_actions=False,
                memory_dims=None,
                assume_agent_mask_is_active_prefix=False,
            ),
            continuous_config=StickySignMagnitudeBetaConfig(stickiness=0.25),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
            max_agents=8,
            compile_modules=False,
        ),
    )

    assert isinstance(policy.action_encoder, nn.Identity)


def test_qcx_policy_rejects_skipping_action_projection_when_dims_differ() -> None:
    action_dim = _DummyMATQCXEnv.action_space.total_agent_action_dim

    with pytest.raises(ValueError, match="project_actions=False requires agent action dim"):
        MATQCXPolicy(
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
                    project_actions=False,
                    memory_dims=None,
                    assume_agent_mask_is_active_prefix=False,
                ),
                continuous_config=StickySignMagnitudeBetaConfig(stickiness=0.25),
                bernoulli_config=BernoulliConfig(initial_prob=0.5),
                max_agents=8,
                compile_modules=False,
            ),
        )
