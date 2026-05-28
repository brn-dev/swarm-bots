import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.sticky_left_right_beta_action_dist import StickyLeftRightBetaConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat_qcc.mat_qcc_decoder import MATQCCDecoderConfig
from swarmbots.learn.algos.mat_qcc.mat_qcc_policy import MATQCCPolicy, MATQCCPolicyConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace


class _DummyMATQCCEnv:
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


def _make_policy(*, assume_agent_mask_is_active_prefix: bool = False) -> MATQCCPolicy:
    return MATQCCPolicy(
        env=_DummyMATQCCEnv(),
        config=MATQCCPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
            ),
            decoder_config=MATQCCDecoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                query_encoder_hidden_dims=[16],
                context_encoder_hidden_dims=[16],
                memory_dims=None,
                assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
            ),
            continuous_config=StickyLeftRightBetaConfig(stickiness=0.25),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
            max_agents=8,
            compile_modules=False,
        ),
    )


def test_qcc_policy_forward_and_evaluate_actions_handle_arbitrary_masks() -> None:
    torch.manual_seed(0)
    policy = _make_policy()

    batch_size = 4
    action_dim = _DummyMATQCCEnv.action_space.total_agent_action_dim
    local_obs = torch.randn(batch_size, _DummyMATQCCEnv.n_agents, _DummyMATQCCEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATQCCEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATQCCEnv.n_agents, _DummyMATQCCEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATQCCEnv.hidden_global_vars_dim)
    previous_actions = torch.randn(batch_size, _DummyMATQCCEnv.n_agents, action_dim)
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


def test_qcc_policy_uses_qcc_hyperparameter_key() -> None:
    policy = _make_policy()

    hyper_parameters = policy.get_hyper_parameters()

    assert "mat_qcc_policy_config" in hyper_parameters
    assert "mat_policy_config" not in hyper_parameters
