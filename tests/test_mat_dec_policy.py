import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.sticky_left_right_beta_action_dist import StickyLeftRightBetaConfig
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
        {
            "cont": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 2), dtype=float),
            "disc": spaces.MultiBinary((n_agents, 1)),
        }
    )


def _make_policy() -> MATDecPolicy:
    return MATDecPolicy(
        env=_DummyMATDecEnv(),
        config=MATDecPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
            ),
            actor_head_hidden_dims=[16],
            continuous_config=StickyLeftRightBetaConfig(stickiness=0.25),
            bernoulli_config=BernoulliConfig(initial_prob=0.5),
            max_agents=8,
        ),
    )


def test_forward_and_value_paths_match_pipeline_shapes() -> None:
    policy = _make_policy()
    torch.manual_seed(0)

    batch_size = 3
    local_obs = torch.randn(batch_size, _DummyMATDecEnv.n_agents, _DummyMATDecEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATDecEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATDecEnv.n_agents, _DummyMATDecEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATDecEnv.hidden_global_vars_dim)
    agent_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, True, True, True],
        ]
    )
    previous_actions = torch.zeros(
        batch_size,
        _DummyMATDecEnv.n_agents,
        _DummyMATDecEnv.action_space.total_agent_action_dim,
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
        _DummyMATDecEnv.n_agents,
        _DummyMATDecEnv.action_space.total_agent_action_dim,
    )
    assert log_probs.shape == (batch_size, _DummyMATDecEnv.n_agents)
    assert values.shape == (batch_size,)
    assert predicted_values.shape == (batch_size,)
    assert torch.allclose(values, predicted_values, rtol=0.0, atol=1e-6)
    assert torch.equal(actions[~agent_mask], torch.zeros_like(actions[~agent_mask]))
    assert torch.equal(log_probs[~agent_mask], torch.zeros_like(log_probs[~agent_mask]))


def test_evaluate_actions_matches_rollout_log_probs_for_masked_agents() -> None:
    policy = _make_policy()
    torch.manual_seed(1)

    batch_size = 4
    action_dim = _DummyMATDecEnv.action_space.total_agent_action_dim
    local_obs = torch.randn(batch_size, _DummyMATDecEnv.n_agents, _DummyMATDecEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATDecEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATDecEnv.n_agents, _DummyMATDecEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATDecEnv.hidden_global_vars_dim)
    agent_mask = torch.tensor(
        [
            [True, True, True, False],
            [True, True, False, False],
            [True, False, False, False],
            [True, True, True, True],
        ]
    )
    previous_actions = torch.randn(batch_size, _DummyMATDecEnv.n_agents, action_dim)
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
