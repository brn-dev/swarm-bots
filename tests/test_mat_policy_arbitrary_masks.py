import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.sticky_left_right_beta_action_dist import StickyLeftRightBetaConfig
from swarmbots.learn.algos.mat.mat_decoder import MATDecoderConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat.mat_policy import MATPolicy, MATPolicyConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace


class _DummyMATEnv:
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


def _make_policy(*, assume_agent_mask_is_active_prefix: bool = False) -> MATPolicy:
    return MATPolicy(
        env=_DummyMATEnv(),
        config=MATPolicyConfig(
            encoder_config=MATEncoderConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
            ),
            decoder_config=MATDecoderConfig(
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


def test_prefix_and_arbitrary_mask_policy_paths_match_for_prefix_masks() -> None:
    torch.manual_seed(2)
    prefix_policy = _make_policy(assume_agent_mask_is_active_prefix=True)
    arbitrary_policy = _make_policy(assume_agent_mask_is_active_prefix=False)
    arbitrary_policy.load_state_dict(prefix_policy.state_dict())
    prefix_policy.eval()
    arbitrary_policy.eval()

    batch_size = 3
    action_dim = _DummyMATEnv.action_space.total_agent_action_dim
    local_obs = torch.randn(batch_size, _DummyMATEnv.n_agents, _DummyMATEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATEnv.n_agents, _DummyMATEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATEnv.hidden_global_vars_dim)
    previous_actions = torch.empty(batch_size, _DummyMATEnv.n_agents, action_dim).uniform_(-0.8, 0.8)
    previous_actions[..., -1:] = torch.randint(0, 2, (batch_size, _DummyMATEnv.n_agents, 1)).to(torch.float32)
    actions = torch.empty(batch_size, _DummyMATEnv.n_agents, action_dim).uniform_(-0.8, 0.8)
    actions[..., -1:] = torch.randint(0, 2, (batch_size, _DummyMATEnv.n_agents, 1)).to(torch.float32)
    agent_mask = torch.tensor(
        [
            [True, True, False, False, False],
            [True, True, True, False, False],
            [True, True, True, True, True],
        ]
    )
    previous_actions = previous_actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

    class _Batch:
        pass

    prefix_batch = _Batch()
    prefix_batch.actions = actions
    prefix_batch.local_obs = local_obs
    prefix_batch.global_obs = global_obs
    prefix_batch.hidden_local_vars = hidden_local_vars
    prefix_batch.hidden_global_vars = hidden_global_vars
    prefix_batch.agent_mask = agent_mask
    prefix_batch.previous_actions = previous_actions

    arbitrary_batch = _Batch()
    arbitrary_batch.actions = actions
    arbitrary_batch.local_obs = local_obs
    arbitrary_batch.global_obs = global_obs
    arbitrary_batch.hidden_local_vars = hidden_local_vars
    arbitrary_batch.hidden_global_vars = hidden_global_vars
    arbitrary_batch.agent_mask = agent_mask
    arbitrary_batch.previous_actions = previous_actions

    with torch.no_grad():
        prefix_actions, prefix_log_probs, prefix_values = prefix_policy(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=True,
        )
        arbitrary_actions, arbitrary_log_probs, arbitrary_values = arbitrary_policy(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=True,
        )
        prefix_eval_log_probs, prefix_eval_values, *_ = prefix_policy.evaluate_actions(prefix_batch)
        arbitrary_eval_log_probs, arbitrary_eval_values, *_ = arbitrary_policy.evaluate_actions(arbitrary_batch)

    torch.testing.assert_close(arbitrary_actions, prefix_actions, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(arbitrary_log_probs, prefix_log_probs, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(arbitrary_values, prefix_values, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(arbitrary_eval_log_probs, prefix_eval_log_probs, rtol=0.0, atol=1e-6)
    torch.testing.assert_close(arbitrary_eval_values, prefix_eval_values, rtol=0.0, atol=1e-6)


def test_forward_handles_arbitrary_agent_masks_without_nan() -> None:
    policy = _make_policy()
    torch.manual_seed(0)

    batch_size = 4
    action_dim = _DummyMATEnv.action_space.total_agent_action_dim
    local_obs = torch.randn(batch_size, _DummyMATEnv.n_agents, _DummyMATEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATEnv.n_agents, _DummyMATEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATEnv.hidden_global_vars_dim)
    previous_actions = torch.randn(batch_size, _DummyMATEnv.n_agents, action_dim)
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
            deterministic=True,
        )
        predicted_values = policy.predict_values(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )

    assert torch.isfinite(actions).all()
    assert torch.isfinite(log_probs).all()
    assert torch.isfinite(values).all()
    assert torch.isfinite(predicted_values).all()
    assert actions.shape == (batch_size, _DummyMATEnv.n_agents, action_dim)
    assert log_probs.shape == (batch_size, _DummyMATEnv.n_agents)
    assert values.shape == (batch_size,)


def test_evaluate_actions_matches_rollout_on_active_arbitrary_agent_slots() -> None:
    policy = _make_policy()
    torch.manual_seed(1)

    batch_size = 4
    action_dim = _DummyMATEnv.action_space.total_agent_action_dim
    local_obs = torch.randn(batch_size, _DummyMATEnv.n_agents, _DummyMATEnv.local_obs_dim)
    global_obs = torch.randn(batch_size, _DummyMATEnv.global_obs_dim)
    hidden_local_vars = torch.randn(batch_size, _DummyMATEnv.n_agents, _DummyMATEnv.hidden_local_vars_dim)
    hidden_global_vars = torch.randn(batch_size, _DummyMATEnv.hidden_global_vars_dim)
    previous_actions = torch.randn(batch_size, _DummyMATEnv.n_agents, action_dim)
    agent_mask = torch.tensor(
        [
            [False, True, False, True, False],
            [False, False, False, True, False],
            [True, False, True, False, True],
            [False, False, True, False, True],
        ]
    )
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

    assert torch.isfinite(eval_log_probs).all()
    assert torch.isfinite(eval_values).all()
    assert torch.allclose(eval_log_probs[agent_mask], rollout_log_probs[agent_mask], rtol=0.0, atol=1e-5)
    assert torch.allclose(eval_values, rollout_values, rtol=0.0, atol=1e-6)
