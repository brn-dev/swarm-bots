from __future__ import annotations

from gymnasium.vector import VectorEnv
import pytest
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution

from examples.train_standalone_ppo import (
    ActorCritic,
    bootstrap_values,
    collect_rollout,
    compute_gae,
    joint_log_prob,
    policy_inputs,
    update_policy,
)
from swarmbots import make_env


@pytest.fixture
def observations() -> dict[str, torch.Tensor]:
    generator = torch.Generator().manual_seed(42)
    return {
        "local_obs": torch.randn(2, 3, 4, generator=generator),
        "global_obs": torch.randn(2, 2, generator=generator),
        "hidden_local_vars": torch.randn(2, 3, 2, generator=generator),
        "hidden_global_vars": torch.randn(2, 3, generator=generator),
        "agent_mask": torch.tensor([[True, True, False], [True, False, True]]),
    }


@pytest.fixture
def model() -> ActorCritic:
    return ActorCritic(input_size=6, action_sizes=(2, 1))


def test_actor_uses_public_observations_and_controls_both_action_fields(
    model: ActorCritic, observations: dict[str, torch.Tensor],
) -> None:
    public_observations = {key: observations[key] for key in ("local_obs", "global_obs", "agent_mask")}
    with torch.no_grad():
        actions = model.act(public_observations, torch.ones(2, dtype=torch.bool))
        observations["hidden_local_vars"].fill_(1000.0)
        observations["hidden_global_vars"].fill_(-1000.0)
        with_privileged_state = model.act(observations, torch.ones(2, dtype=torch.bool))
        distribution, _ = model(*policy_inputs(observations))
        sampled_actions = model.env_actions(distribution.sample(), observations["agent_mask"])
    for key, expected_shape in (("actuators", (2, 3, 2)), ("connectors", (2, 3, 1))):
        torch.testing.assert_close(actions[key], with_privileged_state[key])
        assert sampled_actions[key].shape == expected_shape
        assert torch.isfinite(sampled_actions[key]).all()
        assert (sampled_actions[key].abs() <= 1.0).all()
        assert (sampled_actions[key][~observations["agent_mask"]] == 0.0).all()


def test_padding_cannot_change_team_likelihood_or_value(
    model: ActorCritic, observations: dict[str, torch.Tensor],
) -> None:
    distribution, original_values = model(*policy_inputs(observations))
    raw_actions = distribution.sample()
    original_log_probs = joint_log_prob(distribution, raw_actions, observations["agent_mask"])
    observations["local_obs"][~observations["agent_mask"]] = 1000.0
    observations["hidden_local_vars"][~observations["agent_mask"]] = -1000.0
    raw_actions[~observations["agent_mask"]] = 0.9
    distribution, new_values = model(*policy_inputs(observations))
    new_log_probs = joint_log_prob(distribution, raw_actions, observations["agent_mask"])
    torch.testing.assert_close(new_log_probs, original_log_probs)
    torch.testing.assert_close(new_values, original_values)


def test_gaussian_ratio_matches_squashed_action_density(observations: dict[str, torch.Tensor]) -> None:
    raw_actions = torch.tensor([-1.2, 0.5, 1.8]).expand(2, 3, -1)
    old_distribution = Normal(torch.zeros_like(raw_actions), 1.0)
    new_distribution = Normal(torch.full_like(raw_actions, 0.15), 0.8)
    mask = observations["agent_mask"]
    ratio = (joint_log_prob(new_distribution, raw_actions, mask) - joint_log_prob(old_distribution, raw_actions, mask)).exp()
    old_squashed = TransformedDistribution(old_distribution, TanhTransform())
    new_squashed = TransformedDistribution(new_distribution, TanhTransform())
    log_ratio = new_squashed.log_prob(raw_actions.tanh()) - old_squashed.log_prob(raw_actions.tanh())
    expected = log_ratio.sum(dim=-1).masked_fill(~mask, 0.0).sum(dim=-1).exp()
    torch.testing.assert_close(ratio, expected)


def test_saturated_actions_keep_finite_consistent_likelihoods(
    model: ActorCritic, observations: dict[str, torch.Tensor],
) -> None:
    inputs, mask = policy_inputs(observations)
    distribution, _ = model(inputs, mask)
    raw_actions = torch.full_like(distribution.mean, 30.0)
    old_log_probs = joint_log_prob(distribution, raw_actions, mask)
    recomputed_distribution, _ = model(inputs, mask)
    log_probs = joint_log_prob(recomputed_distribution, raw_actions, mask)
    assert torch.isfinite(log_probs).all()
    torch.testing.assert_close((log_probs - old_log_probs).exp(), torch.ones(2))
    for actions in model.env_actions(raw_actions, mask).values():
        assert (actions[mask] == 1.0).all()


@pytest.mark.parametrize("also_truncated", [False, True])
def test_bootstrap_uses_terminal_state_only_for_time_limits(
    model: ActorCritic, observations: dict[str, torch.Tensor], monkeypatch: pytest.MonkeyPatch, also_truncated: bool,
) -> None:
    def observation_forward(inputs: torch.Tensor, agent_mask: torch.Tensor) -> tuple[Normal, torch.Tensor]:
        return Normal(torch.zeros((*inputs.shape[:2], 3)), 1.0), inputs[:, 0, 0].clone()

    monkeypatch.setattr(model, "forward", observation_forward)
    observations["local_obs"][:, 0, 0] = torch.tensor([100.0, 200.0])
    terminal_observations = {key: value.clone() for key, value in observations.items()}
    terminal_observations["local_obs"][:, 0, 0] = torch.tensor([10.0, 20.0])
    terminations = torch.tensor([True, False])
    truncations = torch.tensor([also_truncated, True])
    info = {"final_obs": terminal_observations, "_final_obs": torch.ones(2, dtype=torch.bool)}
    torch.testing.assert_close(
        bootstrap_values(model, observations, terminations, truncations, info), torch.tensor([0.0, 20.0]),
    )
    torch.testing.assert_close(
        bootstrap_values(model, observations, torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}),
        torch.tensor([100.0, 200.0]),
    )


def test_advantage_traces_stop_at_termination_and_truncation() -> None:
    rollout = {
        "values": torch.tensor([[2.0, 3.0, 4.0], [10.0, 20.0, 5.0]]),
        "rewards": torch.tensor([[1.0, 2.0, 3.0], [100.0, 200.0, 6.0]]),
        "next_values": torch.tensor([[0.0, 8.0, 5.0], [0.0, 0.0, 0.0]]),
        "dones": torch.tensor([[True, True, False], [True, True, True]]),
    }
    advantages, returns = compute_gae(rollout, gamma=1.0, gae_lambda=1.0)
    # The first two lanes start new episodes at step 1; only the third lane can include its reward.
    torch.testing.assert_close(advantages, torch.tensor([[-1.0, 7.0, 5.0], [90.0, 180.0, 1.0]]))
    torch.testing.assert_close(returns, torch.tensor([[1.0, 10.0, 9.0], [100.0, 200.0, 6.0]]))


class ReusingBufferEnv(VectorEnv):
    num_envs = 2

    def __init__(self, observations: dict[str, torch.Tensor]) -> None:
        self.observations = observations
        self.rewards = torch.zeros(self.num_envs)
        self.step_count = 0

    def step(
        self, actions: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
        self.step_count += 1
        self.observations["local_obs"].add_(1.0)
        self.observations["agent_mask"][:, 1:].logical_not_()
        self.rewards.fill_(self.step_count)
        return self.observations, self.rewards, torch.zeros(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool), {}


def test_rollout_snapshots_reused_buffers_and_updates_actor_and_critic(
    model: ActorCritic, observations: dict[str, torch.Tensor],
) -> None:
    original = {key: value.clone() for key, value in observations.items()}
    env = ReusingBufferEnv(observations)
    rollout, next_observations = collect_rollout(env, model, observations, 2)
    assert next_observations is observations
    torch.testing.assert_close(rollout["inputs"][0, ..., :4], original["local_obs"])
    torch.testing.assert_close(rollout["inputs"][1, ..., :4], original["local_obs"] + 1.0)
    torch.testing.assert_close(rollout["agent_mask"][0], original["agent_mask"])
    assert not torch.equal(rollout["agent_mask"][0], rollout["agent_mask"][1])
    torch.testing.assert_close(rollout["rewards"], torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    actor_before = [parameter.detach().clone() for parameter in model.actor.parameters()]
    critic_before = [parameter.detach().clone() for parameter in model.critic.parameters()]
    update_policy(model, torch.optim.Adam(model.parameters(), lr=1e-3), rollout)
    assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, model.actor.parameters()))
    assert any(not torch.equal(before, after) for before, after in zip(critic_before, model.critic.parameters()))


@pytest.mark.integration
def test_real_environment_training_through_autoreset() -> None:
    env = make_env(
        "SwarmBots-MoveToGoal-v0", num_envs=2, device="cpu", seed=7, episode_length=2,
        scenario_kwargs={"compile_reward_kernel": False}, compile_tensor_operations=False,
    )
    try:
        observations, _ = env.reset(seed=7)
        inputs, _ = policy_inputs(observations)
        action_sizes = (env.single_action_space["actuators"].shape[-1], env.single_action_space["connectors"].shape[-1])
        model = ActorCritic(inputs.shape[-1], action_sizes)
        rollout, observations = collect_rollout(env, model, observations, 3)
        assert rollout["dones"][1].all()
        update_policy(model, torch.optim.Adam(model.parameters()), rollout)
        assert all(torch.isfinite(parameter).all() for parameter in model.parameters())
        public_observations = {key: observations[key] for key in ("local_obs", "global_obs", "agent_mask")}
        starts = torch.zeros(2, dtype=torch.bool)
        actions = model.act(public_observations, starts)
        for key, value in actions.items():
            assert value.shape == env.action_space[key].shape
            assert torch.isfinite(value).all()
    finally:
        env.close()
