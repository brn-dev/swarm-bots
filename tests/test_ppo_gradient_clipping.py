from collections.abc import Iterator
from copy import deepcopy
import math
from types import SimpleNamespace

import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamplerConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig, NextObsPredWrapper
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as SwarmTestEnv
from tests.test_model_scale import make_policy
from tests.test_ppo_policy_dropout import _make_env, _make_policy


DEVICES = ["cpu", pytest.param("cuda", marks=(
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable"),
))]


@pytest.fixture(scope="module", autouse=True)
def single_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _two_step_segment(algorithm: PPO, observations: dict[str, torch.Tensor]) -> PPOEpisodeSegment:
    device = algorithm.train_device
    algorithm.policy.to(device).eval()
    with torch.no_grad():
        actions, log_probs, values = algorithm.policy(**observations)
    return PPOEpisodeSegment(
        **observations, actions=actions, log_probs=log_probs, values=values,
        rewards=torch.zeros(2, device=device), returns=torch.ones_like(values),
        advantages=torch.tensor([1.0, -1.0], device=device),
        final_local_obs=None, final_global_obs=None, final_hidden_local_vars=None,
        final_hidden_global_vars=None, final_agent_mask=None, final_value=None,
    )


def _zero_observation_segment(algorithm: PPO) -> PPOEpisodeSegment:
    env, device = algorithm.env, algorithm.train_device
    return _two_step_segment(algorithm, dict(
        local_obs=torch.zeros(2, env.n_agents, env.local_obs_dim, device=device),
        global_obs=torch.zeros(2, env.global_obs_dim, device=device),
        hidden_local_vars=torch.zeros(2, env.n_agents, env.hidden_local_vars_dim, device=device),
        hidden_global_vars=torch.full((2, env.hidden_global_vars_dim), float(env.n_agents), device=device),
        agent_mask=torch.ones(2, env.n_agents, dtype=torch.bool, device=device),
    ))


def _make_nop_policy(env: SimpleNamespace) -> NextObsPredWrapper:
    policy = _make_policy(env)
    return NextObsPredWrapper(policy, world_model_config=NOPWorldModelConfig(
        n_agents=env.n_agents, local_latent_dim=policy.nop_latent_dim,
        action_dim=env.action_space.total_agent_action_dim,
        d_model_transition_model=8, nhead_transition_model=2, num_layers_transition_model=1,
        dim_feedforward_transition_model=16,
        next_obs_pred_config=NextObsPredConfig(local_scalar_target_indices=[0]),
    ))


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value", [4.0, 4e20, float("inf"), float("nan")])
def test_nop_total_includes_auxiliary_gradients_once_without_mutation(device: str, value: float) -> None:
    policy = _make_nop_policy(_make_env()).to(device)
    policy.shared_predictor_alias = policy.local_scalars_predictor
    core_parameter = next(policy.policy.parameters())
    auxiliary_parameter = next(policy.local_scalars_predictor.parameters())
    core_parameter.grad = torch.full_like(core_parameter, 3.0)
    auxiliary_parameter.grad = torch.full_like(auxiliary_parameter, value)
    expected = torch.cat([core_parameter.grad.double().flatten(), auxiliary_parameter.grad.double().flatten()]).norm()
    gradients = {parameter: parameter.grad for parameter in policy.parameters()}
    before = {parameter: gradient.clone() for parameter, gradient in gradients.items() if gradient is not None}

    metrics = policy.get_grad_norms()

    if math.isnan(value):
        assert math.isnan(metrics["total"])
    else:
        assert metrics["total"] == pytest.approx(expected.item(), rel=1e-12)
    for parameter, gradient in gradients.items():
        assert parameter.grad is gradient
        if gradient is not None:
            torch.testing.assert_close(gradient, before[parameter], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("device", DEVICES)
def test_nop_ppo_training_reports_the_full_preclip_total(device: str) -> None:
    torch.manual_seed(101)
    env = _make_env()
    policy = _make_nop_policy(env).to(device)
    algorithm = PPO(
        policy, env, n_epochs=1, target_kl=None,
        sampler_config=PPOWMSamplerConfig(batch_size=2, num_next_steps=1),
        rollout_mode=StepsRolloutMode(2), train_device=device, rollout_device="cpu",
    )
    local_obs = torch.rand(2, env.n_agents, env.local_obs_dim, device=device) - 0.5
    local_obs[1] = local_obs[0] + 0.25
    segment = _two_step_segment(algorithm, dict(
        local_obs=local_obs, global_obs=torch.rand(2, env.global_obs_dim, device=device),
        hidden_local_vars=torch.empty(2, env.n_agents, env.hidden_local_vars_dim, device=device),
        hidden_global_vars=torch.empty(2, env.hidden_global_vars_dim, device=device),
        agent_mask=torch.ones(2, env.n_agents, dtype=torch.bool, device=device),
    ))
    segment.final_local_obs = local_obs[-1] + 0.25
    segment.final_global_obs = segment.global_obs[-1]
    segment.final_hidden_local_vars = segment.hidden_local_vars[-1]
    segment.final_hidden_global_vars = segment.hidden_global_vars[-1]
    segment.final_agent_mask = segment.agent_mask[-1]
    predictors_before = [parameter.detach().clone() for parameter in policy.local_scalars_predictor.parameters()]

    metrics = algorithm.train([segment])

    assert metrics["updates"] == 1
    assert metrics["grad_norm_total"].mean == pytest.approx(metrics["grad_norm"].mean, rel=1e-12)
    assert metrics["grad_norm_wm_local_scalars_predictor"].mean > 0
    assert any(not torch.equal(old, new) for old, new in zip(
        predictors_before, policy.local_scalars_predictor.parameters(), strict=True,
    ))


@pytest.mark.parametrize("device", DEVICES)
def test_deep_ppo_completes_a_finite_update_from_zero_reset_observations(device: str) -> None:
    torch.manual_seed(101)
    env = SwarmBotsLearnEnvWrapper(SyncVectorEnv([
        lambda: SwarmTestEnv(
            n_agents=3, n_local_obs=5, n_global_obs=2, actuators_dim=2, connectors_dim=1,
            n_hidden_local_vars=2, n_hidden_global_vars=1, continuous_connector_actions=True,
        ) for _ in range(2)
    ], autoreset_mode=AutoresetMode.SAME_STEP))
    try:
        policy = make_policy("mat_ind", "10M-4L", use_nop=False).to(device)
        algorithm = PPO(
            policy, env, n_epochs=1, target_kl=None, sampler_config=PPOSamplerConfig(batch_size=2),
            rollout_mode=StepsRolloutMode(2), train_device=device, rollout_device="cpu",
        )
        weights = [parameter.detach().clone() for parameter in policy.parameters()]

        metrics = algorithm.train([_zero_observation_segment(algorithm)])

        assert metrics["updates"] == algorithm.n_total_updates == 1
        assert math.isfinite(metrics["grad_norm"].mean)
        assert metrics["grad_norm_total"].mean == pytest.approx(metrics["grad_norm"].mean, rel=1e-12)
        assert all(math.isfinite(summary.mean) for name, summary in metrics.items() if name.startswith("grad_norm_"))
        gradients = [parameter.grad for parameter in policy.parameters() if parameter.grad is not None]
        clipped_norm = torch.stack([gradient.double().norm() for gradient in gradients]).norm()
        assert clipped_norm <= algorithm.max_grad_norm * (1 + 1e-5)
        assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())
        assert any(not torch.equal(old, new) for old, new in zip(weights, policy.parameters(), strict=True))
        assert algorithm.optimizer.state
        assert all(
            torch.isfinite(value).all()
            for state in algorithm.optimizer.state.values() for value in state.values() if torch.is_tensor(value)
        )
    finally:
        env.close()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_ppo_rejects_nonfinite_gradients_without_updating_parameters_or_adam(device: str, value: float) -> None:
    torch.manual_seed(101)
    env = _make_env()
    algorithm = PPO(
        _make_policy(env), env, n_epochs=1, target_kl=None, sampler_config=PPOSamplerConfig(batch_size=2),
        rollout_mode=StepsRolloutMode(2), train_device=device, rollout_device="cpu",
    )
    algorithm.train([_zero_observation_segment(algorithm)])
    weights = deepcopy(algorithm.policy.state_dict())
    optimizer = deepcopy(algorithm.optimizer.state_dict())
    updates = algorithm.n_total_updates

    def nonfinite_gradient(gradient: torch.Tensor) -> torch.Tensor:
        return torch.full_like(gradient, value)

    handle = next(algorithm.policy.parameters()).register_hook(nonfinite_gradient)
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            algorithm.train([_zero_observation_segment(algorithm)])
    finally:
        handle.remove()

    torch.testing.assert_close(algorithm.policy.state_dict(), weights, rtol=0, atol=0)
    torch.testing.assert_close(algorithm.optimizer.state_dict(), optimizer, rtol=0, atol=0)
    assert algorithm.n_total_updates == updates
