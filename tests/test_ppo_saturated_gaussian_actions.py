import math
from contextlib import ExitStack, nullcontext
from dataclasses import replace
from typing import Any

import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from torch.nn.modules.module import register_module_forward_hook, register_module_forward_pre_hook

from swarmbots.learn.action_dists.beta_action_dist import BetaActionDist
from swarmbots.learn.action_dists.continuous_action_dist import _COMPILES_TENSOR_VERSION
from swarmbots.learn.action_dists.diag_gaussian_action_dist import DiagGaussianActionDist
from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist, GSDEConfig
from swarmbots.learn.action_dists.hybrid_action_dist import ContinuousActionDistConfig, HybridActionDistribution
from swarmbots.learn.action_dists.predicted_std_gaussian_action_dist import (
    PredictedStdGaussianActionDist, PredictedStdGaussianConfig,
)
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import (
    SquashedDiagGaussianActionDist, SquashedDiagGaussianConfig,
)
from swarmbots.learn.action_dists.tanh_bijector import TanhBijector
from swarmbots.learn.algos.ppo.ppo import PPO, StepsRolloutMode, _backward_fp16_loss
from swarmbots.learn.algos.ppo.ppo_policy import PPOActorConfig, PPOPolicy, PPOPolicyConfig
from swarmbots.learn.algos.ppo.ppo_rollout import collect_step_rollout_batch, collect_steps
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPORolloutBuffer
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamplerConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig, NextObsPredWrapper
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEIntervalResetMode
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as _TestingSwarmBotsEnv


_GAUSSIAN_CONFIGS = [
    PredictedStdGaussianConfig(base_std=0.1),
    SquashedDiagGaussianConfig(std=0.1, std_learnable=True),
    GSDEConfig(base_std=0.1),
]


def _set_gaussian_means(action_dist: HybridActionDistribution, mean: float) -> None:
    with torch.no_grad():
        for dist in action_dist.distributions:
            dist.action_net.weight.zero_()
            dist.action_net.bias.fill_(mean)
            if hasattr(dist, "log_std_net"):
                dist.log_std_net.weight.zero_()
                dist.log_std_net.bias.zero_()


def _make_env(obs_dtype: torch.dtype = torch.float32, max_steps: int = 100) -> SwarmBotsLearnEnvWrapper:
    return SwarmBotsLearnEnvWrapper(
        SyncVectorEnv(
            [
                lambda: _TestingSwarmBotsEnv(
                    n_agents=2,
                    n_local_obs=2,
                    n_global_obs=1,
                    actuators_dim=1,
                    connectors_dim=1,
                    continuous_connector_actions=True,
                    max_steps=max_steps,
                )
            ],
            autoreset_mode=AutoresetMode.SAME_STEP,
        ),
        obs_dtype=obs_dtype,
    )


@pytest.mark.parametrize("config", _GAUSSIAN_CONFIGS, ids=lambda config: type(config).__name__)
@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_ppo_endpoint_ratio_tracks_the_change_in_gaussian_tail_probability(config, sign: float) -> None:
    env = _make_env()
    try:
        hybrid = HybridActionDistribution(4, env.action_space, config)
        component = hybrid.distributions[0]
        latent = torch.ones(1, 2, 4)
        cutoff = torch.tensor(math.atanh(1.0 - torch.finfo(latent.dtype).eps / 4))
        _set_gaussian_means(hybrid, sign * cutoff)
        component.update_latent_features(latent)
        endpoint = torch.full_like(component.distribution.mean, sign)
        old = component.log_prob(endpoint)
        sigma = component.distribution.stddev[0, 0, 0].item()

        _set_gaussian_means(hybrid, sign * (cutoff + sigma))
        component.update_latent_features(latent)
        new = component.log_prob(endpoint)
        expected_ratio = (1 + math.erf(1 / math.sqrt(2))) ** endpoint.shape[-1]
        torch.testing.assert_close(old.exp(), torch.full_like(old, 0.5 ** endpoint.shape[-1]))
        torch.testing.assert_close((new - old).exp(), torch.full_like(old, expected_ratio))
        new.sum().backward()
        assert (sign * component.action_net.bias.grad > 0).all()
    finally:
        env.close()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_float32_tanh_saturates_on_the_first_input_above_the_action_cell_boundary(sign: float) -> None:
    cutoff = math.atanh(1.0 - torch.finfo(torch.float32).eps / 4)
    lower = torch.tensor(cutoff, dtype=torch.float32)
    upper = torch.nextafter(lower, torch.tensor(math.inf))
    assert lower.item() < cutoff < upper.item()
    expected = torch.stack((torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)), torch.tensor(1.0)))
    torch.testing.assert_close(TanhBijector.forward(sign * torch.stack((lower, upper))), sign * expected, rtol=0, atol=0)


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=[pytest.mark.cuda, pytest.mark.skipif(
    not torch.cuda.is_available(), reason="requires CUDA",
)])])
@pytest.mark.parametrize("family", ["predicted", "diagonal", "gsde"])
@pytest.mark.parametrize("at_endpoint", [False, True])
def test_gaussian_policy_draws_follow_the_scored_action_cell(device: str, family: str, at_endpoint: bool) -> None:
    mean = math.atanh(1 - torch.finfo(torch.float32).eps / 4) if at_endpoint else 0.375
    sigma = 5e-7 if at_endpoint else 1.6e-8
    configs = {
        "predicted": PredictedStdGaussianConfig(base_std=sigma),
        "diagonal": SquashedDiagGaussianConfig(std=sigma, std_learnable=True),
        "gsde": GSDEConfig(base_std=0.1, epsilon=sigma ** 2),
    }
    env = _make_env()
    try:
        torch.manual_seed(726)
        hybrid = HybridActionDistribution(1, env.action_space, configs[family]).to(device)
        _set_gaussian_means(hybrid, mean)
        count = 16384
        hybrid.reset_temporal_correlations_on_step(batch_shape=(count, 1))
        component = hybrid.distributions[0]
        actions, scores = component.get_on_policy_actions_with_log_probs(torch.zeros(count, 1, 1, device=device))
        y = torch.tensor(1.0 if at_endpoint else math.tanh(mean), device=device)
        if at_endpoint:
            width = 1.0
        else:
            width = ((torch.nextafter(y, y + 1) - torch.nextafter(y, y - 1)) / 2).item()
        sampled = actions[..., 0] == y
        frequency = sampled.float().mean().item()
        mass = math.exp(component.log_prob(y.expand_as(actions))[0].item()) * width
        assert 0.1 < mass < 0.95
        assert frequency == pytest.approx(mass, abs=0.02)
        torch.testing.assert_close(scores[sampled], torch.full_like(scores[sampled], math.log(mass / width)))
    finally:
        env.close()


@pytest.mark.parametrize("sign", [-1.0, 1.0])
def test_ppo_nearest_interior_action_uses_integrated_gaussian_cell(sign: float) -> None:
    component = SquashedDiagGaussianActionDist(1, 1, std=0.1, std_learnable=True)
    y = torch.nextafter(torch.tensor(1.0), torch.tensor(0.0)).item()
    width = 1.0 - y
    low, high = math.atanh(y - width / 2), math.atanh(y + width / 2)
    actions = torch.tensor([[sign * y], [sign], [0.0]])
    masses, scores = [], []
    for mean in (8.9, 8.8):
        with torch.no_grad():
            component.action_net.bias.fill_(sign * mean)
        component.update_latent_features(torch.zeros_like(actions))
        z_low, z_high = (low - mean) / 0.1, (high - mean) / 0.1
        mass = (math.erf(z_high / math.sqrt(2)) - math.erf(z_low / math.sqrt(2))) / 2
        phi_low, phi_high = (math.exp(-z * z / 2) / math.sqrt(2 * math.pi) for z in (z_low, z_high))
        score = component.log_prob(actions)[0]
        assert score.item() == pytest.approx(math.log(mass / width), abs=1e-5)
        mean_grad, log_std_grad = torch.autograd.grad(score, (component.action_net.bias, component.log_stds))
        assert mean_grad.item() == pytest.approx(sign * (phi_low - phi_high) / (0.1 * mass), abs=5e-5)
        assert log_std_grad.item() == pytest.approx((z_low * phi_low - z_high * phi_high) / mass, abs=1e-5)
        masses.append(mass)
        scores.append(score.item())
    assert math.exp(scores[1] - scores[0]) == pytest.approx(masses[1] / masses[0], abs=1e-5)


@pytest.mark.parametrize("mean", [-math.atanh(0.5), -0.375, math.atanh(0.25), 0.375, math.atanh(0.5)])
def test_narrow_ppo_gaussian_scores_the_stored_interior_action_cell(mean: float) -> None:
    action = torch.tensor([[math.tanh(mean)]])
    neighbors = [torch.nextafter(action, torch.full_like(action, direction)).item() for direction in (-1.0, 1.0)]
    low, high = [math.atanh((neighbor + action.item()) / 2) for neighbor in neighbors]
    component = SquashedDiagGaussianActionDist(1, 1, std=5e-9, std_learnable=True)
    with torch.no_grad():
        component.action_net.bias.fill_(mean)
    for std in (5e-9, 2.5e-9):
        component.set_std(std)
        component.update_latent_features(torch.zeros_like(action))
        stored_mean = component.distribution.mean.item()
        sigma = component.distribution.scale.item()
        z_low, z_high = (low - stored_mean) / sigma, (high - stored_mean) / sigma
        mass = (math.erf(z_high / math.sqrt(2)) - math.erf(z_low / math.sqrt(2))) / 2
        cell_width = (neighbors[1] - neighbors[0]) / 2
        score = component.log_prob(action)
        assert score.item() == pytest.approx(math.log(mass / cell_width), abs=3e-6)
        phi_low, phi_high = (math.exp(-z * z / 2) / math.sqrt(2 * math.pi) for z in (z_low, z_high))
        log_std_grad, = torch.autograd.grad(score, component.log_stds)
        assert log_std_grad.item() == pytest.approx((z_low * phi_low - z_high * phi_high) / mass, abs=2e-6)


@pytest.mark.parametrize("config", _GAUSSIAN_CONFIGS, ids=lambda config: type(config).__name__)
@pytest.mark.parametrize("mean", [-10.0, 10.0])
def test_saturated_gaussian_ppo_rollout_has_unit_ratio_and_trains(config, mean: float) -> None:
    torch.manual_seed(123)
    env = _make_env()
    try:
        policy = PPOPolicy(
            env,
            PPOPolicyConfig(
                actor_config=PPOActorConfig(latent_pi_dim_per_agent=8),
                continuous_config=config,
            ),
        )
        _set_gaussian_means(policy.action_dist, mean)
        algorithm = PPO(
            policy,
            env,
            rollout_mode=StepsRolloutMode(2),
            sampler_config=PPOSamplerConfig(batch_size=2),
            n_epochs=1,
            target_kl=0.01,
            gsde_reset_mode=GSDEIntervalResetMode(interval=1),
            train_device="cpu",
            rollout_device="cpu",
        )
        rollout, _infos, _metrics, _state = collect_step_rollout_batch(
            env, policy, algorithm.rollout_buffer, n_steps=2,
            gsde_reset_mode=algorithm.gsde_reset_mode,
        )
        assert torch.equal(rollout.actions, torch.full_like(rollout.actions, 1.0 if mean > 0 else -1.0))
        policy.train()
        sampler = policy.make_rollout_batch_sampler(rollout, algorithm.sampler_config)
        batch = next(iter(sampler.sample()))
        log_probs, values, _losses, _metrics = policy.evaluate_actions(batch)

        assert torch.isfinite(log_probs).all()
        torch.testing.assert_close(log_probs, batch.log_probs, rtol=0, atol=0)
        torch.testing.assert_close(torch.exp(log_probs - batch.log_probs), torch.ones_like(log_probs))
        _loss, approximate_kl, metrics = algorithm.compute_ppo_loss(batch, log_probs, values)
        assert approximate_kl == 0.0
        assert metrics["clip_frac"] == 0.0
        assert algorithm.train(rollout)["updates"] == 1
    finally:
        env.close()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_gaussian_heads_keep_their_float32_rollout_likelihood(dtype: torch.dtype) -> None:
    env = _make_env()
    try:
        policy = PPOPolicy(env, PPOPolicyConfig(
            actor_config=PPOActorConfig(latent_pi_dim_per_agent=8),
            continuous_config=SquashedDiagGaussianConfig(std=0.1, std_learnable=True),
        ))
        if dtype == torch.float64:
            _set_gaussian_means(policy.action_dist, 10.0)
            for dist in policy.action_dist.distributions:
                dist.action_net.register_forward_hook(lambda _module, _inputs, output: output.double())
            precision = nullcontext()
        else:
            _set_gaussian_means(policy.action_dist, 5.5)
            precision = torch.autocast("cpu", dtype=dtype)
        algorithm = PPO(
            policy, env, rollout_mode=StepsRolloutMode(2), sampler_config=PPOSamplerConfig(batch_size=2),
            n_epochs=1, target_kl=0.01, train_device="cpu", rollout_device="cpu",
        )
        with precision:
            rollout, *_ = collect_step_rollout_batch(env, policy, algorithm.rollout_buffer, n_steps=2)
            assert rollout.actions.dtype == torch.float32
            if dtype == torch.float64:
                assert (rollout.actions == 1).all()
            batch = next(policy.make_rollout_batch_sampler(rollout, algorithm.sampler_config).sample())
            policy.train()
            log_probs, *_ = policy.evaluate_actions(batch)
            torch.testing.assert_close(log_probs, batch.log_probs, rtol=0, atol=0)
            assert algorithm.train(rollout)["updates"] == 1
    finally:
        env.close()


def test_double_gaussian_distribution_keeps_native_actions_unless_storage_precision_is_requested() -> None:
    dist = SquashedDiagGaussianActionDist(latent_dim=1, action_dim=1, std=0.1, std_learnable=True).double()
    with torch.no_grad():
        dist.action_net.weight.zero_()
        dist.action_net.bias.fill_(10.0)
    latent = torch.zeros(2, 1, dtype=torch.float64)
    native, native_score = dist.get_on_policy_actions_with_log_probs(latent, deterministic=True)
    sac, sac_score = dist.get_actions_with_log_probs(latent, deterministic=True)
    stored, stored_score = dist.get_on_policy_actions_with_log_probs(latent, deterministic=True, storage_dtype=torch.float32)
    assert native.dtype == native_score.dtype == sac.dtype == sac_score.dtype == torch.float64
    assert (native < 1).all()
    torch.testing.assert_close(native, sac, rtol=0, atol=0)
    assert stored.dtype == torch.float32 and (stored == 1).all()
    torch.testing.assert_close(stored_score, dist.log_prob(stored), rtol=0, atol=0)


@pytest.mark.parametrize(("on_policy", "deterministic", "use_rsample", "offset"), [
    (True, True, False, 0.1),
    (True, False, False, 0.2),
    (False, True, False, 0.1),
    (False, False, False, 0.2),
    (False, False, True, 0.3),
])
def test_joint_gaussian_sampling_honors_draw_and_score_overrides(
        on_policy: bool, deterministic: bool, use_rsample: bool, offset: float,
) -> None:
    class ConditionedGaussian(SquashedDiagGaussianActionDist):
        def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
            return previous_actions + 0.1

        def sample(self, agent: int | None = None, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
            assert agent == 1
            return previous_actions + 0.2

        def rsample(self, agent: int | None = None, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
            assert agent == 1
            return previous_actions + 0.3

        def log_prob(self, actions: torch.Tensor, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
            return (actions - previous_actions).sum(dim=-1)

    dist = ConditionedGaussian(1, 1, std=0.1, std_learnable=True)
    latent = torch.zeros(2, 1)
    previous = torch.tensor([[0.25], [-0.5]])
    if on_policy:
        actions, scores = dist.get_on_policy_actions_with_log_probs(
            latent, deterministic=deterministic, agent=1, previous_actions=previous, storage_dtype=torch.float64,
        )
        assert actions.dtype == torch.float64
    else:
        actions, scores = dist.get_actions_with_log_probs(
            latent, deterministic=deterministic, agent=1, previous_actions=previous, use_rsample=use_rsample,
        )
    torch.testing.assert_close(actions, (previous + offset).to(actions.dtype))
    torch.testing.assert_close(scores, torch.full((2,), offset, dtype=scores.dtype))


def test_sac_native_gaussian_draw_honors_custom_scorer() -> None:
    class CustomScorer(SquashedDiagGaussianActionDist):
        def log_prob(
                self,
                actions: torch.Tensor,
                previous_actions: torch.Tensor | None = None,
                gaussian_actions: torch.Tensor | None = None,
        ) -> torch.Tensor:
            assert gaussian_actions is not None
            torch.testing.assert_close(actions, gaussian_actions.tanh().to(actions.dtype))
            return previous_actions.squeeze(-1)

    dist = CustomScorer(1, 1, std=0.1, std_learnable=True)
    previous = torch.tensor([[0.25], [-0.5]])
    _, scores = dist.get_actions_with_log_probs(
        torch.zeros(2, 1), deterministic=True, previous_actions=previous,
    )
    torch.testing.assert_close(scores, previous.squeeze(-1))


def test_on_policy_sampling_explicitly_uses_non_pathwise_draws() -> None:
    class PathwiseByDefault(BetaActionDist):
        def get_actions(
                self, deterministic: bool = False, agent: int | None = None,
                previous_actions: torch.Tensor | None = None, use_rsample: bool = True,
        ) -> torch.Tensor:
            return super().get_actions(
                deterministic=deterministic, agent=agent, previous_actions=previous_actions, use_rsample=use_rsample,
            )

    dist = PathwiseByDefault(1, 1, None)
    actions, scores = dist.get_on_policy_actions_with_log_probs(torch.zeros(2, 1))
    assert not actions.requires_grad
    assert scores.requires_grad


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test_hybrid_rollout_keeps_in_place_custom_action_clamp(
        compiled: bool, inference: bool, autocast_dtype: torch.dtype,
) -> None:
    class ClampedMode(SquashedDiagGaussianActionDist):
        def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
            return super().mode(previous_actions).clamp_(-0.25, 0.25)

    env = _make_env()
    try:
        config = SquashedDiagGaussianConfig(std=0.1, std_learnable=True)
        hybrid = HybridActionDistribution(1, env.action_space, config)
        hybrid.distributions[0] = ClampedMode(1, hybrid.distributions[0].action_dim, 0.1, True)
        draw = hybrid.get_on_policy_actions_with_log_probs
        if compiled:
            draw = torch.compile(draw, backend="eager", fullgraph=True)
        latent = torch.zeros(1, env.n_agents, 1)
        with torch.inference_mode(inference), torch.autocast("cpu", autocast_dtype):
            for value in (0.75, 0.25003):
                _set_gaussian_means(hybrid, math.atanh(value))
                actions, scores = draw(latent, deterministic=True, storage_dtype=torch.float32)
                clamped, _ = torch.split(actions, hybrid.action_dims, dim=-1)
                assert actions.dtype == torch.float32
                torch.testing.assert_close(clamped, torch.full_like(clamped, 0.25), atol=0, rtol=0)
                torch.testing.assert_close(scores, hybrid.log_prob(actions), atol=0, rtol=0)
    finally:
        env.close()


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("inference", [False, True])
@pytest.mark.parametrize("mutated", [False, True])
def test_custom_gaussian_scorer_receives_the_original_draw_and_actual_action(
        compiled: bool, inference: bool, mutated: bool,
) -> None:
    class CustomModeAndScorer(SquashedDiagGaussianActionDist):
        def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
            actions = super().mode(previous_actions)
            return actions.clamp_(-0.25, 0.25) if mutated else actions

        def log_prob(
                self, actions: torch.Tensor, previous_actions: torch.Tensor | None = None,
                gaussian_actions: torch.Tensor | None = None,
        ) -> torch.Tensor:
            assert gaussian_actions is not None
            return actions.squeeze(-1) + gaussian_actions.squeeze(-1).to(actions.dtype)

    dist = CustomModeAndScorer(1, 1, std=0.1, std_learnable=True)
    with torch.no_grad():
        dist.action_net.bias.fill_(1.0)
    draw = dist.get_actions_with_log_probs
    if compiled:
        draw = torch.compile(draw, backend="eager", fullgraph=True)
    with torch.inference_mode(inference):
        actions, scores = draw(torch.zeros(2, 1), deterministic=True)
    expected_action = 0.25 if mutated else math.tanh(1.0)
    expected_score = expected_action + 1.0
    torch.testing.assert_close(actions, torch.full_like(actions, expected_action))
    torch.testing.assert_close(scores, torch.full_like(scores, expected_score))


@pytest.mark.parametrize("legacy_head", ["component", "hybrid"])
@pytest.mark.parametrize("compiled", [False, True])
def test_ppo_rollout_accepts_action_heads_with_the_previous_sampler_signature(
        legacy_head: str, compiled: bool,
) -> None:
    class LegacySampler:
        def get_on_policy_actions_with_log_probs(
                self,
                latent_pi: torch.Tensor,
                deterministic: bool = False,
                agent: int | None = None,
                previous_actions: torch.Tensor | None = None,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            actions, scores = super().get_on_policy_actions_with_log_probs(
                latent_pi, deterministic, agent=agent, previous_actions=previous_actions,
            )
            return (
                torch.full_like(actions, -0.25, dtype=torch.float64),
                torch.full_like(scores, 0.75, dtype=torch.float64),
            )

    class LegacyComponent(LegacySampler, SquashedDiagGaussianActionDist):
        pass

    class LegacyHybrid(LegacySampler, HybridActionDistribution):
        pass

    class DelegatingMode(SquashedDiagGaussianActionDist):
        def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
            return super().mode(previous_actions=previous_actions)

    env = _make_env()
    try:
        config = SquashedDiagGaussianConfig(std=0.1, std_learnable=True)
        policy = PPOPolicy(env, PPOPolicyConfig(continuous_config=config))
        latent_dim = policy.action_dist.latent_dim
        if legacy_head == "component":
            for idx, component in enumerate((LegacyComponent, DelegatingMode)):
                action_dim = policy.action_dist.distributions[idx].action_dim
                policy.action_dist.distributions[idx] = component(latent_dim, action_dim, 0.1, True)
        else:
            policy.action_dist = LegacyHybrid(latent_dim, env.action_space, config)
        _set_gaussian_means(policy.action_dist, 10.0)
        run_policy = torch.compile(policy, backend="eager", fullgraph=True) if compiled else policy
        with torch.autocast("cpu", torch.float16):
            obs, _ = env.reset()
            actions, log_probs, _ = run_policy(**obs, deterministic=True)
            assert actions.dtype == log_probs.dtype == torch.float64
            expected_log_probs = torch.full_like(log_probs, 0.75)
            if legacy_head == "component":
                custom_action, other_action = torch.split(actions, policy.action_dist.action_dims, dim=-1)
                if not compiled or _COMPILES_TENSOR_VERSION:
                    assert (other_action < 1).all()
                else:
                    torch.testing.assert_close(other_action, torch.ones_like(other_action), atol=0, rtol=0)
                expected_log_probs += policy.action_dist.distributions[1].log_prob(other_action)
            else:
                custom_action = actions
            torch.testing.assert_close(custom_action, torch.full_like(custom_action, -0.25), atol=0, rtol=0)
            torch.testing.assert_close(log_probs, expected_log_probs, atol=0, rtol=0)
    finally:
        env.close()


def test_hybrid_preserves_a_custom_joint_score_when_actions_widen() -> None:
    class CustomJointScore(SquashedDiagGaussianActionDist):
        def get_on_policy_actions_with_log_probs(
                self,
                latent_pi: torch.Tensor,
                deterministic: bool = False,
                agent: int | None = None,
                previous_actions: torch.Tensor | None = None,
                **kwargs: torch.dtype,
        ) -> tuple[torch.Tensor, torch.Tensor]:
            assert kwargs.get("storage_dtype") == torch.float32
            return (
                latent_pi.new_full((*latent_pi.shape[:-1], self.action_dim), 0.375, dtype=torch.float16),
                latent_pi.new_full(latent_pi.shape[:-1], 0.75, dtype=torch.float16),
            )

        def log_prob(self, actions: torch.Tensor, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
            raise AssertionError("The custom joint scorer has already returned its score")

    env = _make_env()
    try:
        dist = HybridActionDistribution(1, env.action_space, SquashedDiagGaussianConfig(0.1, True))
        dist.distributions[0] = CustomJointScore(1, dist.distributions[0].action_dim, 0.1, True)
        latent = torch.zeros(1, env.n_agents, 1)
        actions, scores = dist.get_on_policy_actions_with_log_probs(
            latent, deterministic=True, storage_dtype=torch.float32,
        )
        custom_actions, stock_actions = torch.split(actions, dist.action_dims, dim=-1)
        torch.testing.assert_close(custom_actions, torch.full_like(custom_actions, 0.375), rtol=0, atol=0)
        torch.testing.assert_close(scores, 0.75 + dist.distributions[1].log_prob(stock_actions), rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.parametrize("deterministic", [False, True])
def test_mixed_gaussian_heads_round_once_to_the_emitted_hybrid_dtype(deterministic: bool) -> None:
    env = _make_env()
    try:
        dist = HybridActionDistribution(1, env.action_space, SquashedDiagGaussianConfig(std=0.1, std_learnable=True))
        _set_gaussian_means(dist, 10.0)
        dist.distributions[1].action_net.register_forward_hook(lambda _m, _inputs, output: output.double())
        latent = torch.zeros(1, env.n_agents, 1)
        torch.manual_seed(73)
        actions, scores = dist.get_on_policy_actions_with_log_probs(latent, deterministic=deterministic)
        next_draw = torch.rand(2)
        assert actions.dtype == torch.float64
        assert (actions < 1).all()
        torch.testing.assert_close(scores, dist.log_prob(actions), atol=0, rtol=0)
        torch.manual_seed(73)
        dist.update_latent_features(latent).get_actions(deterministic=deterministic)
        torch.testing.assert_close(next_draw, torch.rand(2), atol=0, rtol=0)
    finally:
        env.close()


@pytest.mark.parametrize("mean", [-10.0, 0.375, 9.0, 10.0])
@pytest.mark.parametrize("std", [1e-20, 1e-25, 1e-39])
def test_narrow_gaussian_occupied_cell_has_finite_zero_gradient(mean: float, std: float) -> None:
    dist = SquashedDiagGaussianActionDist(1, 1, std=std, std_learnable=True)
    with torch.no_grad():
        dist.action_net.bias.fill_(mean)
        dist.action_net.weight.zero_()
    latent = torch.zeros(2, 1)
    actions, _ = dist.get_on_policy_actions_with_log_probs(latent)
    score = dist.log_prob(actions.detach()).mean()
    assert torch.isfinite(score)
    for gradient in torch.autograd.grad(score, (dist.action_net.bias, dist.log_stds)):
        torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=0, rtol=0)


@pytest.mark.parametrize("deterministic", [False, True])
def test_subnormal_gaussian_replays_its_finite_density_near_zero(deterministic: bool) -> None:
    dist = SquashedDiagGaussianActionDist(1, 1, std=1e-39, std_learnable=True)
    torch.manual_seed(123)
    actions, score = dist.get_on_policy_actions_with_log_probs(torch.zeros(1, 1), deterministic=deterministic)
    replay = dist.log_prob(actions.detach())
    torch.testing.assert_close(score, replay, atol=0, rtol=0)
    mean_grad, log_std_grad = torch.autograd.grad(replay.sum(), (dist.action_net.bias, dist.log_stds))
    log_sigma = dist.log_stds.item()
    z = actions.item() / math.exp(log_sigma)
    assert replay.item() == pytest.approx(-0.5 * (z**2 + math.log(2 * math.pi)) - log_sigma, abs=5e-6)
    assert mean_grad.item() == pytest.approx(z / math.exp(log_sigma), rel=1e-5)
    assert log_std_grad.item() == pytest.approx(z**2 - 1, rel=1e-5)


def test_bound_gaussian_retains_its_scale_until_updated() -> None:
    dist = SquashedDiagGaussianActionDist(1, 1, std=0.1, std_learnable=True)
    latent = torch.zeros(1, 1)
    keys = dist.state_dict().keys()
    actions, score = dist.get_on_policy_actions_with_log_probs(latent, deterministic=True)
    assert keys == dist.state_dict().keys()
    dist.set_std(0.2)
    torch.testing.assert_close(dist.log_prob(actions), score, atol=0, rtol=0)
    dist.update_latent_features(latent)
    assert dist.log_prob(actions).item() == pytest.approx(score.item() - math.log(2))


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("family", ["predicted", "diagonal", "gsde"])
def test_half_gaussian_replay_preserves_small_log_scale_gradient(family: str, autocast: bool) -> None:
    sigma = math.exp(-12)
    if family == "predicted":
        dist = PredictedStdGaussianActionDist(1, 1, base_std=1., squash_output=True)
        with torch.no_grad():
            dist.log_std_net.weight.zero_()
            dist.log_std_net.bias.fill_(-12)
        log_std_param = dist.log_std_net.bias
    elif family == "diagonal":
        dist = SquashedDiagGaussianActionDist(1, 1, std=sigma, std_learnable=True)
        log_std_param = dist.log_stds
    else:
        dist = GSDEActionDist(1, 1, base_std=sigma, squash_output=True, epsilon=sigma**2)
        log_std_param = dist.log_stds
    with torch.no_grad():
        dist.action_net.weight.zero_()
        dist.action_net.bias.fill_(0.65)
    if not autocast:
        dist.half()
    latent = torch.ones(1, 1, dtype=torch.float32 if autocast else torch.float16)
    with torch.autocast("cpu", torch.float16, enabled=autocast):
        actions, _ = dist.get_on_policy_actions_with_log_probs(latent, deterministic=True, storage_dtype=torch.float32)
        replay = dist.log_prob(actions.detach())
    gradient, = torch.autograd.grad(replay.sum(), log_std_param)
    # At the mean the score derivative is -1. For gSDE the independent variance floor
    # supplies half the variance and does not depend on the learned scale.
    expected = -0.5 if family == "gsde" else -1.
    torch.testing.assert_close(gradient, torch.full_like(gradient, expected), rtol=0., atol=5e-4)


@pytest.mark.parametrize("autocast", [False, True])
@pytest.mark.parametrize("storage_dtype", [None, torch.float16])
def test_unsquashed_half_gaussian_preserves_sample_precision_and_score_gradient(
        autocast: bool, storage_dtype: torch.dtype | None,
) -> None:
    dist = DiagGaussianActionDist(1, 1, std=1e-3, std_learnable=True)
    with torch.no_grad():
        dist.action_net.bias.zero_()
    if not autocast:
        dist.half()
    latent = torch.zeros(1, 1, dtype=torch.float32 if autocast else torch.float16)
    with torch.autocast("cpu", torch.float16, enabled=autocast):
        torch.manual_seed(123)
        actions, score = dist.get_on_policy_actions_with_log_probs(latent, storage_dtype=storage_dtype)
        expected_dtype = storage_dtype or (torch.float32 if autocast else torch.float16)
        assert actions.dtype == expected_dtype
        torch.testing.assert_close(score, dist.log_prob(actions), rtol=0, atol=0)
        if storage_dtype is None:
            torch.manual_seed(123)
            combined, combined_score = dist.get_actions_with_log_probs(latent)
            torch.testing.assert_close(combined, actions, rtol=0, atol=0)
            torch.testing.assert_close(combined_score, score.to(combined.dtype), rtol=0, atol=0)
            score = combined_score
    mean_grad, log_std_grad = torch.autograd.grad(score.sum(), (dist.action_net.bias, dist.log_stds))
    z = actions.item() / dist.distribution.scale.item()
    assert mean_grad.item() == pytest.approx(z / dist.distribution.scale.item(), rel=1e-3)
    assert log_std_grad.item() == pytest.approx(z**2 - 1., rel=1e-3)


class _CustomGaussianBinding(SquashedDiagGaussianActionDist):
    def update_distribution_params(self, means: torch.Tensor, log_stds: torch.Tensor) -> "_CustomGaussianBinding":
        self.distribution = torch.distributions.Normal(means, log_stds.float().exp())
        return self


def test_custom_gaussian_binding_resumes_native_precision_after_autocast() -> None:
    dist = _CustomGaussianBinding(1, 1, std=0.3, std_learnable=True)
    latent = torch.zeros(1, 1)
    with torch.autocast("cpu", torch.float16):
        assert dist.update_latent_features(latent).mode().dtype == torch.float16
    dist.update_distribution_params(latent.double(), dist.log_stds.double())
    assert dist.mode().dtype == torch.float64
    assert dist.update_latent_features(latent).mode().dtype == torch.float32


def test_gaussian_with_custom_action_head_can_run_inside_autocast() -> None:
    dist = SquashedDiagGaussianActionDist(1, 1, std=0.3, std_learnable=True)
    dist.action_net = torch.nn.Sequential(dist.action_net, torch.nn.Identity())
    with torch.autocast("cpu", torch.float16):
        assert dist.update_latent_features(torch.zeros(1, 1)).mode().dtype == torch.float16


class _WideOutputLinear(torch.nn.Linear):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return super().forward(values).float()


@pytest.mark.parametrize("head_kind", ["subclass", "local_hook", "global_hook"])
def test_gaussian_custom_head_keeps_its_output_precision_under_autocast(head_kind: str) -> None:
    dist = SquashedDiagGaussianActionDist(1, 1, std=0.3, std_learnable=True)
    if head_kind == "subclass":
        dist.action_net = _WideOutputLinear(1, 1)
    with torch.no_grad():
        dist.action_net.bias.fill_(5.)

    def widen_head_output(
            module: torch.nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor,
    ) -> torch.Tensor | None:
        if module is dist.action_net:
            return output.float()
        return None

    with ExitStack() as hooks:
        if head_kind == "local_hook":
            hooks.enter_context(dist.action_net.register_forward_hook(widen_head_output))
        elif head_kind == "global_hook":
            hooks.enter_context(register_module_forward_hook(widen_head_output))
        with torch.autocast("cpu", torch.float16):
            action = dist.update_latent_features(torch.zeros(1, 1)).mode()
    torch.testing.assert_close(action, torch.tensor([[5.]]).tanh())
    assert action.item() < 1.


@pytest.mark.parametrize("global_hook", [False, True], ids=["local_hook", "global_hook"])
def test_gaussian_forward_pre_hook_runs_in_the_callers_autocast(global_hook: bool) -> None:
    dist = SquashedDiagGaussianActionDist(1, 1, std=0.3, std_learnable=True)
    with torch.no_grad():
        dist.action_net.bias.zero_()
        dist.action_net.weight.fill_(1.)

    def cast_head_input(module: torch.nn.Module, inputs: tuple[torch.Tensor]) -> tuple[torch.Tensor] | None:
        if module is dist.action_net:
            return (inputs[0].half() + 0.25,)
        return None

    register_hook = register_module_forward_pre_hook if global_hook else dist.action_net.register_forward_pre_hook
    with register_hook(cast_head_input), torch.autocast("cpu", torch.float16):
        action = dist.update_latent_features(torch.full((1, 1), 0.25)).mode()
    torch.testing.assert_close(action, torch.tensor([[0.5]], dtype=torch.float16).tanh())


def test_fp16_ppo_trunk_gradients_match_a_float32_reference() -> None:
    trunk = torch.nn.Linear(1, 8)
    dist = PredictedStdGaussianActionDist(8, 1, base_std=1., squash_output=True)
    parameters = [*trunk.parameters(), *dist.parameters()]
    optimizer = torch.optim.SGD(parameters, lr=0.01)
    log_std = -16.
    with torch.no_grad():
        trunk.weight.fill_(0.5)
        trunk.bias.zero_()
        dist.action_net.weight.fill_(0.0625)
        dist.action_net.bias.fill_(0.75)
        dist.log_std_net.weight.zero_()
        dist.log_std_net.bias.fill_(log_std)

    inputs = torch.ones(1, 1)
    # An exact mean of 1 and a fixed residual keep the overflow independent of linear-kernel rounding.
    action = torch.tensor([[1. + math.exp(log_std)]], dtype=torch.float64).tanh().float()
    with torch.autocast("cpu", torch.float16):
        score = dist.update_latent_features(trunk(inputs)).log_prob(action)
    _backward_fp16_loss(score.sum(), parameters, optimizer)
    actual_gradients = [param.grad.clone() for param in parameters]

    dist.update_latent_features(trunk(inputs))
    expected_gradients = torch.autograd.grad(dist.log_prob(action).sum(), parameters)
    # This finite reference gradient is too large to pass through an FP16 trunk at unit scale.
    assert expected_gradients[1].abs().max() > torch.finfo(torch.float16).max
    for actual, expected in zip(actual_gradients, expected_gradients, strict=True):
        torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@pytest.mark.parametrize(("seed", "log_std", "mean_weight"), [
    (0, -12., 0.001), (12, -16., 0.001), (12, -16., 0.05), (12, -16., 0.2), (3, -16., 0.5),
])
def test_autocast_ppo_updates_narrow_predicted_gaussian_heads(seed: int, log_std: float, mean_weight: float) -> None:
    torch.manual_seed(seed)
    env = _make_env()
    try:
        policy = PPOPolicy(env, PPOPolicyConfig(
            actor_config=PPOActorConfig(latent_pi_dim_per_agent=8),
            continuous_config=PredictedStdGaussianConfig(base_std=1.),
        ))
        _set_gaussian_means(policy.action_dist, 0.65)
        with torch.no_grad():
            for dist in policy.action_dist.distributions:
                dist.action_net.weight.fill_(mean_weight)
                dist.log_std_net.bias.fill_(log_std)
        algorithm = PPO(
            policy, env, rollout_mode=StepsRolloutMode(4), sampler_config=PPOSamplerConfig(batch_size=4),
            n_epochs=1, target_kl=.01, train_device="cpu", rollout_device="cpu",
        )
        with torch.autocast("cpu", torch.float16):
            rollout, *_ = collect_step_rollout_batch(env, policy, algorithm.rollout_buffer, n_steps=4)
            batch = next(policy.make_rollout_batch_sampler(rollout, algorithm.sampler_config).sample())
            policy.train()
            log_probs, *_ = policy.evaluate_actions(batch)
            torch.testing.assert_close(log_probs, batch.log_probs, rtol=0, atol=0)
            assert algorithm.train(rollout)["updates"] == 1
        for dist in policy.action_dist.distributions:
            assert torch.isfinite(dist.action_net.bias.grad).all()
            assert torch.isfinite(dist.log_std_net.bias.grad).all()
            assert dist.log_std_net.bias.item() != log_std
        assert all(torch.isfinite(param.grad).all() for param in policy.parameters() if param.grad is not None)
    finally:
        env.close()


@pytest.mark.parametrize("nonfinite", ["loss", "gradient"])
def test_autocast_ppo_rejects_a_nonfinite_update(nonfinite: str, monkeypatch: pytest.MonkeyPatch) -> None:
    torch.manual_seed(12)
    env = _make_env()
    try:
        policy = PPOPolicy(env, PPOPolicyConfig(
            actor_config=PPOActorConfig(latent_pi_dim_per_agent=8),
            continuous_config=PredictedStdGaussianConfig(base_std=1.),
        ))
        algorithm = PPO(
            policy, env, rollout_mode=StepsRolloutMode(4), sampler_config=PPOSamplerConfig(batch_size=4),
            n_epochs=1, train_device="cpu", rollout_device="cpu",
        )
        param = policy.action_dist.distributions[0].action_net.bias
        before_update = param.detach().clone()
        compute_loss = algorithm.compute_loss

        def invalid_loss(*args, **kwargs) -> tuple[torch.Tensor, float | torch.Tensor, dict]:
            loss, kl, metrics = compute_loss(*args, **kwargs)
            bad_term = float("nan") if nonfinite == "loss" else (param.sum() * 0.).sqrt()
            return loss + bad_term, kl, metrics

        monkeypatch.setattr(algorithm, "compute_loss", invalid_loss)
        with torch.autocast("cpu", torch.float16):
            rollout, *_ = collect_step_rollout_batch(env, policy, algorithm.rollout_buffer, n_steps=4)
            with pytest.raises(RuntimeError, match="non-finite"):
                algorithm.train(rollout)
        assert not algorithm.optimizer.state
        torch.testing.assert_close(param, before_update, rtol=0, atol=0)
    finally:
        env.close()


@pytest.mark.parametrize("dense", [False, True], ids=["episodes", "dense"])
@pytest.mark.parametrize(("action_dtype", "storage_dtype", "train_dtype"), [
    (torch.float32, torch.float16, torch.float64),
    (torch.float64, torch.float16, torch.float32),
    (torch.float16, torch.float64, torch.float32),
])
def test_collectors_preserve_emitted_actions_and_scores_across_buffer_dtypes(
    dense: bool, action_dtype: torch.dtype, storage_dtype: torch.dtype, train_dtype: torch.dtype,
) -> None:
    env = _make_env(obs_dtype=train_dtype, max_steps=2)
    try:
        actor = PPOPolicy(env, PPOPolicyConfig(
            actor_config=PPOActorConfig(latent_pi_dim_per_agent=8),
            continuous_config=SquashedDiagGaussianConfig(std=0.1, std_learnable=True),
        ))
        actor.rollout_dtype = action_dtype
        _set_gaussian_means(actor.action_dist, 10.0 if action_dtype == torch.float64 else 5.5)
        policy = NextObsPredWrapper(actor, world_model_config=NOPWorldModelConfig(
            n_agents=env.n_agents, local_latent_dim=actor.nop_latent_dim, action_dim=env.action_space.total_agent_action_dim,
            d_model_transition_model=8, nhead_transition_model=2, num_layers_transition_model=1,
            dim_feedforward_transition_model=16,
            next_obs_pred_config=NextObsPredConfig(local_scalar_target_indices=[0]),
        )).to(train_dtype)
        assert policy.rollout_dtype == action_dtype
        emitted = []
        actor.register_forward_hook(lambda _module, _inputs, output: emitted.append(output[:2]))
        buffer = PPORolloutBuffer(
            max_episode_length=3, observation_space=env.observation_space, action_space=env.action_space,
            gamma=0.99, gae_lambda=0.95, rollout_dtype=storage_dtype, train_dtype=train_dtype, train_device="cpu",
        )
        config = PPOWMSamplerConfig(batch_size=3, num_next_steps=2)
        if dense:
            rollout, *_ = collect_step_rollout_batch(env, policy, buffer, n_steps=3)
            stored_actions, stored_scores = rollout.actions[0], rollout.log_probs[0]
            sampler = policy.make_rollout_batch_sampler(rollout, config)
        else:
            episodes, *_ = collect_steps(env, policy, buffer, n_steps=3)
            stored_actions = torch.cat([episode.actions for episode in episodes])
            stored_scores = torch.cat([episode.log_probs for episode in episodes])
            sampler = policy.make_sampler(episodes, config)
        torch.testing.assert_close(stored_actions, torch.cat([actions for actions, _ in emitted]), rtol=0, atol=0)
        torch.testing.assert_close(stored_scores, torch.cat([scores for _, scores in emitted]), rtol=0, atol=0)
        batch = next(sampler.sample())
        assert batch.actions.dtype == action_dtype
        assert batch.local_obs.dtype == train_dtype
        policy.train()
        scores, _, losses, _ = policy.evaluate_actions(batch)
        torch.testing.assert_close(scores, batch.log_probs, rtol=0, atol=0)
        losses["world_model"].backward()
        grad = next(policy.transition_model.parameters()).grad
        assert grad is not None and torch.isfinite(grad).all()
    finally:
        env.close()


@pytest.mark.parametrize("dense", [False, True], ids=["episodes", "dense"])
@pytest.mark.parametrize("batch_size", [1, 3], ids=["shuffled", "full-rollout"])
def test_ppo_scores_mixed_rollout_actions_in_their_emitted_precision(dense: bool, batch_size: int) -> None:
    env = _make_env(max_steps=2)
    try:
        actor = PPOPolicy(env, PPOPolicyConfig(
            actor_config=PPOActorConfig(latent_pi_dim_per_agent=8),
            continuous_config=SquashedDiagGaussianConfig(std=0.1, std_learnable=True),
        ))
        _set_gaussian_means(actor.action_dist, 9.15)
        with torch.no_grad():
            actor.action_dist.distributions[-1].action_net.bias.neg_()
        policy = NextObsPredWrapper(actor, world_model_config=NOPWorldModelConfig(
            n_agents=env.n_agents, local_latent_dim=actor.nop_latent_dim, action_dim=env.action_space.total_agent_action_dim,
            d_model_transition_model=8, nhead_transition_model=2, num_layers_transition_model=1,
            dim_feedforward_transition_model=16,
            next_obs_pred_config=NextObsPredConfig(local_scalar_target_indices=[0]),
        ))
        dtypes = iter((torch.float32, torch.float64, torch.float32))

        def set_rollout_precision(
            _module: torch.nn.Module, args: tuple[Any, ...], kwargs: dict[str, Any],
        ) -> tuple[tuple[Any, ...], dict[str, Any]]:
            actor.rollout_dtype = next(dtypes)
            return args, {**kwargs, "deterministic": True}

        buffer = PPORolloutBuffer(
            max_episode_length=3, observation_space=env.observation_space, action_space=env.action_space,
            gamma=0.99, gae_lambda=0.95, rollout_dtype=torch.float32, train_dtype=torch.float32, train_device="cpu",
        )
        config = PPOWMSamplerConfig(batch_size=batch_size, num_next_steps=2)
        with actor.register_forward_pre_hook(set_rollout_precision, with_kwargs=True):
            if dense:
                rollout, *_ = collect_step_rollout_batch(env, policy, buffer, n_steps=3)
                sampler = policy.make_rollout_batch_sampler(rollout, config)
            else:
                episodes, *_ = collect_steps(env, policy, buffer, n_steps=3)
                sampler = policy.make_sampler(episodes, config)

        batches = list(sampler.sample())
        for batch in batches:
            scores, _, _, _ = policy.evaluate_actions(batch)
            torch.testing.assert_close(scores, batch.log_probs, rtol=1e-6, atol=1e-6, check_dtype=False)

        _set_gaussian_means(actor.action_dist, 9.25)
        for batch in batches:
            scores, _, _, _ = policy.evaluate_actions(batch)
            for dtype, mask in batch.action_dtypes.items():
                if mask.any():
                    native = replace(batch, actions=batch.actions.to(dtype), action_dtypes=None)
                    reference, _, _, _ = policy.evaluate_actions(native)
                    torch.testing.assert_close(scores[mask], reference[mask].to(scores.dtype), rtol=1e-6, atol=1e-6)
            assert not torch.equal(scores, batch.log_probs)
            scores.sum().backward()
        for dist in actor.action_dist.distributions:
            grad = dist.action_net.bias.grad
            assert grad is not None and torch.isfinite(grad).all()
    finally:
        env.close()


@pytest.mark.parametrize("config", _GAUSSIAN_CONFIGS, ids=lambda config: type(config).__name__)
@pytest.mark.parametrize("mean", [-10.0, 10.0])
def test_sac_gaussian_sampling_retains_cached_likelihood_and_gradients(
    config: ContinuousActionDistConfig, mean: float,
) -> None:
    env = _make_env()
    try:
        action_dist = HybridActionDistribution(4, env.action_space, config)
        _set_gaussian_means(action_dist, mean)
        action_dist.reset_temporal_correlations_on_step(batch_shape=(1, 2))
        actions, log_probs = action_dist.get_actions_with_log_probs(torch.ones(1, 2, 4), use_rsample=True)

        assert torch.equal(actions, torch.full_like(actions, 1.0 if mean > 0 else -1.0))
        cached_log_probs = torch.stack(
            [
                dist.log_prob(action, gaussian_actions=dist._last_gaussian_actions)
                for dist, action in zip(action_dist.distributions, actions.split(action_dist.action_dims, dim=-1), strict=True)
            ],
            dim=-1,
        ).sum(dim=-1)
        torch.testing.assert_close(log_probs, cached_log_probs)
        expected_log_probs = torch.stack([
            (
                dist.distribution.log_prob(dist._last_gaussian_actions.double())
                + 2.0 * dist._last_gaussian_actions.double().cosh().log()
            ).sum(dim=-1)
            for dist in action_dist.distributions
        ], dim=-1).sum(dim=-1)
        torch.testing.assert_close(log_probs.double(), expected_log_probs, atol=1e-5, rtol=1e-5)
        assert not torch.allclose(log_probs, action_dist.log_prob(actions))
        log_probs.sum().backward()
        for dist in action_dist.distributions:
            expected_mean_gradient = (2.0 * dist._last_gaussian_actions.tanh().sum(dim=(0, 1))).to(dist.action_net.bias)
            torch.testing.assert_close(dist.action_net.bias.grad, expected_mean_gradient)
            std_parameter = dist.log_std_net.bias if hasattr(dist, "log_std_net") else dist.log_stds
            assert std_parameter.grad is not None
            assert torch.isfinite(std_parameter.grad).all()
            assert std_parameter.grad.abs().sum() > 0
    finally:
        env.close()


@pytest.mark.parametrize("use_rsample", [False, True])
def test_narrow_sac_scores_keep_the_sampled_gaussian_entropy(use_rsample: bool) -> None:
    torch.manual_seed(847)
    component = SquashedDiagGaussianActionDist(1, 1, std=2.5e-9, std_learnable=True)
    with torch.no_grad():
        component.action_net.bias.fill_(0.375)
    actions, scores = component.get_actions_with_log_probs(torch.zeros(16384, 1), use_rsample=use_rsample)
    # At this scale the squash Jacobian is constant to float32 precision.
    expected = -math.log(2.5e-9 * math.sqrt(2 * math.pi)) - 0.5 + 2 * math.log(math.cosh(0.375))
    assert actions.dtype == scores.dtype == torch.float32
    assert scores.mean().item() == pytest.approx(expected, abs=0.02)
    mu_grad, log_std_grad = torch.autograd.grad(scores.mean(), (component.action_net.bias, component.log_stds))
    if use_rsample:
        assert mu_grad.item() == pytest.approx(2 * math.tanh(0.375), abs=2e-6)
        assert log_std_grad.item() == pytest.approx(-1.0, abs=2e-6)
    else:
        scale = component.distribution.scale.detach().double()
        residual = (component._last_gaussian_actions.detach() - component.distribution.loc.detach().double()) / scale
        assert mu_grad.item() == pytest.approx((residual / scale).mean().item(), rel=2e-5)
        assert log_std_grad.item() == pytest.approx((residual.square() - 1).mean().item(), abs=2e-6)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize(("std", "autocast_dtype"), [
    (2.5e-9, None), (1e-25, None), (1e-25, torch.float16), (1e-25, torch.bfloat16),
])
def test_sac_delegating_rsample_keeps_narrow_native_score_gradient(
        compiled: bool, std: float, autocast_dtype: torch.dtype | None,
) -> None:
    class DelegatingRSample(SquashedDiagGaussianActionDist):
        def rsample(self, agent: int | None = None, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
            return super().rsample(agent=agent, previous_actions=previous_actions)

    def sample_and_score(
            dist: SquashedDiagGaussianActionDist, *, compile_draw: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, ...]]:
        with torch.no_grad():
            dist.action_net.bias.fill_(0.375)
        torch.manual_seed(847)
        def draw(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return dist.get_actions_with_log_probs(latent, use_rsample=True)

        if compile_draw:
            draw = torch.compile(draw, backend="aot_eager", fullgraph=True)
        precision = nullcontext() if autocast_dtype is None else torch.autocast("cpu", autocast_dtype)
        with precision:
            actions, scores = draw(torch.zeros(128, 1))
        gradients = torch.autograd.grad(scores.mean(), (dist.action_net.bias, dist.log_stds))
        return actions, scores, gradients

    native = SquashedDiagGaussianActionDist(1, 1, std=std, std_learnable=True)
    custom = DelegatingRSample(1, 1, std=std, std_learnable=True)
    expected = sample_and_score(native, compile_draw=False)
    actual = sample_and_score(custom, compile_draw=compiled)
    expected_actions, expected_scores, expected_gradients = expected
    actual_actions, actual_scores, actual_gradients = actual
    if compiled:
        # The compiled branch can also score a modified action, which needs at least FP32.
        expected_scores = expected_scores.to(torch.promote_types(expected_scores.dtype, torch.float32))
    torch.testing.assert_close(actual_actions, expected_actions, atol=0, rtol=0)
    torch.testing.assert_close(actual_scores, expected_scores, atol=0, rtol=0)
    torch.testing.assert_close(actual_gradients, expected_gradients, atol=0, rtol=0)


@pytest.mark.parametrize("compiled", [False, True])
@pytest.mark.parametrize("in_place", [False, True])
def test_sac_clamped_rsample_keeps_finite_score_in_autocast(compiled: bool, in_place: bool) -> None:
    class ClampedRSample(SquashedDiagGaussianActionDist):
        def rsample(self, agent: int | None = None, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
            actions = super().rsample(agent=agent, previous_actions=previous_actions)
            return actions.clamp_(-0.25, 0.25) if in_place else actions.clamp(-0.25, 0.25)

    dist = ClampedRSample(1, 1, std=1e-3, std_learnable=True)
    with torch.no_grad():
        dist.action_net.bias.fill_(1.0)

    def draw(latent: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return dist.get_actions_with_log_probs(latent, use_rsample=True)

    if compiled:
        draw = torch.compile(draw, backend="aot_eager", fullgraph=True)
    with torch.autocast("cpu", torch.float16):
        actions, scores = draw(torch.zeros(4, 1))

    assert actions.dtype == torch.float16
    torch.testing.assert_close(actions, torch.full_like(actions, 0.25), atol=0, rtol=0)
    expected_scores = dist.log_prob(actions)
    assert expected_scores.dtype == torch.float32
    assert torch.all(expected_scores < -torch.finfo(torch.float16).max)
    torch.testing.assert_close(scores, expected_scores, atol=0, rtol=0)
    gradients = torch.autograd.grad(scores.mean(), (dist.action_net.bias, dist.log_stds))
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
