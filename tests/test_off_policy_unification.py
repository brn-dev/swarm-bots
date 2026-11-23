"""Behavioral regressions for shared off-policy code and opt-in update variants."""

import copy
from collections.abc import Iterable
from dataclasses import fields, replace
from typing import Any
from unittest.mock import patch

import pytest
import torch
from torch import nn

from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.off_policy import collect_off_policy_steps
from swarmbots.learn.algos.off_policy.nop import SACNOPLatentSource
from swarmbots.learn.algos.off_policy.transformer_critic import TransformerCriticConfig
from swarmbots.learn.algos.off_policy.actor_state_critic import ActorStateCriticInputConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoderConfig
from swarmbots.learn.algos.sac import SAC, RecurrentSAC, TMASACPolicy
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import RecurrentTMASACPolicy
from swarmbots.learn.algos.td3 import RecurrentTD3, RecurrentTD3Policy, RecurrentTD3PolicyConfig, TD3
from swarmbots.learn.algos.td3.td3_policy import TD3Policy
from swarmbots.learn.algos.off_policy.actor_heads import TMASACActorHeadConfig, TMASACActorHeadKind
from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from tests.test_sac_algorithm import _make_env, _make_policy
from tests.test_deterministic_baselines import make_policy
from tests.test_recurrent_tmasac import (
    _actor_head_config,
    _make_env as _make_recurrent_env,
    _make_recurrent_critic_algorithm,
    _make_segment_batch,
    _small_nop_config,
    _shared_recurrent_policy_config,
)
from swarmbots.learn.temporal_state import index_temporal_state, clone_temporal_state


@pytest.fixture(scope="module", autouse=True)
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def env():
    instance = _make_env(max_steps=5)
    yield instance
    instance.close()


def algorithm_options():
    return dict(
        buffer_capacity_per_env=16,
        batch_size=2,
        rollout_steps_per_iteration=6,
        learning_starts=0,
        learning_rate_warmup_updates=0,
        replay_storage_device="cpu",
        train_device="cpu",
    )


def make_batch(algorithm):
    collect_off_policy_steps(
        algorithm.env, algorithm.replay_buffer, n_steps=8, policy=algorithm.policy, random_actions=True
    )
    return algorithm.replay_buffer.sample(2)


@pytest.mark.parametrize(("variant", "algorithm_type"), [("tmasac", SAC), ("tmatd3", TD3)])
def test_delayed_actor_gradients_and_frozen_critic_parameter(env, variant, algorithm_type):
    policy = make_policy(env, variant)
    algorithm = algorithm_type(policy, env, **algorithm_options(), policy_delay=2)
    frozen = next(iter(policy.critic_parameters()))
    frozen.requires_grad_(False)
    before = frozen.detach().clone()
    make_batch(algorithm)

    skipped = algorithm.train(gradient_steps=1)
    assert skipped["actor_grad_norm"].n == 0
    assert skipped["critic_grad_norm"].n == 1
    alternating = algorithm.train(gradient_steps=4)
    assert alternating["actor_grad_norm"].n == 2
    assert alternating["actor_grad_norm"].min_value > 0
    assert alternating["critic_grad_norm"].n == 4
    assert not frozen.requires_grad
    torch.testing.assert_close(frozen, before, atol=0, rtol=0)


@pytest.mark.parametrize("invalid", [
    float("nan"), float("inf"), -float("inf"), 1e39, -1e39,
    "auto*1e39", "auto*nan", "auto_inf", "auto*0", "auto_bad",
])
def test_sac_rejects_invalid_target_entropy_before_training(env, invalid):
    with pytest.raises(ValueError, match="target_entropy"):
        SAC(make_policy(env, "tmasac"), env, **algorithm_options(), target_entropy=invalid)


@pytest.mark.parametrize("field,value", [
    ("max_grad_norm", float("nan")), ("ent_coef_learning_rate", float("inf")),
    ("ent_coef", 1e39), ("ent_coef", "auto_1e-48"), ("ent_coef", f"auto_{torch.finfo(torch.float32).max}"),
])
def test_sac_rejects_unusable_optimization_settings(env: SwarmBotsLearnEnvWrapper, field: str, value: Any) -> None:
    with pytest.raises(ValueError):
        SAC(make_policy(env, "tmasac"), env, **{**algorithm_options(), field: value})


def test_live_target_entropy_preserves_last_representable_value(env: SwarmBotsLearnEnvWrapper) -> None:
    boundary = torch.finfo(torch.float32).max
    algorithm = SAC(make_policy(env, "tmasac"), env, **algorithm_options(), target_entropy=-boundary)
    with pytest.raises(ValueError, match="target_entropy"):
        algorithm.target_entropy = -2 * boundary
    batch = make_batch(algorithm)
    target = algorithm._target_entropy(batch=batch, dtype=batch.rewards.dtype, device=batch.rewards.device)
    assert algorithm.target_entropy == -boundary
    torch.testing.assert_close(target, torch.full_like(batch.rewards, -boundary))


def assert_parameters_equal(before: Iterable[torch.Tensor], parameters: Iterable[torch.Tensor]) -> None:
    for old, new in zip(before, parameters, strict=True):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def assert_parameters_changed(before: Iterable[torch.Tensor], parameters: Iterable[torch.Tensor]) -> None:
    assert any(not torch.equal(old, new) for old, new in zip(before, parameters, strict=True))


def parameter_snapshot(parameters: Iterable[torch.Tensor]) -> list[torch.Tensor]:
    return [value.detach().clone() for value in parameters]


def actor_learning_modules(policy: TMASACPolicy) -> dict[str, nn.Module]:
    modules = {"actor_encoder": policy.actor_encoder, "actor_head": policy.actor_head}
    for index, distribution in enumerate(policy.action_dist.distributions):
        modules[f"action_mean_{index}"] = distribution.action_net
        modules[f"action_log_std_{index}"] = distribution.log_std_net
    return modules


def assert_nonzero_finite_gradients(gradients: Iterable[torch.Tensor | None]) -> None:
    gradients = list(gradients)
    assert gradients
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gradients)
    assert any(gradient.count_nonzero() > 0 for gradient in gradients if gradient is not None)


def assert_optimizer_owns(optimizer: torch.optim.Optimizer, modules: Iterable[nn.Module]) -> None:
    expected = {id(parameter) for module in modules for parameter in module.parameters()}
    actual = [parameter for group in optimizer.param_groups for parameter in group["params"]]
    assert len(actual) == len(expected)
    assert {id(parameter) for parameter in actual} == expected


def assert_polyak_parameters(
    previous: Iterable[torch.Tensor], source: nn.Module, target: nn.Module, tau: float,
) -> None:
    for old_target, online, actual in zip(previous, source.parameters(), target.parameters(), strict=True):
        expected = (1.0 - tau) * old_target + tau * online.detach()
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-8)
        assert not actual.requires_grad and actual.grad is None


@pytest.mark.parametrize("variant", ["tmasac", "tmasac_dec", "tmatd3", "tmatd3_dec", "masac__deepset_critic", "matd3__deepset_critic"])
def test_factory_sharing_defaults_and_opt_in_override(env, variant):
    policy = make_policy(env, variant)
    independent_default = False
    assert (policy.critic.encoder2 is not None) == independent_default
    override = make_policy(env, variant, critic_independent_encoders=not independent_default)
    assert (override.critic.encoder2 is not None) != independent_default
    assert policy.critic.nop_source_latent_dim == policy.critic.d_model * (2 if independent_default else 1)


@pytest.mark.parametrize("variant", ["tmasac", "tmasac_dec", "masac__deepset_critic"])
def test_sac_defaults_match_explicit_disabled_options_for_every_update(env, variant):
    torch.manual_seed(123)
    policy = make_policy(env, variant)
    algorithm = SAC(policy, env, **algorithm_options())
    explicit = SAC(
        copy.deepcopy(policy),
        env,
        **algorithm_options(),
        policy_delay=1,
        target_policy_noise=0.0,
        target_noise_clip=0.0,
    )
    batch = make_batch(algorithm)
    for index in range(3):
        torch.manual_seed(456 + index)
        expected = algorithm._train_step(batch, global_update_idx=index)
        torch.manual_seed(456 + index)
        actual = explicit._train_step(batch, global_update_idx=index)
        assert actual == expected
        for name, tensor in algorithm.policy.state_dict().items():
            torch.testing.assert_close(explicit.policy.state_dict()[name], tensor, rtol=0, atol=0)
    assert algorithm.get_hyper_parameters()["policy_delay"] == 1
    assert algorithm.get_hyper_parameters()["target_policy_noise"] == 0
    assert algorithm.get_hyper_parameters()["target_noise_clip"] == 0


@pytest.mark.parametrize("shared_encoder", [False, True])
@pytest.mark.parametrize("actor_head_kind", [TMASACActorHeadKind.INDEPENDENT, TMASACActorHeadKind.QCX])
@pytest.mark.parametrize("action_samples", [1, 3])
@pytest.mark.parametrize("use_nop", [False, True])
def test_sac_delay_freezes_actor_and_temperature_but_updates_critic_and_targets(
    env: SwarmBotsLearnEnvWrapper, shared_encoder: bool, actor_head_kind: TMASACActorHeadKind,
    action_samples: int, use_nop: bool,
) -> None:
    torch.manual_seed(123)
    env.action_space.seed(123)
    nop_config = (
        replace(_small_nop_config(num_next_steps=1), skip_first_transition_for_critic=False) if use_nop else None
    )
    policy = _make_policy(
        env, nop_config=nop_config,
        actor_head_config=_actor_head_config(actor_head_kind),
        shared_encoder=shared_encoder,
    )
    algorithm = SAC(
        policy, env, **algorithm_options(), policy_delay=3,
        actor_action_samples=action_samples, target_action_samples=action_samples, tau=0.25,
    )
    batch = make_batch(algorithm)
    actor_modules = actor_learning_modules(policy)
    critic_modules = {
        "critic_encoder": policy.critic.encoder,
        "q1": policy.critic.q1,
        "q2": policy.critic.q2,
    }
    target_pairs = {"critic": (policy.critic, policy.critic_target)}
    if shared_encoder:
        critic_modules["shared_encoder"] = policy.shared_observation_encoder
        target_pairs["shared_encoder"] = (
            policy.shared_observation_encoder, policy.shared_observation_encoder_target,
        )
    if use_nop:
        actor_modules["actor_nop"] = policy.actor_nop
        critic_modules["critic_nop"] = policy.critic_nop
    assert_optimizer_owns(algorithm.actor_optimizer, actor_modules.values())
    assert_optimizer_owns(algorithm.critic_optimizer, critic_modules.values())
    component_parameters = {
        name: list(module.parameters()) for name, module in {**actor_modules, **critic_modules}.items()
    }
    sampling_gradients: list[tuple[bool, bool]] = []
    sample_actions = policy.action_log_prob

    def record_sampling_gradients(**kwargs: Any) -> tuple[torch.Tensor, torch.Tensor]:
        actions, log_probs = sample_actions(**kwargs)
        sampling_gradients.append((actions.requires_grad, log_probs.requires_grad))
        return actions, log_probs

    with (
        patch.object(policy, "polyak_update_targets", wraps=policy.polyak_update_targets) as targets,
        patch.object(policy, "action_log_prob", side_effect=record_sampling_gradients),
    ):
        for index, actor_updated in enumerate((False, False, True, False, False, True)):
            before = {name: parameter_snapshot(parameters) for name, parameters in component_parameters.items()}
            before_targets = {
                name: parameter_snapshot(target.parameters()) for name, (_, target) in target_pairs.items()
            }
            before_temperature = algorithm.log_ent_coef.detach().clone()
            sampling_gradients.clear()
            metrics, actor_norm, critic_norm = algorithm._train_step(batch, global_update_idx=index)

            assert metrics["actor_updated"] == float(actor_updated)
            assert critic_norm > 0
            assert sampling_gradients == [(actor_updated, actor_updated), (False, False)]
            assert targets.call_count == index + 1
            for name, parameters in component_parameters.items():
                if name in actor_modules and not actor_updated:
                    assert_parameters_equal(before[name], parameters)
                else:
                    assert_parameters_changed(before[name], parameters)
            for name, (online, target) in target_pairs.items():
                assert_polyak_parameters(before_targets[name], online, target, algorithm.tau)
            if actor_updated:
                assert actor_norm > 0
                assert "actor_loss" in metrics and "ent_coef_loss" in metrics
                assert not torch.equal(before_temperature, algorithm.log_ent_coef)
            else:
                assert actor_norm == 0
                assert "actor_loss" not in metrics and "ent_coef_loss" not in metrics
                torch.testing.assert_close(before_temperature, algorithm.log_ent_coef, rtol=0, atol=0)


@pytest.mark.parametrize("shared_encoder", [False, True])
@pytest.mark.parametrize("actor_head_kind", [TMASACActorHeadKind.INDEPENDENT, TMASACActorHeadKind.QCX])
@pytest.mark.parametrize("action_samples", [1, 3])
def test_sac_actor_learns_from_q_without_entropy_or_nop(
    env: SwarmBotsLearnEnvWrapper, shared_encoder: bool, actor_head_kind: TMASACActorHeadKind,
    action_samples: int,
) -> None:
    torch.manual_seed(764)
    env.action_space.seed(764)
    policy = _make_policy(env, actor_head_config=_actor_head_config(actor_head_kind), shared_encoder=shared_encoder)
    algorithm = SAC(
        policy, env, **algorithm_options(), policy_delay=3, ent_coef=0.0,
        actor_action_samples=action_samples, target_action_samples=action_samples,
    )
    batch = make_batch(algorithm)
    assert not policy.has_nop_loss()
    actor_modules = tuple(actor_learning_modules(policy).values())
    for index, actor_updated in enumerate((False, False, True, False, False, True)):
        before = [parameter_snapshot(module.parameters()) for module in actor_modules]
        metrics, actor_norm, _critic_norm = algorithm._train_step(batch, global_update_idx=index)
        assert metrics["actor_updated"] == float(actor_updated)
        assert metrics["ent_coef"] == 0.0
        assert "ent_coef_loss" not in metrics
        if actor_updated:
            assert actor_norm > 0
            for previous, module in zip(before, actor_modules, strict=True):
                assert_parameters_changed(previous, module.parameters())
        else:
            assert actor_norm == 0
            for previous, module in zip(before, actor_modules, strict=True):
                assert_parameters_equal(previous, module.parameters())


@pytest.mark.parametrize("shared_encoder", [False, True])
@pytest.mark.parametrize("loss_source,reuse_critic_latents", [
    pytest.param("actor", False, id="actor"),
    pytest.param("critic", False, id="critic"),
    pytest.param("critic", True, id="critic-cached"),
])
def test_sac_nop_gradients_reach_only_their_source_encoders(
    env: SwarmBotsLearnEnvWrapper, shared_encoder: bool, loss_source: str, reuse_critic_latents: bool,
) -> None:
    torch.manual_seed(831)
    env.action_space.seed(831)
    policy = _make_policy(
        env, shared_encoder=shared_encoder,
        nop_config=replace(_small_nop_config(num_next_steps=1), skip_first_transition_for_critic=False),
    )
    algorithm = SAC(policy, env, **algorithm_options())
    batch = make_batch(algorithm)
    policy.zero_grad(set_to_none=True)
    source_latents = None
    if loss_source == "actor":
        loss, _metrics = policy.compute_actor_nop_loss(batch)
        learning_modules = [policy.actor_encoder, policy.actor_nop]
        untouched_modules = [policy.critic, policy.critic_nop]
        if shared_encoder:
            untouched_modules.append(policy.shared_observation_encoder)
    else:
        if reuse_critic_latents:
            _q1, _q2, source_latents = policy.q_values_with_nop_latents(
                local_obs=batch.local_obs,
                global_obs=batch.global_obs,
                actions=batch.actions,
                hidden_local_vars=batch.hidden_local_vars,
                hidden_global_vars=batch.hidden_global_vars,
                agent_mask=batch.agent_mask,
            )
            assert source_latents is not None and source_latents.requires_grad
            source_latents.retain_grad()
        loss, _metrics = policy.compute_critic_nop_loss(batch, source_latents=source_latents)
        learning_modules = [policy.critic.encoder, policy.critic_nop]
        untouched_modules = [policy.actor_encoder, policy.actor_nop, policy.critic.q1, policy.critic.q2]
        if shared_encoder:
            learning_modules.append(policy.shared_observation_encoder)
    untouched_modules.extend((policy.actor_head, policy.action_dist))
    loss.backward()
    if reuse_critic_latents:
        assert source_latents.grad is not None
        assert_nonzero_finite_gradients((source_latents.grad,))
    for module in learning_modules:
        assert_nonzero_finite_gradients(
            parameter.grad for parameter in module.parameters() if parameter.requires_grad
        )
    for module in untouched_modules:
        assert all(parameter.grad is None for parameter in module.parameters())


@pytest.mark.parametrize("samples", [1, 3])
def test_sac_target_smoothing_preserves_original_policy_entropy_and_masks(env, samples):
    policy = _make_policy(env)
    algorithm = SAC(
        policy,
        env,
        **algorithm_options(),
        ent_coef=0.25,
        target_action_samples=samples,
        target_policy_noise=0.3,
        target_noise_clip=0.2,
    )
    batch = make_batch(algorithm)
    mask = torch.ones(batch.actions.shape[:-1], dtype=torch.bool)
    mask[:, -1] = False
    actions = torch.full_like(batch.actions, 0.95)
    log_probs = torch.full(actions.shape[:-1], -2.0)
    if samples > 1:
        actions = actions.unsqueeze(0).expand(samples, *actions.shape)
        log_probs = log_probs.unsqueeze(0).expand(samples, *log_probs.shape)
    with (
        patch.object(policy, "action_log_prob", return_value=(actions, log_probs)),
        patch.object(policy, "q_values_samples", return_value=(torch.full(log_probs.shape[:-1], 3.0),) * 2) as q,
        patch("torch.randn_like", side_effect=lambda value: torch.full_like(value, 100.0)),
    ):
        target = algorithm._target_forward_phase(
            batch,
            batch.next_local_obs,
            batch.next_global_obs,
            batch.next_hidden_local_vars,
            batch.next_hidden_global_vars,
            mask,
            torch.tensor(0.25),
        )
    smoothed = q.call_args.kwargs["actions"]
    torch.testing.assert_close(smoothed[..., 0, :], torch.ones_like(smoothed[..., 0, :]))
    assert not smoothed[..., -1, :].any()
    expected = torch.where(batch.terminal_mask, batch.rewards, batch.rewards + algorithm.gamma * 3.5)
    torch.testing.assert_close(target, expected)


@pytest.mark.parametrize("shape", [(2, 2, 2), (3, 2, 4, 2, 2)])
def test_smoothing_disabled_consumes_no_randomness(env, shape):
    algorithm = SAC(_make_policy(env), env, **algorithm_options())
    actions = torch.randn(shape)
    before = torch.get_rng_state()
    assert algorithm._smooth_target_actions(actions, None) is actions
    assert torch.equal(before, torch.get_rng_state())


@pytest.mark.parametrize(
    "option,value",
    [
        ("policy_delay", 0),
        ("policy_delay", True),
        ("policy_delay", 1.5),
        ("target_policy_noise", -1),
        ("target_policy_noise", float("nan")),
        ("target_noise_clip", float("inf")),
        ("target_noise_clip", -0.1),
    ],
)
def test_sac_rejects_invalid_opt_in_settings(env, option, value):
    with pytest.raises(ValueError, match=option):
        SAC(_make_policy(env), env, **algorithm_options(), **{option: value})


def recurrent_batch(env):
    return _make_segment_batch(
        batch_size=2,
        sequence_length=4,
        n_agents=env.n_agents,
        local_obs_dim=env.local_obs_dim,
        global_obs_dim=env.global_obs_dim,
        hidden_local_vars_dim=env.hidden_local_vars_dim,
        hidden_global_vars_dim=env.hidden_global_vars_dim,
        action_dim=env.action_space.total_agent_action_dim,
    )


def recurrent_td3(
    env,
    *,
    recurrent_critic=False,
    shared=False,
    nop=False,
    critic_kind="transformer",
    compile_modules=False,
    actor_state_config="auto",
):
    actor_config = RMATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
    critic_config = (
        actor_config if recurrent_critic else MATEncoderConfig(d_model=8, nhead=2, num_layers=1, dim_feedforward=16)
    )
    config = RecurrentTD3PolicyConfig(
        actor_encoder_config=actor_config,
        critic_encoder_config=critic_config,
        actor_head_config=TMASACActorHeadConfig(hidden_dims=[8]),
        critic_kind=critic_kind,
        joint_critic_config=JointCriticConfig(
            kind="deepset", hidden_dims=(8,), element_hidden_dims=(8,), independent_encoders=not shared,
        ),
        recurrent_critic=recurrent_critic,
        actor_state_critic_input_config=actor_state_config,
        compile_modules=compile_modules,
        transformer_critic_config=TransformerCriticConfig(independent_encoders=not shared),
        **({"nop_config": _small_nop_config(latent_source=SACNOPLatentSource.BOTH)} if nop else {}),
    )
    policy = RecurrentTD3Policy(env, config)
    return policy, RecurrentTD3(
        policy, env, **algorithm_options(), burn_in_steps=1, learning_steps=3, temporal_state_store_interval=1
    )


def test_recurrent_sac_delay_and_multi_sample_smoothing(env: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    policy, algorithm = _make_recurrent_critic_algorithm(env)
    delayed = RecurrentSAC(
        policy,
        env,
        **algorithm_options(),
        burn_in_steps=1,
        learning_steps=3,
        temporal_state_store_interval=1,
        policy_delay=2,
        actor_action_samples=2,
        target_action_samples=3,
        target_policy_noise=0.2,
        target_noise_clip=0.5,
    )
    forward = delayed._learning_sequence_actions
    action_requires_grad = []

    def record_actions(**kwargs: Any) -> tuple[Any, ...]:
        result = forward(**kwargs)
        action_requires_grad.append(result[0].requires_grad)
        return result

    monkeypatch.setattr(delayed, "_learning_sequence_actions", record_actions)
    before = parameter_snapshot(policy.actor_parameters())
    metrics, _, _ = delayed._train_step(recurrent_batch(env), global_update_idx=0)
    assert metrics["actor_updated"] == 0
    assert_parameters_equal(before, policy.actor_parameters())
    metrics, actor_norm, critic_norm = delayed._train_step(recurrent_batch(env), global_update_idx=1)
    assert metrics["actor_updated"] == 1 and actor_norm > 0 and critic_norm > 0
    assert action_requires_grad == [False, True]


@pytest.mark.parametrize(
    "recurrent_critic,shared,nop,critic_kind",
    [
        (False, False, False, "transformer"),
        (False, True, True, "transformer"),
        (True, False, True, "transformer"),
        (True, True, True, "transformer"),
        (False, False, True, "deepset"),
        (False, True, True, "deepset"),
        (False, False, False, "mlp"),
    ],
)
def test_recurrent_td3_trains_with_history_and_preserves_delay(env, recurrent_critic, shared, nop, critic_kind):
    policy, algorithm = recurrent_td3(
        env, recurrent_critic=recurrent_critic, shared=shared, nop=nop, critic_kind=critic_kind
    )
    batch = recurrent_batch(env)
    before = parameter_snapshot(policy.actor_parameters())
    target_before = parameter_snapshot(policy.actor_target.parameters())
    metrics, _, _ = algorithm._train_step(batch, global_update_idx=0)
    assert metrics["actor_updated"] == 0
    assert_parameters_equal(before, policy.actor_parameters())
    assert_parameters_equal(target_before, policy.actor_target.parameters())
    metrics, actor_norm, critic_norm = algorithm._train_step(batch, global_update_idx=1)
    assert metrics["actor_updated"] == 1 and actor_norm > 0 and critic_norm > 0
    assert all(torch.isfinite(value).all() for value in policy.parameters())
    if nop:
        assert metrics["actor_nop_loss"] > 0 and metrics["critic_nop_loss"] > 0
    assert all(value.grad is None for value in policy.actor_target.parameters())
    assert all(value.grad is None for value in policy.critic_target.parameters())


def test_recurrent_td3_replay_anchors_online_and_target_histories_separately(env):
    policy, algorithm = recurrent_td3(env)
    with torch.no_grad():
        next(policy.actor_target.encoder.parameters()).add_(0.7)
    collect_off_policy_steps(env, algorithm.replay_buffer, n_steps=12, policy=policy, random_actions=True)
    batch, _, _ = algorithm._sample_training_batches()
    assert set(batch.initial_temporal_state) == {"actor", "target_actor"}
    assert algorithm.train(gradient_steps=2)["updates"] == 2


def test_recurrent_td3_episode_reset_matches_fresh_state(env):
    policy, _algorithm = recurrent_td3(env)
    local = torch.randn(2, env.n_agents, env.local_obs_dim)
    global_obs = torch.randn(2, env.global_obs_dim)
    _, state = policy.act_with_temporal_state(local, global_obs, deterministic=True)
    reset_actions, reset_state = policy.act_with_temporal_state(
        local, global_obs, temporal_state=state, episode_start_mask=torch.ones(2, dtype=torch.bool), deterministic=True
    )
    fresh_actions, _fresh_state = policy.act_with_temporal_state(local, global_obs, deterministic=True)
    torch.testing.assert_close(reset_actions, fresh_actions)
    assert set(reset_state) == {"actor", "target_actor"}


def test_plain_td3_rejects_recurrent_policy_and_factory_opt_in_builds_it(env):
    policy, _algorithm = recurrent_td3(env)
    with pytest.raises(TypeError, match="recurrent TD3"):
        TD3(policy, env, **algorithm_options())
    factory_policy = make_policy(env, "tmatd3", td3_recurrent_actor=True, td3_recurrent_critic=True)
    assert isinstance(factory_policy, RecurrentTD3Policy)
    assert factory_policy.recurrent_critic
    assert isinstance(make_policy(env, "tmatd3"), TD3Policy)


def test_recurrent_sac_shared_encoder_keeps_critic_gradients_on_delayed_steps(
    env: Any, monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = RecurrentTMASACPolicy(
        env,
        _shared_recurrent_policy_config(
            nop_config=_small_nop_config(latent_source=SACNOPLatentSource.SHARED_ENCODER),
        ),
    )
    algorithm = RecurrentSAC(
        policy,
        env,
        **algorithm_options(),
        burn_in_steps=1,
        learning_steps=3,
        temporal_state_store_interval=1,
        policy_delay=2,
    )
    before_actor = parameter_snapshot(policy.actor_parameters())
    before_shared = parameter_snapshot(policy.shared_observation_encoder.parameters())
    forward = policy.action_log_prob_sequence_with_selected_states_and_shared_latents
    gradient_paths = []

    def record_gradients(**kwargs: Any) -> tuple[Any, ...]:
        result = forward(**kwargs)
        actions, _, actor_latents, _, _, shared_latents = result
        gradient_paths.append((actions.requires_grad, actor_latents.requires_grad, shared_latents.requires_grad))
        return result

    monkeypatch.setattr(policy, "action_log_prob_sequence_with_selected_states_and_shared_latents", record_gradients)
    metrics, _, _ = algorithm._train_step(recurrent_batch(env), global_update_idx=0)
    assert metrics["actor_updated"] == 0
    assert_parameters_equal(before_actor, policy.actor_parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before_shared, policy.shared_observation_encoder.parameters()))
    metrics, _, _ = algorithm._train_step(recurrent_batch(env), global_update_idx=1)
    assert metrics["actor_updated"] == 1
    assert gradient_paths == [(False, False, True), (True, True, True)]


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_shared_recurrent_sac_learns_from_identical_finite_initial_observations(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(193)
    environment = _make_recurrent_env()
    try:
        policy = RecurrentTMASACPolicy(environment, _shared_recurrent_policy_config())
        options = algorithm_options()
        options.update(
            train_device=device, sac_compile_optimizer_steps=False, sac_compile_tensor_operations=False,
            learning_rate_warmup_updates=2,
        )
        algorithm = RecurrentSAC(
            policy, environment, **options, burn_in_steps=0, learning_steps=3,
            temporal_state_store_interval=1, policy_delay=2,
        )
        batch = _make_segment_batch(
            batch_size=2, sequence_length=3, n_agents=environment.n_agents,
            local_obs_dim=environment.local_obs_dim, global_obs_dim=environment.global_obs_dim,
            hidden_local_vars_dim=environment.hidden_local_vars_dim,
            hidden_global_vars_dim=environment.hidden_global_vars_dim,
            action_dim=environment.action_space.total_agent_action_dim,
        )
        batch.actions.mul_(0.25)
        batch.previous_actions.zero_()
        batch.previous_actions[:, 1:].copy_(batch.actions[:, :-1])
        batch.rewards.copy_(torch.tensor([1.0, 2.0, 3.0]).expand_as(batch.rewards))
        batch.episode_start_mask[:, 0] = True
        batch = replace(batch, **{
            field.name: value.to(device)
            for field in fields(batch) if torch.is_tensor(value := getattr(batch, field.name))
        })
        before_shared = parameter_snapshot(policy.shared_observation_encoder.parameters())
        before_nop = parameter_snapshot(policy.critic_nop.parameters())

        for index in range(4):
            _metrics, actor_norm, critic_norm = algorithm._train_step(batch, global_update_idx=index)
            assert torch.isfinite(torch.as_tensor(actor_norm))
            assert torch.isfinite(torch.as_tensor(critic_norm))
            assert all(torch.isfinite(parameter).all() for parameter in policy.parameters())

        assert any(not torch.equal(old, new) for old, new in zip(
            before_shared, policy.shared_observation_encoder.parameters(), strict=True,
        ))
        assert any(not torch.equal(old, new) for old, new in zip(before_nop, policy.critic_nop.parameters(), strict=True))
    finally:
        environment.close()


@pytest.mark.parametrize("recurrent,recurrent_critic", [(False, False), (True, False), (True, True)])
def test_delayed_updates_resume_with_optimizer_and_counter_state(env, tmp_path, recurrent, recurrent_critic):
    if recurrent:
        _policy, algorithm = recurrent_td3(env, recurrent_critic=recurrent_critic, nop=True)
        _restored_policy, restored = recurrent_td3(env, recurrent_critic=recurrent_critic, nop=True)
        batch = recurrent_batch(env)
    else:
        algorithm = SAC(_make_policy(env), env, **algorithm_options(), policy_delay=3)
        restored = SAC(_make_policy(env), env, **algorithm_options(), policy_delay=3)
        batch = make_batch(algorithm)
    for index in range(2):
        algorithm._train_step(batch, global_update_idx=index)
    assert algorithm.n_total_updates == 0
    algorithm.n_total_updates = 2
    checkpoint = tmp_path / "off-policy.pt"
    algorithm.save(checkpoint, optimizer_state_dict=algorithm._get_optimizer_state_dict())
    restored.load(checkpoint, restore_env_state=False)
    assert restored.n_total_updates == 2
    torch.manual_seed(123)
    expected = algorithm._train_step(batch, global_update_idx=algorithm.n_total_updates)
    torch.manual_seed(123)
    actual = restored._train_step(batch, global_update_idx=restored.n_total_updates)
    assert actual == expected
    torch.testing.assert_close(restored.policy.state_dict(), algorithm.policy.state_dict(), rtol=0, atol=0)


@pytest.mark.parametrize("field", ["critic_optimizer", "ent_coef_optimizer"])
def test_rejected_off_policy_optimizer_preserves_weights_and_temperature(env, tmp_path, field: str) -> None:
    source = SAC(_make_policy(env), env, **algorithm_options(), ent_coef="auto_0.4")
    source._train_step(make_batch(source), global_update_idx=0)
    path = tmp_path / "invalid.pt"
    source.save(path, optimizer_state_dict=source._get_optimizer_state_dict())
    checkpoint = torch.load(path, weights_only=False)
    checkpoint["optimizer_state_dict"][field]["param_groups"][0]["params"].clear()
    torch.save(checkpoint, path)
    target = SAC(_make_policy(env), env, **algorithm_options(), ent_coef=0.1)
    weights = {name: value.clone() for name, value in target.policy.state_dict().items()}
    with pytest.raises(ValueError, match="parameter group"):
        target.load(path, restore_env_state=False)
    torch.testing.assert_close(target.policy.state_dict(), weights)
    assert target.log_ent_coef is None and target.ent_coef == 0.1
    assert not target.actor_optimizer.state and not target.critic_optimizer.state


def test_recurrent_td3_truncation_bootstraps_from_history_before_reset(env):
    policy, algorithm = recurrent_td3(env)
    algorithm.target_policy_noise = 0
    batch = recurrent_batch(env)
    resets = torch.tensor([[True, False, False, True], [True, False, False, True]])
    batch = replace(
        batch,
        episode_start_mask=resets,
        truncations=torch.tensor([[False, False, True, False], [False, False, True, False]]),
        terminations=torch.zeros_like(batch.terminations),
    )
    state = None
    expected = []
    expected_actor_states = []
    with torch.no_grad():
        for parameter in policy.actor_target.encoder.parameters():
            parameter.add_(0.1)
    with torch.no_grad():
        for time in range(4):
            _latents, state = policy.encode_actor_sequence(
                local_obs=batch.local_obs[:, time],
                global_obs=batch.global_obs[:, time],
                agent_mask=batch.agent_mask[:, time],
                target=True,
                initial_state=state,
                reset_mask=resets[:, time],
            )
            if time > 0:
                actions, _latents, candidate_state = policy.actor_actions_sequence(
                    local_obs=batch.next_local_obs[:, time],
                    global_obs=batch.next_global_obs[:, time],
                    agent_mask=batch.next_agent_mask[:, time],
                    target=True,
                    initial_state=clone_temporal_state(state),
                )
                expected.append(actions)
                expected_actor_states.append(policy.actor_state_critic_input(candidate_state, target=True))
    with patch.object(algorithm, "_target_next_q_values", wraps=algorithm._target_next_q_values) as target_q:
        algorithm._train_step(batch, global_update_idx=0)
    torch.testing.assert_close(target_q.call_args.kwargs["next_actions"], torch.stack(expected, dim=1))
    torch.testing.assert_close(target_q.call_args.kwargs["actor_state"], torch.stack(expected_actor_states, dim=1))


def test_recurrent_td3_online_and_target_states_are_independent(env):
    policy, _algorithm = recurrent_td3(env)
    local = torch.randn(2, env.n_agents, env.local_obs_dim)
    global_obs = torch.randn(2, env.global_obs_dim)
    _, initial = policy.act_with_temporal_state(local, global_obs, deterministic=True)
    with torch.no_grad():
        for parameter in policy.actor.encoder.parameters():
            parameter.add_(0.1)
    _, advanced = policy.act_with_temporal_state(local, global_obs, temporal_state=initial, deterministic=True)
    _latents, expected = policy.encode_actor_sequence(
        local_obs=local, global_obs=global_obs, initial_state=initial["target_actor"], target=True
    )

    def assert_states_equal(a, b):
        if torch.is_tensor(a):
            torch.testing.assert_close(a, b)
        else:
            for lhs, rhs in zip(a, b, strict=True):
                assert_states_equal(lhs, rhs)

    assert_states_equal(advanced["target_actor"], expected)
    # Both state families remain independently indexable by replay's generic helpers.
    assert set(index_temporal_state(advanced, slice(0, 1))) == {"actor", "target_actor"}


@pytest.mark.parametrize(
    "recurrent_critic,shared,critic_kind",
    [(False, False, "transformer"), (True, False, "transformer"), (False, False, "deepset"), (False, True, "deepset")],
)
def test_recurrent_td3_compiled_sequence_entry_points_preserve_updates(env, recurrent_critic, shared, critic_kind):
    torch._dynamo.reset()
    real_compile = torch.compile
    torch.manual_seed(99)
    eager_policy, eager = recurrent_td3(
        env, recurrent_critic=recurrent_critic, shared=shared, critic_kind=critic_kind, nop=True,
    )
    with patch(
        "torch.compile", side_effect=lambda function, **kwargs: real_compile(function, backend="eager", **kwargs)
    ):
        compiled_policy, compiled = recurrent_td3(
            env, recurrent_critic=recurrent_critic, shared=shared, critic_kind=critic_kind, nop=True, compile_modules=True,
        )
    compiled_policy.load_state_dict(eager_policy.state_dict())
    batch = recurrent_batch(env)
    for index in range(2):
        torch.manual_seed(200 + index)
        expected = eager._train_step(batch, global_update_idx=index)
        torch.manual_seed(200 + index)
        actual = compiled._train_step(batch, global_update_idx=index)
        assert actual[0] == pytest.approx(expected[0], rel=1e-5, abs=1e-7)
        assert actual[1:] == pytest.approx(expected[1:], rel=1e-5, abs=1e-7)
    # Adam amplifies float32 roundoff in nearly zero attention-bias gradients.
    torch.testing.assert_close(compiled_policy.state_dict(), eager_policy.state_dict(), rtol=1e-5, atol=3e-6)
    assert compiled_policy._actor_encoder_sequence_forward is not None
    torch._dynamo.reset()


@pytest.mark.cuda
@pytest.mark.parametrize("recurrent_critic", [False, True])
def test_cuda_recurrent_td3_compiled_updates_match_eager(env, recurrent_critic):
    if not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.compiler.reset()
    try:
        torch.manual_seed(99)
        eager_policy, _ = recurrent_td3(env, recurrent_critic=recurrent_critic, nop=True)
        compiled_policy, _ = recurrent_td3(
            env, recurrent_critic=recurrent_critic, nop=True, compile_modules=True,
        )
        compiled_policy.load_state_dict(eager_policy.state_dict())
        options = algorithm_options()
        options.update(train_device="cuda", burn_in_steps=1, learning_steps=3, temporal_state_store_interval=1)
        eager = RecurrentTD3(
            eager_policy, env, **options, sac_compile_tensor_operations=False, sac_compile_optimizer_steps=False,
        )
        compiled = RecurrentTD3(
            compiled_policy, env, **options, sac_compile_tensor_operations=True, sac_compile_optimizer_steps=True,
        )
        batch = recurrent_batch(env)
        batch = replace(batch, **{
            field.name: value.cuda()
            for field in fields(batch) if torch.is_tensor(value := getattr(batch, field.name))
        })
        for index in range(4):
            torch.manual_seed(777 + index)
            expected = eager._train_step(batch, global_update_idx=index)
            torch.manual_seed(777 + index)
            actual = compiled._train_step(batch, global_update_idx=index)
            assert actual[0] == pytest.approx(expected[0], rel=1e-4, abs=1e-5)
            assert actual[1:] == pytest.approx(expected[1:], rel=1e-4, abs=1e-5)
            assert actual[0]["actor_updated"] == float(index % 2 == 1)
            for compiled_parameter, eager_parameter in zip(
                compiled_policy.parameters(), eager_policy.parameters(), strict=True,
            ):
                if eager_parameter.grad is None:
                    assert compiled_parameter.grad is None
                else:
                    torch.testing.assert_close(
                        compiled_parameter.grad, eager_parameter.grad, rtol=1e-4, atol=1e-5,
                    )
            # Adam amplifies float32 roundoff in nearly zero attention-bias gradients.
            torch.testing.assert_close(compiled_policy.state_dict(), eager_policy.state_dict(), rtol=1e-4, atol=3e-5)
        for target in (compiled_policy.actor_target, compiled_policy.critic_target):
            assert all(not parameter.requires_grad and parameter.grad is None for parameter in target.parameters())
    finally:
        torch.compiler.reset()


@pytest.mark.integration
def test_training_helper_selects_recurrent_td3_for_the_opt_in_flags():
    from swarmbots.learn import make_training
    from tests.test_learning_training import _training_kwargs

    settings = _training_kwargs()
    settings["use_nop"] = False
    settings["policy_kwargs"].update(td3_recurrent_actor=True, td3_recurrent_critic=True)
    settings.setdefault("algorithm_kwargs", {}).update(
        burn_in_steps=1, learning_steps=3, temporal_state_store_interval=1, batch_size=2, replay_exposure=3.0
    )
    algorithm = make_training("SwarmBots-WallEasy-v0", "tmatd3", **settings)
    try:
        assert isinstance(algorithm, RecurrentTD3)
        assert algorithm.policy.recurrent_critic
    finally:
        algorithm.env.close()


@pytest.mark.parametrize("variant", ["tmasac__lstm", "tmasac__slstm"])
def test_standard_recurrent_tmasac_presets_condition_the_feedforward_critic_on_actor_state(env, variant):
    policy = make_policy(env, variant)
    assert policy.uses_actor_state_critic_input
    assert not policy.recurrent_critic
    assert hasattr(policy.critic, "actor_state_encoder")


@pytest.mark.parametrize("kind", ["mlp", "deepset", "transformer"])
def test_recurrent_td3_feedforward_critic_uses_detached_actor_state_by_default(env, kind):
    policy, _algorithm = recurrent_td3(env, critic_kind=kind)
    assert policy.uses_actor_state_critic_input
    batch = recurrent_batch(env)
    actor_state = torch.randn(*batch.local_obs.shape[:-1], 16, requires_grad=True)
    inputs = dict(
        local_obs=batch.local_obs,
        global_obs=batch.global_obs,
        actions=batch.actions,
        hidden_local_vars=batch.hidden_local_vars,
        hidden_global_vars=batch.hidden_global_vars,
        agent_mask=batch.agent_mask,
    )
    q1, q2, _latents, _state = policy.q_values_sequence(**inputs, actor_state=actor_state)
    changed_q1, _q2, _latents, _state = policy.q_values_sequence(**inputs, actor_state=actor_state + 2)
    assert not torch.equal(q1, changed_q1)
    (q1.mean() + q2.mean()).backward()
    assert actor_state.grad is None
    assert all(parameter.grad is None for parameter in policy.actor.parameters())
    modules = list(policy.critic) if kind == "mlp" else [policy.critic]
    for module in modules:
        assert any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in module.actor_state_encoder.parameters()
        )
        target_ids = {id(parameter) for parameter in policy.critic_target.parameters()}
        assert not target_ids & {id(parameter) for parameter in policy.critic_parameters()}


@pytest.mark.parametrize("kind", ["mlp", "deepset", "transformer"])
def test_td3_actor_state_opt_out_and_stateless_q_methods(env, kind):
    policy, _algorithm = recurrent_td3(env, critic_kind=kind)
    batch = recurrent_batch(env)
    inputs = dict(
        local_obs=batch.local_obs[:, 0],
        global_obs=batch.global_obs[:, 0],
        actions=batch.actions[:, 0],
        hidden_local_vars=batch.hidden_local_vars[:, 0],
        hidden_global_vars=batch.hidden_global_vars[:, 0],
        agent_mask=batch.agent_mask[:, 0],
    )
    assert all(torch.isfinite(value).all() for value in policy.q_values(**inputs))
    assert all(torch.isfinite(value).all() for value in policy.target_q_values(**inputs))
    disabled, algorithm = recurrent_td3(env, critic_kind=kind, actor_state_config=None)
    assert not disabled.uses_actor_state_critic_input
    algorithm._train_step(batch, global_update_idx=1)


def test_td3_recurrent_critic_keeps_its_own_history_instead_of_actor_state_input(env):
    policy, _algorithm = recurrent_td3(env, recurrent_critic=True)
    assert not policy.uses_actor_state_critic_input
    with pytest.raises(ValueError, match="recurrent_critic=True"):
        recurrent_td3(env, recurrent_critic=True, actor_state_config=ActorStateCriticInputConfig())
    factory_policy = make_policy(env, "tmatd3", td3_recurrent_actor=True)
    assert factory_policy.uses_actor_state_critic_input
    assert not make_policy(
        env, "tmatd3", td3_recurrent_actor=True, td3_actor_state_critic_input_config=None
    ).uses_actor_state_critic_input
