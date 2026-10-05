"""Behavioral regressions for shared off-policy code and opt-in update variants."""

import copy
from dataclasses import fields, replace
from unittest.mock import patch

import pytest
import torch

from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.off_policy import collect_off_policy_steps
from swarmbots.learn.algos.off_policy.base_algorithm import OffPolicyAlgorithm
from swarmbots.learn.algos.off_policy.nop import SACNOPLatentSource
from swarmbots.learn.algos.off_policy.transformer_critic import TransformerCriticConfig, TransformerTwinCritic
from swarmbots.learn.algos.off_policy.actor_state_critic import ActorStateCriticInputConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoderConfig
from swarmbots.learn.algos.sac import SAC, RecurrentSAC, TMASACCriticConfig
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import RecurrentTMASACPolicy
from swarmbots.learn.algos.sac.tmasac_policy import TMASACTwinCritic
from swarmbots.learn.algos.td3 import RecurrentTD3, RecurrentTD3Policy, RecurrentTD3PolicyConfig, TD3
from swarmbots.learn.algos.td3.td3_policy import TD3Policy
from swarmbots.learn.algos.sac.tmasac_actor_heads import TMASACActorHeadConfig
from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig
from tests.test_sac_algorithm import _make_env, _make_policy
from tests.test_deterministic_baselines import make_policy
from tests.test_recurrent_tmasac import (
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


def assert_parameters_equal(before, parameters):
    for old, new in zip(before, parameters, strict=True):
        torch.testing.assert_close(old, new, rtol=0, atol=0)


def parameter_snapshot(parameters):
    return [value.detach().clone() for value in parameters]


def test_objectives_and_legacy_critic_imports_share_the_neutral_components():
    assert issubclass(SAC, OffPolicyAlgorithm)
    assert issubclass(TD3, OffPolicyAlgorithm)
    assert not issubclass(TD3, SAC)
    assert TMASACCriticConfig is TransformerCriticConfig
    assert TMASACTwinCritic is TransformerTwinCritic


@pytest.mark.parametrize("variant", ["tmasac", "tmasac_dec", "tmatd3", "tmatd3_dec"])
def test_factory_sharing_defaults_and_opt_in_override(env, variant):
    policy = make_policy(env, variant)
    independent_default = False
    assert (policy.critic.encoder2 is not None) == independent_default
    override = make_policy(env, variant, critic_independent_encoders=not independent_default)
    assert (override.critic.encoder2 is not None) != independent_default
    assert policy.critic.nop_source_latent_dim == policy.critic.d_model * (2 if independent_default else 1)


@pytest.mark.parametrize("variant", ["tmasac", "tmasac_dec", "masac_deepset"])
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


def test_sac_delay_freezes_actor_and_temperature_but_updates_critic_and_targets(env):
    policy = _make_policy(env)
    algorithm = SAC(policy, env, **algorithm_options(), policy_delay=3)
    batch = make_batch(algorithm)
    before_actor = parameter_snapshot(policy.actor_parameters())
    before_critic = parameter_snapshot(policy.critic_parameters())
    before_temperature = algorithm.log_ent_coef.detach().clone()
    with patch.object(policy, "polyak_update_targets", wraps=policy.polyak_update_targets) as targets:
        for index in (0, 1):
            metrics, actor_norm, _critic_norm = algorithm._train_step(batch, global_update_idx=index)
            assert metrics["actor_updated"] == 0 and actor_norm == 0
            assert "actor_loss" not in metrics and "ent_coef_loss" not in metrics
            assert_parameters_equal(before_actor, policy.actor_parameters())
            torch.testing.assert_close(before_temperature, algorithm.log_ent_coef, rtol=0, atol=0)
        assert targets.call_count == 2
        metrics, _, _ = algorithm._train_step(batch, global_update_idx=2)
        assert metrics["actor_updated"] == 1
        assert "actor_loss" in metrics and "ent_coef_loss" in metrics
    assert any(not torch.equal(a, b) for a, b in zip(before_actor, policy.actor_parameters()))
    assert any(not torch.equal(a, b) for a, b in zip(before_critic, policy.critic_parameters()))
    assert not torch.equal(before_temperature, algorithm.log_ent_coef)


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
        joint_critic_config=JointCriticConfig(kind="deepset", hidden_dims=(8,), element_hidden_dims=(8,)),
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


def test_recurrent_sac_delay_and_multi_sample_smoothing(env):
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
    before = parameter_snapshot(policy.actor_parameters())
    metrics, _, _ = delayed._train_step(recurrent_batch(env), global_update_idx=0)
    assert metrics["actor_updated"] == 0
    assert_parameters_equal(before, policy.actor_parameters())
    metrics, actor_norm, critic_norm = delayed._train_step(recurrent_batch(env), global_update_idx=1)
    assert metrics["actor_updated"] == 1 and actor_norm > 0 and critic_norm > 0


@pytest.mark.parametrize(
    "recurrent_critic,shared,nop,critic_kind",
    [
        (False, False, False, "transformer"),
        (False, True, True, "transformer"),
        (True, False, True, "transformer"),
        (True, True, True, "transformer"),
        (False, False, True, "deepset"),
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


def test_recurrent_sac_shared_encoder_keeps_critic_gradients_on_delayed_steps(env):
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
    metrics, _, _ = algorithm._train_step(recurrent_batch(env), global_update_idx=0)
    assert metrics["actor_updated"] == 0
    assert_parameters_equal(before_actor, policy.actor_parameters())
    assert any(not torch.equal(a, b) for a, b in zip(before_shared, policy.shared_observation_encoder.parameters()))
    metrics, _, _ = algorithm._train_step(recurrent_batch(env), global_update_idx=1)
    assert metrics["actor_updated"] == 1


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


@pytest.mark.parametrize("recurrent_critic", [False, True])
def test_recurrent_td3_compiled_sequence_entry_points_preserve_updates(env, recurrent_critic):
    torch._dynamo.reset()
    real_compile = torch.compile
    torch.manual_seed(99)
    eager_policy, eager = recurrent_td3(env, recurrent_critic=recurrent_critic, nop=True)
    with patch(
        "torch.compile", side_effect=lambda function, **kwargs: real_compile(function, backend="eager", **kwargs)
    ):
        compiled_policy, compiled = recurrent_td3(
            env, recurrent_critic=recurrent_critic, nop=True, compile_modules=True
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
    torch.testing.assert_close(compiled_policy.state_dict(), eager_policy.state_dict(), rtol=1e-5, atol=1e-7)
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
        burn_in_steps=1, learning_steps=3, temporal_state_store_interval=1
    )
    algorithm = make_training("SwarmBots-WallEasy-v0", "tmatd3", **settings)
    try:
        assert isinstance(algorithm, RecurrentTD3)
        assert algorithm.policy.recurrent_critic
    finally:
        algorithm.env.close()


@pytest.mark.parametrize("variant", ["tmasac_lstm", "tmasac_slstm", "tmasac_slstm_swiglu"])
def test_standard_recurrent_tmasac_presets_condition_the_feedforward_critic_on_actor_state(env, variant):
    policy = make_policy(env, variant)
    assert policy.uses_actor_state_critic_input
    assert not policy.recurrent_critic
    assert hasattr(policy.critic, "actor_state_encoder")


@pytest.mark.parametrize("variant", ["tmasac_lstm_no_actor_state", "tmasac_slstm_shared_encoder"])
def test_tmasac_actor_state_ablation_and_shared_temporal_latents_keep_their_existing_inputs(env, variant):
    assert not make_policy(env, variant).uses_actor_state_critic_input


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
    modules = [policy.critic] if kind == "transformer" else list(policy.critic)
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
