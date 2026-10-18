from __future__ import annotations

import json
import csv
import math
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import torch
from gymnasium import spaces
from torch import nn
from loguru import logger

from swarmbots import ALL_BENCHMARK_IDS, evaluate_policy
from swarmbots.learn import as_benchmark_policy, list_variants, make_training, train
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import (
    GumbelSoftmaxSignMagnitudeBetaActionDist,
)
from swarmbots.learn.action_dists.predicted_std_gaussian_action_dist import PredictedStdGaussianActionDist
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaActionDist
from swarmbots.learn.algos.mappo.mappo_policy import MAPPOPolicy
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy
from swarmbots.learn.algos.ppo.ppo import PPO
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples
from swarmbots.learn.algos.sac.recurrent_sac import RecurrentSAC
from swarmbots.learn.algos.sac.sac import SAC
from swarmbots.learn.algos.td3 import TD3
from swarmbots.learn.algos.sac.tmasac_policy import TMASACPolicy
from swarmbots.learn.checkpointing import capture_env_state
from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.nn_components.feed_forward import MLP, MLPConfig
from swarmbots.learn.presets.policy_factory import _is_off_policy_variant, _make_base_policy
from swarmbots.learn.presets.tmasac import make_tmasac_options, TMASACVariant
from swarmbots.learn.presets.transformer import MATInitGains, MATNormalizationConfig
from swarmbots.learn.training import _variant_options
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


OFF_POLICY_VARIANTS = tuple(
    variant for variant in list_variants(include_hidden=True)
    if _is_off_policy_variant(_variant_options(variant)[0]["policy_variant"])
)


@pytest.fixture(scope="module", autouse=True)
def single_torch_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _scenario_kwargs() -> dict[str, Any]:
    return {
        "unit_start_locations": MJWPreConnectedUnitLocationsConfig(
            num_units=2,
            pool_seeds=(42000,),
            max_radius=1.5,
            z_pos=0.5,
        ),
        "compile_reward_kernel": False,
        "reset_settle_time": 0,
    }


def _training_kwargs() -> dict[str, Any]:
    return {
        "model_scale": None,  # Exercise explicit architecture overrides in these tests.
        "num_envs": 2,
        "device": "cpu",
        "episode_length": 4,
        "scenario_kwargs": _scenario_kwargs(),
        "env_kwargs": {"compile_tensor_operations": False},
        "policy_kwargs": {"enc_d_model": 32, "dec_d_model": 16},
    }


@pytest.fixture
def capture_training_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    def make_environment(
        _benchmark_id: str, *, policy_options: dict[str, Any], **_kwargs: Any,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        recurrent = (
            policy_options["policy_variant"] == "tmasac__recurrent"
            or policy_options.get("td3_recurrent_actor", False)
        )
        policy = SimpleNamespace(requires_recurrent_training=lambda: recurrent)
        env = SimpleNamespace(unwrapped=SimpleNamespace(
            device=torch.device("cpu"), episode_length=_kwargs["episode_length"] or 500,
        ))
        return env, policy

    def capture_algorithm(**kwargs: Any) -> dict[str, Any]:
        return kwargs

    monkeypatch.setattr("swarmbots.learn.training._make_policy_env", make_environment)
    for algorithm_name in ("SAC", "RecurrentSAC", "TD3", "RecurrentTD3", "DDPG"):
        monkeypatch.setattr(f"swarmbots.learn.training.{algorithm_name}", capture_algorithm)


@pytest.mark.parametrize("variant", OFF_POLICY_VARIANTS)
def test_off_policy_presets_share_common_training_defaults(
    capture_training_configuration: None, variant: str,
) -> None:
    config = make_training("SwarmBots-WallEasy-v0", variant, num_envs=256)
    assert {key: config[key] for key in (
        "learning_rate", "learning_rate_warmup_updates", "learning_rate_warmup_start_factor",
        "buffer_capacity_per_env", "learning_starts", "rollout_steps_per_iteration", "gradient_steps",
        "gamma", "tau", "max_grad_norm",
    )} == {
        "learning_rate": 5e-5, "learning_rate_warmup_updates": 500, "learning_rate_warmup_start_factor": 0.01,
        "buffer_capacity_per_env": 1000, "learning_starts": 10_000, "rollout_steps_per_iteration": 256,
        "gradient_steps": 8, "gamma": 0.99, "tau": 0.005, "max_grad_norm": 2.0,
    }
    learning_positions = config["batch_size"] * config.get("learning_steps", 1)
    assert learning_positions == 1024


@pytest.mark.parametrize("variant", OFF_POLICY_VARIANTS)
def test_off_policy_presets_honor_common_training_overrides(
    capture_training_configuration: None, variant: str,
) -> None:
    overrides = {
        "learning_rate": 2e-4, "learning_rate_warmup_updates": 20, "learning_rate_warmup_start_factor": 0.1,
        "buffer_capacity_per_env": 256, "learning_starts": 512, "rollout_steps_per_iteration": 512,
        "batch_size": 3, "gradient_steps": 2, "gamma": 0.95, "tau": 0.02, "max_grad_norm": 0.7,
    }
    config = make_training("SwarmBots-WallEasy-v0", variant, num_envs=256, algorithm_kwargs=overrides)
    assert {key: config[key] for key in overrides} == overrides


@pytest.mark.parametrize("variant", ["tmasac__lstm", "tmasac__slstm", "tmatd3"])
@pytest.mark.parametrize(("num_envs", "rollout_steps", "learning_steps"), [
    (64, 1, 64), (256, 1, 64), (1024, 1, 64), (4096, 1, 64), (256, 4, 64), (256, 1, 32), (256, 1, 128),
])
def test_recurrent_default_batches_match_feedforward_replay_exposure(
    capture_training_configuration: None, variant: str, num_envs: int, rollout_steps: int, learning_steps: int,
) -> None:
    common = dict(num_envs=num_envs, rollout_steps_per_env=rollout_steps)
    baseline = make_training("SwarmBots-WallEasy-v0", "tmatd3" if variant == "tmatd3" else "tmasac", **common)
    recurrent = make_training(
        "SwarmBots-WallEasy-v0", variant, **common,
        policy_kwargs={"td3_recurrent_actor": True} if variant == "tmatd3" else {},
        algorithm_kwargs={"learning_steps": learning_steps},
    )
    baseline_positions = baseline["batch_size"]
    recurrent_positions = recurrent["batch_size"] * recurrent["learning_steps"]
    assert baseline_positions == recurrent_positions == 1024
    assert baseline["gradient_steps"] == recurrent["gradient_steps"] == 8
    assert baseline["rollout_steps_per_iteration"] == recurrent["rollout_steps_per_iteration"] == num_envs * rollout_steps
    assert baseline["learning_rate"] == recurrent["learning_rate"] == 5e-5
    assert {key: recurrent[key] for key in (
        "burn_in_steps", "temporal_state_store_interval", "temporal_state_storage_dtype",
    )} == {"burn_in_steps": 32, "temporal_state_store_interval": 16, "temporal_state_storage_dtype": torch.float16}


@pytest.mark.parametrize("learning_steps", [3, 63, 100, 2048])
def test_recurrent_default_batches_round_up_to_whole_learning_sequences(
    capture_training_configuration: None, learning_steps: int,
) -> None:
    config = make_training(
        "SwarmBots-WallEasy-v0", "tmasac__lstm", num_envs=256,
        algorithm_kwargs={"learning_steps": learning_steps, "temporal_state_store_interval": 1},
    )
    positions = config["batch_size"] * config["learning_steps"]
    assert config["batch_size"] >= 1
    assert 0 <= positions - 1024 < learning_steps


@pytest.mark.parametrize("variant", ["tmasac__lstm", "tmasac__slstm", "tmatd3"])
def test_recurrent_default_replay_capacity_supports_custom_horizons(
    capture_training_configuration: None, variant: str,
) -> None:
    config = make_training(
        "SwarmBots-WallEasy-v0", variant, num_envs=4096, episode_length=4,
        policy_kwargs={"td3_recurrent_actor": True} if variant == "tmatd3" else {},
        algorithm_kwargs={"learning_steps": 256, "burn_in_steps": 16, "temporal_state_store_interval": 32},
    )
    assert config["buffer_capacity_per_env"] >= 16 + 256 + 32 - 1


@pytest.mark.parametrize("variant", ["tmasac__lstm", "tmasac__slstm", "tmatd3"])
def test_explicit_recurrent_batch_size_remains_a_sequence_count(
    capture_training_configuration: None, variant: str,
) -> None:
    config = make_training(
        "SwarmBots-WallEasy-v0", variant, num_envs=256,
        policy_kwargs={"td3_recurrent_actor": True} if variant == "tmatd3" else {},
        algorithm_kwargs={"batch_size": 7, "learning_steps": 32},
    )
    assert config["batch_size"] == 7
    assert config["learning_steps"] == 32


@pytest.mark.integration
@pytest.mark.parametrize("variant", [
    "maddpg__deepset_critic", "tmatd3", "tmasac", "tmasac__lstm", "tmasac__slstm",
])
def test_off_policy_training_updates_with_default_batch_budget(variant: str) -> None:
    kwargs = _training_kwargs()
    kwargs["episode_length"] = 128
    recurrent = variant in {"tmasac__lstm", "tmasac__slstm"}
    algorithm = make_training(
        "SwarmBots-WallEasy-v0", variant, **kwargs, rollout_steps_per_env=96 if recurrent else 512,
        algorithm_kwargs={
            "learning_starts": 0, "gradient_steps": 1,
            "buffer_capacity_per_env": 128 if recurrent else 512, "learning_rate_warmup_updates": 0,
        },
    )
    try:
        assert algorithm.batch_size == (16 if recurrent else 1024)
        if recurrent:
            assert algorithm.learning_steps == 64
            assert algorithm.burn_in_steps == 32
        assert algorithm.batch_size * getattr(algorithm, "learning_steps", 1) == 1024
        parameters_before = [parameter.detach().clone() for parameter in algorithm.policy.parameters()]
        algorithm.perform_iteration(ExponentialMovingAverage(0.1), ExponentialMovingAverage(0.1), update_ema=True)
        assert algorithm.n_total_updates > 0
        assert any(
            not torch.equal(before, after)
            for before, after in zip(parameters_before, algorithm.policy.parameters(), strict=True)
        )
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", OFF_POLICY_VARIANTS)
def test_off_policy_presets_retain_algorithm_specific_defaults(variant: str) -> None:
    algorithm = make_training("SwarmBots-WallEasy-v0", variant, **_training_kwargs())
    try:
        assert algorithm.learning_rate == 5e-5
        for optimizer in (algorithm.actor_optimizer, algorithm.critic_optimizer):
            assert all(group["lr"] == pytest.approx(5e-7) for group in optimizer.param_groups)
        assert algorithm.learning_rate_warmup_updates == 500
        assert algorithm.learning_rate_warmup_start_factor == 0.01
        assert algorithm.batch_size * getattr(algorithm, "learning_steps", 1) == 1024
        assert algorithm.gradient_steps == 8
        assert algorithm.nop_batch_size == algorithm.batch_size
        assert not algorithm.independent_nop_sampling
        if isinstance(algorithm, SAC):
            assert algorithm.policy_delay == 1
            assert algorithm.target_update_interval == 1
            assert algorithm.target_policy_noise == algorithm.target_noise_clip == 0
            assert algorithm.actor_action_samples == algorithm.target_action_samples == 1
            assert algorithm.ent_coef == "auto_0.05"
            assert algorithm.target_entropy == "auto_0.1"
            assert algorithm.ent_coef_optimizer.param_groups[0]["lr"] == 5e-5
        else:
            ddpg = variant.startswith("maddpg")
            assert algorithm.policy_delay == (1 if ddpg else 2)
            assert algorithm.target_policy_noise == (0.0 if ddpg else 0.2)
            assert algorithm.target_noise_clip == (0.0 if ddpg else 0.5)
            assert algorithm.exploration_noise == 0.1
        if algorithm.policy.has_nop_loss():
            assert algorithm.policy.get_nop_num_next_steps() == 3
            assert algorithm.policy.critic_nop.nop_loss_coef == 0.1
            assert algorithm.policy.actor_nop is None
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", [
    "masac__deepset_critic", "masac__gated_deepset_critic",
    "tmasac", "tmasac_dec", "tmasac__no_nop", "tmasac__lstm", "tmasac__slstm",
])
@pytest.mark.parametrize("override", [False, True])
def test_sac_presets_share_tmasac_training_defaults_and_honor_overrides(variant: str, override: bool) -> None:
    options: dict[str, Any] = {"learning_rate_warmup_updates": 0}
    learning_rate, temperature_rate, entropy, target = 5e-5, 5e-5, "auto_0.05", "auto_0.1"
    if override:
        learning_rate, temperature_rate, entropy, target = 2e-4, 7e-4, "auto_0.2", "auto_0.3"
        options.update(learning_rate=learning_rate, ent_coef_learning_rate=temperature_rate,
                       ent_coef=entropy, target_entropy=target)
    algorithm = make_training("SwarmBots-WallEasy-v0", variant, algorithm_kwargs=options, **_training_kwargs())
    try:
        assert algorithm.learning_rate == pytest.approx(learning_rate)
        for optimizer in (algorithm.actor_optimizer, algorithm.critic_optimizer):
            assert all(group["lr"] == pytest.approx(learning_rate) for group in optimizer.param_groups)
        assert algorithm.ent_coef_optimizer.param_groups[0]["lr"] == pytest.approx(temperature_rate)
        assert algorithm.ent_coef == entropy
        assert algorithm.target_entropy == target
    finally:
        algorithm.env.close()


def test_variants_expose_only_canonical_architectures_and_ablations() -> None:
    assert set(list_variants(include_hidden=True)) == {
        "maddpg__deepset_critic",
        "matd3__deepset_critic",
        "masac__deepset_critic",
        "maddpg__gated_deepset_critic",
        "matd3__gated_deepset_critic",
        "masac__gated_deepset_critic",
        "tmatd3",
        "tmatd3_dec",
        "ppo",
        "mappo__deepset_critic",
        "mat_orig",
        "mat_ind",
        "mat_qcx",
        "mat_dec",
        "mat_ind__lstm",
        "mat_qcx__lstm",
        "tmasac",
        "tmasac_dec",
        "tmasac__slstm",
        "tmasac__lstm",
        "mat_qcx__no_nop",
        "tmasac__no_nop",
    }
    assert set(list_variants()) == {
        "ppo",
        "mappo__deepset_critic",
        "mat_orig",
        "mat_ind",
        "mat_dec",
        "mat_qcx",
        "mat_ind__lstm",
        "mat_qcx__lstm",
        "maddpg__deepset_critic",
        "matd3__deepset_critic",
        "masac__deepset_critic",
        "tmatd3",
        "tmatd3_dec",
        "tmasac",
        "tmasac_dec",
        "tmasac__slstm",
        "tmasac__lstm",
    }
    assert len(list_variants()) == len(set(list_variants()))


@pytest.mark.parametrize("variant", [
    "mat__orig", "mat__ind", "mat__dec", "mat__qcx", "mat__ind__lstm", "mat__qcx__lstm",
    "mat__ind__no_attention", "mat__qcx__no_nop", "tmasac__dec", "tmatd3__dec",
    "mat_ind_lstm", "mat_qcx_lstm", "maddpg_deepset", "matd3_deepset", "masac_deepset",
    "tmasac_lstm", "tmasac_slstm", "mat_ind_no_attention", "mat_ind__no_attention",
    "mat_qcx_no_nop", "tmasac_no_nop", "tmasac_shared_encoder",
    "maddpg__deepset_critic__gated", "matd3__deepset_critic__gated", "masac__deepset_critic__gated",
])
def test_replaced_variant_names_are_rejected(variant: str) -> None:
    with pytest.raises(ValueError, match="Unknown learning variant"):
        _variant_options(variant)


@pytest.mark.parametrize("variant", ["tmasac__slstm", "tmasac__lstm"])
def test_recurrent_presets_use_common_temporal_residual_and_normalization(variant: str) -> None:
    options, _ = _variant_options(variant)
    assert options["rmat_temporal_residual"] is True
    assert options["rmat_temporal_layer_norm"] is True


def _small_policy(
    variant: str,
    n_agents: int = 3,
    *,
    policy_kwargs: dict[str, Any] | None = None,
) -> TMASACPolicy | MAPPOPolicy:
    env = SimpleNamespace(
        n_agents=n_agents,
        local_obs_dim=5,
        global_obs_dim=2,
        hidden_local_vars_dim=0,
        hidden_global_vars_dim=0,
        action_space=HybridActionSpace({
            "actuators": spaces.Box(-1.0, 1.0, shape=(n_agents, 2)),
            "connectors": spaces.Box(-1.0, 1.0, shape=(n_agents, 1)),
        }),
    )
    options, _ = _variant_options(variant, model_scale=None)
    options.update(policy_kwargs or {})
    options["use_nop"] = False
    return _make_base_policy(
        env=env,
        enc_d_model=32,
        enc_nhead=4,
        dec_d_model=16,
        dec_nhead=2,
        use_popart=False,
        popart_beta=5e-4,
        popart_init_sigma=0.65,
        compile_policy_modules=False,
        policy_compile_mode="default",
        gsde_init_stds=[0.25, 0.30],
        mat_add_agent_embeddings=options.pop("mat_add_agent_embeddings", True),
        act_fn_cls=nn.GELU,
        mat_init_gains=MATInitGains(),
        mat_normalization=MATNormalizationConfig(),
        **options,
    )


@pytest.mark.parametrize("algorithm", ["maddpg", "matd3", "masac"])
@pytest.mark.parametrize("head_gating", [False, True])
def test_gated_deepset_presets_gate_encoder_and_toggle_q_heads(algorithm: str, head_gating: bool) -> None:
    baseline = _small_policy(f"{algorithm}__deepset_critic")
    gated = _small_policy(
        f"{algorithm}__gated_deepset_critic",
        policy_kwargs={"critic_deepset_head_gating": True} if head_gating else None,
    )
    baseline_shapes = {name: parameter.shape for name, parameter in baseline.critic.named_parameters()}
    gated_shapes = {name: parameter.shape for name, parameter in gated.critic.named_parameters()}
    gate_names = {name for name in gated_shapes if ".reception_gate." in name}
    assert gate_names
    encoder_prefix = "0.encoder." if algorithm == "maddpg" else "encoder."
    encoder_gate_names = {name for name in gate_names if name.startswith(encoder_prefix)}
    assert encoder_gate_names
    assert bool(gate_names - encoder_gate_names) is head_gating
    assert {name: shape for name, shape in gated_shapes.items() if name not in gate_names} == baseline_shapes

    settings = gated.get_hyper_parameters()
    config_name = "masac_policy_config" if algorithm == "masac" else "td3_policy_config"
    assert settings[config_name]["joint_critic_config"]["interaction_config"]["gated"] is True
    assert settings[config_name]["joint_critic_config"]["head_config"]["interaction_config"]["gated"] is head_gating

    agent_mask = torch.tensor([[True, True, False], [True, True, True]])
    values = gated.q_values(
        local_obs=torch.randn(2, 3, 5), global_obs=torch.randn(2, 2),
        actions=torch.randn(2, 3, 3), agent_mask=agent_mask,
    )
    sum(value.square().mean() for value in values).backward()
    gate_gradients = [
        parameter.grad for name, parameter in gated.critic.named_parameters() if name in gate_names
    ]
    assert all(gradient is not None and torch.isfinite(gradient).all() for gradient in gate_gradients)
    assert any(torch.count_nonzero(gradient) for gradient in gate_gradients)


@pytest.mark.parametrize("variant", [
    "tmasac__ac_shared_observation_encoder", "tmasac__slstm__ac_shared_observation_encoder",
])
def test_low_level_shared_observation_presets_remain_available_but_unregistered(variant: TMASACVariant) -> None:
    options, _ = make_tmasac_options(variant)
    policy = _small_policy("tmasac", policy_kwargs=options)

    assert policy.shared_observation_encoder is not None
    assert policy.shared_observation_encoder_target is not None
    assert variant not in list_variants(include_hidden=True)
    with pytest.raises(ValueError, match="Unknown learning variant"):
        _variant_options(variant)


@pytest.mark.parametrize("variant", ["tmasac_shared_encoder", "tmasac_slstm_shared_encoder"])
def test_low_level_shared_observation_presets_reject_old_names(variant: str) -> None:
    with pytest.raises(ValueError, match="Unknown TMASAC variant"):
        make_tmasac_options(variant)


@pytest.mark.parametrize("variant", ["tmasac__slstm", "tmasac__lstm"])
def test_recurrent_tmasac_presets_split_actor_ff_and_preserve_full_critic_ff(variant: str) -> None:
    options, _ = _variant_options(variant, model_scale=None)
    assert options["mat_encoder_transformer_ff_config"] == MLPConfig(hidden_dims=[512, 512])
    assert options["rmat_actor_transformer_ff_config"] == MLPConfig(hidden_dims=[512])
    assert options["rmat_actor_inter_module_mlp"] is True
    policy = _small_policy(variant)
    for critic in (policy.critic, policy.critic_target):
        for layer in critic.encoder.layers:
            assert isinstance(layer.feedforward, MLP)
            widths = [(module.in_features, module.out_features) for module in layer.feedforward if isinstance(module, nn.Linear)]
            assert widths == [(32, 512), (512, 512), (512, 32)]
    for layer in policy.actor_encoder.layers:
        for feedforward in (layer.inter_module_feedforward, layer.feedforward):
            assert isinstance(feedforward, MLP)
            widths = [(module.in_features, module.out_features) for module in feedforward if isinstance(module, nn.Linear)]
            assert widths == [(32, 512), (512, 32)]


@pytest.mark.parametrize("variant", ["tmasac__slstm", "tmasac__lstm"])
def test_recurrent_tmasac_executes_and_trains_both_ff_stages_around_memory(variant: str) -> None:
    policy = _small_policy(variant, policy_kwargs={"rmat_experimental_compile_lstm": False})
    layer = policy.actor_encoder.layers[0]
    execution_order = []
    handles = [
        module.register_forward_hook(lambda _module, _inputs, _output, name=name: execution_order.append(name))
        for name, module in (
            ("attention", layer.self_attn), ("pre_mlp", layer.inter_module_feedforward),
            ("memory", layer.temporal_model), ("post_mlp", layer.feedforward),
        )
    ]
    try:
        output, _ = policy.actor_encoder(
            local_obs=torch.randn(1, 2, 3, 5), global_obs=torch.randn(1, 2, 2),
            agent_mask=torch.ones(1, 2, 3, dtype=torch.bool),
        )
    finally:
        for handle in handles:
            handle.remove()
    assert execution_order == ["attention", "pre_mlp", "memory", "post_mlp"]
    output[..., 0].square().mean().backward()
    for encoder_layer in policy.actor_encoder.layers:
        for feedforward in (encoder_layer.inter_module_feedforward, encoder_layer.feedforward):
            assert any(parameter.grad is not None and torch.count_nonzero(parameter.grad) for parameter in feedforward.parameters())


@pytest.mark.parametrize("variant,recurrent_critic", [("matd3__deepset_critic", False), ("tmatd3", False), ("tmatd3_dec", False), ("tmatd3", True), ("tmatd3_dec", True)])
def test_td3_split_actor_ff_override_preserves_full_critic_ff(variant: str, recurrent_critic: bool) -> None:
    baseline = _small_policy(variant)
    policy = _small_policy(variant, policy_kwargs={
        "td3_recurrent_actor": True,
        "td3_recurrent_critic": recurrent_critic,
        "rmat_actor_transformer_ff_config": MLPConfig(hidden_dims=[512]),
        "rmat_actor_inter_module_mlp": True,
        # Isolate FF routing from the separate actor-state conditioning input.
        "td3_actor_state_critic_input_config": None,
    })
    for actor in (policy.actor, policy.actor_target):
        for layer in actor.encoder.layers:
            for block in (layer.inter_module_feedforward, layer.feedforward):
                assert isinstance(block, MLP)
                widths = [(module.in_features, module.out_features) for module in block if isinstance(module, nn.Linear)]
                assert widths == [(32, 512), (512, 32)]
    if variant.startswith("tmatd3"):
        for critic in (policy.critic, policy.critic_target):
            for layer in critic.encoder.layers:
                if recurrent_critic:
                    assert layer.inter_module_feedforward is None
                assert isinstance(layer.feedforward, MLP)
                widths = [(module.in_features, module.out_features) for module in layer.feedforward if isinstance(module, nn.Linear)]
                assert widths == [(32, 512), (512, 512), (512, 32)]
    else:
        expected_shapes = {name: parameter.shape for name, parameter in baseline.critic.named_parameters()}
        for critic in (policy.critic, policy.critic_target):
            assert {name: parameter.shape for name, parameter in critic.named_parameters()} == expected_shapes


@pytest.mark.parametrize("variant", ["mat_ind", "mat_qcx"])
def test_recurrent_mat_preserves_base_feedforward_networks_and_critic_heads(variant: str) -> None:
    settings = {"mat_add_agent_embeddings": False}
    baseline = _small_policy(variant, policy_kwargs=settings)
    recurrent = _small_policy(variant + "__lstm", policy_kwargs=settings)
    expected_shapes = {name: parameter.shape for name, parameter in baseline.critic.named_parameters()}
    assert {name: parameter.shape for name, parameter in recurrent.critic.named_parameters()} == expected_shapes
    for baseline_layer, recurrent_layer in zip(baseline.actor_encoder.layers, recurrent.actor_encoder.layers, strict=True):
        expected_ff_shapes = {name: parameter.shape for name, parameter in baseline_layer.feedforward.named_parameters()}
        assert recurrent_layer.inter_module_feedforward is None
        assert {name: parameter.shape for name, parameter in recurrent_layer.feedforward.named_parameters()} == expected_ff_shapes


@pytest.mark.parametrize("variant", ["tmasac", "tmasac_dec"])
def test_sac_presets_produce_actions_for_swarms_larger_than_twenty_agents(variant: str) -> None:
    n_agents = 21
    policy = _small_policy(variant, n_agents)
    with torch.no_grad():
        actions = policy.act(
            local_obs=torch.zeros(1, n_agents, policy.local_obs_dim),
            global_obs=torch.zeros(1, policy.global_obs_dim),
            agent_mask=torch.ones(1, n_agents, dtype=torch.bool),
            deterministic=True,
        )
    assert actions.shape == (1, n_agents, 3)
    assert torch.isfinite(actions).all()
    assert (actions >= -1).all() and (actions <= 1).all()


def test_tmasac_dec_actor_is_independent_and_critic_mixes_agents() -> None:
    torch.manual_seed(0)
    policy = _small_policy("tmasac_dec")
    baseline = _small_policy("tmasac")
    assert not any(isinstance(module, nn.MultiheadAttention) for module in policy.actor_encoder.modules())
    assert any(isinstance(module, nn.MultiheadAttention) for module in policy.critic.modules())
    assert policy.critic_encoder_config == baseline.critic_encoder_config
    assert policy.config.critic_config == baseline.config.critic_config
    assert _variant_options("tmasac_dec")[1] == _variant_options("tmasac")[1]

    local_obs = torch.randn(2, 3, policy.local_obs_dim)
    global_obs = torch.randn(2, policy.global_obs_dim)
    agent_mask = torch.tensor([[True, True, False], [True, True, True]])
    changed_local_obs = local_obs.clone()
    changed_local_obs[:, 1] += torch.tensor([10.0, -20.0, 30.0, -40.0, 50.0])
    with torch.no_grad():
        actions = policy.act(local_obs, global_obs, agent_mask=agent_mask, deterministic=True)
        changed_actions = policy.act(changed_local_obs, global_obs, agent_mask=agent_mask, deterministic=True)
        torch.testing.assert_close(actions[:, 0], changed_actions[:, 0], rtol=0, atol=0)
        assert not torch.allclose(actions[:, 1], changed_actions[:, 1])
        assert (actions[~agent_mask] == 0).all()

        critic_inputs = dict(
            global_inputs=global_obs, actions=actions, hidden_local_vars=None,
            hidden_global_vars=None, agent_mask=agent_mask,
        )
        latents = policy.critic.encode(local_inputs=local_obs, **critic_inputs)
        changed_latents = policy.critic.encode(local_inputs=changed_local_obs, **critic_inputs)
        assert not torch.allclose(latents[:, 0], changed_latents[:, 0])

    sampled_actions, log_probs = policy.action_log_prob(
        local_obs=local_obs, global_obs=global_obs, agent_mask=agent_mask,
    )
    q1, q2 = policy.q_values(
        local_obs=local_obs, global_obs=global_obs, actions=sampled_actions, agent_mask=agent_mask,
    )
    (log_probs.mean() - torch.minimum(q1, q2).mean()).backward()
    for parameters in (policy.actor_parameters(), policy.critic_parameters()):
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        assert gradients
        assert all(torch.isfinite(gradient).all() for gradient in gradients)
        assert any(torch.count_nonzero(gradient) for gradient in gradients)


@pytest.mark.integration
@pytest.mark.parametrize(
    ("variant", "expected_ratios"),
    [("ppo", (1.04**2, 1.04)), ("mappo__deepset_critic", (1.04, 1.04, 1.04)), ("mat_ind", (1.04, 1.04, 1.04))],
)
def test_ppo_presets_clip_and_measure_kl_at_the_intended_action_scope(
    variant: str, expected_ratios: tuple[float, ...]
) -> None:
    algorithm = make_training(
        "SwarmBots-WallEasy-v0",
        variant,
        algorithm_kwargs={"use_popart": False, "mc_ent_coef": 0.0},
        **_training_kwargs(),
    )
    try:
        batch = PPOSamples(
            local_obs=torch.empty(2, 2, 0),
            global_obs=torch.empty(2, 0),
            hidden_local_vars=torch.empty(2, 2, 0),
            hidden_global_vars=torch.empty(2, 0),
            agent_mask=torch.tensor([[True, True], [True, False]]),
            previous_actions=None,
            actions=torch.zeros(2, 2, 1),
            log_probs=torch.zeros(2, 2),
            values=torch.zeros(2),
            returns=torch.zeros(2),
            advantages=torch.ones(2),
        )
        ratios = torch.tensor([[1.04, 1.04], [1.04, 10.0]])
        loss, approximate_kl, metrics = algorithm.compute_ppo_loss(
            batch, ratios.log(), torch.zeros(2), normalize_advantage=False
        )
        expected = torch.tensor(expected_ratios)
        torch.testing.assert_close(loss, -expected.clamp(max=1.05).mean())
        assert approximate_kl == pytest.approx((expected - 1 - expected.log()).mean().item(), abs=1e-7)
        assert metrics["clip_frac"] == pytest.approx((expected > 1.05).float().mean().item())
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("coefficient", [None, 0.0, 0.02])
def test_ppo_preset_entropy_default_and_override(coefficient: float | None) -> None:
    algorithm_kwargs = {} if coefficient is None else {"mc_ent_coef": coefficient}
    algorithm = make_training(
        "SwarmBots-WallEasy-v0", "mat_ind", algorithm_kwargs=algorithm_kwargs, **_training_kwargs(),
    )
    try:
        assert algorithm.mc_ent_coef == pytest.approx(1e-3 if coefficient is None else coefficient)
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["mat_ind", "tmasac"])
def test_training_persists_numeric_distribution_entropy_diagnostics(variant: str, tmp_path: Path) -> None:
    algorithm_kwargs = (
        {"learning_starts": 0, "batch_size": 2, "gradient_steps": 1, "buffer_capacity_per_env": 16}
        if variant == "tmasac"
        else {"n_epochs": 1, "target_kl": None, "learning_rate": 1e-3}
    )
    train(
        "SwarmBots-WallEasy-v0", variant, total_timesteps=8, rollout_steps_per_env=4,
        run_dir=tmp_path, use_nop=False, algorithm_kwargs=algorithm_kwargs,
        learn_kwargs={"logging_console_keys": []}, **_training_kwargs(),
    )
    with (tmp_path / "log.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    assert rows
    prefix = "rollout_act" if variant == "tmasac" else "act"
    entropy_columns = [name for name in rows[-1] if name.startswith(prefix) and "_ent" in name]
    assert entropy_columns
    for row in rows:
        assert all(math.isfinite(float(row[name])) for name in entropy_columns)


@pytest.mark.integration
@pytest.mark.parametrize("variant", list_variants())
def test_each_variant_produces_finite_benchmark_actions(variant: str) -> None:
    algorithm = make_training("SwarmBots-WallEasy-v0", variant, **_training_kwargs())
    try:
        expected_distribution = (
            PredictedStdGaussianActionDist if isinstance(algorithm, SAC) else SignMagnitudeBetaActionDist
        )
        distributions = () if isinstance(algorithm, TD3) else algorithm.policy.action_dist.distributions
        for distribution in distributions:
            assert isinstance(distribution, expected_distribution)
            if isinstance(distribution, PredictedStdGaussianActionDist):
                assert distribution.squash_output
        observations, _ = algorithm.env.unwrapped.reset(seed=17)
        policy = as_benchmark_policy(algorithm)
        actions = policy(observations, torch.ones(2, dtype=torch.bool))
        assert actions.keys() == {"actuators", "connectors"}
        for key, space in algorithm.env.unwrapped.action_space.items():
            assert actions[key].shape == space.shape
            assert torch.isfinite(actions[key]).all()
            assert (actions[key] >= -1).all() and (actions[key] <= 1).all()
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["tmasac", "tmasac_dec", "tmasac__slstm"])
def test_sac_action_distribution_can_be_overridden_without_changing_the_architecture(variant: str) -> None:
    algorithm = make_training(
        "SwarmBots-WallEasy-v0", variant,
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        **_training_kwargs(),
    )
    try:
        assert all(
            isinstance(distribution, GumbelSoftmaxSignMagnitudeBetaActionDist)
            for distribution in algorithm.policy.action_dist.distributions
        )
        observations, _ = algorithm.env.unwrapped.reset(seed=17)
        actions = as_benchmark_policy(algorithm)(observations, torch.ones(2, dtype=torch.bool))
        assert all(torch.isfinite(action).all() for action in actions.values())
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
def test_nop_can_be_constructed_for_every_registered_scenario(benchmark_id: str) -> None:
    algorithm = make_training(benchmark_id, "mat_qcx", **_training_kwargs())
    try:
        observations, _ = algorithm.env.reset(seed=17)
        with torch.no_grad():
            actions = algorithm.policy.act(
                local_obs=observations["local_obs"],
                global_obs=observations["global_obs"],
                agent_mask=observations["agent_mask"],
            )
        assert torch.isfinite(actions).all()
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", list_variants())
def test_learning_updates_parameters_on_actual_benchmark_transitions(variant: str) -> None:
    algorithm_options: dict[str, Any]
    policy_options, _ = _variant_options(variant)
    if _is_off_policy_variant(policy_options["policy_variant"]):
        algorithm_options = {
            "learning_starts": 0,
            "batch_size": 2,
            "buffer_capacity_per_env": 32,
            "gradient_steps": 1,
            "learning_rate": 1e-3,
            "learning_rate_warmup_updates": 0,
        }
        if policy_options["policy_variant"] in {"tmasac__recurrent"}:
            algorithm_options.update(burn_in_steps=1, learning_steps=3, temporal_state_store_interval=1)
    else:
        algorithm_options = {"learning_rate": 1e-3, "n_epochs": 1, "target_kl": None}
    algorithm = make_training(
        "SwarmBots-WallEasy-v0",
        variant,
        rollout_steps_per_env=4,
        algorithm_kwargs=algorithm_options,
        **_training_kwargs(),
    )
    try:
        parameters_before = [parameter.detach().clone() for parameter in algorithm.policy.parameters()]
        metrics, transitions = algorithm.perform_iteration(
            ExponentialMovingAverage(0.1),
            ExponentialMovingAverage(0.1),
            update_ema=True,
        )
        assert transitions == 8
        assert metrics
        assert algorithm.n_total_updates > 0
        assert any(
            not torch.equal(before, after)
            for before, after in zip(parameters_before, algorithm.policy.parameters(), strict=True)
        )
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["matd3__deepset_critic", "tmatd3_dec"])
def test_new_off_policy_train_helper_saves_resumable_targets(variant: str, tmp_path: Path) -> None:
    kwargs = _training_kwargs()
    kwargs["policy_kwargs"].update(
        baseline_critic_hidden_dims=(32, 32), baseline_critic_element_hidden_dims=(32, 32),
    )
    kwargs["algorithm_kwargs"] = {
        "learning_starts": 0, "batch_size": 2, "buffer_capacity_per_env": 16,
        "gradient_steps": 2, "learning_rate_warmup_updates": 0,
    }
    algorithm = train(
        "SwarmBots-WallEasy-v0", variant, total_timesteps=4, run_dir=tmp_path / "first", **kwargs,
    )
    checkpoint = next((tmp_path / "first/models").glob("*_final.pt"))
    restored = make_training("SwarmBots-WallEasy-v0", variant, **kwargs)
    try:
        restored.load(checkpoint)
        assert restored.n_total_updates == algorithm.n_total_updates == 4
        assert restored.actor_optimizer.state_dict()["state"]
        assert restored.critic_optimizer.state_dict()["state"]
        for name, parameter in restored.policy.state_dict().items():
            torch.testing.assert_close(parameter, algorithm.policy.state_dict()[name], rtol=0, atol=0)
        observations, _ = restored.env.unwrapped.reset(seed=17)
        actions = as_benchmark_policy(restored)(observations, torch.ones(2, dtype=torch.bool))
        assert all(torch.isfinite(action).all() for action in actions.values())
    finally:
        restored.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["maddpg__deepset_critic", "matd3__deepset_critic", "masac__deepset_critic", "tmatd3", "tmatd3_dec"])
@pytest.mark.parametrize("source", ["critic", "both"])
def test_off_policy_nop_is_enabled_by_default_and_can_train(variant: str, source: str) -> None:
    kwargs = _training_kwargs()
    kwargs["policy_kwargs"].update(
        baseline_critic_hidden_dims=(32, 32), baseline_critic_element_hidden_dims=(32, 32),
        transition_model_d_model=16, transition_model_nhead=2,
    )
    if source == "both":
        kwargs["policy_kwargs"]["tmasac_nop_latent_source"] = source
    kwargs["algorithm_kwargs"] = {
        "learning_starts": 0, "batch_size": 2, "buffer_capacity_per_env": 16,
        "gradient_steps": 2, "learning_rate_warmup_updates": 0,
    }
    algorithm = make_training("SwarmBots-WallEasy-v0", variant, **kwargs)
    try:
        assert algorithm.policy.has_nop_loss()
        assert algorithm.policy.critic_nop is not None
        assert (algorithm.policy.actor_nop is not None) == (source == "both")
        metrics, transitions = algorithm.perform_iteration(
            ExponentialMovingAverage(0.1), ExponentialMovingAverage(0.1), update_ema=True,
        )
        assert transitions == 2 and algorithm.n_total_updates == 2
        assert "critic_nop_loss" in metrics
        assert ("actor_nop_loss" in metrics) == (source == "both")
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["mappo__deepset_critic"])
@pytest.mark.parametrize("model_scale", [None, "5M-2L"])
def test_mappo_deepset_critic_trains_and_restores_matching_checkpoints(
    variant: str, model_scale: str | None, tmp_path: Path,
) -> None:
    kwargs = _training_kwargs()
    kwargs["model_scale"] = model_scale
    kwargs["policy_kwargs"].update(transition_model_d_model=16, transition_model_nhead=2)
    kwargs["algorithm_kwargs"] = {"learning_rate": 1e-3, "n_epochs": 1, "target_kl": None}
    algorithm = make_training("SwarmBots-WallEasy-v0", variant, **kwargs)
    try:
        policy = algorithm.policy.policy
        assert isinstance(policy, MAPPOPolicy) and policy.hidden_global_vars_dim > 0
        assert policy.critic_encoder.global_obs_dim == policy.global_obs_dim + policy.hidden_global_vars_dim
        assert policy.critic.num_global_features == 0
        before = {name: parameter.detach().clone() for name, parameter in policy.critic.named_parameters()}
        metrics, transitions = algorithm.perform_iteration(
            ExponentialMovingAverage(0.1), ExponentialMovingAverage(0.1), update_ema=True,
        )
        assert transitions == 8 and algorithm.n_total_updates > 0 and "wm_loss" in metrics
        assert any(not torch.equal(parameter, before[name]) for name, parameter in policy.critic.named_parameters())
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
        path = tmp_path / "mappo.pt"
        algorithm.save(path, optimizer_state_dict=algorithm._get_optimizer_state_dict())
        checkpoint = torch.load(path, weights_only=False)
        saved_config = checkpoint["policy_hyper_parameters"]["mappo_policy_config"]
        assert not saved_config["critic_encoder_config"]["use_agent_attention"]
        assert "critic_interaction_config" in saved_config
        restored = make_training("SwarmBots-WallEasy-v0", variant, **kwargs)
        try:
            restored.load(path)
            assert restored.n_total_updates == algorithm.n_total_updates
            assert restored.optimizer.state_dict()["state"]
            torch.testing.assert_close(restored.policy.state_dict(), algorithm.policy.state_dict(), rtol=0, atol=0)
        finally:
            restored.env.close()
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", [
    "mat_ind",
    "mat_qcx",
    "mat_dec",
    "mat_ind__lstm",
    "mat_qcx__lstm",
    "tmasac",
    "tmasac_dec",
    "tmasac__slstm",
    "tmasac__lstm",
    "tmatd3",
    "tmatd3_dec",
    "maddpg__deepset_critic",
    "matd3__deepset_critic",
    "masac__deepset_critic",
    "maddpg__gated_deepset_critic",
    "matd3__gated_deepset_critic",
    "masac__gated_deepset_critic",
])
def test_joint_observation_embedding_trains_and_restores_existing_presets(variant: str, tmp_path: Path) -> None:
    kwargs = _training_kwargs()
    kwargs["policy_kwargs"].update(
        mat_joint_obs_embedding=True, rmat_actor_d_model=32, rmat_experimental_compile_lstm=False,
        baseline_critic_hidden_dims=(32, 32), baseline_critic_element_hidden_dims=(32, 32),
        transition_model_d_model=16, transition_model_nhead=2,
    )
    policy_options, _ = _variant_options(variant)
    if _is_off_policy_variant(policy_options["policy_variant"]):
        kwargs["algorithm_kwargs"] = {
            "learning_starts": 0, "batch_size": 2, "buffer_capacity_per_env": 16,
            "gradient_steps": 1, "learning_rate_warmup_updates": 0,
            "sac_compile_tensor_operations": False, "sac_compile_optimizer_steps": False,
        }
        if policy_options["policy_variant"] in {"tmasac__recurrent"}:
            kwargs["algorithm_kwargs"].update(burn_in_steps=1, learning_steps=3, temporal_state_store_interval=1)
    else:
        kwargs["algorithm_kwargs"] = {"learning_rate": 1e-3, "n_epochs": 1, "target_kl": None}
    algorithm = make_training("SwarmBots-PayloadPlane-v0", variant, rollout_steps_per_env=4, **kwargs)
    try:
        assert algorithm.env.global_obs_dim > 0
        assert any(getattr(module, "joint_obs_embedding", False) for module in algorithm.policy.modules())

        def includes_joint_setting(value):
            if isinstance(value, dict):
                return value.get("joint_obs_embedding") is True or any(includes_joint_setting(v) for v in value.values())
            if isinstance(value, (list, tuple)):
                return any(includes_joint_setting(v) for v in value)
            return False

        assert includes_joint_setting(algorithm.policy.get_hyper_parameters())
        before = [parameter.detach().clone() for parameter in algorithm.policy.parameters()]
        metrics, transitions = algorithm.perform_iteration(
            ExponentialMovingAverage(0.1), ExponentialMovingAverage(0.1), update_ema=True,
        )
        assert metrics and transitions == 8 and algorithm.n_total_updates > 0
        assert any(
            not torch.equal(parameter, old)
            for parameter, old in zip(algorithm.policy.parameters(), before, strict=True)
        )
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
        path = tmp_path / "joint.pt"
        algorithm.save(path, optimizer_state_dict=algorithm._get_optimizer_state_dict())
        checkpoint = torch.load(path, weights_only=False)
        assert includes_joint_setting(checkpoint["policy_hyper_parameters"])
        restored = make_training("SwarmBots-PayloadPlane-v0", variant, rollout_steps_per_env=4, **kwargs)
        try:
            restored.load(path)
            assert restored.n_total_updates == algorithm.n_total_updates
            torch.testing.assert_close(restored.policy.state_dict(), algorithm.policy.state_dict(), rtol=0, atol=0)
            observations, _ = restored.env.unwrapped.reset(seed=17)
            actions = as_benchmark_policy(restored)(observations, torch.ones(2, dtype=torch.bool))
            assert all(torch.isfinite(action).all() for action in actions.values())
        finally:
            restored.env.close()
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["mappo__deepset_critic", "matd3__deepset_critic", "tmasac__slstm"])
def test_training_levels_apply_independently_to_each_algorithm_family(variant: str, tmp_path: Path) -> None:
    kwargs = _training_kwargs()
    kwargs["use_nop"] = False
    policy_options, _ = _variant_options(variant)
    off_policy = _is_off_policy_variant(policy_options["policy_variant"])
    if off_policy:
        kwargs["algorithm_kwargs"] = {
            "learning_starts": 0,
            "batch_size": 2,
            "buffer_capacity_per_env": 16,
            "gradient_steps": 1,
            "learning_rate_warmup_updates": 0,
        }
        if policy_options["policy_variant"] == "tmasac__recurrent":
            kwargs["algorithm_kwargs"].update(burn_in_steps=1, learning_steps=3, temporal_state_store_interval=1)
    else:
        kwargs["algorithm_kwargs"] = {"n_epochs": 1, "target_kl": None, "learning_rate": 1e-3}
    console_messages: list[str] = []
    sink_id = logger.add(
        lambda message: console_messages.append(message.record["message"]),
        filter=lambda record: record["function"] == "_log_to_console",
    )
    try:
        train(
            "SwarmBots-WallEasy-v0",
            variant,
            total_timesteps=8,
            rollout_steps_per_env=4,
            run_dir=tmp_path,
            learn_kwargs={"logging_console_level": "full", "logging_persistence_level": "return_success"},
            **kwargs,
        )
    finally:
        logger.remove(sink_id)
    loss_key = "critic_loss" if off_policy else "val_loss_scaled"
    assert any(loss_key in message for message in console_messages)
    with (tmp_path / "log.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle, delimiter=";"))
    assert rows and int(rows[-1]["timesteps"]) == 8
    assert {"timestamp", "iteration", "total_updates", "ep_rew__mean", "ep_rew_ema", "ep_success_rate_ema"} <= rows[-1].keys()
    assert f"{loss_key}__mean" not in rows[-1]
    assert "learning_rate" not in rows[-1]


@pytest.mark.integration
def test_training_saves_and_resumes_then_evaluates_without_updating_normalization(tmp_path: Path) -> None:
    kwargs = _training_kwargs()
    algorithm = train(
        "SwarmBots-WallEasy-v0",
        "mat_ind",
        total_timesteps=8,
        run_dir=tmp_path / "first",
        learn_kwargs={"extra_run_metadata": {"experiment_label": "metadata-forwarding"}},
        algorithm_kwargs={"n_epochs": 1},
        **kwargs,
    )
    checkpoint = next((tmp_path / "first/models").glob("*_final.pt"))
    metadata_path = next((tmp_path / "first").glob("run_metadata*.json"))
    metadata = json.loads(metadata_path.read_text())
    assert metadata["experiment_label"] == "metadata-forwarding"
    assert metadata["benchmark_id"] == "SwarmBots-WallEasy-v0"
    assert metadata["learning_variant"] == "mat_ind"
    resumed = train(
        "SwarmBots-WallEasy-v0",
        "mat_ind",
        total_timesteps=16,
        run_dir=tmp_path / "resumed",
        load_path=checkpoint,
        algorithm_kwargs={"n_epochs": 1},
        **kwargs,
    )
    assert algorithm.n_total_timesteps == 8
    assert resumed.n_total_timesteps == 16
    before = capture_env_state(resumed.env)
    actor = as_benchmark_policy(resumed)
    result = evaluate_policy(
        actor,
        "SwarmBots-WallEasy-v0",
        num_envs=2,
        num_episodes=2,
        device="cpu",
        episode_length=2,
        scenario_kwargs=kwargs["scenario_kwargs"],
        env_kwargs=kwargs["env_kwargs"],
        action_mode="deterministic",
    )
    assert len(result.episode_returns) == 2
    after = capture_env_state(resumed.env)
    for old_wrapper, new_wrapper in zip(before, after, strict=True):
        for key in old_wrapper:
            if key.endswith("_rms"):
                for field in ("mean", "var", "count"):
                    torch.testing.assert_close(getattr(old_wrapper[key], field), getattr(new_wrapper[key], field))


@pytest.mark.integration
def test_presets_select_the_intended_algorithm_and_actor() -> None:
    for variant, expected_class in (("mappo__deepset_critic", PPO), ("mat_ind", PPO), ("tmasac", SAC), ("tmasac_dec", SAC), ("tmasac__slstm", RecurrentSAC), ("tmasac__lstm", RecurrentSAC)):
        algorithm = make_training("SwarmBots-WallEasy-v0", variant, **_training_kwargs())
        try:
            assert isinstance(algorithm, expected_class)
            if variant in {"mappo__deepset_critic"}:
                assert isinstance(algorithm.policy.policy, MAPPOPolicy)
            elif variant == "mat_ind":
                assert isinstance(algorithm.policy.policy, MATIndPolicy)
                assert algorithm.policy.policy.config.encoder_config.use_agent_attention is True
            elif variant in {"tmasac__slstm", "tmasac__lstm"}:
                assert algorithm.policy.actor_encoder.config.temporal_residual is True
                assert algorithm.policy.actor_encoder.config.temporal_layer_norm is True
        finally:
            algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("phase", ["warmup_stopped", "growth_pending", "disabled"])
def test_ppo_checkpoint_continues_the_same_learning_rate_schedule(tmp_path: Path, phase: str) -> None:
    original = make_training("SwarmBots-WallEasy-v0", "mat_ind", **_training_kwargs())
    try:
        original.n_total_iterations = 100
        original._maybe_update_automatic_lr(0.004, 0, {})
        if phase == "growth_pending":
            original._maybe_update_automatic_lr(None, None, {})
        elif phase == "disabled":
            assert original.execute_command("disable_auto_lr", "", None)
        checkpoint = tmp_path / "model.pt"
        original.save(checkpoint, optimizer_state_dict=original._get_optimizer_state_dict())

        resumed = make_training("SwarmBots-WallEasy-v0", "mat_ind", **_training_kwargs())
        try:
            resumed.load(checkpoint)
            assert resumed.learning_rate == pytest.approx(original.learning_rate)
            for _ in range(3):
                original.n_total_iterations += 1
                resumed.n_total_iterations += 1
                expected = original._maybe_update_automatic_lr(None, None, {})
                actual = resumed._maybe_update_automatic_lr(None, None, {})
                assert actual == expected
                assert [group["lr"] for group in resumed.optimizer.param_groups] == pytest.approx(
                    [group["lr"] for group in original.optimizer.param_groups]
                )
        finally:
            resumed.env.close()
    finally:
        original.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("target_kl", [0.0002, 0.002, 0.02])
def test_ppo_learning_rate_controller_uses_the_selected_kl_target(target_kl: float) -> None:
    algorithm = make_training(
        "SwarmBots-WallEasy-v0",
        "mat_ind",
        algorithm_kwargs={"target_kl": target_kl},
        **_training_kwargs(),
    )
    try:
        algorithm.n_total_iterations = 300
        initial_lr = algorithm.learning_rate
        algorithm._maybe_update_automatic_lr(target_kl * 1.51, 7, {})
        assert algorithm.learning_rate == pytest.approx(initial_lr)
        algorithm._maybe_update_automatic_lr(target_kl * 1.6, 7, {})
        assert algorithm.learning_rate < initial_lr
    finally:
        algorithm.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["mat_ind__lstm", "tmasac__slstm", "tmasac__lstm"])
def test_benchmark_adapter_resets_memory_for_the_selected_lanes(variant: str) -> None:
    algorithm = make_training("SwarmBots-WallEasy-v0", variant, **_training_kwargs())
    try:
        observations, _ = algorithm.env.unwrapped.reset(seed=17)
        adapter = as_benchmark_policy(algorithm)
        first = adapter(observations, torch.ones(2, dtype=torch.bool))
        adapter(observations, torch.zeros(2, dtype=torch.bool))
        after_reset = adapter(observations, torch.tensor([True, False]))
        for key in first:
            torch.testing.assert_close(after_reset[key][0], first[key][0])
    finally:
        algorithm.env.close()


@pytest.mark.integration
def test_stochastic_gsde_evaluation_preserves_training_noise_state() -> None:
    algorithm = make_training(
        "SwarmBots-WallEasy-v0", "mat_qcx", continuous_action_dist="gsde", **_training_kwargs(),
    )
    try:
        observations, _ = algorithm.env.unwrapped.reset(seed=17)
        action_dist = algorithm.policy.action_dist
        action_dist.reset_temporal_correlations_on_step(batch_shape=(2, 2))
        before = action_dist.get_temporal_correlation_state()
        adapter = as_benchmark_policy(algorithm, deterministic=False)
        for episode_starts in (torch.ones(2, dtype=torch.bool), torch.zeros(2, dtype=torch.bool)):
            actions = adapter(observations, episode_starts)
            assert all(torch.isfinite(values).all() for values in actions.values())
            torch.testing.assert_close(action_dist.get_temporal_correlation_state(), before)
    finally:
        algorithm.env.close()


@pytest.mark.integration
def test_training_accepts_independent_nop_and_value_normalization_options() -> None:
    kwargs = _training_kwargs()
    kwargs["policy_kwargs"].update(world_model_loss_coef=0.0, transition_model_d_model=16)
    algorithm = make_training(
        "SwarmBots-WallEasy-v0",
        "mat_qcx",
        algorithm_kwargs={"use_popart": False, "n_epochs": 1},
        **kwargs,
    )
    try:
        assert algorithm.policy.world_model_loss_coef == 0.0
        assert algorithm.use_popart is False
        algorithm.learn(max_total_timesteps=8, enable_command_prompt=False)
        assert algorithm.n_total_updates > 0
        assert all(torch.isfinite(parameter).all() for parameter in algorithm.policy.parameters())
    finally:
        algorithm.env.close()


@pytest.mark.integration
def test_benchmark_evaluation_preserves_mixed_module_training_modes() -> None:
    algorithm = make_training("SwarmBots-WallEasy-v0", "mat_ind", **_training_kwargs())
    try:
        algorithm.policy.train()
        algorithm.policy.action_dist.eval()
        modes_before = {module: module.training for module in algorithm.policy.modules()}
        observations, _ = algorithm.env.unwrapped.reset(seed=17)
        as_benchmark_policy(algorithm)(observations, torch.ones(2, dtype=torch.bool))
        assert {module: module.training for module in algorithm.policy.modules()} == modes_before
    finally:
        algorithm.env.close()


def test_conflicting_value_normalization_options_fail_before_creating_an_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unexpected_environment(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Should not create an environment for inconsistent PopArt options")

    monkeypatch.setattr("swarmbots.learn.training.make_env", unexpected_environment)
    with pytest.raises(ValueError, match="use_popart must agree"):
        make_training(
            "SwarmBots-WallEasy-v0",
            "mat_ind",
            policy_kwargs={"use_popart": True},
            algorithm_kwargs={"use_popart": False},
        )


def test_unknown_variant_fails_before_constructing_an_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected_environment(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("Should not create an environment for an unknown variant")

    monkeypatch.setattr("swarmbots.learn.training.make_env", unexpected_environment)
    with pytest.raises(ValueError, match="Unknown learning variant"):
        make_training("SwarmBots-WallEasy-v0", "typo")
