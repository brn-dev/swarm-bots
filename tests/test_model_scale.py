from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn import list_variants, list_model_scales
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.parameter_counts import count_policy_parameters
from swarmbots.learn.presets.policy_factory import _is_off_policy_variant, _make_base_policy
from swarmbots.learn.presets.transformer import MATInitGains, MATNormalizationConfig
from swarmbots.learn.presets.model_scale import DEFAULT_MODEL_SCALE, SCALE_LAYOUTS, normalize_model_scale
from swarmbots.learn.training import _add_ppo_nop, _variant_options
from swarmbots.learn.serialization_utils import serialize_dataclass


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def make_policy(variant, scale="5M NOP1M", *, n_agents=3, **overrides):
    env = SimpleNamespace(
        n_agents=n_agents,
        local_obs_dim=5,
        global_obs_dim=2,
        hidden_local_vars_dim=2,
        hidden_global_vars_dim=1,
        action_space=HybridActionSpace(
            {
                "actuators": spaces.Box(-1.0, 1.0, shape=(n_agents, 2)),
                "connectors": spaces.Box(-1.0, 1.0, shape=(n_agents, 1)),
            }
        ),
    )
    indices = ObsIndices(**{field.name: [] for field in fields(ObsIndices)})
    indices = ObsIndices(
        **{
            **{field.name: getattr(indices, field.name) for field in fields(indices)},
            "local_scalar_indices": [0, 1, 2, 3],
            "local_binary_indices": [4],
        }
    )
    options, _ = _variant_options(variant, model_scale=scale)
    options.update(
        enc_d_model=256,
        enc_nhead=4,
        dec_d_model=128,
        dec_nhead=2,
        use_popart=False,
        popart_beta=5e-4,
        popart_init_sigma=0.65,
        compile_policy_modules=False,
        compile_world_model_modules=False,
        policy_compile_mode="default",
        gsde_init_stds=[0.25, 0.30],
        mat_add_agent_embeddings=False,
        act_fn_cls=nn.GELU,
        mat_init_gains=MATInitGains(),
        mat_normalization=MATNormalizationConfig(),
    )
    options.update(overrides)
    policy = _make_base_policy(env=env, obs_indices=indices, **options)
    if options["use_nop"] and not _is_off_policy_variant(options["policy_variant"]):
        policy = _add_ppo_nop(policy, env, indices, options)
    return policy


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("5M NOP1M", DEFAULT_MODEL_SCALE),
        ("5+1M", DEFAULT_MODEL_SCALE),
        ("5M+1M", DEFAULT_MODEL_SCALE),
        (" 5m nop1m ", DEFAULT_MODEL_SCALE),
        ("2.5M", "2.5M NOP0.75M"),
        ("2.5+0.75M", "2.5M NOP0.75M"),
        ("10M", "10M NOP2M"),
        ("10+2M", "10M NOP2M"),
    ],
)
def test_only_supported_scale_aliases(value, expected):
    assert normalize_model_scale(value) == expected


@pytest.mark.parametrize("value", ["1M", "2M NOP200K", "2.5M NOP0.5M", "10M NOP1M", "20M", "", 5])
def test_unsupported_scales_fail_before_building_environment(value):
    with pytest.raises(ValueError, match="Supported fixed"):
        _variant_options("mappo", model_scale=value)


def test_custom_width_mode_is_available():
    assert normalize_model_scale(None) is None


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant", list_variants())
def test_core_presets_have_explicit_conventional_layouts(variant, scale):
    with torch.device("meta"):
        policy = make_policy(variant, scale)
    layout = SCALE_LAYOUTS[scale]
    counts = count_policy_parameters(policy)
    assert 0.7 * layout.main_parameters < counts["trainable_minus_nop"] < 1.3 * layout.main_parameters
    assert policy.model_scale["name"] == scale
    assert policy.model_scale["layout_version"] == layout.version
    shared = counts["roles"]["shared_encoder"]["trainable"] > 0
    assert policy.model_scale["role_targets"] == {
        "shared_encoder": layout.main_parameters * 2 // 5 if shared else 0,
        "actor": layout.main_parameters // 5 if shared else layout.main_parameters * 2 // 5,
        "critic": layout.main_parameters * 2 // 5 if shared else layout.main_parameters * 3 // 5,
    }
    allowed = {128, 192, 256, 384, 512, 768, 1024, 1536}

    def check(config):
        if not is_dataclass(config):
            return
        for field in fields(config):
            value = getattr(config, field.name)
            if (
                field.name
                in {
                    "d_model",
                    "dim_feedforward",
                    "hidden_dim",
                    "latent_pi_dim",
                    "latent_pi_dim_per_agent",
                    "shared_encoder_latent_dim",
                    "shared_encoder_latent_dim_per_agent",
                    "projection_dim",
                    "nop_latent_dim",
                    "transition_model_d_model",
                    "transition_model_dim_feedforward",
                }
                and value is not None
            ):
                assert value in allowed, (field.name, value)
            elif (
                value
                and isinstance(value, (list, tuple))
                and (field.name.endswith("_dims") or field.name == "hidden_dims")
                and all(isinstance(width, int) for width in value)
            ):
                assert set(value) <= allowed, (field.name, value)
            elif is_dataclass(value):
                check(value)

    check(getattr(policy, "policy", policy).config)
    base = getattr(policy, "policy", policy)
    config = base.config
    if hasattr(config, "encoder_config"):
        assert len(base.encoder.layers) == layout.layers
    if hasattr(config, "decoder_config"):
        blocks = getattr(base.decoder, "layers", getattr(base.decoder, "blocks", None))
        assert len(blocks) == layout.layers
    if hasattr(config, "actor_encoder_config"):
        actor_encoder = base.actor_encoder if hasattr(base, "actor_encoder") else base.actor.encoder
        if getattr(config, "shared_encoder_config", None) is not None:
            actor_layers = layout.private_layers
        elif hasattr(config.actor_encoder_config, "temporal_model_cls"):
            actor_layers = layout.layers
        else:
            actor_layers = (
                layout.layers if actor_encoder.layers[0].self_attn is not None else layout.decentralized_layers
            )
        assert len(actor_encoder.layers) == actor_layers
    if getattr(config, "shared_encoder_config", None) is not None:
        assert len(base.shared_observation_encoder.layers) == layout.layers
    if hasattr(base.critic, "encoder"):
        expected = (
            layout.private_layers if getattr(config, "shared_encoder_config", None) is not None else layout.layers
        )
        assert len(base.critic.encoder.layers) == expected
    transition = getattr(
        policy, "transition_model", getattr(getattr(base, "critic_nop", None), "transition_model", None)
    )
    assert len(transition.encoder.layers) == layout.nop_layers
    nop_count = counts["roles"]["next_obs_prediction"]["trainable"]
    if layout.main_parameters == 2_500_000:
        assert 600_000 < nop_count < 850_000
    else:
        assert 0.75 * layout.nop_parameters < nop_count < 1.35 * layout.nop_parameters


@pytest.mark.parametrize(
    "variant", [name for name in list_variants(include_hidden=True) if name not in list_variants()]
)
def test_hidden_presets_remain_explicitly_available(variant):
    with torch.device("meta"):
        policy = make_policy(variant)
    assert policy.model_scale["layout_version"] == SCALE_LAYOUTS[DEFAULT_MODEL_SCALE].version


@pytest.mark.parametrize("variant", ["ppo", "mappo", "mat_qcx", "tmasac_lstm"])
def test_task_shapes_do_not_resize_the_layout(variant):
    with torch.device("meta"):
        small = make_policy(variant, n_agents=2)
        large = make_policy(variant, n_agents=9)
    small_config = getattr(small, "policy", small).config
    large_config = getattr(large, "policy", large).config
    if hasattr(small_config, "max_agents"):
        small_config = replace(small_config, max_agents=None)
        large_config = replace(large_config, max_agents=None)
    assert serialize_dataclass(small_config) == serialize_dataclass(large_config)
    assert small.model_scale == large.model_scale


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant", ["tmasac", "tmasac_shared_encoder"])
def test_critic_encoder_sharing_keeps_actor_dimensions_and_seeded_weights(variant, scale):
    torch.manual_seed(72)
    shared = make_policy(variant, scale, use_nop=False, critic_independent_encoders=False)
    torch.manual_seed(72)
    independent = make_policy(variant, scale, use_nop=False, critic_independent_encoders=True)
    assert shared.config.actor_encoder_config == independent.config.actor_encoder_config
    assert shared.config.actor_head_config == independent.config.actor_head_config
    for name in ("actor_encoder", "actor_head", "action_dist"):
        torch.testing.assert_close(
            getattr(shared, name).state_dict(), getattr(independent, name).state_dict(), rtol=0, atol=0
        )


@pytest.mark.parametrize("scale", list_model_scales())
def test_off_policy_decentralized_actor_is_common_across_algorithms(scale):
    with torch.device("meta"):
        policies = [
            make_policy(variant, scale, use_nop=False)
            for variant in ("maddpg_deepset", "matd3_deepset", "masac_deepset", "tmatd3_dec", "tmasac_dec")
        ]
    first = policies[0].config
    for policy in policies[1:]:
        assert policy.config.actor_encoder_config == first.actor_encoder_config
        assert (
            policy.config.actor_head_config.hidden_dims
            == first.actor_head_config.hidden_dims
            == [SCALE_LAYOUTS[scale].head_width]
        )
        assert policy.config.actor_head_config.normalize_input == first.actor_head_config.normalize_input
        assert policy.config.actor_head_config.init_gain == first.actor_head_config.init_gain


@pytest.mark.parametrize("scale", list_model_scales())
def test_shared_mat_v_critics_and_decoders_use_common_dimensions(scale):
    with torch.device("meta"):
        policies = [
            make_policy(name, scale, use_nop=False) for name in ("mat_ind", "mat_qcx", "mat_ind_lstm", "mat_qcx_lstm")
        ]
    assert all(policy.config.critic_config == policies[0].config.critic_config for policy in policies)
    assert policies[1].config.decoder_config == policies[3].config.decoder_config
    assert policies[0].config.actor_head_hidden_dims == policies[2].config.actor_head_hidden_dims
    layout = SCALE_LAYOUTS[scale]
    for policy in (policies[1], policies[3]):
        assert policy.config.decoder_config.dim_feedforward == layout.qcx_ff_width
        for block in policy.decoder.layers:
            assert (block.linear1.in_features, block.linear1.out_features, block.linear2.out_features) == (
                layout.qcx_decoder_width,
                layout.qcx_ff_width,
                layout.qcx_decoder_width,
            )


def test_nop_layout_is_common_and_does_not_change_main_weights():
    torch.manual_seed(72)
    enabled = make_policy("mappo")
    torch.manual_seed(72)
    disabled = make_policy("mappo", use_nop=False)
    torch.testing.assert_close(enabled.policy.state_dict(), disabled.state_dict(), rtol=0, atol=0)
    with torch.device("meta"):
        policies = [make_policy(name) for name in ("tmasac", "tmasac_lstm", "matd3_deepset")]
    for policy in policies:
        assert policy.config.nop_config.transition_model_d_model == enabled._wm_pre_transition_dims[-1] == 192
        assert policy.config.nop_config.latent_projection_hidden_dims == [256, 256]
        assert policy.config.nop_config.transition_model_num_layers == 2


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant", ["tmasac", "tmatd3"])
def test_default_twin_critics_share_one_encoder_at_every_scale(variant, scale):
    with torch.device("meta"):
        policy = make_policy(variant, scale)
    assert policy.critic.encoder2 is None
    assert policy.critic.q1 is not policy.critic.q2


def test_small_nop_retains_projection_and_prediction_capacity():
    with torch.device("meta"):
        small = make_policy("tmasac", "2.5M")
        medium = make_policy("tmasac")
    assert (
        small.config.nop_config.latent_projection_hidden_dims == medium.config.nop_config.latent_projection_hidden_dims
    )
    assert small.config.nop_config.pre_predictors_hidden_dims == medium.config.nop_config.pre_predictors_hidden_dims
    assert count_policy_parameters(small)["roles"]["next_obs_prediction"]["total"] > (
        count_policy_parameters(medium)["roles"]["next_obs_prediction"]["total"] * 0.6
    )


@pytest.mark.integration
@pytest.mark.parametrize(
    ("variant", "scale"),
    [
        (variant, DEFAULT_MODEL_SCALE)
        for variant in [
            "ppo",
            "mappo",
            "mappo_mlp",
            "mat_orig",
            "mat_ind",
            "mat_dec",
            "mat_qcx",
            "mat_ind_lstm",
            "mat_qcx_lstm",
            "tmasac_slstm",
            "tmasac_lstm",
        ]
    ]
    + [
        (variant, scale)
        for scale in list_model_scales()
        if scale != DEFAULT_MODEL_SCALE
        for variant in ("mappo", "mat_qcx_lstm", "tmasac_slstm", "tmasac_lstm")
    ],
)
def test_scaled_auxiliary_training_and_checkpoint_restore(variant, tmp_path, scale):
    from swarmbots.learn import make_training
    from swarmbots.learn.exponential_moving_average import ExponentialMovingAverage

    recurrent_sac = variant.startswith("tmasac")
    kwargs = dict(
        model_scale=scale,
        num_envs=2,
        device="cpu",
        episode_length=4,
        rollout_steps_per_env=4,
        scenario_kwargs={"compile_reward_kernel": False, "reset_settle_time": 0},
        env_kwargs={"compile_tensor_operations": False},
        policy_kwargs={"rmat_experimental_compile_lstm": False},
        algorithm_kwargs=(
            {
                "learning_starts": 0,
                "batch_size": 2,
                "buffer_capacity_per_env": 16,
                "gradient_steps": 1,
                "learning_rate_warmup_updates": 0,
                "burn_in_steps": 0,
                "learning_steps": 3,
                "temporal_state_store_interval": 1,
            }
            if recurrent_sac
            else {"learning_rate": 1e-3, "n_epochs": 1, "target_kl": None}
        ),
    )
    trainer = make_training("SwarmBots-WallEasy-v0", variant, **kwargs)
    try:
        before = {name: parameter.detach().clone() for name, parameter in trainer.policy.named_parameters()}
        metrics, _ = trainer.perform_iteration(
            ExponentialMovingAverage(0.1),
            ExponentialMovingAverage(0.1),
            update_ema=True,
        )
        assert trainer.n_total_updates > 0
        assert any("nop_loss" in key or key == "wm_loss" for key in metrics)
        assert any(not torch.equal(parameter, before[name]) for name, parameter in trainer.policy.named_parameters())
        assert all(torch.isfinite(parameter).all() for parameter in trainer.policy.parameters())
        path = tmp_path / "scaled.pt"
        trainer.save(path, optimizer_state_dict=trainer._get_optimizer_state_dict())
        checkpoint = torch.load(path, weights_only=False)
        layout = SCALE_LAYOUTS[scale]
        assert checkpoint["model_scale"]["requested"] == {
            "policy": layout.main_parameters,
            "nop": layout.nop_parameters,
        }
        assert checkpoint["model_scale"]["layout_version"] == layout.version
        restored = make_training("SwarmBots-WallEasy-v0", variant, **kwargs)
        try:
            restored.load(path)
            torch.testing.assert_close(restored.policy.state_dict(), trainer.policy.state_dict(), rtol=0, atol=0)
        finally:
            restored.env.close()
    finally:
        trainer.env.close()
