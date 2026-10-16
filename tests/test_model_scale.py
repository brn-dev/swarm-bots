from dataclasses import fields, is_dataclass, replace
from types import SimpleNamespace

import pytest
import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn import list_variants, list_model_scales
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig
from swarmbots.learn.algos.off_policy.transformer_critic import TransformerCriticConfig
from swarmbots.learn.algos.r_mat import RMATEncoder
from swarmbots.learn.algos.sac.masac_policy import MASACPolicy, MASACPolicyConfig
from swarmbots.learn.algos.td3.recurrent_td3_policy import RecurrentTD3PolicyConfig
from swarmbots.learn.algos.td3.td3_policy import TD3Policy, TD3PolicyConfig
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.parameter_counts import count_policy_parameters
from swarmbots.learn.presets.policy_factory import _is_off_policy_variant, _make_base_policy
from swarmbots.learn.presets.transformer import MATInitGains, MATNormalizationConfig
from swarmbots.learn.presets.model_scale import (
    DEFAULT_MODEL_SCALE, SCALE_LAYOUTS, make_policy_at_scale, model_config_at_scale, normalize_model_scale,
)
from swarmbots.learn.training import _add_ppo_nop, _variant_options
from swarmbots.learn.serialization_utils import serialize_dataclass
from swarmbots.learn.nn_components.deep_set import DeepSet, DeepSetInteractionConfig, GatedDeepSetInteraction


@pytest.fixture(autouse=True, scope="module")
def single_thread():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def _make_env(n_agents: int = 3) -> SimpleNamespace:
    return SimpleNamespace(
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


def make_policy(variant, scale="5M-2L NOP1M", *, n_agents=3, **overrides):
    env = _make_env(n_agents)
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
        ("5M-2L NOP1M", DEFAULT_MODEL_SCALE),
        ("5M-2L+1M", DEFAULT_MODEL_SCALE),
        (" 5m-2l nop1m ", DEFAULT_MODEL_SCALE),
        ("2.5M", "2.5M NOP0.75M"),
        ("2.5+0.75M", "2.5M NOP0.75M"),
        ("10M-3L", "10M-3L NOP2M"),
        ("10M-3L+2M", "10M-3L NOP2M"),
        ("10M-4L", "10M-4L NOP2M"),
        ("10M-5L", "10M-5L NOP2M"),
        (" 10m-5l nop2m ", "10M-5L NOP2M"),
        ("5M-2L", "5M-2L NOP1M"),
        ("5M-3L", "5M-3L NOP1M"),
        (" 5m-3l nop1m ", "5M-3L NOP1M"),
    ],
)
def test_only_supported_scale_aliases(value, expected):
    assert normalize_model_scale(value) == expected


@pytest.mark.parametrize(
    "value", ["1M", "2M NOP200K", "2.5M NOP0.5M", "10M NOP1M", "20M", "", 5,
              "5M", "5M NOP1M", "5+1M", "5M+1M", "10M", "10M NOP2M", "10+2M", "10M+2M"],
)
def test_unsupported_scales_fail_before_building_environment(value):
    with pytest.raises(ValueError, match="Supported fixed"):
        _variant_options("mappo", model_scale=value)


def test_model_scales_are_ordered_by_budget_and_depth() -> None:
    assert list_model_scales() == (
        "2.5M NOP0.75M", "5M-2L NOP1M", "5M-3L NOP1M",
        "10M-3L NOP2M", "10M-4L NOP2M", "10M-5L NOP2M",
    )


def test_custom_width_mode_is_available():
    assert normalize_model_scale(None) is None


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant", ["mat_ind__lstm", "mat_qcx__lstm", "tmasac__lstm", "tmasac__slstm"])
@pytest.mark.parametrize("custom_memory_width", [None, 8])
def test_fixed_scales_override_existing_recurrent_memory_widths(
    scale: str, variant: str, custom_memory_width: int | None,
) -> None:
    encoder_names = (
        ("encoder_config", "critic_encoder_config") if variant.startswith("mat_") else ("actor_encoder_config",)
    )
    with torch.device("meta"):
        source = make_policy(variant, "5M-3L", use_nop=False)
        reference = make_policy(variant, scale, use_nop=False)
        source_config = source.config
        if custom_memory_width is not None:
            source_config = replace(source_config, **{
                name: replace(getattr(source_config, name), temporal_hidden_dim=custom_memory_width)
                for name in encoder_names
            })
        scaled = make_policy_at_scale(type(source), _make_env(), source_config, scale)

    for name in ("actor_encoder", "critic_encoder"):
        reference_encoder = getattr(reference, name, None)
        if not isinstance(reference_encoder, RMATEncoder):
            continue
        scaled_encoder = getattr(scaled, name)
        assert [layer.temporal_model.hidden_dim for layer in scaled_encoder.layers] == [
            layer.temporal_model.hidden_dim for layer in reference_encoder.layers
        ]
        assert {name: parameter.shape for name, parameter in scaled_encoder.named_parameters()} == {
            name: parameter.shape for name, parameter in reference_encoder.named_parameters()
        }


@pytest.mark.parametrize("variant", list_variants(include_hidden=True))
@pytest.mark.parametrize(
    ("scales", "depths", "decentralized_depth"),
    [
        (("5M-2L", "5M-3L"), (2, 3), 3),
        (("10M-3L", "10M-4L", "10M-5L"), (3, 4, 5), 4),
    ],
)
def test_depth_choices_preserve_heads_and_auxiliary_architecture(
    variant: str, scales: tuple[str, ...], depths: tuple[int, ...], decentralized_depth: int,
) -> None:
    with torch.device("meta"):
        policies = [make_policy(variant, scale) for scale in scales]
    bases = [getattr(policy, "policy", policy) for policy in policies]
    for depth, base in zip(depths, bases, strict=True):
        config = base.config
        if hasattr(config, "actor_config"):
            assert config == bases[0].config
        else:
            actor_encoder = base.actor_encoder if hasattr(base, "actor_encoder") else base.actor.encoder
            uses_attention = actor_encoder.layers[0].self_attn is not None
            assert len(actor_encoder.layers) == (depth if uses_attention else decentralized_depth)
            if hasattr(base, "critic_encoder"):
                critic_encoder = base.critic_encoder
            else:
                critic = base.critic[0] if isinstance(base.critic, nn.ModuleList) else base.critic
                critic_encoder = critic.encoder
                if not hasattr(critic_encoder, "layers"):
                    critic_encoder = critic_encoder.network
            assert len(critic_encoder.layers) == depth
    first = bases[0]
    for second in bases[1:]:
        if hasattr(first.config, "actor_config"):
            assert first.config == second.config
        else:
            actor_config_name = "encoder_config" if hasattr(first.config, "encoder_config") else "actor_encoder_config"
            if variant == "mat_dec" or not getattr(first.config, actor_config_name).use_agent_attention:
                assert getattr(first.config, actor_config_name) == getattr(second.config, actor_config_name)
        for config_name in (
            "decoder_config", "actor_head_config", "nop_config", "critic_config", "transformer_critic_config",
        ):
            if hasattr(first.config, config_name):
                assert getattr(first.config, config_name) == getattr(second.config, config_name)
    if hasattr(policies[0], "world_model_config"):
        assert all(policy.world_model_config == policies[0].world_model_config for policy in policies[1:])
    counts = [count_policy_parameters(policy)["trainable_minus_nop"] for policy in policies]
    tolerance = 0.01 if variant.endswith(("__lstm", "__slstm")) else 0.05
    assert all(abs(count / counts[0] - 1) < tolerance for count in counts[1:])


@pytest.mark.parametrize("scale", [None, *list_model_scales()])
def test_actor_attention_ablation_preserves_critic_architecture(scale: str | None) -> None:
    with torch.device("meta"):
        baseline = make_policy("mat_ind", scale, use_nop=False)
        ablation = make_policy("mat_ind", scale, use_nop=False, mat_use_agent_attention=False)

    assert baseline.actor_encoder_config.use_agent_attention
    assert not ablation.actor_encoder_config.use_agent_attention
    assert ablation.critic_encoder_config == baseline.critic_encoder_config
    assert ablation.config.critic_config == baseline.config.critic_config
    assert count_policy_parameters(ablation)["roles"]["critic"] == count_policy_parameters(baseline)["roles"]["critic"]


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("config_class", [TD3PolicyConfig, RecurrentTD3PolicyConfig])
@pytest.mark.parametrize("critic_kind", ["mlp", "deepset"])
@pytest.mark.parametrize("n_critics", [1, 2])
def test_td3_scale_uses_top_level_critic_kind(
    scale: str, config_class: type[TD3PolicyConfig], critic_kind: str, n_critics: int,
) -> None:
    config = config_class(
        critic_kind=critic_kind,
        n_critics=n_critics,
        joint_critic_config=JointCriticConfig(kind="mlp" if critic_kind == "deepset" else "deepset"),
    )
    matching_config = replace(config, joint_critic_config=replace(config.joint_critic_config, kind=critic_kind))

    assert model_config_at_scale(config, scale) == model_config_at_scale(matching_config, scale)


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("family", ["ddpg", "td3", "sac"])
def test_scaled_deepset_policies_preserve_explicit_encoder_and_head_controls(family: str, scale: str) -> None:
    encoder_config = MATEncoderConfig(
        d_model=16, nhead=4, joint_obs_embedding=True, normalize_tokens=True,
        norm_first=False, bias=False, layer_norm_eps=1e-4, act_fn_cls=nn.ReLU,
        transformer_ff_init_gain=0.7,
    )
    head_config = TransformerCriticConfig(
        separate_observation_action_encoders=True, action_encoder_dim=7, value_head_init_gain=0.123,
        interaction_config=DeepSetInteractionConfig(
            gated=True, message_dim=24, gate_hidden_dim=5, normalize=False, local_update="linear",
        ),
    )
    joint_config = JointCriticConfig(kind="deepset", encoder_config=encoder_config, head_config=head_config)
    top_level_encoder = MATEncoderConfig(act_fn_cls=nn.Tanh)
    if family == "sac":
        config = MASACPolicyConfig(
            joint_critic_config=joint_config, critic_encoder_config=top_level_encoder, act_fn_cls=nn.SiLU,
        )
        policy_class = MASACPolicy
        head_config_name = "critic_config"
        activation = nn.SiLU
    else:
        config = TD3PolicyConfig(
            critic_kind="deepset", n_critics=1 if family == "ddpg" else 2,
            joint_critic_config=joint_config, critic_encoder_config=top_level_encoder,
        )
        policy_class = TD3Policy
        head_config_name = "transformer_critic_config"
        activation = nn.Tanh
    env = _make_env()
    with torch.device("meta"):
        baseline = policy_class(env=env, config=config)
        scaled = make_policy_at_scale(policy_class, env, config, scale)
    single_q = family == "ddpg"
    baseline_encoder = baseline.critic[0].encoder.network if single_q else baseline.critic.encoder.network
    encoder = scaled.critic[0].encoder.network if single_q else scaled.critic.encoder.network
    heads = [scaled.critic[0].network] if single_q else [scaled.critic.q1, scaled.critic.q2]
    layout = SCALE_LAYOUTS[scale]

    assert encoder.joint_obs_embedding == baseline_encoder.joint_obs_embedding is True
    assert isinstance(encoder.token_norm, nn.LayerNorm)
    assert encoder.observation_action_encoder is not None
    assert encoder.observation_action_encoder.action_encoder_dim == 7
    assert encoder.d_model == layout.critic_width and len(encoder.layers) == layout.layers
    assert all(not layer.norm_first and layer.norm1.bias is None for layer in encoder.layers)
    assert encoder.norm.eps == 1e-4
    assert any(isinstance(module, activation) for module in encoder.local_action_encoder.modules())
    for head in heads:
        assert all(block.features == layout.deepset_q_head_width for block in head.deepset.interactions)
        assert all(isinstance(block, GatedDeepSetInteraction) for block in head.deepset.interactions)
        assert all(block.message_dim == 24 and block.gate_hidden_dim == 5 for block in head.deepset.interactions)
        assert all(isinstance(block.global_norm, nn.Identity) for block in head.deepset.interactions)
        assert all(block.local_update == "linear" for block in head.deepset.interactions)
    resolved = scaled.config.joint_critic_config
    assert resolved.encoder_config == scaled.config.critic_encoder_config
    assert resolved.encoder_config.transformer_ff_init_gain == 0.7
    assert resolved.head_config == getattr(scaled.config, head_config_name)
    assert resolved.head_config.value_head_init_gain == 0.123


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("family", ["td3", "sac"])
def test_scaled_deepset_configs_keep_top_level_templates_when_nested_configs_are_absent(
    family: str, scale: str,
) -> None:
    encoder = MATEncoderConfig(joint_obs_embedding=True, normalize_tokens=True)
    head = TransformerCriticConfig(
        separate_observation_action_encoders=True, value_head_init_gain=0.123,
        interaction_config=DeepSetInteractionConfig(gated=True),
    )
    joint = JointCriticConfig(kind="deepset")
    config = (
        MASACPolicyConfig(joint_critic_config=joint, critic_encoder_config=encoder, critic_config=head)
        if family == "sac" else TD3PolicyConfig(
            critic_kind="deepset", joint_critic_config=joint,
            critic_encoder_config=encoder, transformer_critic_config=head,
        )
    )

    resolved = model_config_at_scale(config, scale).joint_critic_config

    assert resolved.encoder_config.joint_obs_embedding and resolved.encoder_config.normalize_tokens
    assert resolved.head_config.separate_observation_action_encoders
    assert resolved.head_config.interaction_config.gated
    assert resolved.head_config.value_head_init_gain == 0.123


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant", list_variants())
def test_core_presets_have_explicit_conventional_layouts(variant, scale):
    with torch.device("meta"):
        policy = make_policy(variant, scale)
    layout = SCALE_LAYOUTS[scale]
    counts = count_policy_parameters(policy)
    reference_counts = counts
    if variant.endswith(("__lstm", "__slstm")):
        with torch.device("meta"):
            reference = make_policy(variant.rsplit("__", 1)[0], scale)
        reference_counts = count_policy_parameters(reference)
        assert counts["trainable_minus_nop"] > reference_counts["trainable_minus_nop"]
    assert 0.7 * layout.main_parameters < reference_counts["trainable_minus_nop"] < 1.3 * layout.main_parameters
    assert policy.model_scale["name"] == scale
    assert policy.model_scale["layout_version"] == layout.version
    shared = counts["roles"]["shared_encoder"]["trainable"] > 0
    assert policy.model_scale["role_targets"] == {
        "shared_encoder": layout.main_parameters * 2 // 5 if shared else 0,
        "actor": layout.main_parameters // 5 if shared else layout.main_parameters * 2 // 5,
        "critic": layout.main_parameters * 2 // 5 if shared else layout.main_parameters * 3 // 5,
    }
    allowed = {128, 192, 256, 320, 384, 400, 480, 512, 608, 768, 960, 1024, 1536}

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
                    "latent_dim",
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
        assert len(base.actor_encoder.layers) == (layout.decentralized_layers if variant == "mat_dec" else layout.layers)
        assert len(base.critic_encoder.layers) == layout.layers
    if hasattr(config, "decoder_config"):
        blocks = getattr(base.decoder, "layers", getattr(base.decoder, "blocks", None))
        assert len(blocks) == (layout.decoder_layers or layout.layers)
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
    if hasattr(base.critic, "encoder") and hasattr(base.critic.encoder, "layers"):
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


@pytest.mark.parametrize("variant", ["ppo", "mappo", "mat_qcx", "tmasac__lstm"])
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
@pytest.mark.parametrize("variant", ["tmasac"])
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
            for variant in ("maddpg__deepset_critic", "matd3__deepset_critic", "masac__deepset_critic", "tmatd3_dec", "tmasac_dec")
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
def test_all_off_policy_q_heads_share_the_same_architecture(scale: str) -> None:
    names = (
        "maddpg__deepset_critic", "matd3__deepset_critic", "masac__deepset_critic",
        "tmatd3", "tmatd3_dec", "tmasac", "tmasac_dec", "tmasac__lstm", "tmasac__slstm",
    )
    with torch.device("meta"):
        policies = [make_policy(name, scale, use_nop=False) for name in names]
    reference = policies[0].critic[0].network
    shapes = {name: parameter.shape for name, parameter in reference.named_parameters()}
    for policy in policies[1:]:
        for head in (policy.critic.q1, policy.critic.q2):
            assert {name: parameter.shape for name, parameter in head.named_parameters()} == shapes
    assert policies[0].config.joint_critic_config.encoder_config == policies[1].config.joint_critic_config.encoder_config
    single_count = count_policy_parameters(policies[0])["roles"]["critic"]["trainable"]
    twin_count = count_policy_parameters(policies[1])["roles"]["critic"]["trainable"]
    assert twin_count - single_count == sum(parameter.numel() for parameter in reference.parameters())


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant", ["tmasac", "tmatd3", "masac__deepset_critic", "matd3__deepset_critic"])
def test_independent_critic_encoders_duplicate_without_resizing(variant: str, scale: str) -> None:
    with torch.device("meta"):
        shared = make_policy(variant, scale, use_nop=False)
        independent = make_policy(variant, scale, use_nop=False, critic_independent_encoders=True)
    for name in ("encoder", "q1", "q2"):
        assert {
            key: parameter.shape for key, parameter in getattr(shared.critic, name).named_parameters()
        } == {key: parameter.shape for key, parameter in getattr(independent.critic, name).named_parameters()}
    assert count_policy_parameters(independent)["trainable_minus_nop"] - count_policy_parameters(shared)[
        "trainable_minus_nop"
    ] == sum(parameter.numel() for parameter in independent.critic.encoder2.parameters())


@pytest.mark.parametrize("variant", ["maddpg__deepset_critic", "matd3__deepset_critic", "masac__deepset_critic"])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("scale", [None, *list_model_scales()])
def test_custom_deepset_widths_apply_only_without_a_fixed_scale(
    variant: str, gated: bool, scale: str | None,
) -> None:
    name = variant + ("__gated" if gated else "")
    with torch.device("meta"):
        policy = make_policy(
            name, scale, use_nop=False,
            baseline_critic_element_hidden_dims=(17, 19), baseline_critic_hidden_dims=(23, 29),
        )
        if scale is not None:
            baseline = make_policy(name, scale, use_nop=False)
            assert policy.config.joint_critic_config == baseline.config.joint_critic_config
            assert policy.config.critic_encoder_config == baseline.config.critic_encoder_config
            assert {key: parameter.shape for key, parameter in policy.critic.named_parameters()} == {
                key: parameter.shape for key, parameter in baseline.critic.named_parameters()
            }
            return
    single = variant.startswith("maddpg")
    encoder = policy.critic[0].encoder.network if single else policy.critic.encoder.network
    heads = [policy.critic[0].network] if single else [policy.critic.q1, policy.critic.q2]
    assert encoder.d_model == 19
    assert [module.out_features for module in encoder.local_action_encoder if isinstance(module, nn.Linear)] == [17, 19]
    for head in heads:
        assert [
            module.out_features for module in head.deepset.set_decoder.modules() if isinstance(module, nn.Linear)
        ] == [23, 29, 1]
    interaction = encoder.layers[0].interaction
    for module_name in ("message_encoder", "global_update", "message_projection"):
        first = next(getattr(interaction, module_name).parameters())
        second = next(getattr(encoder.layers[1].interaction, module_name).parameters())
        assert first is not second


@pytest.mark.parametrize("scale", list_model_scales())
def test_shared_mat_v_critics_and_decoders_use_common_dimensions(scale):
    with torch.device("meta"):
        policies = [
            make_policy(name, scale, use_nop=False) for name in ("mat_ind", "mat_qcx", "mat_ind__lstm", "mat_qcx__lstm")
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
        policies = [make_policy(name) for name in ("tmasac", "tmasac__lstm", "matd3__deepset_critic")]
    for policy in policies:
        assert policy.config.nop_config.transition_model_d_model == enabled._wm_pre_transition_dims[-1] == 192
        assert policy.config.nop_config.latent_projection_hidden_dims == [256, 256]
        assert policy.config.nop_config.transition_model_num_layers == 2


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant", [
    "mappo", "mat_ind", "mat_qcx", "mat_dec", "mat_ind__lstm", "mat_qcx__lstm",
    "maddpg__deepset_critic", "matd3__deepset_critic", "masac__deepset_critic", "tmatd3", "tmasac", "tmasac__lstm",
])
@pytest.mark.parametrize("custom", [False, True])
def test_deepset_presets_use_family_defaults_and_honor_overrides(
    scale: str, variant: str, custom: bool,
) -> None:
    settings = {"num_layers": 3, "local_update": "linear", "normalize": False, "hidden_dims": [64]}
    with torch.device("meta"):
        policy = make_policy(variant, scale, **({"deepset_interaction_config": settings} if custom else {}))
    critics = [module for module in policy.modules() if isinstance(module, DeepSet)]
    assert critics
    expected = DeepSetInteractionConfig(**settings) if custom else DeepSetInteractionConfig()
    for critic in critics:
        assert len(critic.interactions) == expected.num_layers
        assert all(block.local_update == expected.local_update for block in critic.interactions)
        assert all(isinstance(block.global_norm, nn.LayerNorm) == expected.normalize for block in critic.interactions)
        assert all(isinstance(block.local_norm, nn.LayerNorm) == expected.normalize for block in critic.interactions)
    metadata = next(value for key, value in policy.get_hyper_parameters().items() if key.endswith("policy_config"))
    critic_config = (
        metadata["joint_critic_config"]["head_config"] if variant.endswith("__deepset_critic")
        else metadata.get("transformer_critic_config") or metadata["critic_config"]
    )
    interaction = (critic_config.get("deep_set_config") or critic_config)["interaction_config"]
    for name in ("num_layers", "local_update", "normalize"):
        assert interaction[name] == getattr(expected, name)
    if custom:
        assert interaction["hidden_dims"] == settings["hidden_dims"]
    else:
        assert interaction["message_dim"] is not None


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant", ["tmasac", "tmatd3", "masac__deepset_critic", "matd3__deepset_critic"])
def test_default_twin_critics_share_one_encoder_at_every_scale(variant, scale):
    with torch.device("meta"):
        policy = make_policy(variant, scale)
    assert policy.critic.encoder2 is None
    assert policy.critic.q1 is not policy.critic.q2


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant,comparison", [("masac__deepset_critic", "tmasac_dec"), ("matd3__deepset_critic", "tmatd3_dec")])
def test_shared_deepset_and_transformer_critics_have_comparable_parameter_budgets(
    scale: str, variant: str, comparison: str,
) -> None:
    with torch.device("meta"):
        deepset = make_policy(variant, scale, use_nop=False)
        transformer = make_policy(comparison, scale, use_nop=False)
        independent = make_policy(variant, scale, use_nop=False, critic_independent_encoders=True)
    deepset_counts = count_policy_parameters(deepset)["roles"]
    transformer_counts = count_policy_parameters(transformer)["roles"]
    assert deepset_counts["actor"]["trainable"] == transformer_counts["actor"]["trainable"]
    assert deepset.config.actor_encoder_config == independent.config.actor_encoder_config
    assert 0.85 < deepset_counts["critic"]["trainable"] / transformer_counts["critic"]["trainable"] < 1.15
    assert independent.critic.encoder2 is not None


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
        for variant in ["ppo", "mappo", "mat_orig", "mat_ind", "mat_dec", "mat_qcx", "mat_ind__lstm", "mat_qcx__lstm", "tmasac__slstm", "tmasac__lstm"]
    ]
    + [
        (variant, scale)
        for scale in list_model_scales()
        if scale != DEFAULT_MODEL_SCALE
        for variant in ("mappo", "mat_qcx__lstm", "tmasac__slstm", "tmasac__lstm")
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
