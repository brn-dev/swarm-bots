from dataclasses import fields, replace
from types import SimpleNamespace

import pytest
import torch
from gymnasium import spaces
from torch import nn

from swarmbots.learn import list_variants, list_model_scales
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.off_policy.actor_heads import TMASACActorHeadKind
from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig
from swarmbots.learn.algos.off_policy.transformer_critic import TransformerCriticConfig
from swarmbots.learn.algos.r_mat import RMATEncoder
from swarmbots.learn.algos.r_mat.temporal_sequence_model import (
    LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.sac.masac_policy import MASACPolicy, MASACPolicyConfig
from swarmbots.learn.algos.td3.recurrent_td3_policy import RecurrentTD3PolicyConfig
from swarmbots.learn.algos.td3.td3_policy import TD3Policy, TD3PolicyConfig
from swarmbots.learn.algos.xlstm.slstm import SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig
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
from swarmbots.learn.nn_components.residual_mlp import ResidualMLP


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


@pytest.fixture
def wall_easy_actor_shape(monkeypatch: pytest.MonkeyPatch) -> None:
    def make_environment(n_agents: int = 5) -> SimpleNamespace:
        return SimpleNamespace(
            n_agents=n_agents, local_obs_dim=71, global_obs_dim=0,
            hidden_local_vars_dim=4, hidden_global_vars_dim=2,
            action_space=HybridActionSpace({
                "actuators": spaces.Box(-1.0, 1.0, shape=(n_agents, 8)),
                "connectors": spaces.Box(-1.0, 1.0, shape=(n_agents, 4)),
            }),
        )

    monkeypatch.setattr(f"{__name__}._make_env", make_environment)


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
              "5M", "5M NOP1M", "5+1M", "5M+1M", "10M", "10M NOP2M", "10+2M", "10M+2M", "legacy"],
)
def test_unsupported_scales_fail_before_building_environment(value):
    with pytest.raises(ValueError, match="Supported fixed"):
        _variant_options("mappo__deepset_critic", model_scale=value)


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
    ("scales", "depths"),
    [
        (("5M-2L", "5M-3L"), (2, 3)),
        (("10M-3L", "10M-4L", "10M-5L"), (3, 4, 5)),
    ],
)
def test_depth_choices_preserve_parameter_and_auxiliary_budgets(
    variant: str, scales: tuple[str, ...], depths: tuple[int, ...],
) -> None:
    with torch.device("meta"):
        policies = [make_policy(variant, scale) for scale in scales]
    bases = [getattr(policy, "policy", policy) for policy in policies]
    for depth, base in zip(depths, bases, strict=True):
        config = base.config
        layout = SCALE_LAYOUTS[base.model_scale["name"]]
        if hasattr(config, "actor_config"):
            if variant == "ppo":
                assert config == bases[0].config
            else:
                assert len(base.actor_encoder.layers) == depth
                assert len(base.actor.mlp.layers) == layout.actor_head_blocks
                assert len(base.critic_encoder.layers) == depth
        elif hasattr(config, "encoder_config"):
            assert len(base.actor_encoder.layers) == depth
            assert len(base.critic_encoder.layers) == depth
        else:
            actor_encoder = base.actor_encoder if hasattr(base, "actor_encoder") else base.actor.encoder
            assert len(actor_encoder.layers) == depth
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
            assert first.config.critic_config == second.config.critic_config
        if hasattr(first.config, "nop_config"):
            assert first.config.nop_config == second.config.nop_config
    if hasattr(policies[0], "world_model_config"):
        assert all(policy.world_model_config == policies[0].world_model_config for policy in policies[1:])
    counts = [count_policy_parameters(policy)["trainable_minus_nop"] for policy in policies]
    tolerance = 0.03 if scales[0].startswith("10M-") and "__gated_" in variant else 0.01
    assert all(abs(count / counts[0] - 1) < tolerance for count in counts[1:])


@pytest.mark.parametrize("variant", [name for name in list_variants(include_hidden=True) if name != "ppo"])
@pytest.mark.parametrize("scales", [("5M-2L", "5M-3L"), ("10M-3L", "10M-4L", "10M-5L")])
def test_depth_choices_preserve_actor_width_and_output_head(variant: str, scales: tuple[str, ...]) -> None:
    with torch.device("meta"):
        policies = [getattr(policy, "policy", policy) for policy in (
            make_policy(variant, scale, use_nop=False) for scale in scales
        )]
    encoders = [policy.actor_encoder if hasattr(policy, "actor_encoder") else policy.actor.encoder for policy in policies]
    widths = [encoder.norm.normalized_shape[0] for encoder in encoders]
    assert len(set(widths)) == 1
    for policy in policies[1:]:
        if hasattr(policies[0].config, "decoder_config"):
            assert policy.config.decoder_config == policies[0].config.decoder_config
        elif variant == "mappo__deepset_critic":
            assert policy.config.actor_config == policies[0].config.actor_config
        elif hasattr(policy.config, "actor_head_config"):
            assert policy.config.actor_head_config == policies[0].config.actor_head_config
        else:
            assert policy.config.residual_mlp_config == policies[0].config.residual_mlp_config
            assert policy.config.actor_head_hidden_dims == policies[0].config.actor_head_hidden_dims


@pytest.mark.parametrize("variant", ["mat_ind", "mat_qcx", "mat_orig", "tmasac", "tmatd3"])
def test_two_layer_actors_have_two_attention_blocks_and_encoder_dominated_capacity(
    variant: str, wall_easy_actor_shape: None,
) -> None:
    with torch.device("meta"):
        policy = make_policy(variant, "5M-2L", n_agents=5, use_nop=False)
    encoder = policy.actor_encoder if hasattr(policy, "actor_encoder") else policy.actor.encoder
    assert len(encoder.layers) == 2
    assert all(isinstance(layer.self_attn, nn.MultiheadAttention) for layer in encoder.layers)
    encoder_count = sum(parameter.numel() for parameter in encoder.parameters())
    counts = count_policy_parameters(policy)
    base_actor_count = counts["roles"]["actor"]["total"] - counts["processing"]["actor_decoder_attention_parameters"]
    assert encoder_count > 0.8 * base_actor_count


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
    if scale is not None:
        actor_counts = [count_policy_parameters(policy)["roles"]["actor"]["total"] for policy in (baseline, ablation)]
        assert max(actor_counts) / min(actor_counts) < 1.001


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
def test_single_flattened_critic_omits_one_unchanged_mlp_head(scale: str) -> None:
    with torch.device("meta"):
        single = make_policy_at_scale(TD3Policy, _make_env(), TD3PolicyConfig(critic_kind="mlp", n_critics=1), scale)
        twin = make_policy_at_scale(TD3Policy, _make_env(), TD3PolicyConfig(critic_kind="mlp", n_critics=2), scale)
    shapes = {name: parameter.shape for name, parameter in single.critic[0].named_parameters()}
    assert len(single.critic) == 1 and len(twin.critic) == 2
    for head in twin.critic:
        assert {name: parameter.shape for name, parameter in head.named_parameters()} == shapes
    single_count = count_policy_parameters(single)["roles"]["critic"]["total"]
    twin_count = count_policy_parameters(twin)["roles"]["critic"]["total"]
    assert twin_count == 2 * single_count


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
        assert all(block.features == layout.critic_head_width for block in head.deepset.interactions)
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
    base = getattr(policy, "policy", policy)
    config = base.config
    if hasattr(config, "encoder_config"):
        assert len(base.actor_encoder.layers) == layout.actor_encoder_layers
        assert len(base.critic_encoder.layers) == layout.layers
    if hasattr(config, "decoder_config"):
        blocks = getattr(base.decoder, "layers", getattr(base.decoder, "blocks", None))
        assert len(blocks) == layout.decoder_layers
    if hasattr(config, "actor_encoder_config"):
        actor_encoder = base.actor_encoder if hasattr(base, "actor_encoder") else base.actor.encoder
        if getattr(config, "shared_encoder_config", None) is not None:
            actor_layers = layout.private_layers
        else:
            actor_layers = layout.actor_encoder_layers
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


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant", ["mat_orig", "mat_ind", "mat_dec", "mat_qcx", "mat_ind__lstm", "mat_qcx__lstm"])
def test_mat_and_off_policy_critics_use_matching_observation_encoder_layouts(variant: str, scale: str) -> None:
    with torch.device("meta"):
        policy = make_policy(variant, scale, use_nop=False)
        reference = make_policy("tmasac", scale, use_nop=False)
    config = policy.critic_encoder_config
    reference_config = reference.config.critic_encoder_config
    for field_name in (
        "d_model", "nhead", "num_layers", "dim_feedforward", "transformer_ff_config",
        "local_obs_encoder_config", "global_obs_encoder_config", "joint_obs_embedding", "normalize_tokens",
    ):
        assert getattr(config, field_name) == getattr(reference_config, field_name), field_name
    first_local = next(module for module in policy.critic_encoder.local_obs_encoder.modules() if isinstance(module, nn.Linear))
    assert first_local.in_features == policy.local_obs_dim + policy.hidden_local_vars_dim
    first_global = next(module for module in policy.critic_encoder.global_obs_encoder.modules() if isinstance(module, nn.Linear))
    assert first_global.in_features == policy.global_obs_dim + policy.hidden_global_vars_dim
    assert policy.nop_latent_dim == config.d_model


@pytest.mark.parametrize(
    "variant", [name for name in list_variants(include_hidden=True) if name not in list_variants()]
)
def test_hidden_presets_remain_explicitly_available(variant):
    with torch.device("meta"):
        policy = make_policy(variant)
    assert policy.model_scale["layout_version"] == SCALE_LAYOUTS[DEFAULT_MODEL_SCALE].version


@pytest.mark.parametrize("variant", ["ppo", "mappo__deepset_critic", "mat_qcx", "tmasac__lstm"])
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
        )
        assert policy.config.actor_head_config.hidden_dims[-1] == SCALE_LAYOUTS[scale].width
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


@pytest.mark.parametrize(("algorithm", "independent_encoders"), [
    ("maddpg", False), ("matd3", False), ("masac", False), ("matd3", True), ("masac", True),
])
@pytest.mark.parametrize("scale", [None, *list_model_scales()])
def test_gated_deepset_presets_preserve_regular_q_heads_at_every_scale(
    algorithm: str, independent_encoders: bool, scale: str | None,
) -> None:
    with torch.device("meta"):
        baseline = make_policy(
            f"{algorithm}__deepset_critic", scale, use_nop=False, critic_independent_encoders=independent_encoders,
        )
        gated = make_policy(
            f"{algorithm}__gated_deepset_critic", scale, use_nop=False,
            critic_independent_encoders=independent_encoders,
        )
    assert gated.config.joint_critic_config.interaction_config.gated
    assert gated.config.joint_critic_config.head_config == baseline.config.joint_critic_config.head_config
    baseline_shapes = {name: parameter.shape for name, parameter in baseline.named_parameters()}
    gated_shapes = {name: parameter.shape for name, parameter in gated.named_parameters()}
    gate_names = {name for name in gated_shapes if ".reception_gate." in name}
    encoder_names = ("0.encoder",) if algorithm == "maddpg" else ("encoder",)
    if independent_encoders:
        encoder_names += ("encoder2",)
    encoder_prefixes = tuple(
        f"{critic}.{encoder}." for critic in ("critic", "critic_target") for encoder in encoder_names
    )
    assert gate_names
    assert all(name.startswith(encoder_prefixes) for name in gate_names)
    assert all(any(name.startswith(prefix) for name in gate_names) for prefix in encoder_prefixes)
    assert {name: shape for name, shape in gated_shapes.items() if name not in gate_names} == baseline_shapes


@pytest.mark.parametrize("algorithm", ["maddpg", "matd3", "masac"])
@pytest.mark.parametrize("gated", [False, True])
@pytest.mark.parametrize("scale", [None, *list_model_scales()])
def test_custom_deepset_widths_apply_only_without_a_fixed_scale(
    algorithm: str, gated: bool, scale: str | None,
) -> None:
    critic = "gated_deepset_critic" if gated else "deepset_critic"
    name = f"{algorithm}__{critic}"
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
    single = algorithm == "maddpg"
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


@pytest.mark.parametrize("variant", [
    "maddpg__deepset_critic", "maddpg__gated_deepset_critic",
    "matd3__deepset_critic", "matd3__gated_deepset_critic",
    "masac__deepset_critic", "masac__gated_deepset_critic", "tmatd3", "tmasac",
])
@pytest.mark.parametrize("head_gating", [False, True])
@pytest.mark.parametrize("scale", [None, *list_model_scales()])
def test_q_head_gating_is_independent_of_encoder_gating(
    variant: str, head_gating: bool, scale: str | None,
) -> None:
    with torch.device("meta"):
        policy = make_policy(variant, scale, use_nop=False, critic_deepset_head_gating=head_gating)
    single_q = variant.startswith("maddpg")
    encoder = policy.critic[0].encoder if single_q else policy.critic.encoder
    heads = [policy.critic[0].network] if single_q else [policy.critic.q1, policy.critic.q2]
    for head in heads:
        assert all(isinstance(block, GatedDeepSetInteraction) is head_gating for block in head.deepset.interactions)
    encoder_gates = [module for module in encoder.modules() if isinstance(module, GatedDeepSetInteraction)]
    assert bool(encoder_gates) is variant.endswith("__gated_deepset_critic")


@pytest.mark.parametrize("scale", [None, *list_model_scales()])
def test_eight_head_slstm_produces_actions_at_every_scale(scale: str | None) -> None:
    policy = make_policy(
        "tmasac__slstm", scale, use_nop=False,
        rmat_temporal_model_config=SLSTMTemporalSequenceModelConfig(num_heads=8),
    )
    with torch.no_grad():
        actions, log_probs = policy.action_log_prob(
            local_obs=torch.randn(2, 3, 5), global_obs=torch.randn(2, 2),
            agent_mask=torch.ones(2, 3, dtype=torch.bool), deterministic=True,
        )
    assert actions.shape == (2, 3, 3) and log_probs.shape == (2, 3)
    assert torch.isfinite(actions).all() and torch.isfinite(log_probs).all()
    assert all(layer.temporal_model.config.num_heads == 8 for layer in policy.actor_encoder.layers)


@pytest.mark.parametrize("scale", [None, *list_model_scales()])
@pytest.mark.parametrize("variant", ["mat_ind__lstm", "mat_qcx__lstm"])
@pytest.mark.parametrize("specification", ["classes", "configs", "both"])
def test_recurrent_mat_preserves_independent_per_layer_temporal_specs(
    scale: str | None, variant: str, specification: str,
) -> None:
    layout = None if scale is None else SCALE_LAYOUTS[scale]
    actor_depth = 2 if layout is None else layout.actor_encoder_layers
    critic_depth = 2 if layout is None else layout.layers
    options = {}
    expected = {}
    for role, depth in (("actor", actor_depth), ("critic", critic_depth)):
        classes = [LSTMTemporalSequenceModel] * depth
        configs = [LSTMTemporalSequenceModelConfig(num_layers=1 if role == "actor" else 2)] * depth
        if specification == "both":
            classes[::2] = [SLSTMTemporalSequenceModel] * len(classes[::2])
            configs[::2] = [SLSTMTemporalSequenceModelConfig()] * len(configs[::2])
        prefix = "rmat" if role == "actor" else "rmat_critic"
        if specification != "configs":
            options[prefix + "_temporal_model_cls"] = classes
        if specification != "classes":
            options[prefix + "_temporal_model_config"] = configs
        else:
            configs = [LSTMTemporalSequenceModelConfig()] * depth
        expected[role] = list(zip(classes, configs, strict=True))
    with torch.device("meta"):
        policy = make_policy(variant, scale, use_nop=False, **options)
    for role, temporal_specs in expected.items():
        encoder = getattr(policy, role + "_encoder")
        for layer, (model_class, config) in zip(encoder.layers, temporal_specs, strict=True):
            assert isinstance(layer.temporal_model, model_class)
            assert layer.temporal_model.config == config


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("num_heads", [4, 8])
def test_per_layer_slstm_specs_match_repeated_scalar_specs(scale: str, num_heads: int) -> None:
    config = SLSTMTemporalSequenceModelConfig(num_heads=num_heads)
    depth = SCALE_LAYOUTS[scale].actor_encoder_layers
    with torch.device("meta"):
        scalar = make_policy("tmasac__slstm", scale, use_nop=False, rmat_temporal_model_config=config)
        per_layer = make_policy(
            "tmasac__slstm", scale, use_nop=False,
            rmat_temporal_model_cls=[SLSTMTemporalSequenceModel] * depth,
            rmat_temporal_model_config=[config] * depth,
        )
    assert {name: parameter.shape for name, parameter in per_layer.named_parameters()} == {
        name: parameter.shape for name, parameter in scalar.named_parameters()
    }


@pytest.mark.parametrize("scale", ["5M-3L NOP1M", "10M-5L NOP2M"])
def test_mixed_tmasac_temporal_specs_support_eight_head_slstm_layers(scale: str) -> None:
    depth = SCALE_LAYOUTS[scale].actor_encoder_layers
    with torch.device("meta"):
        policy = make_policy(
            "tmasac__slstm", scale, use_nop=False,
            rmat_temporal_model_cls=[LSTMTemporalSequenceModel] * (depth - 1) + [SLSTMTemporalSequenceModel],
            rmat_temporal_model_config=[LSTMTemporalSequenceModelConfig()] * (depth - 1)
            + [SLSTMTemporalSequenceModelConfig(num_heads=8)],
        )
    assert isinstance(policy.actor_encoder.layers[0].temporal_model, LSTMTemporalSequenceModel)
    assert isinstance(policy.actor_encoder.layers[-1].temporal_model, SLSTMTemporalSequenceModel)


@pytest.mark.parametrize("variant", ["mat_ind__lstm", "mat_qcx__lstm"])
def test_recurrent_mat_rollout_uses_independent_actor_and_critic_temporal_specs(variant: str) -> None:
    policy = make_policy(
        variant, "5M-2L", use_nop=False,
        rmat_temporal_model_config=[LSTMTemporalSequenceModelConfig(), LSTMTemporalSequenceModelConfig()],
        rmat_critic_temporal_model_cls=[SLSTMTemporalSequenceModel, LSTMTemporalSequenceModel],
        rmat_critic_temporal_model_config=[SLSTMTemporalSequenceModelConfig(), LSTMTemporalSequenceModelConfig()],
    )
    actions, log_probs, values, state = policy.forward_with_temporal_state(
        local_obs=torch.randn(2, 3, 5), global_obs=torch.randn(2, 2),
        hidden_local_vars=torch.randn(2, 3, 2), hidden_global_vars=torch.randn(2, 1),
        agent_mask=torch.ones(2, 3, dtype=torch.bool), deterministic=True,
    )
    assert actions.shape == (2, 3, 3) and log_probs.shape == (2, 3) and values.shape == (2,)
    assert torch.isfinite(actions).all() and torch.isfinite(values).all()
    assert len(state["actor"]) == len(state["critic"]) == 2
    values.square().mean().backward()
    for layer in policy.critic_encoder.layers:
        assert any(parameter.grad is not None and parameter.grad.abs().sum() > 0
                   for parameter in layer.temporal_model.parameters())
    assert all(parameter.grad is None for parameter in policy.actor_encoder.parameters())


@pytest.mark.parametrize("variant", ["mat_ind__lstm", "mat_qcx__lstm", "tmatd3"])
@pytest.mark.parametrize("scale", [None, DEFAULT_MODEL_SCALE])
def test_recurrent_critics_inherit_shared_temporal_defaults(variant: str, scale: str | None) -> None:
    shared_config = LSTMTemporalSequenceModelConfig(bias=True)
    options = {"td3_recurrent_actor": True, "td3_recurrent_critic": True} if variant == "tmatd3" else {}
    with torch.device("meta"):
        policy = make_policy(
            variant, scale, use_nop=False, rmat_temporal_model_config=shared_config, **options,
        )
    critic_encoder = policy.critic.encoder if variant == "tmatd3" else policy.critic_encoder
    actor_encoder = policy.actor.encoder if variant == "tmatd3" else policy.actor_encoder
    for encoder in (actor_encoder, critic_encoder):
        assert all(layer.temporal_model.config == shared_config for layer in encoder.layers)


def test_recurrent_td3_preserves_critic_temporal_overrides() -> None:
    critic_config = SLSTMTemporalSequenceModelConfig()
    with torch.device("meta"):
        policy = make_policy(
            "tmatd3", use_nop=False, td3_recurrent_actor=True, td3_recurrent_critic=True,
            rmat_critic_temporal_model_cls=SLSTMTemporalSequenceModel,
            rmat_critic_temporal_model_config=critic_config,
        )
    assert all(isinstance(layer.temporal_model, LSTMTemporalSequenceModel) for layer in policy.actor.encoder.layers)
    for layer in policy.critic.encoder.layers:
        assert isinstance(layer.temporal_model, SLSTMTemporalSequenceModel)
        assert layer.temporal_model.config == critic_config


@pytest.mark.parametrize("scale", [None, *list_model_scales()])
@pytest.mark.parametrize("variant", ["tmasac", "tmasac__lstm", "tmasac__slstm"])
def test_tmasac_qcx_actor_head_produces_actions_at_every_scale(variant: str, scale: str | None) -> None:
    policy = make_policy(
        variant, scale, use_nop=False, tmasac_actor_head_kind="qcx", rmat_experimental_compile_lstm=False,
    )
    with torch.no_grad():
        actions, log_probs = policy.action_log_prob(
            local_obs=torch.randn(2, 3, 5), global_obs=torch.randn(2, 2),
            agent_mask=torch.ones(2, 3, dtype=torch.bool), deterministic=True,
        )
    assert policy.actor_head_kind is TMASACActorHeadKind.QCX
    assert actions.shape == (2, 3, 3)
    assert log_probs.shape == (2, 3)
    assert torch.isfinite(actions).all() and torch.isfinite(log_probs).all()


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
                layout.decoder_width,
                layout.qcx_ff_width,
                layout.decoder_width,
            )


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("recurrent", [False, True])
def test_mat_heads_use_identical_encoders_and_matched_non_attention_budgets(
    scale: str, recurrent: bool, wall_easy_actor_shape: None,
) -> None:
    variants = ("mat_ind__lstm", "mat_qcx__lstm") if recurrent else ("mat_ind", "mat_qcx", "mat_orig")
    with torch.device("meta"):
        policies = [make_policy(variant, scale, n_agents=5, use_nop=False) for variant in variants]
    reference = policies[0]
    reference_shapes = {name: parameter.shape for name, parameter in reference.actor_encoder.named_parameters()}
    head_counts = []
    for policy in policies:
        assert policy.config.encoder_config == reference.config.encoder_config
        assert {name: parameter.shape for name, parameter in policy.actor_encoder.named_parameters()} == reference_shapes
        encoder_count = sum(parameter.numel() for parameter in policy.actor_encoder.parameters())
        counts = count_policy_parameters(policy)
        actor_count = counts["roles"]["actor"]["trainable"]
        head_counts.append(actor_count - encoder_count - counts["processing"]["actor_decoder_attention_parameters"])
    assert max(head_counts) / min(head_counts) < 1.01
    assert policies[1].config.critic_config == reference.config.critic_config
    assert policies[1].critic_encoder_config == reference.critic_encoder_config


@pytest.mark.parametrize("scale", list_model_scales())
def test_actor_roles_and_encoder_head_depth_are_controlled_across_methods(
    scale: str, wall_easy_actor_shape: None,
) -> None:
    variants = ("mappo__deepset_critic", "mat_ind", "mat_dec", "mat_qcx", "mat_orig", "tmasac", "tmasac_dec",
                "masac__deepset_critic", "tmatd3", "tmatd3_dec", "matd3__deepset_critic", "maddpg__deepset_critic")
    with torch.device("meta"):
        policies = {variant: make_policy(variant, scale, n_agents=5, use_nop=False) for variant in variants}
    counts = {variant: count_policy_parameters(policy) for variant, policy in policies.items()}
    actor_counts = [
        count["roles"]["actor"]["trainable"] - count["processing"]["actor_decoder_attention_parameters"]
        for count in counts.values()
    ]
    assert max(actor_counts) / min(actor_counts) < 1.015
    target = SCALE_LAYOUTS[scale].main_parameters * 2 // 5
    assert all(abs(count / target - 1) < 0.015 for count in actor_counts)
    encoder_layers = SCALE_LAYOUTS[scale].layers
    for variant in variants:
        policy = policies[variant]
        encoder = policy.actor_encoder if hasattr(policy, "actor_encoder") else policy.actor.encoder
        encoder_count = sum(parameter.numel() for parameter in encoder.parameters())
        base_actor_count = (
            counts[variant]["roles"]["actor"]["trainable"]
            - counts[variant]["processing"]["actor_decoder_attention_parameters"]
        )
        assert 0.8 < encoder_count / base_actor_count < 0.92
        assert len(encoder.layers) == encoder_layers
        if variant == "mappo__deepset_critic":
            assert len(policy.actor.mlp.layers) == SCALE_LAYOUTS[scale].actor_head_blocks
        elif variant in ("mat_qcx", "mat_orig"):
            blocks = policy.decoder.layers if variant == "mat_qcx" else policy.decoder.blocks
            assert len(blocks) == 1
        else:
            head = policy.actor_head if hasattr(policy, "actor_head") else policy.actor.head
            if not isinstance(head, ResidualMLP):
                head = head.head
            assert len(head.layers) == SCALE_LAYOUTS[scale].actor_head_blocks
        reference_critic = counts["mat_ind"]["roles"]["critic"]["trainable"]
        critic_count = counts[variant]["roles"]["critic"]["trainable"]
        if hasattr(policy.critic, "q2"):
            critic_count -= sum(parameter.numel() for parameter in policy.critic.q2.parameters())
        assert abs(critic_count / reference_critic - 1) < 0.01
    assert policies["mat_ind"].config.residual_mlp_config == (
        policies["mappo__deepset_critic"].config.actor_config.residual_mlp_config
    )


@pytest.mark.parametrize("scale", list_model_scales())
def test_mappo_critic_uses_common_deepset_blocks_and_mat_value_head(scale: str) -> None:
    with torch.device("meta"):
        mappo = make_policy("mappo__deepset_critic", scale, use_nop=False)
        masac = make_policy("masac__deepset_critic", scale, use_nop=False)
        mat = make_policy("mat_ind", scale, use_nop=False)
    for source, reference in (
        (mappo.critic_encoder.layers, masac.critic.encoder.network.layers),
        (mappo.critic, mat.critic),
    ):
        assert {name: parameter.shape for name, parameter in source.named_parameters()} == {
            name: parameter.shape for name, parameter in reference.named_parameters()
        }
    assert all(layer.self_attn is None for layer in mappo.critic_encoder.layers)
    assert mappo.critic_encoder.local_obs_dim == mappo.local_obs_dim + mappo.hidden_local_vars_dim
    assert mappo.critic_encoder.global_obs_dim == mappo.global_obs_dim + mappo.hidden_global_vars_dim
    assert mappo.nop_latent_dim == SCALE_LAYOUTS[scale].critic_width


def test_ind_and_mappo_heads_use_common_initialization_gains() -> None:
    policies = [make_policy(variant, "2.5M", use_nop=False) for variant in ("mat_ind", "mappo__deepset_critic")]
    for policy in policies:
        head = policy.actor_head if hasattr(policy, "actor_head") else policy.actor.mlp
        for layer in head.modules():
            if isinstance(layer, nn.Linear):
                assert layer.weight.square().sum().item() / min(layer.weight.shape) == pytest.approx(1.0, rel=1e-5)
        for layer in policy.action_dist.modules():
            if isinstance(layer, nn.Linear):
                assert layer.weight.square().sum().item() / min(layer.weight.shape) == pytest.approx(0.01 ** 2, rel=1e-5)


@pytest.mark.parametrize("scale", ["5M-2L", "10M-5L"])
def test_mappo_actor_features_retain_scale_and_gradients_through_deep_encoders(scale: str) -> None:
    policy = make_policy("mappo__deepset_critic", scale, use_nop=False)
    local_obs = torch.randn(4, 3, 5, requires_grad=True)
    latents = policy.actor_encoder(local_obs, torch.randn(4, 2))
    assert 0.9 < latents.std().item() < 1.1
    head_output = policy.actor(latents)
    assert head_output.std().item() > 1e-3
    head_output[..., 0].sum().backward()
    assert local_obs.grad is not None and torch.isfinite(local_obs.grad).all()
    assert local_obs.grad.abs().max().item() > 1e-6


def test_nop_layout_is_common_and_does_not_change_main_weights():
    torch.manual_seed(72)
    enabled = make_policy("mappo__deepset_critic")
    torch.manual_seed(72)
    disabled = make_policy("mappo__deepset_critic", use_nop=False)
    torch.testing.assert_close(enabled.policy.state_dict(), disabled.state_dict(), rtol=0, atol=0)
    with torch.device("meta"):
        policies = [make_policy(name) for name in ("tmasac", "tmasac__lstm", "matd3__deepset_critic")]
    for policy in policies:
        assert policy.config.nop_config.transition_model_d_model == enabled._wm_pre_transition_dims[-1] == 192
        assert policy.config.nop_config.latent_projection_hidden_dims == [256, 256]
        assert policy.config.nop_config.transition_model_num_layers == 2


@pytest.mark.parametrize("scale", [scale for scale in list_model_scales() if scale.startswith("10M-")])
@pytest.mark.parametrize("variant", list_variants())
def test_ten_million_nop_balances_wider_attention_and_expanding_mlp(variant: str, scale: str) -> None:
    with torch.device("meta"):
        policy = make_policy(variant, scale)
        without_nop = make_policy(variant, scale, use_nop=False)
        smaller = make_policy(variant, "5M-2L")
    nop = policy if hasattr(policy, "transition_model") else policy.critic_nop
    smaller_nop = smaller if hasattr(smaller, "transition_model") else smaller.critic_nop
    transition = nop.transition_model
    assert transition.d_model == transition.latent_dim == 256
    assert transition.d_model > smaller_nop.transition_model.d_model
    assert transition.nhead == 4 and len(transition.encoder.layers) == 3
    for layer in transition.encoder.layers:
        assert (layer.linear1.in_features, layer.linear1.out_features) == (256, 384)
        assert (layer.linear2.in_features, layer.linear2.out_features) == (384, 256)
        attention_count = sum(parameter.numel() for parameter in layer.self_attn.parameters())
        smaller_attention = smaller_nop.transition_model.encoder.layers[0].self_attn
        assert attention_count > 1.7 * sum(parameter.numel() for parameter in smaller_attention.parameters())
    counts = count_policy_parameters(policy)
    assert 1_950_000 < counts["roles"]["next_obs_prediction"]["trainable"] < 2_100_000
    assert counts["trainable_minus_nop"] == count_policy_parameters(without_nop)["trainable_minus_nop"]
    assert transition.get_hyper_parameters()["num_feedforward_hidden_layers"] == 1


@pytest.mark.parametrize("scale", list_model_scales())
@pytest.mark.parametrize("variant", [
    "mappo__deepset_critic", "mat_ind", "mat_qcx", "mat_dec", "mat_ind__lstm", "mat_qcx__lstm",
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
    joint_config = metadata.get("joint_critic_config", {})
    critic_config = (
        joint_config["head_config"] if metadata.get("critic_kind", joint_config.get("kind")) == "deepset"
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
        for variant in ["ppo", "mappo__deepset_critic", "mat_orig", "mat_ind", "mat_dec", "mat_qcx", "mat_ind__lstm", "mat_qcx__lstm", "tmasac__slstm", "tmasac__lstm"]
    ]
    + [
        (variant, scale)
        for scale in list_model_scales()
        if scale != DEFAULT_MODEL_SCALE
        for variant in ("mappo__deepset_critic", "mat_qcx__lstm", "tmasac__slstm", "tmasac__lstm")
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
