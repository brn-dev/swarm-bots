"""Explicit architecture layouts for the supported 2.5M, 5M and 10M scales.

No parameter counting, width search, or task-dependent resizing happens here.
Counts are audited separately by the inspector.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from swarmbots.learn.algos.mat.mat_dec_policy import MATDecPolicyConfig
from swarmbots.learn.algos.mat_qc_base_policy import MATCriticConfig
from swarmbots.learn.algos.off_policy.actor_heads import TMASACActorHeadKind
from swarmbots.learn.algos.off_policy.actor_state_critic import ActorStateCriticInputConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig
from swarmbots.learn.algos.xlstm.slstm import SLSTMTemporalSequenceModel
from swarmbots.learn.nn_components.feed_forward import MLPConfig
from swarmbots.learn.nn_components.deep_set import DeepSetInteractionConfig

DEFAULT_MODEL_SCALE = "5M-2L NOP1M"


@dataclass(frozen=True)
class ScaleLayout:
    name: str
    main_parameters: int
    nop_parameters: int
    version: str
    width: int
    mlp_latent_width: int
    ff_width: int
    layers: int
    decentralized_layers: int
    private_layers: int
    private_ff_width: int
    private_critic_ff: int
    autoregressive_actor_width: int
    autoregressive_actor_ff: int
    head_width: int
    decoder_width: int
    qcx_decoder_width: int
    qcx_ff_width: int
    critic_width: int
    critic_ff: int
    state_bridge_width: int
    single_joint_width: int
    twin_joint_width: int
    deepset_v_width: int
    deepset_dec_width: int
    deepset_q_head_width: int
    gated_width: int
    ppo_backbone: tuple[int, ...]
    mappo_actor_backbone: tuple[int, ...]
    mappo_critic_backbone: tuple[int, ...]
    private_actor: tuple[int, ...]
    shared_v: tuple[int, ...]
    separate_v_regressor: tuple[int, ...]
    orig_v: tuple[int, ...]
    nop_width: int
    nop_ff: int
    nop_layers: int
    nop_projection: tuple[int, ...]
    nop_predictors: tuple[int, ...]
    decentralized_ff_width: int | None = None
    decoder_layers: int | None = None
    temporal_hidden_dim: int | None = None
    autoregressive_temporal_hidden_dim: int | None = None
    lstm_actor_temporal_hidden_dim: int | None = None
    slstm_actor_temporal_hidden_dim: int | None = None

    @property
    def slug(self) -> str:
        return self.name.split(" ", 1)[0]


# These are reviewed declarations, not widths generated from a multiplier.
# Small NOP retains the 256-wide bridge and 192-wide prediction MLPs and
# uses FF=512 at D=128 to avoid shrinking all auxiliary processing together.
SCALE_LAYOUTS = {
    "2.5M NOP0.75M": ScaleLayout(
        name="2.5M NOP0.75M",
        main_parameters=2_500_000,
        nop_parameters=750_000,
        version="2.5+0.75M-v7",
        width=192,
        mlp_latent_width=192,
        ff_width=384,
        layers=2,
        decentralized_layers=3,
        private_layers=1,
        private_ff_width=384,
        private_critic_ff=384,
        autoregressive_actor_width=128,
        autoregressive_actor_ff=256,
        head_width=128,
        decoder_width=128,
        qcx_decoder_width=128,
        qcx_ff_width=128,
        critic_width=256,
        critic_ff=384,
        state_bridge_width=192,
        single_joint_width=512,
        twin_joint_width=384,
        deepset_v_width=192,
        deepset_dec_width=128,
        deepset_q_head_width=128,
        gated_width=256,
        ppo_backbone=(512, 512, 384, 384),
        mappo_actor_backbone=(512, 384, 384),
        mappo_critic_backbone=(512, 512, 384, 384),
        private_actor=(512, 384, 384),
        shared_v=(768, 384),
        separate_v_regressor=(512, 384),
        orig_v=(512, 512, 384),
        nop_width=128,
        nop_ff=512,
        nop_layers=2,
        nop_projection=(256, 256),
        nop_predictors=(192, 192),
    ),
    "5M-2L NOP1M": ScaleLayout(
        name="5M-2L NOP1M",
        main_parameters=5_000_000,
        nop_parameters=1_000_000,
        version="5M-2L+1M-v2",
        width=256,
        mlp_latent_width=256,
        ff_width=512,
        layers=2,
        decentralized_layers=3,
        private_layers=1,
        private_ff_width=512,
        private_critic_ff=512,
        autoregressive_actor_width=192,
        autoregressive_actor_ff=384,
        head_width=128,
        decoder_width=192,
        qcx_decoder_width=192,
        qcx_ff_width=256,
        critic_width=384,
        critic_ff=512,
        state_bridge_width=256,
        single_joint_width=768,
        twin_joint_width=512,
        deepset_v_width=256,
        deepset_dec_width=192,
        deepset_q_head_width=128,
        gated_width=384,
        ppo_backbone=(768, 768, 512, 512),
        mappo_actor_backbone=(768, 512, 512, 384),
        mappo_critic_backbone=(1024, 768, 512),
        private_actor=(768, 512, 512),
        shared_v=(1024, 512),
        separate_v_regressor=(768, 512),
        orig_v=(768, 768, 512),
        nop_width=192,
        nop_ff=384,
        nop_layers=2,
        nop_projection=(256, 256),
        nop_predictors=(192, 192),
    ),
}

SCALE_LAYOUTS.update({
    "5M-3L NOP1M": replace(
        SCALE_LAYOUTS[DEFAULT_MODEL_SCALE],
        name="5M-3L NOP1M",
        version="5M-3L+1M-v3",
        layers=3,
        decentralized_layers=3,
        ff_width=320,
        decentralized_ff_width=512,
        autoregressive_actor_ff=256,
        decoder_layers=2,
        critic_width=320,
        critic_ff=384,
        temporal_hidden_dim=192,
        autoregressive_temporal_hidden_dim=136,
        lstm_actor_temporal_hidden_dim=192,
        slstm_actor_temporal_hidden_dim=168,
    ),
    "10M-3L NOP2M": ScaleLayout(
        name="10M-3L NOP2M",
        main_parameters=10_000_000,
        nop_parameters=2_000_000,
        version="10M-3L+2M-v1",
        width=256,
        mlp_latent_width=384,
        ff_width=768,
        layers=3,
        decentralized_layers=4,
        private_layers=2,
        private_ff_width=512,
        private_critic_ff=1024,
        autoregressive_actor_width=192,
        autoregressive_actor_ff=512,
        head_width=192,
        decoder_width=256,
        qcx_decoder_width=192,
        qcx_ff_width=512,
        critic_width=384,
        critic_ff=1536,
        state_bridge_width=384,
        single_joint_width=1024,
        twin_joint_width=768,
        deepset_v_width=384,
        deepset_dec_width=256,
        deepset_q_head_width=192,
        gated_width=512,
        ppo_backbone=(1024, 1024, 768, 768),
        mappo_actor_backbone=(1024, 768, 768, 512),
        mappo_critic_backbone=(1024, 1024, 768),
        private_actor=(1024, 768, 768),
        shared_v=(1536, 768),
        separate_v_regressor=(1024, 768),
        orig_v=(1024, 1024, 768),
        nop_width=192,
        nop_ff=768,
        nop_layers=3,
        nop_projection=(384, 384),
        nop_predictors=(256, 256),
    ),
})

SCALE_LAYOUTS.update({
    "10M-4L NOP2M": replace(
        SCALE_LAYOUTS["10M-3L NOP2M"],
        name="10M-4L NOP2M",
        version="10M-4L+2M-v2",
        layers=4,
        ff_width=608,
        decentralized_ff_width=768,
        autoregressive_actor_ff=400,
        decoder_layers=3,
        critic_ff=960,
        temporal_hidden_dim=204,
        autoregressive_temporal_hidden_dim=144,
        lstm_actor_temporal_hidden_dim=192,
        slstm_actor_temporal_hidden_dim=144,
    ),
    "10M-5L NOP2M": replace(
        SCALE_LAYOUTS["10M-3L NOP2M"],
        name="10M-5L NOP2M",
        version="10M-5L+2M-v2",
        layers=5,
        ff_width=480,
        decentralized_ff_width=768,
        autoregressive_actor_ff=320,
        decoder_layers=3,
        critic_ff=608,
        temporal_hidden_dim=184,
        autoregressive_temporal_hidden_dim=128,
        lstm_actor_temporal_hidden_dim=176,
        slstm_actor_temporal_hidden_dim=136,
    ),
})


def list_model_scales() -> tuple[str, ...]:
    return tuple(SCALE_LAYOUTS)


def normalize_model_scale(value: str | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        compact = "".join(value.upper().split())
        for name in SCALE_LAYOUTS:
            main, nop = name.split(" NOP")
            if compact in {name.replace(" ", ""), main, main.removesuffix("M") + "+" + nop, main + "+" + nop}:
                return name
    raise ValueError(
        f"Supported fixed model scales: {', '.join(SCALE_LAYOUTS)}; use model_scale=None for custom widths"
    )


def _layout(scale: str) -> ScaleLayout:
    return SCALE_LAYOUTS[normalize_model_scale(scale)]


def _encoder(
    config: Any, width: int, ff: list[int], layers: int, layout: ScaleLayout,
    *, temporal_hidden_dim: int | None = None,
) -> Any:
    existing_ff = config.transformer_ff_config
    if existing_ff is not None and hasattr(existing_ff, "hidden_dim"):
        ff_config = replace(existing_ff, hidden_dim=layout.gated_width)
    else:
        ff_config = MLPConfig(hidden_dims=ff)
    temporal_options = (
        {"temporal_hidden_dim": temporal_hidden_dim}
        if hasattr(config, "temporal_hidden_dim") else {}
    )
    return replace(
        config,
        d_model=width,
        nhead=4,
        num_layers=layers,
        dim_feedforward=ff[0],
        transformer_ff_config=ff_config,
        local_obs_encoder_config=MLPConfig(hidden_dims=[width, width]),
        global_obs_encoder_config=MLPConfig(hidden_dims=[width]),
        **temporal_options,
    )


def _decoder(config: Any, layout: ScaleLayout) -> Any:
    width = layout.qcx_decoder_width if hasattr(config, "dim_feedforward") else layout.decoder_width
    updates = dict(d_model=width, nhead=4, num_layers=layout.decoder_layers or layout.layers)
    if hasattr(config, "dim_feedforward"):
        updates.update(
            dim_feedforward=layout.qcx_ff_width, context_encoder_hidden_dims=[width], action_encoder_dims=[width, width]
        )
    if hasattr(config, "latent_pi_dim"):
        updates["latent_pi_dim"] = width
    return replace(config, **updates)


def _nop(config: Any, layout: ScaleLayout) -> Any:
    return replace(
        config,
        nop_latent_dim=layout.nop_width,
        latent_projection_hidden_dims=list(layout.nop_projection),
        pre_predictors_hidden_dims=list(layout.nop_predictors),
        transition_model_d_model=layout.nop_width,
        transition_model_nhead=4,
        transition_model_num_layers=layout.nop_layers,
        transition_model_dim_feedforward=layout.nop_ff,
        transition_model_coembed_hidden_dims=[layout.nop_width],
    )


def _interaction_at_width(
    config: DeepSetInteractionConfig, width: int, ff_width: int | None = None,
) -> DeepSetInteractionConfig:
    return replace(
        config,
        message_dim=width * 5 // 4 if config.message_dim is None else config.message_dim,
        hidden_dims=(ff_width or width,) if config.hidden_dims is None else config.hidden_dims,
        gate_hidden_dim=max(1, width // 4) if config.gated and config.gate_hidden_dim is None else config.gate_hidden_dim,
    )


def ppo_nop_at_scale(config: NOPWorldModelConfig, scale: str = DEFAULT_MODEL_SCALE) -> NOPWorldModelConfig:
    """Same projection/transition/prediction widths as off-policy NOP."""
    layout = _layout(scale)
    return replace(
        config,
        wm_pre_transition_dims=[*layout.nop_projection, layout.nop_width],
        wm_pre_predictors_dims=list(layout.nop_predictors),
        d_model_transition_model=layout.nop_width,
        nhead_transition_model=4,
        num_layers_transition_model=layout.nop_layers,
        dim_feedforward_transition_model=layout.nop_ff,
        transition_model_coembed_hidden_dims=[layout.nop_width],
    )


def _mlp_on_policy(config: Any, layout: ScaleLayout) -> Any:
    actor = config.actor_config
    if hasattr(actor, "latent_pi_dim_per_agent"):
        actor = replace(
            actor,
            hidden_dims=list(layout.ppo_backbone),
            shared_encoder_latent_dim_per_agent=layout.mlp_latent_width,
            actor_head_hidden_dims=list(layout.private_actor[:-1]),
            latent_pi_dim_per_agent=layout.private_actor[-1],
        )
        critic = replace(config.critic_config, hidden_dims=list(layout.shared_v))
    else:
        actor = replace(
            actor,
            hidden_dims=list(layout.mappo_actor_backbone),
            latent_dim=layout.mlp_latent_width,
            actor_head_hidden_dims=list(layout.private_actor[:-1]),
            latent_pi_dim=layout.private_actor[-1],
        )
        critic = config.critic_config
        if critic.deep_set_config is None:
            critic = replace(critic, mlp_hidden_dims=list(layout.shared_v))
        else:
            critic = replace(
                critic,
                deep_set_config=replace(
                    critic.deep_set_config,
                    local_projection_hidden_dims=[layout.deepset_v_width],
                    value_regressor_hidden_dims=list(layout.shared_v),
                    interaction_config=_interaction_at_width(
                        critic.deep_set_config.interaction_config, layout.deepset_v_width,
                    ),
                ),
            )
        return replace(
            config,
            actor_config=actor,
            critic_encoder_config=replace(
                config.critic_encoder_config,
                hidden_dims=list(layout.mappo_critic_backbone),
                latent_dim=layout.mlp_latent_width,
            ),
            critic_config=critic,
        )
    return replace(config, actor_config=actor, critic_config=critic)


def _mat(config: Any, layout: ScaleLayout) -> Any:
    decentralized = isinstance(config, MATDecPolicyConfig)
    autoregressive = hasattr(config, "decoder_config")
    actor_width = layout.autoregressive_actor_width if autoregressive else layout.width
    actor_ff = layout.autoregressive_actor_ff if autoregressive else layout.ff_width
    if decentralized:
        actor_ff = layout.decentralized_ff_width or actor_ff
    actor = _encoder(
        config.encoder_config, actor_width, [actor_ff, actor_ff],
        layout.decentralized_layers if decentralized else layout.layers, layout,
        temporal_hidden_dim=(
            layout.autoregressive_temporal_hidden_dim if autoregressive else layout.temporal_hidden_dim
        ),
    )
    critic_source = config.critic_encoder_config or config.encoder_config
    critic_encoder = _encoder(
        critic_source,
        layout.width,
        [layout.ff_width, layout.ff_width],
        layout.layers, layout,
        temporal_hidden_dim=layout.temporal_hidden_dim,
    )
    critic_encoder = replace(critic_encoder, use_agent_attention=True)
    updates = {"encoder_config": actor, "critic_encoder_config": critic_encoder}
    critic = config.critic_config
    if isinstance(critic, MATCriticConfig):
        critic = replace(
            critic,
            n_local_projection_hidden_layers=1,
            n_value_regressor_hidden_layers=2,
            local_projection_hidden_dims=[layout.deepset_dec_width],
            value_regressor_hidden_dims=list(layout.separate_v_regressor),
            interaction_config=_interaction_at_width(critic.interaction_config, layout.deepset_dec_width),
        )
    else:
        critic = replace(critic, value_head_hidden_dims=list(layout.orig_v))
    updates["critic_config"] = critic
    if autoregressive:
        updates["decoder_config"] = _decoder(config.decoder_config, layout)
    if hasattr(config, "actor_head_hidden_dims"):
        updates["actor_head_hidden_dims"] = [layout.head_width]
    return replace(config, **updates)


def _off_policy(config: Any, layout: ScaleLayout) -> Any:
    actor_config = config.actor_encoder_config
    shared = getattr(config, "shared_encoder_config", None) is not None
    critic_kind = (
        getattr(config, "critic_kind", config.joint_critic_config.kind)
        if hasattr(config, "joint_critic_config") else "transformer"
    )
    joint = critic_kind != "transformer"
    decentralized = (
        joint
        or not actor_config.use_agent_attention
        or TMASACActorHeadKind(config.actor_head_config.kind) is TMASACActorHeadKind.DECENTRALIZED
    )
    if decentralized:
        actor_config = replace(actor_config, use_agent_attention=False)
    actor_temporal_hidden_dim = (
        layout.slstm_actor_temporal_hidden_dim
        if getattr(actor_config, "temporal_model_cls", None) is SLSTMTemporalSequenceModel
        else layout.lstm_actor_temporal_hidden_dim
    )
    if shared:
        actor = _encoder(
            actor_config,
            layout.width,
            [layout.private_ff_width, layout.private_ff_width],
            layout.private_layers,
            layout,
            temporal_hidden_dim=actor_temporal_hidden_dim,
        )
    else:
        actor_ff = (layout.decentralized_ff_width or layout.ff_width) if decentralized else layout.ff_width
        actor = _encoder(
            actor_config,
            layout.width,
            [actor_ff, actor_ff],
            layout.decentralized_layers if decentralized else layout.layers,
            layout,
            temporal_hidden_dim=actor_temporal_hidden_dim,
        )
    head = replace(
        config.actor_head_config,
        hidden_dims=[layout.head_width],
        qcx_decoder_config=_decoder(config.actor_head_config.qcx_decoder_config, layout),
    )
    critic_name = "transformer_critic_config" if hasattr(config, "transformer_critic_config") else "critic_config"
    critic = getattr(config, critic_name)
    critic_encoder_source = config.critic_encoder_config
    if critic_kind == "deepset":
        joint_config = config.joint_critic_config
        # Missing nested configs retain the existing fixed-scale templates.
        if joint_config.encoder_config is not None:
            act_fn_cls = getattr(config, "act_fn_cls", critic_encoder_source.act_fn_cls)
            critic_encoder_source = joint_config.resolved_encoder_config(act_fn_cls)
        if joint_config.head_config is not None:
            critic = joint_config.resolved_head_config()
    recurrent_critic = bool(getattr(config, "recurrent_critic", False))
    critic_encoder = _encoder(
        critic_encoder_source,
        layout.critic_width,
        [layout.private_critic_ff if shared else layout.critic_ff],
        layout.private_layers if shared else layout.layers,
        layout,
        temporal_hidden_dim=layout.temporal_hidden_dim,
    )
    head_interaction = _interaction_at_width(critic.interaction_config, layout.deepset_q_head_width)
    critic = replace(
        critic,
        local_projection_hidden_dims=[layout.deepset_q_head_width],
        value_regressor_hidden_dims=[layout.deepset_q_head_width, layout.deepset_q_head_width],
        interaction_config=head_interaction,
    )
    updates = {
        "actor_encoder_config": actor,
        "actor_head_config": head,
        "critic_encoder_config": critic_encoder,
        "nop_config": _nop(config.nop_config, layout),
        critic_name: critic,
    }
    if shared:
        updates["shared_encoder_config"] = _encoder(
            config.shared_encoder_config, layout.width, [layout.ff_width, layout.ff_width], layout.layers, layout,
            temporal_hidden_dim=actor_temporal_hidden_dim,
        )
    if joint:
        width = layout.single_joint_width if getattr(config, "n_critics", 2) == 1 else layout.twin_joint_width
        joint_config = config.joint_critic_config
        updates["joint_critic_config"] = replace(
            joint_config,
            kind=critic_kind,
            hidden_dims=(layout.deepset_q_head_width, layout.deepset_q_head_width) if critic_kind == "deepset"
            else (width, width, width),
            element_hidden_dims=(layout.critic_width,) if critic_kind == "deepset" else (width, width, width),
            encoder_config=critic_encoder if critic_kind == "deepset" else None,
            head_config=critic if critic_kind == "deepset" else None,
            interaction_config=replace(
                _interaction_at_width(joint_config.interaction_config, layout.critic_width, layout.critic_ff),
                num_layers=layout.layers,
            ) if critic_kind == "deepset" else joint_config.interaction_config,
        )
    if hasattr(config, "actor_state_critic_input_config") and isinstance(
        config.actor_state_critic_input_config, ActorStateCriticInputConfig
    ):
        updates["actor_state_critic_input_config"] = replace(
            config.actor_state_critic_input_config,
            projection_dim=layout.state_bridge_width,
            projection_hidden_dims=(layout.state_bridge_width,),
        )
    elif getattr(config, "actor_state_critic_input_config", None) == "auto":
        updates["actor_state_critic_input_config"] = (
            None
            if recurrent_critic
            else ActorStateCriticInputConfig(
                projection_dim=layout.state_bridge_width,
                projection_hidden_dims=(layout.state_bridge_width,),
            )
        )
    return replace(config, **updates)


def model_config_at_scale(config: Any, scale: str = DEFAULT_MODEL_SCALE) -> Any:
    """Choose a reviewed layout by architecture, never by observed counts."""
    layout = _layout(scale)
    if hasattr(config, "actor_config"):
        return _mlp_on_policy(config, layout)
    if hasattr(config, "actor_encoder_config") and not isinstance(config, MATDecPolicyConfig):
        return _off_policy(config, layout)
    return _mat(config, layout)


def make_policy_at_scale(policy_class: type, env: Any, config: Any, scale: str) -> Any:
    layout = _layout(scale)
    resolved = model_config_at_scale(config, scale)
    policy = policy_class(env=env, config=resolved)
    shared = (
        (hasattr(config, "actor_config") and not hasattr(config, "critic_encoder_config"))
        or getattr(config, "shared_encoder_config", None) is not None
    )
    policy.model_scale = {
        "name": layout.name,
        "layout_version": layout.version,
        "requested": {"policy": layout.main_parameters, "nop": layout.nop_parameters},
        "role_targets": {
            "shared_encoder": layout.main_parameters * 2 // 5 if shared else 0,
            "actor": layout.main_parameters // 5 if shared else layout.main_parameters * 2 // 5,
            "critic": layout.main_parameters * 2 // 5 if shared else layout.main_parameters * 3 // 5,
        },
        "attributed_targets": {"actor": layout.main_parameters * 2 // 5, "critic": layout.main_parameters * 3 // 5},
    }
    return policy
