"""Construct policy families from resolved preset options."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Literal, TypedDict

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.hybrid_action_dist import ContinuousActionDistConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActorConfig
from swarmbots.learn.algos.mappo.mappo_policy import MAPPOPolicy, MAPPOPolicyConfig
from swarmbots.learn.algos.mat import MATEncoderConfig, MLPConfig
from swarmbots.learn.algos.mat.mat_dec_policy import MATDecPolicy, MATDecPolicyConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.algos.mat_orig.mat_orig_decoder import MATOrigDecoderConfig
from swarmbots.learn.algos.mat_orig.mat_orig_policy import MATOrigCriticConfig, MATOrigPolicy, MATOrigPolicyConfig
from swarmbots.learn.algos.mat_qc_base_policy import MATCriticConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_policy import MATQCXPolicy, MATQCXPolicyConfig
from swarmbots.learn.algos.off_policy.actor_heads import (
    TMASACActorHeadConfig,
    TMASACActorHeadKind,
)
from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig
from swarmbots.learn.algos.off_policy.nop import SACNOPConfig
from swarmbots.learn.algos.off_policy.transformer_critic import TransformerCriticConfig
from swarmbots.learn.algos.ppo.ppo_policy import (
    PopArtConfig,
    PPOActorConfig,
    PPOCriticConfig,
    PPOPolicy,
    PPOPolicyConfig,
)
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoderConfig
from swarmbots.learn.algos.r_mat.r_mat_ind_policy import RMATIndPolicy, RMATIndPolicyConfig
from swarmbots.learn.algos.r_mat.r_mat_qcx_policy import RMATQCXPolicy, RMATQCXPolicyConfig
from swarmbots.learn.algos.sac.masac_policy import MASACPolicy, MASACPolicyConfig
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import (
    ActorStateCriticInputConfig,
    RecurrentTMASACPolicy,
    RecurrentTMASACPolicyConfig,
)
from swarmbots.learn.algos.sac.segment_tmasac_policy import SegmentTMASACPolicy
from swarmbots.learn.algos.sac.tmasac_policy import TMASACPolicy, TMASACPolicyConfig
from swarmbots.learn.algos.td3.recurrent_td3_policy import RecurrentTD3Policy, RecurrentTD3PolicyConfig
from swarmbots.learn.algos.td3.td3_policy import TD3Policy, TD3PolicyConfig
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSetInteractionConfig
from swarmbots.learn.presets.transformer import (
    MATInitGains,
    MATNormalizationConfig,
)


class ModuleOptions(TypedDict):
    compile_modules: bool
    compile_mode: str
    action_net_init_gain: float


class DistributionOptions(TypedDict):
    continuous_config: ContinuousActionDistConfig | None
    bernoulli_config: BernoulliConfig


class TokenPolicyOptions(ModuleOptions, DistributionOptions):
    dropout: float
    act_fn_cls: ActivationFactory
    max_agents: int


@dataclass(frozen=True)
class PolicyBuildContext:
    continuous_config: ContinuousActionDistConfig | None
    bernoulli_config: BernoulliConfig
    use_popart: bool
    popart_config: PopArtConfig
    act_fn_cls: ActivationFactory
    init_gains: MATInitGains
    normalization: MATNormalizationConfig
    compile_modules: bool
    compile_mode: str
    model_scale: str | None

    def build(self, policy_class: type[BasePolicy], env: Any, config: Any) -> BasePolicy:
        if self.model_scale is None:
            return policy_class(env=env, config=config)
        from swarmbots.learn.presets.model_scale import make_policy_at_scale

        return make_policy_at_scale(policy_class, env, config, self.model_scale)

    def module_options(self) -> ModuleOptions:
        return ModuleOptions(
            compile_modules=self.compile_modules,
            compile_mode=self.compile_mode,
            action_net_init_gain=self.init_gains.action_net,
        )

    def distribution_options(self) -> DistributionOptions:
        return DistributionOptions(continuous_config=self.continuous_config, bernoulli_config=self.bernoulli_config)


@dataclass(frozen=True)
class PolicyEncoders:
    feedforward: MATEncoderConfig
    recurrent: RMATEncoderConfig
    recurrent_actor: RMATEncoderConfig
    recurrent_critic: RMATEncoderConfig


@dataclass(frozen=True)
class DecoderOptions:
    d_model: int
    nhead: int
    add_agent_embeddings: bool
    assume_agent_mask_is_active_prefix: bool

    def qcx(self, context: PolicyBuildContext, *, include_actor_head: bool) -> MATQCXDecoderConfig:
        gains = context.init_gains
        normalization = context.normalization
        config = MATQCXDecoderConfig(
            d_model=self.d_model,
            nhead=self.nhead,
            num_layers=2,
            dim_feedforward=self.d_model * 2,
            token_encoder_init_gain=gains.decoder_token_encoder,
            token_encoder_projection_init_gain=gains.decoder_token_encoder_projection,
            transformer_ff_init_gain=gains.decoder_transformer_ff,
            context_encoder_hidden_dims=[self.d_model],
            action_encoder_dims=[self.d_model, self.d_model],
            memory_dims=None,
            normalize_context_input=normalization.normalize_context_input,
            normalize_context_tokens=normalization.normalize_context_tokens,
            normalize_action_input=normalization.normalize_action_input,
            normalize_memory_input=normalization.normalize_memory_input,
            normalize_action_tokens=normalization.normalize_action_tokens,
            normalize_memory_tokens=normalization.normalize_memory_tokens,
            assume_agent_mask_is_active_prefix=self.assume_agent_mask_is_active_prefix,
        )
        if include_actor_head:
            config = replace(
                config,
                add_agent_embeddings=self.add_agent_embeddings,
                actor_head_init_gain=gains.actor_head,
                normalize_actor_head_input=normalization.normalize_actor_head_input,
            )
        return config


@dataclass(frozen=True)
class OffPolicyOptions:
    nop_config: SACNOPConfig
    td3_recurrent_actor: bool
    td3_recurrent_critic: bool
    td3_actor_state_critic_input_config: ActorStateCriticInputConfig | Literal["auto"] | None
    tmasac_recurrent_shared_encoder: bool
    tmasac_shared_encoder_num_layers: int | None
    tmasac_actor_encoder_num_layers: int
    tmasac_critic_encoder_num_layers: int
    tmasac_actor_head_kind: TMASACActorHeadKind | str
    tmasac_actor_state_critic_input_config: ActorStateCriticInputConfig | None
    rmat_experimental_compile_lstm: bool


def build_td3_policy(
    *,
    env: Any,
    policy_variant: str,
    context: PolicyBuildContext,
    encoders: PolicyEncoders,
    decoder: DecoderOptions,
    options: OffPolicyOptions,
    critic_config: TransformerCriticConfig,
    joint_critic_config: JointCriticConfig,
) -> BasePolicy:
    policy_class = RecurrentTD3Policy if options.td3_recurrent_actor else TD3Policy
    config_class = RecurrentTD3PolicyConfig if options.td3_recurrent_actor else TD3PolicyConfig
    recurrent_options = (
        {
            "recurrent_critic": options.td3_recurrent_critic,
            "actor_state_critic_input_config": options.td3_actor_state_critic_input_config,
        }
        if options.td3_recurrent_actor
        else {}
    )
    return context.build(
        policy_class,
        env,
        config_class(
            **recurrent_options,
            actor_encoder_config=replace(
                encoders.recurrent_actor if options.td3_recurrent_actor else encoders.feedforward,
                use_agent_attention=policy_variant == "tmatd3",
            ),
            actor_head_config=TMASACActorHeadConfig(
                hidden_dims=[decoder.d_model],
                normalize_input=context.normalization.normalize_actor_head_input,
                init_gain=context.init_gains.actor_head,
            ),
            critic_encoder_config=replace(
                encoders.recurrent_critic if options.td3_recurrent_critic else encoders.feedforward,
                use_agent_attention=True,
            ),
            transformer_critic_config=critic_config,
            critic_kind="transformer" if policy_variant.startswith("tmatd3") else joint_critic_config.kind,
            joint_critic_config=joint_critic_config,
            n_critics=1 if policy_variant.startswith("maddpg") else 2,
            nop_config=options.nop_config,
            **context.module_options(),
        ),
    )


def build_sac_policy(
    *,
    env: Any,
    policy_variant: str,
    context: PolicyBuildContext,
    encoders: PolicyEncoders,
    decoder: DecoderOptions,
    options: OffPolicyOptions,
    critic_config: TransformerCriticConfig,
    joint_critic_config: JointCriticConfig,
) -> BasePolicy:
    tmasac_config_kwargs = {
        "actor_encoder_config": replace(
            encoders.recurrent_actor
            if policy_variant == "tmasac__recurrent" and (not options.tmasac_recurrent_shared_encoder)
            else encoders.feedforward,
            num_layers=options.tmasac_actor_encoder_num_layers,
        ),
        "critic_encoder_config": replace(encoders.feedforward, num_layers=options.tmasac_critic_encoder_num_layers),
        "shared_encoder_config": None
        if options.tmasac_shared_encoder_num_layers is None
        else replace(
            encoders.recurrent_actor if options.tmasac_recurrent_shared_encoder else encoders.feedforward,
            num_layers=options.tmasac_shared_encoder_num_layers,
        ),
        "actor_head_config": TMASACActorHeadConfig(
            kind=TMASACActorHeadKind.DECENTRALIZED
            if policy_variant in {"tmasac_dec", "masac__mlp_critic", "masac__deepset_critic"}
            else options.tmasac_actor_head_kind,
            hidden_dims=[decoder.d_model],
            normalize_input=context.normalization.normalize_actor_head_input,
            init_gain=context.init_gains.actor_head,
            qcx_decoder_config=decoder.qcx(context, include_actor_head=False),
        ),
        "critic_config": critic_config,
        "dropout": 0.0,
        "act_fn_cls": context.act_fn_cls,
        "continuous_config": context.continuous_config,
        "max_agents": env.n_agents,
        "nop_config": options.nop_config,
        **context.module_options(),
    }
    if policy_variant in {"masac__mlp_critic", "masac__deepset_critic"}:
        return context.build(
            MASACPolicy, env, MASACPolicyConfig(joint_critic_config=joint_critic_config, **tmasac_config_kwargs)
        )
    if policy_variant == "tmasac__recurrent":
        return context.build(
            RecurrentTMASACPolicy,
            env=env,
            config=RecurrentTMASACPolicyConfig(
                recurrent_critic=False,
                recurrent_shared_encoder=options.tmasac_recurrent_shared_encoder,
                actor_state_critic_input_config=options.tmasac_actor_state_critic_input_config,
                experimental_compile_lstm=options.rmat_experimental_compile_lstm,
                **tmasac_config_kwargs,
            ),
        )
    if policy_variant == "tmasac__segment":
        return context.build(SegmentTMASACPolicy, env=env, config=TMASACPolicyConfig(**tmasac_config_kwargs))
    return context.build(TMASACPolicy, env=env, config=TMASACPolicyConfig(**tmasac_config_kwargs))


def build_on_policy(
    *,
    env: Any,
    policy_variant: str,
    context: PolicyBuildContext,
    encoders: PolicyEncoders,
    decoder: DecoderOptions,
    interaction_config: DeepSetInteractionConfig,
) -> BasePolicy:
    if policy_variant == "ppo":
        return context.build(
            PPOPolicy,
            env=env,
            config=PPOPolicyConfig(
                actor_config=PPOActorConfig(
                    hidden_dims=[512, 384, 256, 256],
                    shared_encoder_latent_dim_per_agent=192,
                    actor_head_hidden_dims=[128],
                    latent_pi_dim_per_agent=96,
                    act_fun_class=context.act_fn_cls,
                    actor_head_init_gain=context.init_gains.actor_head,
                    shared_encoder_init_gain=context.init_gains.obs_encoder,
                    shared_encoder_projection_init_gain=context.init_gains.obs_encoder_projection,
                    actor_head_hidden_init_gain=context.init_gains.actor_head,
                ),
                critic_config=PPOCriticConfig(
                    hidden_dims=[256, 256],
                    act_fun_class=context.act_fn_cls,
                    use_popart=context.use_popart,
                    popart_config=context.popart_config,
                    value_head_init_gain=context.init_gains.critic_value_head,
                    value_features_init_gain=context.init_gains.critic_value_regressor,
                ),
                **context.module_options(),
                **context.distribution_options(),
            ),
        )
    if policy_variant == "mappo__deepset_critic":
        return context.build(
            MAPPOPolicy,
            env=env,
            config=MAPPOPolicyConfig(
                actor_config=MAPPOActorConfig(
                    hidden_dims=[768, 512, 512, 384],
                    latent_dim=256,
                    actor_head_hidden_dims=[128],
                    latent_pi_dim=128,
                    act_fun_class=context.act_fn_cls,
                    actor_head_init_gain=context.init_gains.actor_head,
                    layer_norm=True,
                ),
                critic_encoder_config=replace(
                    encoders.feedforward,
                    use_agent_attention=False,
                    local_obs_encoder_config=MLPConfig(hidden_dims=[encoders.feedforward.d_model]),
                ),
                critic_interaction_config=interaction_config,
                critic_config=MATCriticConfig(
                    local_projection_hidden_dims=[128],
                    value_regressor_hidden_dims=[256, 128],
                    interaction_config=interaction_config,
                    use_popart=context.use_popart,
                    popart_config=context.popart_config,
                    local_projection_init_gain=context.init_gains.critic_local_projection,
                    value_regressor_init_gain=context.init_gains.critic_value_regressor,
                    value_head_init_gain=context.init_gains.critic_value_head,
                ),
                **context.module_options(),
                **context.distribution_options(),
            ),
        )
    rmat_critic_encoder_config = replace(
        encoders.recurrent_critic,
        use_agent_attention=True,
        local_obs_encoder_config=MLPConfig(hidden_dims=[encoders.feedforward.d_model]),
    )
    mat_critic_config = MATCriticConfig(
        interaction_config=interaction_config,
        n_local_projection_hidden_layers=2,
        n_value_regressor_hidden_layers=1,
        use_popart=context.use_popart,
        popart_config=context.popart_config,
        local_projection_init_gain=context.init_gains.critic_local_projection,
        value_regressor_init_gain=context.init_gains.critic_value_regressor,
        value_head_init_gain=context.init_gains.critic_value_head,
    )
    token_options = TokenPolicyOptions(
        dropout=0.0,
        act_fn_cls=context.act_fn_cls,
        max_agents=env.n_agents,
        **context.module_options(),
        **context.distribution_options(),
    )
    recurrent = policy_variant in {"mat_qcx__lstm", "mat_ind__lstm"}
    if policy_variant in {"mat_qcx", "mat_qcx__lstm"}:
        policy_class = RMATQCXPolicy if recurrent else MATQCXPolicy
        config_class = RMATQCXPolicyConfig if recurrent else MATQCXPolicyConfig
        return context.build(
            policy_class,
            env=env,
            config=config_class(
                encoder_config=encoders.recurrent if recurrent else encoders.feedforward,
                critic_encoder_config=rmat_critic_encoder_config if recurrent else None,
                decoder_config=decoder.qcx(context, include_actor_head=True),
                critic_config=mat_critic_config,
                **token_options,
            ),
        )
    if policy_variant == "mat_dec":
        return context.build(
            MATDecPolicy,
            env=env,
            config=MATDecPolicyConfig(
                encoder_config=encoders.feedforward,
                critic_encoder_config=replace(
                    encoders.feedforward, local_obs_encoder_config=MLPConfig(hidden_dims=[encoders.feedforward.d_model])
                ),
                critic_config=mat_critic_config,
                actor_head_hidden_dims=[decoder.d_model, decoder.d_model],
                actor_head_init_gain=context.init_gains.actor_head,
                **token_options,
            ),
        )
    if policy_variant in {"mat_ind", "mat_ind__lstm"}:
        policy_class = RMATIndPolicy if recurrent else MATIndPolicy
        config_class = RMATIndPolicyConfig if recurrent else MATIndPolicyConfig
        return context.build(
            policy_class,
            env=env,
            config=config_class(
                encoder_config=encoders.recurrent if recurrent else encoders.feedforward,
                critic_encoder_config=rmat_critic_encoder_config
                if recurrent
                else replace(
                    encoders.feedforward,
                    use_agent_attention=True,
                    local_obs_encoder_config=MLPConfig(hidden_dims=[encoders.feedforward.d_model]),
                ),
                critic_config=mat_critic_config,
                actor_head_hidden_dims=[decoder.d_model],
                actor_head_init_gain=context.init_gains.actor_head,
                **token_options,
            ),
        )
    if policy_variant == "mat_orig":
        return context.build(
            MATOrigPolicy,
            env=env,
            config=MATOrigPolicyConfig(
                encoder_config=encoders.feedforward,
                decoder_config=MATOrigDecoderConfig(
                    d_model=decoder.d_model,
                    nhead=decoder.nhead,
                    num_layers=2,
                    latent_pi_dim=decoder.d_model,
                    token_encoder_init_gain=context.init_gains.decoder_token_encoder,
                    transformer_ff_init_gain=context.init_gains.decoder_transformer_ff,
                    actor_head_init_gain=context.init_gains.actor_head,
                ),
                critic_config=MATOrigCriticConfig(
                    use_popart=context.use_popart,
                    popart_config=context.popart_config,
                    pool_mode="mean",
                    value_head_hidden_dims=[encoders.feedforward.d_model, encoders.feedforward.d_model],
                    local_projection_init_gain=context.init_gains.critic_local_projection,
                    value_regressor_init_gain=context.init_gains.critic_value_regressor,
                    value_head_init_gain=context.init_gains.critic_value_head,
                ),
                max_agents=env.n_agents,
                encoder_decoder_projection_init_gain=context.init_gains.decoder_token_encoder
                if context.init_gains.decoder_token_encoder_projection is None
                else context.init_gains.decoder_token_encoder_projection,
                **context.module_options(),
                **context.distribution_options(),
            ),
        )
    raise ValueError(f"Unknown policy_variant: {policy_variant}")
