"""Resolve preset options and delegate policy-family construction."""

from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Literal

import torch
from loguru import logger

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.beta_action_dist import BetaConfig
from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import GumbelSoftmaxSignMagnitudeBetaConfig
from swarmbots.learn.action_dists.hybrid_action_dist import ContinuousActionDistConfig
from swarmbots.learn.action_dists.predicted_std_gaussian_action_dist import PredictedStdGaussianConfig
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.algos.mat import FeedForwardConfig, MATEncoderConfig, MLPConfig
from swarmbots.learn.algos.off_policy.actor_heads import TMASACActorHeadKind
from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig
from swarmbots.learn.algos.off_policy.nop import SACNOPConfig, SACNOPLatentSource
from swarmbots.learn.algos.off_policy.transformer_critic import TransformerCriticConfig
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoderConfig
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import ActorStateCriticInputConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSetInteractionConfig
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.presets.policy_builders import (
    DecoderOptions,
    OffPolicyOptions,
    PolicyBuildContext,
    PolicyEncoders,
    build_on_policy,
    build_sac_policy,
    build_td3_policy,
)
from swarmbots.learn.presets.transformer import (
    MATInitGains,
    MATNormalizationConfig,
    NOPInitGains,
)

ContinuousActionDistVariant = Literal[
    "sign_magnitude_beta",
    "gumbel_softmax_sign_magnitude_beta",
    "beta",
    "predicted_std_gaussian",
    "gsde",
    "squashed_diag_gaussian",
]

PolicyVariant = Literal[
    "maddpg__mlp_critic", "maddpg__deepset_critic", "matd3__mlp_critic", "matd3__deepset_critic",
    "masac__mlp_critic", "masac__deepset_critic", "tmatd3", "tmatd3_dec",
    "mat_qcx",
    "mat_dec",
    "mat_ind",
    "mat_orig",
    "mat_qcx__lstm",
    "mat_ind__lstm",
    "ppo",
    "mappo__deepset_critic",
    "tmasac",
    "tmasac_dec",
    "tmasac__recurrent",
    "tmasac__segment",
]


def _is_sac_policy_variant(policy_variant: PolicyVariant) -> bool:
    return policy_variant in {"masac__mlp_critic", "masac__deepset_critic", "tmasac", "tmasac_dec", "tmasac__recurrent", "tmasac__segment"}


def _is_deterministic_policy_variant(policy_variant: PolicyVariant) -> bool:
    return policy_variant in {"maddpg__mlp_critic", "maddpg__deepset_critic", "matd3__mlp_critic", "matd3__deepset_critic", "tmatd3", "tmatd3_dec"}


def _is_off_policy_variant(policy_variant: PolicyVariant) -> bool:
    return _is_sac_policy_variant(policy_variant) or _is_deterministic_policy_variant(policy_variant)


def wrap_vec_env(
    *,
    vector_env: Any,
    obs_indices: ObsIndices,
    gamma: float,
    use_popart: bool,
    rollout_device: torch.device,
    normalize_prev_binary_actions: bool = False,
    use_transition_obs: bool = False,
    shuffle_agents: bool = False,
    preserve_inactive_prefix_structure: bool = False,
    disable_connector_actions: bool = False,
) -> Any:
    from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
    from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
    from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import TorchNormalizeRewardWrapper
    from swarmbots.learn.env_wrappers.torch_record_episode_statistics_wrapper import TorchRecordEpisodeStatisticsWrapper
    from swarmbots.learn.env_wrappers.torch_shuffle_agents_wrapper import TorchShuffleAgentsWrapper
    from swarmbots.learn.env_wrappers.torch_task_ep_stats_wrapper import (
        TorchTaskEpisodeStatsWrapper,
    )
    from swarmbots.learn.env_wrappers.torch_transition_obs_wrapper import TorchTransitionObsWrapper

    env = SwarmBotsLearnEnvWrapper(
        vector_env,
        device=rollout_device,
        disable_connector_actions=disable_connector_actions,
    )
    if shuffle_agents:
        env = TorchShuffleAgentsWrapper(
            env,
            preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
        )
    env = TorchRecordEpisodeStatisticsWrapper(env)
    env = TorchTaskEpisodeStatsWrapper(env)
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="local_obs",
        scalar_feature_indices=obs_indices.local_scalar_indices,
        quaternion_indices=obs_indices.local_quaternion_indices,
    )
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="global_obs",
        scalar_feature_indices=obs_indices.global_scalar_indices,
        quaternion_indices=obs_indices.global_quaternion_indices,
    )
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="hidden_local_vars",
        scalar_feature_indices=obs_indices.hidden_local_vars_scalar_indices,
        quaternion_indices=obs_indices.hidden_local_vars_quaternion_indices,
    )
    env = TorchFeatureWiseObsNormWrapper(
        env,
        obs_key="hidden_global_vars",
        scalar_feature_indices=obs_indices.hidden_global_vars_scalar_indices,
        quaternion_indices=obs_indices.hidden_global_vars_quaternion_indices,
    )
    if use_transition_obs:
        env = TorchTransitionObsWrapper(env, normalize_prev_binary_actions=normalize_prev_binary_actions)
    if not use_popart:
        env = TorchNormalizeRewardWrapper(env, gamma=gamma)
    return env


def set_actuator_gsde_init_joint_stds(
    *,
    policy: Any,
    actuators_per_limb: int,
    joint_stds: list[float],
) -> None:
    from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist

    if not policy.gsde_enabled:
        return
    gsde_dist = next(
        (dist for dist in policy.action_dist.distributions if isinstance(dist, GSDEActionDist)),
        None,
    )

    if gsde_dist is None:
        return

    if len(joint_stds) != actuators_per_limb:
        raise ValueError(
            f"Expected one gSDE initial std per actuator joint, got {len(joint_stds)=} {actuators_per_limb=}"
        )

    with torch.no_grad():
        for i, joint_std in enumerate(joint_stds):
            gsde_dist.log_stds[:, i::actuators_per_limb] = math.log(joint_std)


def make_continuous_config(
    *,
    variant: ContinuousActionDistVariant,
    gsde_init_stds: list[float],
    action_net_init_gain: float = 0.01,
) -> ContinuousActionDistConfig:
    if variant == "sign_magnitude_beta":
        return SignMagnitudeBetaConfig()
    if variant == "gumbel_softmax_sign_magnitude_beta":
        return GumbelSoftmaxSignMagnitudeBetaConfig()
    if variant == "beta":
        return BetaConfig()
    if variant == "predicted_std_gaussian":
        return PredictedStdGaussianConfig(
            base_std=gsde_init_stds[0],
            log_std_net_initialization=make_init_linear_orthogonal(action_net_init_gain),
        )
    if variant == "gsde":
        return GSDEConfig(
            base_std=gsde_init_stds[0],
            latent_sde_dim=None,
            std_learnable=True,
            full_std=True,
            sde_learn_features=False,
            normalize_latent_sde_by_dim=True,
            log_std_clamp_range=(-20.0, 2.0),
        )
    if variant == "squashed_diag_gaussian":
        return SquashedDiagGaussianConfig(
            std=gsde_init_stds[0],
            std_learnable=True,
        )
    raise ValueError(f"Unknown continuous action dist variant: {variant}")


def _make_mat_parameter_lr_multipliers(
    *,
    policy_variant: PolicyVariant,
    mat_decoder_lr_multiplier: float,
    include_actor_head_lr_multiplier: bool = False,
) -> dict[str, float]:
    if mat_decoder_lr_multiplier == 1.0:
        return {}

    def make_prefix_multipliers(*prefixes: str, include_actor_head_input_norm: bool = False) -> dict[str, float]:
        effective_prefixes = list(prefixes)
        if include_actor_head_lr_multiplier:
            if include_actor_head_input_norm:
                effective_prefixes.append("actor_head_input_norm")
            effective_prefixes.append("actor_head")
        return {prefix: mat_decoder_lr_multiplier for prefix in effective_prefixes}

    if policy_variant == "mat_orig":
        return {
            "decoder": mat_decoder_lr_multiplier,
            "encoder_decoder_projection": mat_decoder_lr_multiplier,
        }

    if policy_variant in {"mat_qcx", "mat_qcx__lstm"}:
        return make_prefix_multipliers(
            "decoder",
            "action_input_norm",
            "action_encoder",
            "action_token_norm",
            "memory_input_norm",
            "memory_encoder",
            "memory_token_norm",
            include_actor_head_input_norm=True,
        )

    if policy_variant in {"mat_dec", "mat_ind", "mat_ind__lstm"} and include_actor_head_lr_multiplier:
        return make_prefix_multipliers()

    logger.warning(f"Ignoring decoder LR multiplier for policy_variant={policy_variant!r}")
    return {}


def _make_sac_nop_config(
    *,
    use_nop: bool,
    obs_indices: ObsIndices | None,
    source_latent_dim: int,
    nop_init_gains: NOPInitGains,
    nop_add_agent_embeddings_transition_model: bool,
    nop_skip_first_transition_for_critic: bool,
    latent_source: SACNOPLatentSource | str,
    compile_world_model_modules: bool,
    policy_compile_mode: str,
    world_model_loss_coef: float,
    num_next_steps: int,
    transition_model_d_model: int,
    transition_model_nhead: int,
    act_fn_cls: ActivationFactory,
) -> SACNOPConfig:
    if not use_nop:
        return SACNOPConfig(enabled=False, num_next_steps=num_next_steps)
    if obs_indices is None:
        raise ValueError("obs_indices is required when building TMASAC with NOP enabled.")

    return SACNOPConfig(
        enabled=True,
        num_next_steps=num_next_steps,
        latent_source=latent_source,
        skip_first_transition_for_critic=nop_skip_first_transition_for_critic,
        nop_loss_coef=world_model_loss_coef,
        compile_modules=compile_world_model_modules,
        compile_mode=policy_compile_mode,
        act_fn_cls=act_fn_cls,
        latent_projection_hidden_dims=[source_latent_dim, source_latent_dim],
        pre_predictors_hidden_dims=[transition_model_d_model, transition_model_d_model],
        scalar_predictor_hidden_dims=[],
        angle_predictor_hidden_dims=[],
        rot6d_predictor_hidden_dims=[],
        binary_predictor_hidden_dims=[],
        latent_projection_init_gain=nop_init_gains.pre_transition,
        pre_predictors_init_gain=nop_init_gains.pre_predictors,
        predictor_init_gain=nop_init_gains.predictors,
        transition_model_dropout=0.0,
        transition_model_d_model=transition_model_d_model,
        transition_model_nhead=transition_model_nhead,
        transition_model_num_layers=2,
        transition_model_dim_feedforward=transition_model_d_model * 2,
        transition_model_add_agent_embeddings=nop_add_agent_embeddings_transition_model,
        transition_model_coembed_hidden_dims=[transition_model_d_model],
        transition_model_coembed_init_gain=nop_init_gains.transition_coembed,
        transition_model_head_init_gain=nop_init_gains.transition_head,
        transition_model_transformer_ff_init_gain=nop_init_gains.transition_transformer_ff,
        scalar_loss_fn="smooth_l1",
        next_obs_pred_config=NextObsPredConfig(
            local_scalar_target_indices=obs_indices.local_scalar_indices,
            local_angle_target_indices=obs_indices.local_angle_indices,
            local_rot6d_target_indices=obs_indices.local_rot6d_indices,
            local_binary_target_indices=obs_indices.local_binary_indices,
            scalar_loss_weight=1.0,
            angle_loss_weight=1.0,
            rot6d_loss_weight=1.0,
            binary_loss_weight=1.0,
        ),
    )


def _resolve_tmasac_nop_latent_source(
    *,
    shared_encoder_num_layers: int | None,
    latent_source: SACNOPLatentSource | str | None,
) -> SACNOPLatentSource | str:
    if latent_source is not None:
        return latent_source
    if shared_encoder_num_layers is not None:
        return SACNOPLatentSource.SHARED_ENCODER
    return SACNOPLatentSource.CRITIC


def _make_base_policy(
    *,
    env: Any,
    policy_variant: PolicyVariant,
    enc_d_model: int,
    enc_nhead: int,
    dec_d_model: int,
    dec_nhead: int,
    use_popart: bool,
    popart_beta: float,
    popart_init_sigma: float,
    compile_policy_modules: bool,
    policy_compile_mode: str,
    continuous_action_dist: ContinuousActionDistVariant | None,
    gsde_init_stds: list[float],
    mat_add_agent_embeddings: bool,
    mat_use_agent_attention: bool,
    act_fn_cls: ActivationFactory,
    mat_init_gains: MATInitGains,
    mat_normalization: MATNormalizationConfig,
    nop_init_gains: NOPInitGains = NOPInitGains(),
    use_nop: bool = False,
    nop_add_agent_embeddings_transition_model: bool = False,
    nop_skip_first_transition_for_critic: bool = True,
    compile_world_model_modules: bool = False,
    world_model_loss_coef: float = 0.1,
    world_model_num_next_steps: int = 4,
    transition_model_d_model: int = 128,
    transition_model_nhead: int = 2,
    mat_encoder_transformer_ff_config: FeedForwardConfig | None = None,
    rmat_actor_d_model: int | None = None,
    rmat_actor_transformer_ff_config: FeedForwardConfig | None = None,
    rmat_actor_inter_module_mlp: bool = False,
    tmasac_actor_state_critic_input_config: ActorStateCriticInputConfig | None = None,
    tmasac_shared_encoder_num_layers: int | None = None,
    tmasac_recurrent_shared_encoder: bool = False,
    tmasac_actor_encoder_num_layers: int = 2,
    tmasac_critic_encoder_num_layers: int = 2,
    tmasac_nop_latent_source: SACNOPLatentSource | str | None = None,
    tmasac_actor_head_kind: TMASACActorHeadKind | str = TMASACActorHeadKind.INDEPENDENT,
    tmasac_separate_observation_action_encoders: bool = False,
    obs_indices: ObsIndices | None = None,
    rmat_temporal_model_cls: Any = None,
    rmat_temporal_model_config: Any = None,
    rmat_critic_temporal_model_cls: Any = None,
    rmat_critic_temporal_model_config: Any = None,
    rmat_temporal_residual: bool = True,
    rmat_temporal_layer_norm: bool = True,
    rmat_use_temporal_output_projection: bool = True,
    rmat_experimental_compile_lstm: bool = False,
    assume_agent_mask_is_active_prefix: bool = False,
    bernoulli_initial_prob: float = 0.8,
    baseline_critic_hidden_dims: tuple[int, ...] | None = None,
    baseline_critic_element_hidden_dims: tuple[int, ...] | None = None,
    baseline_critic_context_in_elements: bool = True,
    deepset_interaction_config: DeepSetInteractionConfig | dict[str, Any] | None = None,
    critic_deepset_head_gating: bool = False,
    mat_joint_obs_embedding: bool = False,
    critic_independent_encoders: bool | None = None,
    td3_recurrent_actor: bool = False,
    td3_recurrent_critic: bool = False,
    td3_actor_state_critic_input_config: ActorStateCriticInputConfig | Literal["auto"] | None = "auto",
    model_scale: str | None = None,
) -> BasePolicy:
    sac_policy_variant = _is_sac_policy_variant(policy_variant)
    deterministic_policy_variant = _is_deterministic_policy_variant(policy_variant)
    if td3_recurrent_actor or td3_recurrent_critic:
        if not deterministic_policy_variant or policy_variant.startswith("maddpg"):
            raise ValueError("TD3 recurrence options require a TD3 preset")
        if td3_recurrent_critic and (not td3_recurrent_actor or not policy_variant.startswith("tmatd3")):
            raise ValueError("A recurrent TD3 critic requires a TMATD3 preset and td3_recurrent_actor=True")
    if deterministic_policy_variant:
        if continuous_action_dist is not None:
            raise ValueError("DDPG/TD3 use deterministic actors; configure exploration_noise in algorithm_kwargs")
        if use_popart:
            raise ValueError("DDPG/TD3 baselines require use_popart=False")
    if use_nop and policy_variant in {"maddpg__mlp_critic", "matd3__mlp_critic", "masac__mlp_critic"}:
        raise ValueError("MLP critics do not support NOP; use a Deep Set or transformer critic")
    continuous_config = make_continuous_config(
        variant=continuous_action_dist,
        gsde_init_stds=gsde_init_stds,
        action_net_init_gain=mat_init_gains.action_net,
    ) if not deterministic_policy_variant else None
    bernoulli_config = BernoulliConfig(
        initial_prob=bernoulli_initial_prob,
    )
    popart_config = PopArtConfig(
        beta=popart_beta,
        init_sigma=popart_init_sigma,
    )
    encoder_options = dict(
        d_model=enc_d_model,
        nhead=enc_nhead,
        num_layers=2,
        dim_feedforward=enc_d_model * 2,
        act_fn_cls=act_fn_cls,
        add_agent_embeddings=mat_add_agent_embeddings,
        linear_init_gain=mat_init_gains.obs_encoder,
        linear_projection_init_gain=mat_init_gains.obs_encoder_projection,
        transformer_ff_init_gain=mat_init_gains.encoder_transformer_ff,
        transformer_ff_config=mat_encoder_transformer_ff_config,
        local_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model, enc_d_model]),
        global_obs_encoder_config=MLPConfig(hidden_dims=[enc_d_model]),
        normalize_tokens=mat_normalization.normalize_encoder_tokens,
        use_agent_attention=mat_use_agent_attention,
        joint_obs_embedding=mat_joint_obs_embedding,
    )
    mat_encoder_config = MATEncoderConfig(**encoder_options)
    rmat_encoder_config = RMATEncoderConfig(
        **encoder_options,
        temporal_residual=rmat_temporal_residual,
        temporal_layer_norm=rmat_temporal_layer_norm,
        inter_module_mlp=False,
        use_temporal_output_projection=rmat_use_temporal_output_projection,
        **({} if rmat_temporal_model_cls is None else {"temporal_model_cls": rmat_temporal_model_cls}),
        **({} if rmat_temporal_model_config is None else {"temporal_model_config": rmat_temporal_model_config}),
    )
    rmat_critic_encoder_config = replace(
        rmat_encoder_config,
        **({} if rmat_critic_temporal_model_cls is None else {"temporal_model_cls": rmat_critic_temporal_model_cls}),
        **({} if rmat_critic_temporal_model_config is None
           else {"temporal_model_config": rmat_critic_temporal_model_config}),
    )
    resolved_rmat_actor_d_model = enc_d_model if rmat_actor_d_model is None else rmat_actor_d_model
    resolved_rmat_actor_ff_config = (
        mat_encoder_transformer_ff_config
        if rmat_actor_transformer_ff_config is None
        else rmat_actor_transformer_ff_config
    )
    rmat_actor_encoder_config = replace(
        rmat_encoder_config,
        d_model=resolved_rmat_actor_d_model,
        dim_feedforward=resolved_rmat_actor_d_model * 2,
        transformer_ff_config=resolved_rmat_actor_ff_config,
        local_obs_encoder_config=MLPConfig(hidden_dims=[resolved_rmat_actor_d_model, resolved_rmat_actor_d_model]),
        global_obs_encoder_config=MLPConfig(hidden_dims=[resolved_rmat_actor_d_model]),
        inter_module_mlp=rmat_actor_inter_module_mlp,
    )

    if isinstance(deepset_interaction_config, dict):
        deepset_interaction_config = DeepSetInteractionConfig(**deepset_interaction_config)
    interaction_config = deepset_interaction_config or DeepSetInteractionConfig()
    critic_config = TransformerCriticConfig(
        interaction_config=replace(
            interaction_config, gated=critic_deepset_head_gating,
            gate_hidden_dim=interaction_config.gate_hidden_dim if critic_deepset_head_gating else None,
            gate_input_dim_cap=interaction_config.gate_input_dim_cap if critic_deepset_head_gating else None,
        ),
        independent_encoders=False if critic_independent_encoders is None else critic_independent_encoders,
        local_projection_hidden_dims=[dec_d_model],
        value_regressor_hidden_dims=[dec_d_model, dec_d_model],
        separate_observation_action_encoders=tmasac_separate_observation_action_encoders,
        use_popart=False,
        popart_config=popart_config,
        local_projection_init_gain=mat_init_gains.critic_local_projection,
        value_regressor_init_gain=mat_init_gains.critic_value_regressor,
        value_head_init_gain=mat_init_gains.critic_value_head,
    )
    deepset_critic = policy_variant.endswith("__deepset_critic")
    default_regressor_dims = (dec_d_model, dec_d_model) if deepset_critic else (512, 512, 512)
    default_element_dims = (enc_d_model, enc_d_model) if deepset_critic else (512, 512, 512)
    joint_critic_config = JointCriticConfig(
        kind="deepset" if deepset_critic else "mlp",
        hidden_dims=default_regressor_dims if baseline_critic_hidden_dims is None else tuple(baseline_critic_hidden_dims),
        element_hidden_dims=default_element_dims if baseline_critic_element_hidden_dims is None
        else tuple(baseline_critic_element_hidden_dims),
        context_in_elements=baseline_critic_context_in_elements,
        interaction_config=interaction_config,
        encoder_config=replace(mat_encoder_config, use_agent_attention=True),
        head_config=critic_config,
        independent_encoders=False if critic_independent_encoders is None else critic_independent_encoders,
    )
    if joint_critic_config.kind == "deepset" and model_scale is None:
        joint_encoder = joint_critic_config.encoder_config
        joint_head = critic_config
        if baseline_critic_element_hidden_dims is not None:
            token_width = joint_critic_config.element_hidden_dims[-1]
            joint_encoder = replace(
                joint_encoder,
                d_model=token_width,
                local_obs_encoder_config=MLPConfig(hidden_dims=list(joint_critic_config.element_hidden_dims)),
                global_obs_encoder_config=MLPConfig(hidden_dims=[token_width]),
            )
            joint_head = replace(
                joint_head, action_coembed_hidden_dims=list(joint_critic_config.element_hidden_dims[:-1]),
            )
        if baseline_critic_hidden_dims is not None:
            joint_head = replace(joint_head, value_regressor_hidden_dims=list(joint_critic_config.hidden_dims))
        joint_critic_config = replace(joint_critic_config, encoder_config=joint_encoder, head_config=joint_head)
    context = PolicyBuildContext(
        continuous_config=continuous_config, bernoulli_config=bernoulli_config,
        use_popart=use_popart, popart_config=popart_config, act_fn_cls=act_fn_cls,
        init_gains=mat_init_gains, normalization=mat_normalization,
        compile_modules=compile_policy_modules, compile_mode=policy_compile_mode, model_scale=model_scale,
    )
    encoders = PolicyEncoders(
        feedforward=mat_encoder_config, recurrent=rmat_encoder_config,
        recurrent_actor=rmat_actor_encoder_config, recurrent_critic=rmat_critic_encoder_config,
    )
    decoder = DecoderOptions(
        d_model=dec_d_model, nhead=dec_nhead, add_agent_embeddings=mat_add_agent_embeddings,
        assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
    )
    if not (deterministic_policy_variant or sac_policy_variant):
        return build_on_policy(
            env=env, policy_variant=policy_variant, context=context, encoders=encoders,
            decoder=decoder, interaction_config=interaction_config,
        )
    if sac_policy_variant and use_popart:
        raise ValueError("TMASACPolicy does not support PopArt; call with use_popart=False.")
    nop_config = _make_sac_nop_config(
        use_nop=use_nop, obs_indices=obs_indices, source_latent_dim=enc_d_model,
        nop_init_gains=nop_init_gains,
        nop_add_agent_embeddings_transition_model=nop_add_agent_embeddings_transition_model,
        nop_skip_first_transition_for_critic=nop_skip_first_transition_for_critic,
        latent_source=_resolve_tmasac_nop_latent_source(
            shared_encoder_num_layers=tmasac_shared_encoder_num_layers, latent_source=tmasac_nop_latent_source,
        ),
        compile_world_model_modules=compile_world_model_modules, policy_compile_mode=policy_compile_mode,
        world_model_loss_coef=world_model_loss_coef, num_next_steps=world_model_num_next_steps,
        transition_model_d_model=transition_model_d_model, transition_model_nhead=transition_model_nhead,
        act_fn_cls=act_fn_cls,
    )
    off_policy_options = OffPolicyOptions(
        nop_config=nop_config, td3_recurrent_actor=td3_recurrent_actor, td3_recurrent_critic=td3_recurrent_critic,
        td3_actor_state_critic_input_config=td3_actor_state_critic_input_config,
        tmasac_recurrent_shared_encoder=tmasac_recurrent_shared_encoder,
        tmasac_shared_encoder_num_layers=tmasac_shared_encoder_num_layers,
        tmasac_actor_encoder_num_layers=tmasac_actor_encoder_num_layers,
        tmasac_critic_encoder_num_layers=tmasac_critic_encoder_num_layers,
        tmasac_actor_head_kind=tmasac_actor_head_kind,
        tmasac_actor_state_critic_input_config=tmasac_actor_state_critic_input_config,
        rmat_experimental_compile_lstm=rmat_experimental_compile_lstm,
    )
    builder = build_td3_policy if deterministic_policy_variant else build_sac_policy
    return builder(
        env=env, policy_variant=policy_variant, context=context, encoders=encoders, decoder=decoder,
        options=off_policy_options, critic_config=critic_config, joint_critic_config=joint_critic_config,
    )
