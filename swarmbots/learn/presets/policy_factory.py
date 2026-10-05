from __future__ import annotations

import math
from dataclasses import replace
from typing import Any, Literal

import torch
from loguru import logger

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.beta_action_dist import BetaConfig
from swarmbots.learn.action_dists.entropy_utils import AgentActionsReduction, EntropyLossConfig
from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.action_dists.gumbel_softmax_sign_magnitude_action_dist import GumbelSoftmaxSignMagnitudeBetaConfig
from swarmbots.learn.action_dists.hybrid_action_dist import ContinuousActionDistConfig
from swarmbots.learn.action_dists.predicted_std_gaussian_action_dist import PredictedStdGaussianConfig
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActorConfig
from swarmbots.learn.algos.mappo.mappo_policy import MAPPOCriticConfig, MAPPOPolicy, MAPPOPolicyConfig
from swarmbots.learn.algos.mat import FeedForwardConfig, MATEncoderConfig, MLPConfig
from swarmbots.learn.algos.mat_qc_base_policy import MATCriticConfig
from swarmbots.learn.algos.mat.mat_dec_policy import MATDecPolicy, MATDecPolicyConfig
from swarmbots.learn.algos.mat.mat_ind_policy import MATIndPolicy, MATIndPolicyConfig
from swarmbots.learn.algos.mat_orig.mat_orig_decoder import MATOrigDecoderConfig
from swarmbots.learn.algos.mat_orig.mat_orig_policy import MATOrigCriticConfig, MATOrigPolicy, MATOrigPolicyConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoderConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_policy import MATQCXPolicy, MATQCXPolicyConfig
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
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import (
    ActorStateCriticInputConfig,
    RecurrentTMASACPolicy,
    RecurrentTMASACPolicyConfig,
)
from swarmbots.learn.algos.sac.sac_nop import SACNOPConfig, SACNOPLatentSource
from swarmbots.learn.algos.sac.segment_tmasac_policy import SegmentTMASACPolicy
from swarmbots.learn.algos.sac.tmasac_actor_heads import (
    TMASACActorHeadConfig,
    TMASACActorHeadKind,
)
from swarmbots.learn.algos.sac.tmasac_policy import (
    TMASACCriticConfig,
    TMASACPolicy,
    TMASACPolicyConfig,
)
from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig
from swarmbots.learn.algos.td3.recurrent_td3_policy import RecurrentTD3Policy, RecurrentTD3PolicyConfig
from swarmbots.learn.algos.sac.masac_policy import MASACPolicy, MASACPolicyConfig
from swarmbots.learn.algos.td3.td3_policy import TD3Policy, TD3PolicyConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSetCriticConfig
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal
from swarmbots.learn.obs_indices import ObsIndices
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
    "maddpg_mlp", "maddpg_deepset", "matd3_mlp", "matd3_deepset",
    "masac_mlp", "masac_deepset", "tmatd3", "tmatd3_dec",
    "mat_qcx",
    "mat_dec",
    "mat_ind",
    "mat_orig",
    "mat_qcx_lstm",
    "mat_ind_lstm",
    "ppo",
    "ppo_small",
    "mappo",
    "mappo_small",
    "mappo_mlp",
    "mappo_mlp_small",
    "tmasac",
    "tmasac_dec",
    "tmasac_recurrent",
    "tmasac_segment",
]


def _is_sac_policy_variant(policy_variant: PolicyVariant) -> bool:
    return policy_variant in {"masac_mlp", "masac_deepset", "tmasac", "tmasac_dec", "tmasac_recurrent", "tmasac_segment"}


def _is_deterministic_policy_variant(policy_variant: PolicyVariant) -> bool:
    return policy_variant in {"maddpg_mlp", "maddpg_deepset", "matd3_mlp", "matd3_deepset", "tmatd3", "tmatd3_dec"}


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
    from swarmbots.learn.env_wrappers.torch_progress_guidance_ep_stats_wrapper import (
        TorchProgressGuidanceEpisodeStatsWrapper,
    )
    from swarmbots.learn.env_wrappers.torch_record_episode_statistics_wrapper import TorchRecordEpisodeStatisticsWrapper
    from swarmbots.learn.env_wrappers.torch_shuffle_agents_wrapper import TorchShuffleAgentsWrapper
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
    env = TorchProgressGuidanceEpisodeStatsWrapper(env)
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


def make_sign_magnitude_categorical_entropy_config() -> EntropyLossConfig:
    return EntropyLossConfig(
        entropy_floor=0.35,
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )


def make_sign_magnitude_magnitude_entropy_config() -> EntropyLossConfig:
    return EntropyLossConfig(
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )


def make_continuous_config(
    *,
    variant: ContinuousActionDistVariant,
    gsde_init_stds: list[float],
    action_net_init_gain: float = 0.01,
    ent_loss_coef: float = 1e-3,
) -> ContinuousActionDistConfig:
    if variant == "sign_magnitude_beta":
        return SignMagnitudeBetaConfig(
            ent_loss_coef=ent_loss_coef,
            beta_ent_scale=0.75,
            categorical_ent_loss_config=make_sign_magnitude_categorical_entropy_config(),
            beta_ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "gumbel_softmax_sign_magnitude_beta":
        return GumbelSoftmaxSignMagnitudeBetaConfig(
            ent_loss_coef=ent_loss_coef,
            beta_ent_scale=0.75,
            categorical_ent_loss_config=make_sign_magnitude_categorical_entropy_config(),
            beta_ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "beta":
        return BetaConfig(
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=make_sign_magnitude_magnitude_entropy_config(),
        )
    if variant == "predicted_std_gaussian":
        return PredictedStdGaussianConfig(
            base_std=gsde_init_stds[0],
            log_std_net_initialization=make_init_linear_orthogonal(action_net_init_gain),
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=EntropyLossConfig(
                agent_actions_reduction=AgentActionsReduction.SUM,
                metrics_reduction=AgentActionsReduction.MEAN,
            ),
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
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=EntropyLossConfig(
                agent_actions_reduction=AgentActionsReduction.SUM,
                metrics_reduction=AgentActionsReduction.MEAN,
            ),
        )
    if variant == "squashed_diag_gaussian":
        return SquashedDiagGaussianConfig(
            std=gsde_init_stds[0],
            std_learnable=True,
            ent_loss_coef=ent_loss_coef,
            ent_loss_config=EntropyLossConfig(
                agent_actions_reduction=AgentActionsReduction.SUM,
                metrics_reduction=AgentActionsReduction.MEAN,
            ),
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

    if policy_variant in {"mat_qcx", "mat_qcx_lstm"}:
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

    if policy_variant in {"mat_dec", "mat_ind", "mat_ind_lstm"} and include_actor_head_lr_multiplier:
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
    rmat_temporal_residual: bool = False,
    rmat_temporal_layer_norm: bool = False,
    rmat_use_temporal_output_projection: bool = True,
    rmat_experimental_compile_lstm: bool = False,
    assume_agent_mask_is_active_prefix: bool = False,
    bernoulli_initial_prob: float = 0.8,
    baseline_critic_hidden_dims: tuple[int, ...] = (512, 512, 512),
    baseline_critic_element_hidden_dims: tuple[int, ...] = (512, 512, 512),
    baseline_critic_context_in_elements: bool = True,
    mappo_critic_context_in_elements: bool = True,
    mat_joint_obs_embedding: bool = False,
    critic_independent_encoders: bool | None = None,
    td3_recurrent_actor: bool = False,
    td3_recurrent_critic: bool = False,
    td3_actor_state_critic_input_config: ActorStateCriticInputConfig | Literal["auto"] | None = "auto",
) -> (
    PPOPolicy
    | MAPPOPolicy
    | MATQCXPolicy
    | MATDecPolicy
    | MATIndPolicy
    | MATOrigPolicy
    | RMATQCXPolicy
    | RMATIndPolicy
    | TMASACPolicy
    | RecurrentTMASACPolicy
    | SegmentTMASACPolicy
    | TD3Policy
):
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
    if use_nop and policy_variant in {"maddpg_mlp", "matd3_mlp", "masac_mlp"}:
        raise ValueError("MLP critics do not support NOP; use a Deep Set or transformer critic")
    continuous_config = make_continuous_config(
        variant=continuous_action_dist,
        gsde_init_stds=gsde_init_stds,
        action_net_init_gain=mat_init_gains.action_net,
        ent_loss_coef=0.0 if sac_policy_variant else 1e-3,
    ) if not deterministic_policy_variant else None
    bernoulli_config = BernoulliConfig(
        initial_prob=bernoulli_initial_prob,
        ent_loss_coef=1e-3,
        ent_loss_config=EntropyLossConfig(
            agent_actions_reduction=AgentActionsReduction.SUM,
            metrics_reduction=AgentActionsReduction.MEAN,
        ),
    )
    popart_config = PopArtConfig(
        beta=popart_beta,
        init_sigma=popart_init_sigma,
    )
    mat_encoder_config = MATEncoderConfig(
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
        normalize_obs_inputs=mat_normalization.normalize_obs_inputs,
        normalize_tokens=mat_normalization.normalize_encoder_tokens,
        use_agent_attention=mat_use_agent_attention,
        joint_obs_embedding=mat_joint_obs_embedding,
    )
    rmat_encoder_config = RMATEncoderConfig(
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
        normalize_obs_inputs=mat_normalization.normalize_obs_inputs,
        normalize_tokens=mat_normalization.normalize_encoder_tokens,
        use_agent_attention=mat_use_agent_attention,
        joint_obs_embedding=mat_joint_obs_embedding,
        temporal_residual=rmat_temporal_residual,
        temporal_layer_norm=rmat_temporal_layer_norm,
        use_temporal_output_projection=rmat_use_temporal_output_projection,
        **({} if rmat_temporal_model_cls is None else {"temporal_model_cls": rmat_temporal_model_cls}),
        **({} if rmat_temporal_model_config is None else {"temporal_model_config": rmat_temporal_model_config}),
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

    joint_critic_config = JointCriticConfig(
        kind="deepset" if policy_variant.endswith("_deepset") else "mlp",
        hidden_dims=tuple(baseline_critic_hidden_dims),
        element_hidden_dims=tuple(baseline_critic_element_hidden_dims),
        context_in_elements=baseline_critic_context_in_elements,
    )
    if deterministic_policy_variant:
        policy_class = RecurrentTD3Policy if td3_recurrent_actor else TD3Policy
        config_class = RecurrentTD3PolicyConfig if td3_recurrent_actor else TD3PolicyConfig
        return policy_class(env, config_class(
            **({
                "recurrent_critic": td3_recurrent_critic,
                "actor_state_critic_input_config": td3_actor_state_critic_input_config,
            } if td3_recurrent_actor else {}),
            actor_encoder_config=replace(
                rmat_actor_encoder_config if td3_recurrent_actor else mat_encoder_config,
                use_agent_attention=policy_variant == "tmatd3",
            ),
            actor_head_config=TMASACActorHeadConfig(
                hidden_dims=[dec_d_model], normalize_input=mat_normalization.normalize_actor_head_input,
                init_gain=mat_init_gains.actor_head,
            ),
            critic_encoder_config=replace(
                rmat_encoder_config if td3_recurrent_critic else mat_encoder_config, use_agent_attention=True,
            ),
            transformer_critic_config=TMASACCriticConfig(
                independent_encoders=False if critic_independent_encoders is None else critic_independent_encoders,
                n_local_projection_hidden_layers=2,
                n_value_regressor_hidden_layers=1,
                local_projection_init_gain=mat_init_gains.critic_local_projection,
                value_regressor_init_gain=mat_init_gains.critic_value_regressor,
                value_head_init_gain=mat_init_gains.critic_value_head,
            ),
            critic_kind="transformer" if policy_variant.startswith("tmatd3") else joint_critic_config.kind,
            joint_critic_config=joint_critic_config,
            n_critics=1 if policy_variant.startswith("maddpg") else 2,
            nop_config=_make_sac_nop_config(
                use_nop=use_nop, obs_indices=obs_indices, source_latent_dim=enc_d_model,
                nop_init_gains=nop_init_gains,
                nop_add_agent_embeddings_transition_model=nop_add_agent_embeddings_transition_model,
                nop_skip_first_transition_for_critic=nop_skip_first_transition_for_critic,
                latent_source=_resolve_tmasac_nop_latent_source(
                    shared_encoder_num_layers=tmasac_shared_encoder_num_layers,
                    latent_source=tmasac_nop_latent_source,
                ),
                compile_world_model_modules=compile_world_model_modules,
                policy_compile_mode=policy_compile_mode, world_model_loss_coef=world_model_loss_coef,
                num_next_steps=world_model_num_next_steps, transition_model_d_model=transition_model_d_model,
                transition_model_nhead=transition_model_nhead, act_fn_cls=act_fn_cls,
            ),
            action_net_init_gain=mat_init_gains.action_net,
            compile_modules=compile_policy_modules, compile_mode=policy_compile_mode,
        ))

    if sac_policy_variant:
        if use_popart:
            raise ValueError("TMASACPolicy does not support PopArt; call with use_popart=False.")
        resolved_tmasac_nop_latent_source = _resolve_tmasac_nop_latent_source(
            shared_encoder_num_layers=tmasac_shared_encoder_num_layers,
            latent_source=tmasac_nop_latent_source,
        )
        tmasac_config_kwargs = {
            "actor_encoder_config": (
                replace(
                    (
                        rmat_actor_encoder_config
                        if policy_variant == "tmasac_recurrent" and not tmasac_recurrent_shared_encoder
                        else mat_encoder_config
                    ),
                    num_layers=tmasac_actor_encoder_num_layers,
                )
            ),
            "critic_encoder_config": replace(
                mat_encoder_config,
                num_layers=tmasac_critic_encoder_num_layers,
            ),
            "shared_encoder_config": (
                None
                if tmasac_shared_encoder_num_layers is None
                else replace(
                    (rmat_actor_encoder_config if tmasac_recurrent_shared_encoder else mat_encoder_config),
                    num_layers=tmasac_shared_encoder_num_layers,
                )
            ),
            "actor_head_config": TMASACActorHeadConfig(
                kind=(
                    TMASACActorHeadKind.DECENTRALIZED
                    if policy_variant in {"tmasac_dec", "masac_mlp", "masac_deepset"}
                    else tmasac_actor_head_kind
                ),
                hidden_dims=[dec_d_model],
                normalize_input=mat_normalization.normalize_actor_head_input,
                init_gain=mat_init_gains.actor_head,
                qcx_decoder_config=MATQCXDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    token_encoder_init_gain=mat_init_gains.decoder_token_encoder,
                    token_encoder_projection_init_gain=mat_init_gains.decoder_token_encoder_projection,
                    transformer_ff_init_gain=mat_init_gains.decoder_transformer_ff,
                    context_encoder_hidden_dims=[dec_d_model],
                    action_encoder_dims=[dec_d_model, dec_d_model],
                    memory_dims=None,
                    normalize_context_input=mat_normalization.normalize_context_input,
                    normalize_action_input=mat_normalization.normalize_action_input,
                    normalize_memory_input=mat_normalization.normalize_memory_input,
                    normalize_action_tokens=mat_normalization.normalize_action_tokens,
                    normalize_memory_tokens=mat_normalization.normalize_memory_tokens,
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                ),
            ),
            "critic_config": TMASACCriticConfig(
                independent_encoders=False if critic_independent_encoders is None else critic_independent_encoders,
                n_local_projection_hidden_layers=2,
                n_value_regressor_hidden_layers=1,
                separate_observation_action_encoders=tmasac_separate_observation_action_encoders,
                use_popart=False,
                popart_config=popart_config,
                local_projection_init_gain=mat_init_gains.critic_local_projection,
                value_regressor_init_gain=mat_init_gains.critic_value_regressor,
                value_head_init_gain=mat_init_gains.critic_value_head,
            ),
            "dropout": 0.0,
            "act_fn_cls": act_fn_cls,
            "continuous_config": continuous_config,
            "max_agents": env.n_agents,
            "nop_config": _make_sac_nop_config(
                use_nop=use_nop,
                obs_indices=obs_indices,
                source_latent_dim=enc_d_model,
                nop_init_gains=nop_init_gains,
                nop_add_agent_embeddings_transition_model=nop_add_agent_embeddings_transition_model,
                nop_skip_first_transition_for_critic=nop_skip_first_transition_for_critic,
                latent_source=resolved_tmasac_nop_latent_source,
                compile_world_model_modules=compile_world_model_modules,
                policy_compile_mode=policy_compile_mode,
                world_model_loss_coef=world_model_loss_coef,
                num_next_steps=world_model_num_next_steps,
                transition_model_d_model=transition_model_d_model,
                transition_model_nhead=transition_model_nhead,
                act_fn_cls=act_fn_cls,
            ),
            "compile_modules": compile_policy_modules,
            "compile_mode": policy_compile_mode,
            "action_net_init_gain": mat_init_gains.action_net,
        }
        if policy_variant in {"masac_mlp", "masac_deepset"}:
            return MASACPolicy(env, MASACPolicyConfig(
                joint_critic_config=joint_critic_config, **tmasac_config_kwargs,
            ))
        if policy_variant == "tmasac_recurrent":
            return RecurrentTMASACPolicy(
                env=env,
                config=RecurrentTMASACPolicyConfig(
                    recurrent_critic=False,
                    recurrent_shared_encoder=tmasac_recurrent_shared_encoder,
                    actor_state_critic_input_config=tmasac_actor_state_critic_input_config,
                    experimental_compile_lstm=rmat_experimental_compile_lstm,
                    **tmasac_config_kwargs,
                ),
            )
        if policy_variant == "tmasac_segment":
            return SegmentTMASACPolicy(
                env=env,
                config=TMASACPolicyConfig(**tmasac_config_kwargs),
            )
        return TMASACPolicy(
            env=env,
            config=TMASACPolicyConfig(**tmasac_config_kwargs),
        )

    if policy_variant == "ppo":
        return PPOPolicy(
            env=env,
            config=PPOPolicyConfig(
                actor_config=PPOActorConfig(
                    hidden_dims=[512, 384, 256, 256],
                    shared_encoder_latent_dim_per_agent=192,
                    actor_head_hidden_dims=[128],
                    latent_pi_dim_per_agent=96,
                    act_fun_class=act_fn_cls,
                ),
                critic_config=PPOCriticConfig(
                    hidden_dims=[256, 256],
                    act_fun_class=act_fn_cls,
                    use_popart=use_popart,
                ),
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
            ),
        )

    if policy_variant == "ppo_small":
        return PPOPolicy(
            env=env,
            config=PPOPolicyConfig(
                actor_config=PPOActorConfig(
                    hidden_dims=[384, 320, 256, 192],
                    shared_encoder_latent_dim_per_agent=160,
                    actor_head_hidden_dims=[128, 96],
                    latent_pi_dim_per_agent=64,
                    act_fun_class=act_fn_cls,
                ),
                critic_config=PPOCriticConfig(
                    hidden_dims=[160, 160],
                    act_fun_class=act_fn_cls,
                    use_popart=use_popart,
                ),
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
            ),
        )

    if policy_variant in {"mappo", "mappo_small", "mappo_mlp", "mappo_mlp_small"}:
        small = policy_variant.endswith("_small")
        mlp_critic = policy_variant in {"mappo_mlp", "mappo_mlp_small"}
        value_regressor_hidden_dims = [192, 128] if small else [256, 128]
        return MAPPOPolicy(
            env=env,
            config=MAPPOPolicyConfig(
                actor_config=MAPPOActorConfig(
                    hidden_dims=[512, 512, 384, 256] if small else [768, 512, 512, 384],
                    shared_encoder_latent_dim=256,
                    actor_head_hidden_dims=[128],
                    latent_pi_dim=112 if small else 128,
                    act_fun_class=act_fn_cls,
                ),
                critic_config=MAPPOCriticConfig(
                    mlp_hidden_dims=value_regressor_hidden_dims if mlp_critic else [],
                    deep_set_config=None if mlp_critic else DeepSetCriticConfig(
                        local_projection_hidden_dims=[224, 112] if small else [256, 128],
                        value_regressor_hidden_dims=value_regressor_hidden_dims,
                    ),
                    context_in_elements=mappo_critic_context_in_elements,
                    act_fun_class=act_fn_cls,
                    use_popart=use_popart,
                ),
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
            ),
        )

    mat_critic_config = MATCriticConfig(
        n_local_projection_hidden_layers=2,
        n_value_regressor_hidden_layers=1,
        use_popart=use_popart,
        popart_config=popart_config,
        local_projection_init_gain=mat_init_gains.critic_local_projection,
        value_regressor_init_gain=mat_init_gains.critic_value_regressor,
        value_head_init_gain=mat_init_gains.critic_value_head,
    )

    if policy_variant == "mat_qcx":
        return MATQCXPolicy(
            env=env,
            config=MATQCXPolicyConfig(
                encoder_config=mat_encoder_config,
                decoder_config=MATQCXDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    add_agent_embeddings=mat_add_agent_embeddings,
                    token_encoder_init_gain=mat_init_gains.decoder_token_encoder,
                    token_encoder_projection_init_gain=mat_init_gains.decoder_token_encoder_projection,
                    transformer_ff_init_gain=mat_init_gains.decoder_transformer_ff,
                    actor_head_init_gain=mat_init_gains.actor_head,
                    context_encoder_hidden_dims=[dec_d_model],
                    action_encoder_dims=[dec_d_model, dec_d_model],
                    memory_dims=None,
                    normalize_context_input=mat_normalization.normalize_context_input,
                    normalize_action_input=mat_normalization.normalize_action_input,
                    normalize_memory_input=mat_normalization.normalize_memory_input,
                    normalize_action_tokens=mat_normalization.normalize_action_tokens,
                    normalize_memory_tokens=mat_normalization.normalize_memory_tokens,
                    normalize_actor_head_input=mat_normalization.normalize_actor_head_input,
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                ),
                critic_config=mat_critic_config,
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=env.n_agents,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "mat_qcx_lstm":
        return RMATQCXPolicy(
            env=env,
            config=RMATQCXPolicyConfig(
                encoder_config=rmat_encoder_config,
                decoder_config=MATQCXDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    add_agent_embeddings=mat_add_agent_embeddings,
                    token_encoder_init_gain=mat_init_gains.decoder_token_encoder,
                    token_encoder_projection_init_gain=mat_init_gains.decoder_token_encoder_projection,
                    transformer_ff_init_gain=mat_init_gains.decoder_transformer_ff,
                    actor_head_init_gain=mat_init_gains.actor_head,
                    context_encoder_hidden_dims=[dec_d_model],
                    action_encoder_dims=[dec_d_model, dec_d_model],
                    memory_dims=None,
                    normalize_context_input=mat_normalization.normalize_context_input,
                    normalize_action_input=mat_normalization.normalize_action_input,
                    normalize_memory_input=mat_normalization.normalize_memory_input,
                    normalize_action_tokens=mat_normalization.normalize_action_tokens,
                    normalize_memory_tokens=mat_normalization.normalize_memory_tokens,
                    normalize_actor_head_input=mat_normalization.normalize_actor_head_input,
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                ),
                critic_config=mat_critic_config,
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=env.n_agents,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "mat_dec":
        return MATDecPolicy(
            env=env,
            config=MATDecPolicyConfig(
                encoder_config=mat_encoder_config,
                critic_config=mat_critic_config,
                actor_head_hidden_dims=[dec_d_model, dec_d_model],
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=env.n_agents,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                actor_head_init_gain=mat_init_gains.actor_head,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "mat_ind":
        return MATIndPolicy(
            env=env,
            config=MATIndPolicyConfig(
                encoder_config=mat_encoder_config,
                critic_config=mat_critic_config,
                actor_head_hidden_dims=[dec_d_model],
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=env.n_agents,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                actor_head_init_gain=mat_init_gains.actor_head,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "mat_ind_lstm":
        return RMATIndPolicy(
            env=env,
            config=RMATIndPolicyConfig(
                encoder_config=rmat_encoder_config,
                critic_config=mat_critic_config,
                actor_head_hidden_dims=[dec_d_model],
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=env.n_agents,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                actor_head_init_gain=mat_init_gains.actor_head,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "mat_orig":
        return MATOrigPolicy(
            env=env,
            config=MATOrigPolicyConfig(
                encoder_config=mat_encoder_config,
                decoder_config=MATOrigDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    latent_pi_dim=dec_d_model,
                ),
                critic_config=MATOrigCriticConfig(
                    use_popart=use_popart,
                    popart_config=popart_config,
                    pool_mode="mean",
                    value_head_hidden_dims=[enc_d_model, enc_d_model],
                ),
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=env.n_agents,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                encoder_decoder_projection_init_gain=(
                    mat_init_gains.decoder_token_encoder
                    if mat_init_gains.decoder_token_encoder_projection is None
                    else mat_init_gains.decoder_token_encoder_projection
                ),
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    raise ValueError(f"Unknown policy_variant: {policy_variant}")
