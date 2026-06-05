from __future__ import annotations

import math
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal

import torch
from loguru import logger
from torch import nn

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import swarmbots.mjw_env.scenarios.mjw_scenario_presets as mjw_scenario_presets
from swarmbots.learn.action_dists.beta_action_dist import BetaConfig
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.entropy_utils import AgentActionsReduction, EntropyLossConfig
from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import SignMagnitudeBetaConfig
from swarmbots.learn.action_dists.sticky_action_dist import StickyActionDist
from swarmbots.learn.action_dists.sticky_sign_magnitude_beta_action_dist import StickySignMagnitudeBetaConfig
from swarmbots.learn.action_dists.squashed_diag_gaussian_action_dist import SquashedDiagGaussianConfig
from swarmbots.learn.algos.mat.mat_dec_policy import MATDecPolicy, MATDecPolicyConfig
from swarmbots.learn.algos.mat_qcc.mat_qcc_decoder import MATQCCDecoderConfig
from swarmbots.learn.algos.mat_qcc.mat_qcc_policy import MATQCCPolicy, MATQCCPolicyConfig
from swarmbots.learn.algos.mappo.mappo_actor import MAPPOActorConfig
from swarmbots.learn.algos.mappo.mappo_policy import MAPPOCriticConfig, MAPPOPolicy, MAPPOPolicyConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderConfig, MATQCSDecoderSelfAttentionMode
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat_qcs.mat_qcs_policy import MATQCSCriticConfig, MATQCSPolicy, MATQCSPolicyConfig
from swarmbots.learn.algos.mat_orig.mat_orig_decoder import MATOrigDecoderConfig
from swarmbots.learn.algos.mat_orig.mat_orig_policy import MATOrigCriticConfig, MATOrigPolicy, MATOrigPolicyConfig
from swarmbots.learn.algos.ppo.ppo import AutomaticLearningRate, PPO, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_policy import PPOActorConfig, PPOCriticConfig, PPOPolicy, PPOPolicyConfig, PopArtConfig
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamplerConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig, NextObsPredWrapper
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
from swarmbots.learn.discord_notifications import run_with_discord_notification
from swarmbots.learn.gsde_reset import GSDEProbabilityResetMode
from swarmbots.learn.nn_components.activations import ActivationFactory, activation_factory_name
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.scheduling.auto_lr_updater import make_auto_lr_updater
from swarmbots.learn.scheduling.cosine_scheduler import CosineSchedulerConfig
from swarmbots.learn.scheduling.linear_scheduler import LinearScheduler
from swarmbots.learn.scheduling.schedulers import ScheduledHyperParameter, SchedulerManager, ScheduleUnit
from swarmbots.learn.summary_statistics import SummaryStatisticsFormat
from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.learn.nn_components.deep_set import DeepSetCriticConfig
from swarmbots.mjw_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_dual_payload_plane, default_wall
from swarmbots.utils.recording_schedule import DEFAULT_LIVE_RECORDING_SCHEDULE, install_scheduled_recordings
from swarmbots.utils.run_paths import get_run_id_from_checkpoint_path

ContinuousActionDistVariant = Literal["sticky_sign_magnitude_beta", "sign_magnitude_beta", "beta", "gsde", "squashed_diag_gaussian"]
PolicyVariant = Literal["mat_qcs", "mat_qcc", "mat_dec", "mat_orig", "ppo", "mappo"]
MJWScenarioName = Literal["wall", "dual_payload"]


@dataclass(frozen=True)
class MATInitGains:
    obs_encoder: float = 1.0
    obs_encoder_projection: float | None = 1.0
    encoder_transformer_ff: float | None = 1.0
    decoder_token_encoder: float = 1.0
    decoder_token_encoder_projection: float | None = 1.0
    decoder_transformer_ff: float | None = 1.0
    actor_head: float = 1.0
    action_net: float = 0.01
    critic_local_projection: float = 1.0
    critic_value_regressor: float = 1.0
    critic_value_head: float = 0.01


@dataclass(frozen=True)
class NOPInitGains:
    pre_transition: float = 1.0
    transition_coembed: float = 1.0
    transition_transformer_ff: float | None = 1.0
    transition_head: float = 0.01
    pre_predictors: float = 1.0
    predictors: float = 0.01


@dataclass(frozen=True)
class MATNormalizationConfig:
    normalize_obs_inputs: bool = False
    normalize_encoder_tokens: bool = False
    normalize_query_input: bool = False
    normalize_context_input: bool = False
    normalize_memory_input: bool = False
    normalize_query_tokens: bool = False
    normalize_context_tokens: bool = False
    normalize_memory_tokens: bool = False
    normalize_actor_head_input: bool = False
    normalize_prev_binary_actions: bool = False


def configure_float32_matmul_precision() -> None:
    if torch.cuda.is_available():
        torch.set_float32_matmul_precision("high")


def _make_scenario(*, scenario_name: MJWScenarioName, scenario_kwargs: dict[str, object] | None) -> Any:
    scenario_factory = {
        "wall": default_wall,
        "dual_payload": default_dual_payload_plane,
    }[scenario_name]
    return scenario_factory(**({} if scenario_kwargs is None else scenario_kwargs))


def _scenario_display_name(*, scenario_name: MJWScenarioName) -> str:
    return "dual-payload" if scenario_name == "dual_payload" else "wall"


def _default_experiment_run_name(*, scenario_name: MJWScenarioName) -> str:
    return f"mat_nop_swarm_bots_{scenario_name}_mjw"


def _default_ccd_iterations(*, scenario_name: MJWScenarioName) -> int | None:
    return 128 if scenario_name == "dual_payload" else None


def make_vector_env(
    *,
    episode_length: int,
    num_envs: int,
    first_episode_lengths: list[int] | None,
    settle_initial_reset: bool,
    device: torch.device,
    scenario_name: MJWScenarioName = "wall",
    ccd_iterations: int | None = None,
    scenario_kwargs: dict[str, object] | None = None,
) -> MJWSwarmBotsVectorEnv:
    return MJWSwarmBotsVectorEnv(
        scenario=_make_scenario(scenario_name=scenario_name, scenario_kwargs=scenario_kwargs),
        num_envs=num_envs,
        episode_length=episode_length,
        first_episode_lengths=first_episode_lengths,
        settle_initial_reset=settle_initial_reset,
        device=device,
        ccd_iterations=ccd_iterations,
    )


def wrap_vec_env(
    *,
    vector_env: Any,
    obs_indices: ObsIndices,
    gamma: float,
    use_popart: bool,
    rollout_device: torch.device,
    normalize_prev_binary_actions: bool = False,
    shuffle_agents: bool = False,
    preserve_inactive_prefix_structure: bool = False,
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

    env = SwarmBotsLearnEnvWrapper(vector_env, device=rollout_device)
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
    env = TorchTransitionObsWrapper(env, normalize_prev_binary_actions=normalize_prev_binary_actions)
    if not use_popart:
        env = TorchNormalizeRewardWrapper(env, gamma=gamma)
    return env


def split_actuator_joints(actions: Any, actuators_per_limb: int) -> dict[str, Any]:
    return {f"j{i}": actions[..., i::actuators_per_limb] for i in range(actuators_per_limb)}


def set_actuator_gsde_init_joint_stds(
    *,
    policy: Any,
    actuators_per_limb: int,
    joint_stds: list[float],
) -> None:
    from swarmbots.learn.action_dists.gsde_action_dist import GSDEActionDist

    if len(joint_stds) != actuators_per_limb:
        raise ValueError()

    gsde_dist = next((dist for dist in policy.action_dist.distributions if isinstance(dist, GSDEActionDist)), None)

    if gsde_dist is None:
        logger.warning("No gSDE dist found, skipping log std init")
        return

    with torch.no_grad():
        for i, joint_std in enumerate(joint_stds):
            gsde_dist.log_stds[:, i::actuators_per_limb] = math.log(joint_std)


def make_sign_magnitude_beta_entropy_config() -> EntropyLossConfig:
    return EntropyLossConfig(
        agent_actions_reduction=AgentActionsReduction.SUM,
        metrics_reduction=AgentActionsReduction.MEAN,
    )


def make_continuous_config(
        *,
        variant: ContinuousActionDistVariant,
        initial_stickiness: float,
        gsde_init_stds: list[float],
) -> StickySignMagnitudeBetaConfig | SignMagnitudeBetaConfig | BetaConfig | GSDEConfig | SquashedDiagGaussianConfig:
    if variant == "sticky_sign_magnitude_beta":
        return StickySignMagnitudeBetaConfig(
            stickiness=initial_stickiness,
            ent_loss_coef=1e-3,
            beta_ent_scale=0.75,
            categorical_ent_loss_config=make_sign_magnitude_beta_entropy_config(),
            beta_ent_loss_config=make_sign_magnitude_beta_entropy_config(),
        )
    if variant == "sign_magnitude_beta":
        return SignMagnitudeBetaConfig(
            ent_loss_coef=1e-3,
            beta_ent_scale=0.75,
            categorical_ent_loss_config=make_sign_magnitude_beta_entropy_config(),
            beta_ent_loss_config=make_sign_magnitude_beta_entropy_config(),
        )
    if variant == "beta":
        return BetaConfig(
            ent_loss_coef=1e-3,
            ent_loss_config=make_sign_magnitude_beta_entropy_config(),
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
            ent_loss_coef=1e-3,
            ent_loss_config=EntropyLossConfig(
                agent_actions_reduction=AgentActionsReduction.SUM,
                metrics_reduction=AgentActionsReduction.MEAN,
            ),
        )
    if variant == "squashed_diag_gaussian":
        return SquashedDiagGaussianConfig(
            std=gsde_init_stds[0],
            std_learnable=True,
            ent_loss_coef=1e-3,
            ent_loss_config=EntropyLossConfig(
                agent_actions_reduction=AgentActionsReduction.SUM,
                metrics_reduction=AgentActionsReduction.MEAN,
            ),
        )
    raise ValueError(f"Unknown continuous action dist variant: {variant}")


def run_experiment(
        *,
        num_envs: int,
        rollout_steps_per_env: int,
        variant_name: str,
        entrypoint_path: Path,
        virtual_mini_batches: int = 1,
        n_epochs: int = 8,
        continuous_action_dist: ContinuousActionDistVariant = "sign_magnitude_beta",
        policy_variant: PolicyVariant = "mat_qcs",
        mat_add_agent_embeddings: bool = False,
        mat_decoder_self_attention_mode: MATQCSDecoderSelfAttentionMode = MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
        act_fn_cls: ActivationFactory = nn.GELU,
        mat_init_gains: MATInitGains = MATInitGains(),
        nop_init_gains: NOPInitGains = NOPInitGains(),
        mat_normalization: MATNormalizationConfig = MATNormalizationConfig(),
        use_nop: bool = True,
        nop_add_agent_embeddings_transition_model: bool = False,
        shuffle_agents: bool = False,
        preserve_inactive_prefix_structure: bool = False,
        mat_decoder_lr_multiplier: float = 0.25,
        mat_query_context_lr_multiplier: float = 0.25,
        experiment_run_name: str | None = None,
        scenario_name: MJWScenarioName = "wall",
        ccd_iterations: int | None = None,
        scenario_kwargs: dict[str, object] | None = None,
) -> None:
    from swarmbots.learn.torch_logging import enable_torch_compile_logging

    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <level>{message}</level>",
    )
    enable_torch_compile_logging()
    configure_float32_matmul_precision()

    if not torch.cuda.is_available():
        raise RuntimeError("MJW batch env sweep requires CUDA.")

    rollout_samples = num_envs * rollout_steps_per_env
    if rollout_samples % virtual_mini_batches != 0:
        raise ValueError(
            f"Expected rollout_samples divisible by virtual_mini_batches, got "
            f"{rollout_samples=} {virtual_mini_batches=}"
        )

    episode_length = 512
    rollout_warmup_steps_per_env = episode_length
    total_timesteps = 100_000_000
    save_interval = None

    use_popart = True
    popart_beta = 5e-4
    popart_init_sigma = 0.65

    vf_coef = 2.0 if use_popart else 0.5
    world_model_loss_coef = 0.1
    world_model_num_next_steps = 3

    initial_stickiness = 0.25
    final_stickiness = 0.0
    stickiness_anneal_steps = 15_000_000
    gsde_init_stds = [0.25, 0.30]

    compile_policy_modules = True
    policy_compile_mode = "default"
    compile_world_model_modules = True

    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    load_path: str | Path | None = None

    rollout_device = torch.device("cuda")
    train_device = torch.device("cuda")
    record_device = torch.device("cuda")

    logger.info(f"{rollout_device = }")
    logger.info(f"{train_device = }")
    mat_decoder_self_attention_mode_metadata = (
        mat_decoder_self_attention_mode.name
        if policy_variant == "mat_qcs"
        else None
    )
    variant_log_message = (
        f"MJW {_scenario_display_name(scenario_name=scenario_name)} batch env sweep variant {variant_name}: "
        f"{num_envs} envs x {rollout_steps_per_env} steps/env = {rollout_samples}, "
        f"virtual_mini_batches={virtual_mini_batches}, n_epochs={n_epochs}, "
        f"continuous_action_dist={continuous_action_dist}, use_nop={use_nop}, "
        f"compile_policy_modules={compile_policy_modules}, policy_compile_mode={policy_compile_mode}, "
        f"nop_add_agent_embeddings_transition_model={nop_add_agent_embeddings_transition_model}, "
        f"act_fn_cls={activation_factory_name(act_fn_cls)}, "
        f"mat_init_gains={mat_init_gains}, nop_init_gains={nop_init_gains}, "
        f"mat_normalization={mat_normalization}, "
        f"mat_add_agent_embeddings={mat_add_agent_embeddings}, "
        f"mat_decoder_self_attention_mode={mat_decoder_self_attention_mode_metadata}, "
        f"shuffle_agents={shuffle_agents}, "
        f"preserve_inactive_prefix_structure={preserve_inactive_prefix_structure}, "
        f"ccd_iterations={ccd_iterations}, "
        f"scenario_kwargs={scenario_kwargs}"
    )
    if policy_variant != "mat_qcs":
        variant_log_message = f"{variant_log_message}, policy_variant={policy_variant}"
    logger.info(variant_log_message)
    scenario_display_name_text = _scenario_display_name(scenario_name=scenario_name)
    logger.info(
        f"MJW {scenario_display_name_text} training uses one batched GPU env directly; worker-pool vectorization is disabled."
    )
    logger.info(
        f"MJW {scenario_display_name_text} training supports live exact-state recording via the `record` command "
        "(for example: record:{\"episodes\":8,\"parallel\":4,\"frame_stride\":4})."
    )
    logger.info(
        f"MJW {scenario_display_name_text} env uses per-env first-episode staggering so episode ends are spread across time from startup."
    )
    logger.info("MJW env also settles all worlds once on the initial reset, which increases startup latency.")

    if load_path is not None:
        if Path(load_path).suffix != ".pt":
            logger.error("load_path is missing .pt")
            raise ValueError()
        logger.info(f"{load_path = }")
        run_id = get_run_id_from_checkpoint_path(load_path)
    logger.info(f"{run_id = }")

    if experiment_run_name is None:
        experiment_run_name = _default_experiment_run_name(scenario_name=scenario_name)
    if ccd_iterations is None:
        ccd_iterations = _default_ccd_iterations(scenario_name=scenario_name)

    run_dir = REPO_ROOT / "runs" / experiment_run_name / variant_name / run_id
    save_optimizer = True

    first_episode_lengths = [int((i + 1) * episode_length / num_envs) for i in range(num_envs)]

    print("Creating MJW vector env...")
    vector_env = make_vector_env(
        episode_length=episode_length,
        num_envs=num_envs,
        first_episode_lengths=first_episode_lengths,
        settle_initial_reset=True,
        device=rollout_device,
        scenario_name=scenario_name,
        ccd_iterations=ccd_iterations,
        scenario_kwargs=scenario_kwargs,
    )
    print(f"Created {type(vector_env)} with {num_envs} environments.")

    env_settings = vector_env.get_settings()
    obs_indices: ObsIndices = build_obs_indices(
        env_settings=env_settings,
        local_obs_dim=int(vector_env.single_observation_space["local_obs"].shape[-1]),
        global_obs_dim=int(vector_env.single_observation_space["global_obs"].shape[-1]),
        hidden_local_vars_dim=int(vector_env.single_observation_space["hidden_local_vars"].shape[-1]),
        hidden_global_vars_dim=int(vector_env.single_observation_space["hidden_global_vars"].shape[-1]),
    )

    gamma = 0.99

    print("Wrapping...")
    env = wrap_vec_env(
        vector_env=vector_env,
        obs_indices=obs_indices,
        gamma=gamma,
        use_popart=use_popart,
        rollout_device=rollout_device,
        normalize_prev_binary_actions=mat_normalization.normalize_prev_binary_actions,
        shuffle_agents=shuffle_agents,
        preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
    )

    print("Environment initialized.")
    print(f"n_agents: {env.n_agents}")
    print(f"local_obs_dim: {env.local_obs_dim}")
    print(f"global_obs_dim: {env.global_obs_dim}")
    print(f"actuators_dim: {env.actuators_dim}")
    print(f"connectors_dim: {env.connectors_dim}")
    if env.connectors_dim <= 0 or env.actuators_dim % env.connectors_dim != 0:
        raise ValueError(
            f"Expected actuators_dim divisible by connectors_dim, got {env.actuators_dim=} {env.connectors_dim=}"
        )
    actuators_per_limb = env.actuators_dim // env.connectors_dim
    print(f"actuators_per_limb: {actuators_per_limb}")

    enc_d_model = 256
    dec_d_model = 128
    transition_model_d_model = 128

    enc_nhead = 4
    dec_nhead = 2
    transition_model_nhead = 2

    print("Initializing Policy...")
    base_policy = _make_base_policy(
        env=env,
        policy_variant=policy_variant,
        enc_d_model=enc_d_model,
        enc_nhead=enc_nhead,
        dec_d_model=dec_d_model,
        dec_nhead=dec_nhead,
        use_popart=use_popart,
        popart_beta=popart_beta,
        popart_init_sigma=popart_init_sigma,
        compile_policy_modules=compile_policy_modules,
        policy_compile_mode=policy_compile_mode,
        continuous_action_dist=continuous_action_dist,
        initial_stickiness=initial_stickiness,
        gsde_init_stds=gsde_init_stds,
        mat_add_agent_embeddings=mat_add_agent_embeddings,
        mat_decoder_self_attention_mode=mat_decoder_self_attention_mode,
        act_fn_cls=act_fn_cls,
        mat_init_gains=mat_init_gains,
        mat_normalization=mat_normalization,
        assume_agent_mask_is_active_prefix=not shuffle_agents or preserve_inactive_prefix_structure,
    )
    policy_local_latent_dim = int(getattr(base_policy, "local_latent_dim", enc_d_model))
    policy = base_policy
    if use_nop:
        policy = NextObsPredWrapper(
            policy=base_policy,
            world_model_config=NOPWorldModelConfig(
                n_agents=env.n_agents,
                local_latent_dim=policy_local_latent_dim,
                action_dim=env.action_space.total_agent_action_dim,
                world_model_loss_coef=world_model_loss_coef,
                compile_modules=compile_world_model_modules,
                compile_mode=policy_compile_mode,
                act_fn_cls=act_fn_cls,
                wm_pre_transition_init_gain=nop_init_gains.pre_transition,
                transition_model_coembed_init_gain=nop_init_gains.transition_coembed,
                transition_model_transformer_ff_init_gain=nop_init_gains.transition_transformer_ff,
                transition_model_head_init_gain=nop_init_gains.transition_head,
                wm_pre_predictors_init_gain=nop_init_gains.pre_predictors,
                wm_predictor_init_gain=nop_init_gains.predictors,
                transition_model_dropout=0.0,
                wm_pre_transition_dims=[policy_local_latent_dim],
                d_model_transition_model=transition_model_d_model,
                nhead_transition_model=transition_model_nhead,
                num_layers_transition_model=2,
                dim_feedforward_transition_model=transition_model_d_model * 2,
                add_agent_embeddings_transition_model=nop_add_agent_embeddings_transition_model,
                transition_model_coembed_hidden_dims=[transition_model_d_model],
                wm_pre_predictors_dims=[transition_model_d_model, transition_model_d_model],
                wm_scalar_predictor_hidden_dims=[],
                wm_angle_predictor_hidden_dims=[],
                wm_rot6d_predictor_hidden_dims=[],
                wm_binary_predictor_hidden_dims=[],
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
            ),
        )
    set_actuator_gsde_init_joint_stds(
        policy=policy,
        actuators_per_limb=actuators_per_limb,
        joint_stds=gsde_init_stds,
    )
    print(policy)
    print(f"learnable_params: {policy.num_parameters():,}")

    print("Initializing PPO Algorithm...")

    warm_lr = 1e-4
    warmup_iterations = 200
    cold_lr = warm_lr * 5e-3 if warmup_iterations > 0 else warm_lr
    clip_range = 0.05
    target_kl = 0.002

    auto_lr = AutomaticLearningRate(
        initial_lr=cold_lr,
        max_lr=8e-4,
        updater=make_auto_lr_updater(
            early_stop_epoch_decay_limit=math.ceil(n_epochs * 0.75),
            max_kl_div=target_kl * 1.55,
            warm_scheduler_config=CosineSchedulerConfig(
                unit=ScheduleUnit.ITERATIONS,
                duration=warmup_iterations,
                start_value=cold_lr,
                final_value=warm_lr,
            )
            if warmup_iterations > 0
            else None,
        ),
    )

    scheduler_manager: SchedulerManager | None = None
    continuous_dist = policy.action_dist.distributions[0]
    if continuous_action_dist == "sticky_sign_magnitude_beta" and isinstance(continuous_dist, StickyActionDist):
        sticky_dist: StickyActionDist = continuous_dist
        scheduler_manager = SchedulerManager(
            [
                ScheduledHyperParameter(
                    name="act0_stickiness",
                    scheduler=LinearScheduler(
                        unit=ScheduleUnit.TIMESTEPS,
                        duration=stickiness_anneal_steps,
                        start_value=initial_stickiness,
                        final_value=final_stickiness,
                        name="act0_stickiness",
                    ),
                    get_value=lambda: sticky_dist.get_stickiness(),
                    apply=lambda new_value: sticky_dist.set_stickiness(new_value),
                )
            ]
        )
    elif continuous_action_dist == "sticky_sign_magnitude_beta":
        act0_dist_type = type(continuous_dist) if policy.action_dist.distributions else None
        logger.warning(f"Skipping act0_stickiness scheduler: action dist[0] is {act0_dist_type}")

    parameter_lr_multipliers = _make_mat_parameter_lr_multipliers(
        policy_variant=policy_variant,
        decoder_lr_multiplier=mat_decoder_lr_multiplier,
        query_context_lr_multiplier=mat_query_context_lr_multiplier,
    )
    ppo = PPO(
        policy=policy,
        env=env,
        learning_rate=auto_lr,
        rollout_mode=StepsRolloutMode(rollout_samples),
        rollout_warmup_steps_per_env=rollout_warmup_steps_per_env,
        max_episode_length=episode_length,
        sampler_config=PPOWMSamplerConfig(
            batch_size=rollout_samples,
            num_next_steps=world_model_num_next_steps,
            compile_wm_window_helper=True,
        ) if use_nop else PPOSamplerConfig(batch_size=rollout_samples),
        n_epochs=n_epochs,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=clip_range,
        target_kl=target_kl,
        max_grad_norm=2.0,
        gsde_reset_mode=GSDEProbabilityResetMode(probability=1 / 6),
        mc_ent_coef=0e-5,
        vf_coef=vf_coef,
        value_loss_fn=nn.MSELoss(reduction="none"),
        train_device=train_device,
        rollout_device=rollout_device,
        record_device=record_device,
        use_popart=use_popart,
        agent_logprob_reduction="sum" if policy_variant == "ppo" else None,
        metrics_action_splitters=[lambda actions: split_actuator_joints(actions, actuators_per_limb), None],
        scheduler_manager=scheduler_manager,
        virtual_mini_batches=virtual_mini_batches,
        parameter_lr_multipliers=parameter_lr_multipliers,
    )

    if load_path:
        logger.info(f"Loading model from {load_path}")
        ppo.load(load_path, recover_best_return_ema=False, strict_load_state_dict=True)

    scheduled_recording_hook = install_scheduled_recordings(
        algorithm=ppo,
        total_timesteps=total_timesteps,
        schedule=DEFAULT_LIVE_RECORDING_SCHEDULE,
    )

    print("Starting training...")
    logging_console_keys: list[
        tuple[str, str | SummaryStatisticsFormat | None] | tuple[str, str | SummaryStatisticsFormat | None, str]
    ] = [
        ("iteration", "5", "it"),
        ("timesteps", "8", "steps"),
        ("total_updates", "6", "tot_upd"),
    ]
    logging_console_keys.extend((f"act0_j{i}", SummaryStatisticsFormat(histogram=11)) for i in range(actuators_per_limb))
    logging_console_keys.extend(
        (f"std0_j{i}", SummaryStatisticsFormat(mean=".3f", std=".3f", min_value=".3f", max_value=".3f"))
        for i in range(actuators_per_limb)
    )
    logging_console_keys.extend(
        [
            ("act1", SummaryStatisticsFormat(histogram=2)),
            ("updates", "3", "upd"),
            ("approx_kl", SummaryStatisticsFormat(mean=".3f", std=".3f", max_value=".3f")),
            ("clip_frac", None),
            ("ratio", SummaryStatisticsFormat(mean=".3f", std=".3f", min_value=".1e", max_value=".3f")),
        ]
    )
    if use_nop:
        logging_console_keys.append(("wm_loss_scaled", None, "wm_loss"))
    logging_console_keys.extend(
        [
            ("val_loss_scaled", None, "val_loss"),
            ("expl_var", ".3f"),
            ("popart_mu", ".3f", "pa_mu"),
            ("popart_sigma", ".3f", "pa_sigma"),
            ("ep_rew", SummaryStatisticsFormat(mean=" .2f", std=".2f", max_value=" .2f", n="1")),
            ("ep_rew_ema", " .3f"),
            ("best_ep_rew_ema", " .3f", "best_ema"),
            ("ep_success_rate_ema", "5.2f", "success_pct_ema"),
            ("fps", None),
        ]
    )

    extra_run_metadata = {
        "load_path": load_path,
        "env_settings": env_settings,
        "script": entrypoint_path.read_text(encoding="utf-8"),
        "shared_experiment_script": Path(__file__).read_text(encoding="utf-8"),
        "script_scenario_presets": Path(mjw_scenario_presets.__file__).read_text(encoding="utf-8"),
        "backend": "mjw_env",
        "recording_enabled": "live_mjw_exact_state",
        "rollout_samples": rollout_samples,
        "rollout_steps_per_env": rollout_steps_per_env,
        "rollout_warmup_steps_per_env": rollout_warmup_steps_per_env,
        "virtual_mini_batches": virtual_mini_batches,
        "n_epochs": n_epochs,
        "num_envs": num_envs,
        "variant_name": variant_name,
        "continuous_action_dist": continuous_action_dist,
        "use_nop": use_nop,
        "nop_add_agent_embeddings_transition_model": nop_add_agent_embeddings_transition_model,
        "act_fn_cls": activation_factory_name(act_fn_cls),
        "mat_init_gains": asdict(mat_init_gains),
        "nop_init_gains": asdict(nop_init_gains),
        "mat_normalization": asdict(mat_normalization),
        "mat_add_agent_embeddings": mat_add_agent_embeddings,
        "mat_decoder_self_attention_mode": mat_decoder_self_attention_mode_metadata,
        "mat_decoder_lr_multiplier": mat_decoder_lr_multiplier,
        "mat_query_context_lr_multiplier": mat_query_context_lr_multiplier,
        "parameter_lr_multipliers": parameter_lr_multipliers,
        "shuffle_agents": shuffle_agents,
        "preserve_inactive_prefix_structure": preserve_inactive_prefix_structure,
        "experiment_run_name": experiment_run_name,
        "scenario_name": scenario_name,
        "ccd_iterations": ccd_iterations,
        "settle_initial_reset": True,
        "scenario_kwargs": scenario_kwargs,
    }
    if policy_variant != "mat_qcs":
        extra_run_metadata["policy_variant"] = policy_variant

    run_with_discord_notification(
        run_name=f"{experiment_run_name}/{variant_name}/{run_id}",
        run_dir=str(run_dir),
        total_timesteps=total_timesteps,
        algorithm=ppo,
        run=lambda: ppo.learn(
            max_total_timesteps=total_timesteps,
            run_dir=str(run_dir),
            log_interval=1,
            save_interval=save_interval,
            save_optimizer=save_optimizer,
            best_rotation_n=1,
            extra_run_metadata=extra_run_metadata,
            logging_console_keys=logging_console_keys,
            post_iteration_hooks=[scheduled_recording_hook],
        ),
    )

    print("Training Finished.")
    env.close()


def _make_mat_parameter_lr_multipliers(
        *,
        policy_variant: PolicyVariant,
        decoder_lr_multiplier: float,
        query_context_lr_multiplier: float,
) -> dict[str, float]:
    parameter_lr_multipliers: dict[str, float] = {}
    if decoder_lr_multiplier != 1.0:
        if policy_variant in {"mat_qcs", "mat_qcc", "mat_orig"}:
            parameter_lr_multipliers["decoder"] = decoder_lr_multiplier
        else:
            logger.warning(f"Ignoring decoder LR multiplier for policy_variant={policy_variant!r}")

    if query_context_lr_multiplier != 1.0:
        if policy_variant in {"mat_qcs", "mat_qcc"}:
            for prefix in (
                    "query_input_norm",
                    "query_encoder",
                    "query_token_norm",
                    "context_input_norm",
                    "context_encoder",
                    "context_token_norm",
            ):
                parameter_lr_multipliers[prefix] = query_context_lr_multiplier
        else:
            logger.warning(f"Ignoring query/context LR multiplier for policy_variant={policy_variant!r}")

    return parameter_lr_multipliers


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
        continuous_action_dist: ContinuousActionDistVariant,
        initial_stickiness: float,
        gsde_init_stds: list[float],
        mat_add_agent_embeddings: bool,
        mat_decoder_self_attention_mode: MATQCSDecoderSelfAttentionMode,
        act_fn_cls: ActivationFactory,
        mat_init_gains: MATInitGains,
        mat_normalization: MATNormalizationConfig,
        assume_agent_mask_is_active_prefix: bool,
) -> PPOPolicy | MAPPOPolicy | MATQCSPolicy | MATQCCPolicy | MATDecPolicy | MATOrigPolicy:
    continuous_config = make_continuous_config(
        variant=continuous_action_dist,
        initial_stickiness=initial_stickiness,
        gsde_init_stds=gsde_init_stds,
    )
    bernoulli_config = BernoulliConfig(
        initial_prob=0.8,
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
        local_obs_encoder_hidden_dims=[enc_d_model, enc_d_model],
        global_obs_encoder_hidden_dims=[enc_d_model],
        normalize_obs_inputs=mat_normalization.normalize_obs_inputs,
        normalize_tokens=mat_normalization.normalize_encoder_tokens,
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

    if policy_variant == "mappo":
        return MAPPOPolicy(
            env=env,
            config=MAPPOPolicyConfig(
                actor_config=MAPPOActorConfig(
                    hidden_dims=[768, 512, 512, 384],
                    shared_encoder_latent_dim=256,
                    actor_head_hidden_dims=[128],
                    latent_pi_dim=128,
                    act_fun_class=act_fn_cls,
                ),
                critic_config=MAPPOCriticConfig(
                    deep_set_config=DeepSetCriticConfig(
                        local_projection_hidden_dims=[256, 128],
                        value_regressor_hidden_dims=[256, 128],
                    ),
                    act_fun_class=act_fn_cls,
                    use_popart=use_popart,
                ),
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
            ),
        )

    mat_qcs_critic_config = MATQCSCriticConfig(
        n_local_projection_hidden_layers=2,
        n_value_regressor_hidden_layers=1,
        use_popart=use_popart,
        popart_config=popart_config,
        local_projection_init_gain=mat_init_gains.critic_local_projection,
        value_regressor_init_gain=mat_init_gains.critic_value_regressor,
        value_head_init_gain=mat_init_gains.critic_value_head,
    )

    if policy_variant == "mat_qcs":
        return MATQCSPolicy(
            env=env,
            config=MATQCSPolicyConfig(
                encoder_config=mat_encoder_config,
                decoder_config=MATQCSDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    add_agent_embeddings=mat_add_agent_embeddings,
                    token_encoder_init_gain=mat_init_gains.decoder_token_encoder,
                    token_encoder_projection_init_gain=mat_init_gains.decoder_token_encoder_projection,
                    transformer_ff_init_gain=mat_init_gains.decoder_transformer_ff,
                    actor_head_init_gain=mat_init_gains.actor_head,
                    query_encoder_hidden_dims=[dec_d_model],
                    context_encoder_hidden_dims=[dec_d_model],
                    memory_dims=None,
                    self_attention_mode=mat_decoder_self_attention_mode,
                    normalize_query_input=mat_normalization.normalize_query_input,
                    normalize_context_input=mat_normalization.normalize_context_input,
                    normalize_memory_input=mat_normalization.normalize_memory_input,
                    normalize_query_tokens=mat_normalization.normalize_query_tokens,
                    normalize_context_tokens=mat_normalization.normalize_context_tokens,
                    normalize_memory_tokens=mat_normalization.normalize_memory_tokens,
                    normalize_actor_head_input=mat_normalization.normalize_actor_head_input,
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                ),
                critic_config=mat_qcs_critic_config,
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
                compile_modules=compile_policy_modules,
                compile_mode=policy_compile_mode,
                action_net_init_gain=mat_init_gains.action_net,
            ),
        )

    if policy_variant == "mat_qcc":
        return MATQCCPolicy(
            env=env,
            config=MATQCCPolicyConfig(
                encoder_config=mat_encoder_config,
                decoder_config=MATQCCDecoderConfig(
                    d_model=dec_d_model,
                    nhead=dec_nhead,
                    num_layers=2,
                    dim_feedforward=dec_d_model * 2,
                    add_agent_embeddings=mat_add_agent_embeddings,
                    token_encoder_init_gain=mat_init_gains.decoder_token_encoder,
                    token_encoder_projection_init_gain=mat_init_gains.decoder_token_encoder_projection,
                    transformer_ff_init_gain=mat_init_gains.decoder_transformer_ff,
                    actor_head_init_gain=mat_init_gains.actor_head,
                    query_encoder_hidden_dims=[dec_d_model],
                    context_encoder_hidden_dims=[dec_d_model],
                    memory_dims=None,
                    normalize_query_input=mat_normalization.normalize_query_input,
                    normalize_context_input=mat_normalization.normalize_context_input,
                    normalize_memory_input=mat_normalization.normalize_memory_input,
                    normalize_query_tokens=mat_normalization.normalize_query_tokens,
                    normalize_context_tokens=mat_normalization.normalize_context_tokens,
                    normalize_memory_tokens=mat_normalization.normalize_memory_tokens,
                    normalize_actor_head_input=mat_normalization.normalize_actor_head_input,
                    assume_agent_mask_is_active_prefix=assume_agent_mask_is_active_prefix,
                ),
                critic_config=mat_qcs_critic_config,
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
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
                critic_config=mat_qcs_critic_config,
                actor_head_hidden_dims=[dec_d_model],
                dropout=0.0,
                act_fn_cls=act_fn_cls,
                continuous_config=continuous_config,
                bernoulli_config=bernoulli_config,
                max_agents=20,
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
                max_agents=20,
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
