"""Build learning algorithms and policy presets for registered benchmark tasks."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn

from swarmbots.benchmark.registry import make_env
from swarmbots.learn.algos.base_algorithm import BaseAlgorithm
from swarmbots.learn.algos.ppo.ppo import PPO, AutomaticLearningRate, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamplerConfig
from swarmbots.learn.algos.r_mat.r_ppo_wm_sampler import RPPOWMSamplerConfig
from swarmbots.learn.algos.sac.recurrent_sac import RecurrentSAC
from swarmbots.learn.algos.sac.sac import SAC
from swarmbots.learn.algos.td3 import DDPG, TD3, RecurrentTD3
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NextObsPredWrapper, NOPWorldModelConfig
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEProbabilityResetMode
from swarmbots.learn.nn_components.feed_forward import MLPConfig
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.presets.policy_factory import (
    ContinuousActionDistVariant,
    _is_deterministic_policy_variant,
    _is_off_policy_variant,
    _is_sac_policy_variant,
    _make_base_policy,
    _make_mat_parameter_lr_multipliers,
    set_actuator_gsde_init_joint_stds,
    wrap_vec_env,
)
from swarmbots.learn.presets.model_scale import DEFAULT_MODEL_SCALE, normalize_model_scale, ppo_nop_at_scale
from swarmbots.learn.presets.tmasac import make_tmasac_options
from swarmbots.learn.presets.transformer import MATInitGains, MATNormalizationConfig, NOPInitGains
from swarmbots.learn.presets.variants import VARIANT_CONFIGS
from swarmbots.learn.scheduling.auto_lr_updater import make_auto_lr_updater
from swarmbots.learn.scheduling.cosine_scheduler import CosineSchedulerConfig
from swarmbots.learn.scheduling.schedulers import ScheduleUnit
from swarmbots.learn.swarmbots_obs_indices import build_obs_indices


def list_variants(*, include_hidden: bool = False) -> tuple[str, ...]:
    """List core presets; opt in to MLP baselines and optional controls."""
    return tuple(name for name, config in VARIANT_CONFIGS.items() if include_hidden or not config.hidden)


def _variant_options(variant: str, *, model_scale: str | None = DEFAULT_MODEL_SCALE) -> tuple[dict[str, Any], dict[str, Any]]:
    if variant not in VARIANT_CONFIGS:
        raise ValueError(f"Unknown learning variant {variant!r}. Available: {', '.join(list_variants())}")
    model_scale = normalize_model_scale(model_scale)
    config = VARIANT_CONFIGS[variant]
    if config.tmasac_variant is not None:
        policy_options, algorithm_options = make_tmasac_options(config.tmasac_variant)
    else:
        policy_options = {"policy_variant": config.policy_variant}
        algorithm_options = {}
    deterministic = _is_deterministic_policy_variant(policy_options["policy_variant"])
    if deterministic or variant in {"masac_mlp", "masac_deepset"}:
        policy_options["mat_encoder_transformer_ff_config"] = MLPConfig(hidden_dims=[512, 512])
    default_distribution = None if deterministic else (
        "predicted_std_gaussian"
        if _is_sac_policy_variant(policy_options["policy_variant"])
        else "sign_magnitude_beta"
    )
    policy_options.update(
        model_scale=model_scale,
        continuous_action_dist=default_distribution,
        use_nop=config.use_nop,
        mat_use_agent_attention=config.use_agent_attention,
    )
    return policy_options, algorithm_options


def _observation_indices(env: Any) -> ObsIndices:
    dimensions = {
        name: int(env.single_observation_space[key].shape[-1])
        for name, key in (
            ("local_obs_dim", "local_obs"),
            ("global_obs_dim", "global_obs"),
            ("hidden_local_vars_dim", "hidden_local_vars"),
            ("hidden_global_vars_dim", "hidden_global_vars"),
        )
    }
    return build_obs_indices(env_settings=env.get_settings(), **dimensions)


def _add_ppo_nop(
    policy: BasePolicy, env: BaseLearnEnvWrapper, indices: ObsIndices, options: Mapping[str, Any]
) -> BasePolicy:
    latent_dim = getattr(policy, "local_latent_dim", None)
    if latent_dim is None:
        latent_dim = policy.config.encoder_config.d_model
    latent_dim = int(latent_dim)
    transition_dim = options.get("transition_model_d_model", 128)
    init_gains = options.get("nop_init_gains", NOPInitGains())
    world_model_config = NOPWorldModelConfig(
        n_agents=env.n_agents,
        local_latent_dim=latent_dim,
        action_dim=env.action_space.total_agent_action_dim,
        world_model_loss_coef=options.get("world_model_loss_coef", 0.1),
        compile_modules=options["compile_world_model_modules"],
        compile_mode=options["policy_compile_mode"],
        act_fn_cls=options["act_fn_cls"],
        wm_pre_transition_init_gain=init_gains.pre_transition,
        transition_model_coembed_init_gain=init_gains.transition_coembed,
        transition_model_transformer_ff_init_gain=init_gains.transition_transformer_ff,
        transition_model_head_init_gain=init_gains.transition_head,
        wm_pre_predictors_init_gain=init_gains.pre_predictors,
        wm_predictor_init_gain=init_gains.predictors,
        wm_pre_transition_dims=[latent_dim],
        d_model_transition_model=transition_dim,
        nhead_transition_model=options.get("transition_model_nhead", 2),
        num_layers_transition_model=2,
        dim_feedforward_transition_model=transition_dim * 2,
        add_agent_embeddings_transition_model=options.get("nop_add_agent_embeddings_transition_model", False),
        transition_model_coembed_hidden_dims=[transition_dim],
        wm_pre_predictors_dims=[transition_dim, transition_dim],
        wm_scalar_predictor_hidden_dims=[],
        wm_angle_predictor_hidden_dims=[],
        wm_rot6d_predictor_hidden_dims=[],
        wm_binary_predictor_hidden_dims=[],
        scalar_loss_fn="smooth_l1",
        next_obs_pred_config=NextObsPredConfig(
            local_scalar_target_indices=indices.local_scalar_indices,
            local_angle_target_indices=indices.local_angle_indices,
            local_rot6d_target_indices=indices.local_rot6d_indices,
            local_binary_target_indices=indices.local_binary_indices,
        ),
    )
    if options.get("model_scale") is not None:
        world_model_config = ppo_nop_at_scale(world_model_config, options["model_scale"])
    wrapped = NextObsPredWrapper(policy=policy, world_model_config=world_model_config)
    if hasattr(policy, "model_scale"):
        wrapped.model_scale = policy.model_scale
    return wrapped


def _make_policy_env(
    benchmark_id: str,
    *,
    num_envs: int,
    device: str | torch.device,
    seed: int,
    episode_length: int | None,
    scenario_kwargs: Mapping[str, object] | None,
    env_kwargs: Mapping[str, Any] | None,
    policy_options: Mapping[str, Any],
    gamma: float = 0.99,
    compile_modules: bool = False,
) -> tuple[BaseLearnEnvWrapper, BasePolicy]:
    policy_options = dict(policy_options)
    policy_options["model_scale"] = normalize_model_scale(policy_options.get("model_scale"))
    is_off_policy = _is_off_policy_variant(policy_options["policy_variant"])
    use_popart = policy_options.get("use_popart", not is_off_policy)
    torch.manual_seed(seed)
    vector_env = make_env(
        benchmark_id,
        num_envs=num_envs,
        device=device,
        seed=seed,
        episode_length=episode_length,
        scenario_kwargs=scenario_kwargs,
        **dict(env_kwargs or {}),
    )
    env = vector_env
    try:
        indices = _observation_indices(vector_env)
        env = wrap_vec_env(
            vector_env=vector_env,
            obs_indices=indices,
            gamma=gamma,
            use_popart=use_popart,
            rollout_device=vector_env.device,
        )
        actuators_per_limb = env.actuators_dim // env.connectors_dim
        env.action_space.seed(seed)
        gsde_stds = [0.25, 0.30] if actuators_per_limb == 2 else [0.25] * actuators_per_limb
        policy_builder_options = {
            "use_popart": use_popart,
            "enc_d_model": 256,
            "enc_nhead": 4,
            "dec_d_model": 128,
            "dec_nhead": 2,
            "popart_beta": 5e-4,
            "popart_init_sigma": 0.65,
            "compile_policy_modules": compile_modules,
            "policy_compile_mode": "default",
            "compile_world_model_modules": compile_modules,
            "gsde_init_stds": gsde_stds,
            "mat_add_agent_embeddings": False,
            "act_fn_cls": nn.GELU,
            "mat_init_gains": MATInitGains(),
            "mat_normalization": MATNormalizationConfig(),
            "world_model_num_next_steps": 3,
            "assume_agent_mask_is_active_prefix": True,
        }
        policy_builder_options.update(policy_options)
        policy = _make_base_policy(env=env, obs_indices=indices, **policy_builder_options)
        if policy_builder_options["use_nop"] and not is_off_policy:
            policy = _add_ppo_nop(policy, env, indices, policy_builder_options)
        set_actuator_gsde_init_joint_stds(
            policy=policy,
            actuators_per_limb=actuators_per_limb,
            joint_stds=policy_builder_options["gsde_init_stds"],
        )
        return env, policy.to(vector_env.device)
    except BaseException:
        env.close()
        raise


def make_training(
    benchmark_id: str,
    variant: str = "mat_qcx",
    *,
    num_envs: int = 1024,
    device: str | torch.device = "cuda",
    seed: int = 42,
    episode_length: int | None = None,
    scenario_kwargs: Mapping[str, object] | None = None,
    env_kwargs: Mapping[str, Any] | None = None,
    continuous_action_dist: ContinuousActionDistVariant | None = None,
    use_nop: bool | None = None,
    model_scale: str | None = DEFAULT_MODEL_SCALE,
    compile_modules: bool = False,
    rollout_steps_per_env: int | None = None,
    policy_kwargs: Mapping[str, Any] | None = None,
    algorithm_kwargs: Mapping[str, Any] | None = None,
) -> PPO | SAC | RecurrentSAC | TD3 | DDPG:
    """Create a trainer owning its wrapped environment; the caller must close it.

    Architectures and optimizer defaults come from the selected preset. Task
    settings, including episode length and settled resets, come from the registry.
    ``model_scale`` defaults to "5M NOP1M": online main-policy parameters
    excluding frozen targets, plus a separate NOP budget. Pass None to use
    explicit architecture widths. Fixed 2.5M/0.75M and 10M/2M layouts are also available;
    sizes are declared per architecture and are not fitted to task shapes.
    ``use_nop=False`` disables the auxiliary model without changing main widths.
    ``policy_kwargs`` overrides the arguments to the preset policy builder;
    ``algorithm_kwargs`` overrides arguments to PPO/SAC/RecurrentSAC/TD3/DDPG.
    """
    policy_options, algorithm_options = _variant_options(variant, model_scale=model_scale)
    policy_options.update(policy_kwargs or {})
    algorithm_options.update(algorithm_kwargs or {})
    if continuous_action_dist is not None:
        policy_options["continuous_action_dist"] = continuous_action_dist
    if use_nop is not None:
        policy_options["use_nop"] = use_nop
    policy_variant = policy_options["policy_variant"]
    is_sac = _is_sac_policy_variant(policy_variant)
    is_deterministic = _is_deterministic_policy_variant(policy_variant)
    is_off_policy = is_sac or is_deterministic
    is_recurrent_ppo = policy_variant in {"mat_ind_lstm", "mat_qcx_lstm"}
    if (
        "use_popart" in policy_options
        and "use_popart" in algorithm_options
        and policy_options["use_popart"] != algorithm_options["use_popart"]
    ):
        raise ValueError("use_popart must agree in policy_kwargs and algorithm_kwargs")
    use_popart = policy_options.setdefault("use_popart", algorithm_options.pop("use_popart", not is_off_policy))
    gamma = algorithm_options.get("gamma", 0.99)
    if rollout_steps_per_env is None:
        rollout_steps_per_env = 1 if is_off_policy else 4
    if rollout_steps_per_env <= 0:
        raise ValueError("rollout_steps_per_env must be positive")
    env, policy = _make_policy_env(
        benchmark_id,
        num_envs=num_envs,
        device=device,
        seed=seed,
        episode_length=episode_length,
        scenario_kwargs=scenario_kwargs,
        env_kwargs=env_kwargs,
        policy_options=policy_options,
        gamma=gamma,
        compile_modules=compile_modules,
    )
    vector_env = env.unwrapped
    device = vector_env.device
    try:
        shared_options = {
            "policy": policy,
            "env": env,
            "gamma": gamma,
            "train_device": device,
            "rollout_device": device,
            "record_device": device,
            "max_grad_norm": 2.0,
            "gsde_reset_mode": GSDEProbabilityResetMode(probability=1 / 6),
        }
        rollout_samples = num_envs * rollout_steps_per_env
        if is_off_policy:
            recurrent = policy.requires_recurrent_training()
            algorithm_class = RecurrentSAC if recurrent else SAC
            if is_deterministic:
                algorithm_class = DDPG if policy_variant.startswith("maddpg") else RecurrentTD3 if recurrent else TD3
            algorithm_config = {
                "learning_rate": 3e-4,
                "buffer_capacity_per_env": vector_env.episode_length * 2,
                "learning_starts": max(10_000, rollout_samples * 4),
                "batch_size": rollout_samples,
                "rollout_steps_per_iteration": rollout_samples,
                "gradient_steps": 8,
                "replay_storage_device": device,
                "tau": 0.005,
            }
            if is_sac:
                algorithm_config.update(ent_coef="auto_0.05", ent_coef_learning_rate=1e-3, target_entropy="auto_0.5")
            if is_deterministic and recurrent:
                algorithm_config.update(
                    batch_size=16,
                    burn_in_steps=32,
                    learning_steps=64,
                    temporal_state_store_interval=16,
                    temporal_state_storage_dtype=torch.float16,
                )
            algorithm_config.update(shared_options)
            algorithm_config.update(algorithm_options)
            if "buffer_capacity_per_env" not in (algorithm_kwargs or {}):
                algorithm_config["buffer_capacity_per_env"] = max(
                    algorithm_config["buffer_capacity_per_env"],
                    math.ceil(max(algorithm_config["learning_starts"], algorithm_config["batch_size"]) / num_envs),
                )
            return algorithm_class(**algorithm_config)
        n_epochs = algorithm_options.get("n_epochs", 8)
        target_kl = algorithm_options.get("target_kl", 0.002)
        if "learning_rate" in algorithm_options:
            learning_rate = algorithm_options["learning_rate"]
        else:
            if target_kl is None:
                raise ValueError("AutomaticLearningRate requires target_kl; supply a fixed learning_rate to disable it.")
            warm_lr = 1e-4
            cold_lr = warm_lr * 5e-3
            learning_rate = AutomaticLearningRate(
                initial_lr=cold_lr,
                max_lr=8e-4,
                updater=make_auto_lr_updater(
                    early_stop_epoch_decay_limit=math.ceil(n_epochs * 0.75),
                    max_kl_div=target_kl * 1.55,
                    warm_scheduler_config=CosineSchedulerConfig(
                        unit=ScheduleUnit.ITERATIONS,
                        duration=200,
                        start_value=cold_lr,
                        final_value=warm_lr,
                    ),
                ),
            )
        sampler_options = {
            "batch_size": num_envs if is_recurrent_ppo else rollout_samples,
            "num_next_steps": policy_options.get("world_model_num_next_steps", 3),
            "compile_wm_window_helper": compile_modules,
        }
        if is_recurrent_ppo:
            sampler = RPPOWMSamplerConfig(sequence_length=rollout_steps_per_env, **sampler_options)
        elif policy_options["use_nop"]:
            sampler = PPOWMSamplerConfig(**sampler_options)
        else:
            sampler = PPOSamplerConfig(batch_size=rollout_samples)
        algorithm_config = {
            "learning_rate": learning_rate,
            "rollout_mode": StepsRolloutMode(rollout_samples),
            "max_episode_length": vector_env.episode_length,
            "sampler_config": sampler,
            "n_epochs": n_epochs,
            "gae_lambda": 0.95,
            "clip_range": 0.05,
            "target_kl": target_kl,
            "mc_ent_coef": 0.0,
            "vf_coef": 2.0 if use_popart else 0.5,
            "use_popart": use_popart,
            "agent_logprob_reduction": "sum" if policy_variant == "ppo" else None,
            "parameter_lr_multipliers": _make_mat_parameter_lr_multipliers(
                policy_variant=policy_variant,
                mat_decoder_lr_multiplier=0.25,
                include_actor_head_lr_multiplier=False,
            ),
        }
        algorithm_config.update(shared_options)
        algorithm_config.update(algorithm_options)
        return PPO(**algorithm_config)
    except BaseException:
        env.close()
        raise


def train(
    benchmark_id: str,
    variant: str = "mat_qcx",
    *,
    total_timesteps: int = 100_000_000,
    run_dir: str | Path | None = None,
    load_path: str | Path | None = None,
    learn_kwargs: Mapping[str, Any] | None = None,
    **training_kwargs: Any,
) -> BaseAlgorithm:
    """Train a variant on a registered task, save its run, and close the environment.

    The returned trainer retains its trained policy and normalization statistics.
    ``total_timesteps`` is an absolute final counter when resuming a checkpoint.
    """
    algorithm = make_training(benchmark_id, variant, **training_kwargs)
    try:
        if load_path is not None:
            algorithm.load(load_path)
        options = dict(learn_kwargs or {})
        options.setdefault("enable_command_prompt", False)
        metadata = dict(options.pop("extra_run_metadata", None) or {})
        metadata.update(benchmark_id=benchmark_id, learning_variant=variant)
        options.update(
            max_total_timesteps=total_timesteps,
            run_dir=run_dir,
            extra_run_metadata=metadata,
        )
        algorithm.learn(**options)
        return algorithm
    finally:
        algorithm.env.close()
