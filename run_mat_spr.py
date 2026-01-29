import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics, NormalizeReward
from loguru import logger
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import GSDEParams
from swarmbots.learn.algos.mat.wm.mat_spr_policy import MATSPRPolicy
from swarmbots.learn.algos.ppo.wm.ppo_wm import PPOWM
from swarmbots.learn.algos.ppo.ppo import AutomaticLearningRate
from swarmbots.learn.env_wrappers.obs_normalization.feature_wise_obs_norm_wrapper import (
    FeatureWiseObsNormWrapper,
)
from swarmbots.learn.env_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.transition_obs_wrapper import TransitionObsWrapper
from swarmbots.learn.gsde_reset import GSDEProbabilityResetMode
from swarmbots.learn.summary_statistics import SummaryStatisticsFormat
from swarmbots.mj_env.float_or_dist import UniformDistParams
from swarmbots.mj_env.scenarios.obstacle_street_scenario import ObstacleStreetScenario
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm, RandomUnitLocationsConfig
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


def make_env_fn(
    episode_length: int,
    scenario_kwargs: dict[str, Any] | None = None,
    render_mode: str | None = None,
) -> Callable[[], SwarmBotsEnv]:
    if scenario_kwargs is None:
        scenario_kwargs = {}
        
    def _init() -> SwarmBotsEnv:
        scenario = ObstacleStreetScenario.no_payload_no_opening_one_wall_no_poles(
            # swarm=HomogeneousSwarm(
            #     unit_start_locations=RandomUnitLocationsConfig(
            #         num_units=4,
            #         pairwise_distance=0.605,
            #         max_distance=1.5,
            #     ),
            #     randomize_unit_orientations=True,
            # ),
            randomize_unit_orientations=False,
            first_wall_distance=UniformDistParams(1.0, 3.0),
            **scenario_kwargs
        )
        return SwarmBotsEnv(
            scenario=scenario,
            episode_length=episode_length,
            render_mode=render_mode,
            camera=0
        )
    return _init



def _build_feature_wise_obs_norm_indices(
    env_settings: dict[str, Any],
    local_obs_dim: int,
    global_obs_dim: int,
) -> tuple[list[int], list[int], list[int], list[int]]:
    scenario_settings = env_settings["scenario"]
    swarm_config = scenario_settings["swarm"]["config"]
    limbs_per_unit = len(swarm_config["unit_config"])
    include_connectors_xpos_in_obs = bool(scenario_settings["include_connectors_xpos_in_obs"])
    include_connectors_xquat_in_obs = bool(scenario_settings["include_connectors_xquat_in_obs"])
    quat_rot6d_representation = bool(scenario_settings.get("quat_rot6d_representation", False))

    num_hinges = limbs_per_unit * 2
    free_joint_rot_dim = 6 if quat_rot6d_representation else 4
    qpos_obs_dim = 3 + free_joint_rot_dim + 2 * num_hinges
    qvel_dim = 6 + num_hinges
    connector_obs_dim = limbs_per_unit * 5
    connectors_xpos_dim = limbs_per_unit * 3 if include_connectors_xpos_in_obs else 0
    connectors_xquat_dim = (
        limbs_per_unit * free_joint_rot_dim if include_connectors_xquat_in_obs else 0
    )
    expected_local_dim = (
        qpos_obs_dim + qvel_dim + connector_obs_dim + connectors_xpos_dim + connectors_xquat_dim
    )
    if local_obs_dim != expected_local_dim:
        raise ValueError(
            "Unexpected local_obs_dim for feature-wise normalization. "
            f"{local_obs_dim=} {expected_local_dim=} {limbs_per_unit=}"
        )
    
    connectors_xpos_offset = qpos_obs_dim + qvel_dim + connector_obs_dim
    connectors_xquat_offset = connectors_xpos_offset + connectors_xpos_dim

    local_scalar_indices = (
            list(range(3)) 
            + list(range(qpos_obs_dim, qpos_obs_dim + qvel_dim)) 
            + list(range(connectors_xpos_offset, connectors_xpos_offset + connectors_xpos_dim))
        )
    local_quat_starts = [] if quat_rot6d_representation else [3]
    if include_connectors_xquat_in_obs and not quat_rot6d_representation:
        local_quat_starts.extend(
            connectors_xquat_offset + 4 * i for i in range(limbs_per_unit)
        )
    

    if scenario_settings.get("payload_type") is not None:
        expected_global_obs_dim = 3 + free_joint_rot_dim
        if global_obs_dim < expected_global_obs_dim:
            raise ValueError(
                "Expected global_obs to include payload pos+rotation when payload_type is set. "
                f"{global_obs_dim=}"
            )
        global_scalar_indices = list(range(3))
        global_quat_starts = [] if quat_rot6d_representation else [3]
    else:
        global_scalar_indices = []
        global_quat_starts = []
    
    return (
        local_scalar_indices,
        local_quat_starts,
        global_scalar_indices,
        global_quat_starts,
    )


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <level>{message}</level>",
    )

    n_envs = 6
    # unit_start_locations = [
    #     (0.0, 0.0, 0.0),
    #     (-0.6, 0, 0),
    # ]
    episode_length = 512
    total_timesteps = 100_000_000
    save_interval = 500
    world_model_num_next_steps = 3
    world_model_loss_coef = 0.5
    world_model_target_tau = 0.005

    # =====  ID  =====
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ===== LOAD =====
    load_path: str | None = None
    # load_path = "runs/mat_spr_swarm_bots/2026-01-27_22-50-41/models/model_6048768_steps_stopped.pt"
    std: float | None = None

    # ===== DEVICE =====
    use_cuda = False and torch.cuda.is_available()
    rollout_device = torch.device("cpu")
    train_device = torch.device("cuda" if use_cuda else "cpu")

    logger.info(f'{rollout_device = }')
    logger.info(f'{train_device = }')

    if load_path is not None:
        if not load_path.endswith('.pt'):
            logger.error('load_path is missing .pt')
        logger.info(f'{load_path = }')
        run_id = load_path.split('/')[2]
        logger.info(f'{run_id = }')

    run_dir = f"runs/mat_spr_swarm_bots/{run_id}/"
    save_optimizer = True

    scenario_kwargs = {
    }

    env_fns = [
        make_env_fn(
            episode_length=episode_length,
            scenario_kwargs=scenario_kwargs,
            render_mode=None
        )
        for _ in range(n_envs)
    ]

    print("Creating dummy env for capturing settings...")
    dummy_env = env_fns[0]()
    env_settings = dummy_env.get_settings()
    local_obs_dim = int(dummy_env.observation_space["local_obs"].shape[-1])
    global_obs_dim = int(dummy_env.observation_space["global_obs"].shape[-1])
    hidden_vars_dim = int(dummy_env.observation_space["hidden_vars"].shape[-1])
    dummy_env.close()
    del dummy_env
    print("Env settings captured.")

    (
        local_scalar_feature_indices,
        local_quaternion_indices,
        global_scalar_feature_indices,
        global_quaternion_indices,
    ) = _build_feature_wise_obs_norm_indices(
        env_settings=env_settings,
        local_obs_dim=local_obs_dim,
        global_obs_dim=global_obs_dim,
    )
    hidden_vars_scalar_feature_indices = list(range(hidden_vars_dim))
    hidden_vars_quaternion_indices: list[int] = []

    def make_record_env() -> SwarmBotsLearnEnvWrapper:
        record_env = SyncVectorEnv([
            make_env_fn(
                episode_length=episode_length,
                scenario_kwargs=scenario_kwargs,
                render_mode='rgb_array'
            )
        ])
        record_env = RecordEpisodeStatistics(record_env)
        record_env = FeatureWiseObsNormWrapper(
            record_env,
            obs_key="local_obs",
            scalar_feature_indices=local_scalar_feature_indices,
            quaternion_indices=local_quaternion_indices,
        )
        record_env = FeatureWiseObsNormWrapper(
            record_env,
            obs_key="global_obs",
            scalar_feature_indices=global_scalar_feature_indices,
            quaternion_indices=global_quaternion_indices,
        )
        record_env = FeatureWiseObsNormWrapper(
            record_env,
            obs_key="hidden_vars",
            scalar_feature_indices=hidden_vars_scalar_feature_indices,
            quaternion_indices=hidden_vars_quaternion_indices,
        )
        record_env = TransitionObsWrapper(record_env)
        record_env = NormalizeReward(record_env, gamma=gamma)
        record_env = SwarmBotsLearnEnvWrapper(record_env, device=rollout_device)

        return record_env

    print('Creating vector env...')
    if sys.gettrace() is None:
        vector_env = AsyncVectorEnv(env_fns)
    else:
        vector_env = SyncVectorEnv(env_fns[:1])
    print(f"Created {type(vector_env)} with {n_envs} environments.")

    if isinstance(vector_env, SyncVectorEnv):
        for _ in range(10):
            logger.warning('USING SYNC VECTOR ENV')

    gamma = 0.987
    
    print("Wrapping...")
    vector_env = RecordEpisodeStatistics(vector_env)
    vector_env = FeatureWiseObsNormWrapper(
        vector_env,
        obs_key="local_obs",
        scalar_feature_indices=local_scalar_feature_indices,
        quaternion_indices=local_quaternion_indices,
    )
    vector_env = FeatureWiseObsNormWrapper(
        vector_env,
        obs_key="global_obs",
        scalar_feature_indices=global_scalar_feature_indices,
        quaternion_indices=global_quaternion_indices,
    )
    vector_env = FeatureWiseObsNormWrapper(
        vector_env,
        obs_key="hidden_vars",
        scalar_feature_indices=hidden_vars_scalar_feature_indices,
        quaternion_indices=hidden_vars_quaternion_indices,
    )
    vector_env = TransitionObsWrapper(vector_env)
    vector_env = NormalizeReward(vector_env, gamma=gamma)

    print("Wrapping with SwarmBotsLearnEnvWrapper...")
    env = SwarmBotsLearnEnvWrapper(vector_env, device=rollout_device)
    
    print(f"Environment initialized.")
    print(f"n_agents: {env.n_agents}")
    print(f"local_obs_dim: {env.local_obs_dim}")
    print(f"global_obs_dim: {env.global_obs_dim}")
    print(f"actuators_dim: {env.actuators_dim}")
    print(f"connectors_dim: {env.connectors_dim}")

    print("Initializing Policy...")
    policy = MATSPRPolicy(
        env=env,
        local_obs_encoder_hidden_dims=[128, 128],
        action_encoder_hidden_dims=[32],
        d_model=64,
        d_model_decoder=32,
        nhead_encoder=2,
        nhead_decoder=1,
        num_layers_encoder=2,
        num_layers_decoder=2,
        dim_feedforward_encoder=128,
        dim_feedforward_decoder=64,
        dropout=0.0,
        n_critic_local_projection_hidden_layers=1,
        n_critic_value_regressor_hidden_layers=1,
        cross_attn_first=True,
        act_fn_cls=nn.GELU,
        continuous_config=GSDEParams(
            base_std=0.45,
            latent_sde_dim=None,
            std_learnable=True,
            full_std=True,
            sde_learn_features=False,
            log_std_clamp_range=(-20.0, 2.0),
            normalize_latent_sde_by_dim=True
        ),
        bernoulli_initial_prob=0.75,
        # SPR
        d_model_transition_model=64,
        nhead_transition_model=2,
        num_layers_transition_model=2,
        dim_feedforward_transition_model=128,
        transition_model_coembed_hidden_dims=[96],
        spr_projection_dims=[48],
        residual_predictor=True
    )
    print(policy)

    lr = 1e-4
    # if load_path is not None:
    #     lr = 2e-5
    #     logger.warning(f'Setting {lr = :.2e}')

    print("Initializing PPO Algorithm...")
    auto_lr = AutomaticLearningRate(
        initial_lr=lr,
        max_lr=3e-4,
        max_kl=0.1,
        max_kl_hit_decay_factor=lambda kl: np.clip(0.9 - kl, 0.4, 0.8),
        min_epochs=2,
        min_epochs_hit_decay_factor=lambda epoch: 0.9 if epoch == 1 else 0.75,
        increase_after_n_iters=2,
        increase_factor=1.3,
    )
    ppo = PPOWM(
        policy=policy,
        env=env,
        learning_rate=auto_lr,
        n_episodes_per_rollout=n_envs,
        max_episode_length=episode_length,
        batch_size=256,
        n_epochs=5,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.04,
        gsde_reset_mode=GSDEProbabilityResetMode(probability=1/6),
        ent_coef=0.001,
        value_loss_fn=nn.SmoothL1Loss(),
        train_device=train_device,
        rollout_device=rollout_device,
        world_model_num_next_steps=world_model_num_next_steps,
        world_model_loss_coef=world_model_loss_coef,
        world_model_target_tau=world_model_target_tau,
    )

    if load_path:
        logger.info(f"Loading model from {load_path}")
        ppo.load(load_path)

        if std:
            logger.warning(f'Setting {std = }')
            ppo.policy.action_dist.set_std(std)

    print("Starting training...")
    ppo.learn(
        max_total_timesteps=total_timesteps,
        run_dir=run_dir,
        log_interval=1,
        save_interval=save_interval,
        save_optimizer=save_optimizer,
        best_rotation_n=3,
        extra_run_metadata={
            'load_path': load_path,
            'env_settings': env_settings,
            'script': Path(__file__).read_text(encoding='utf-8')
        },
        logging_console_keys=[
            ('iteration', '5', 'it'),
            ('timesteps', '8', 'steps'),
            ('total_updates', '6', 'tot_upd'),
            ('act0', SummaryStatisticsFormat(histogram=10)),
            ('act1', SummaryStatisticsFormat(histogram=2)),
            ('std0', SummaryStatisticsFormat(mean='.3f', std='.3f', min_value='.3f', max_value='.3f')),
            ('updates', '3', 'upd'),
            ('approx_kl', SummaryStatisticsFormat(mean='.3f', std='.3f', max_value='.3f')),
            ('clip_frac', None),
            ('ratio', SummaryStatisticsFormat(mean='.3f', std='.3f', min_value='.1e', max_value='.3f')),
            ('wm_loss_scaled', None, 'wm_loss'),
            ('val_loss_scaled', None, 'val_loss'),
            ('expl_var', '.3f'),
            ('ep_rew', SummaryStatisticsFormat(mean=' .2f', std='.2f', max_value=' .2f')),
            ('ep_rew_ema', ' .3f'),
            ('best_ep_rew_ema', ' .3f', 'best_ema'),
            ('fps', None),
        ],
        make_record_env=make_record_env
    )
    
    print("Training Finished.")

    env.close()


if __name__ == "__main__":
    main()
