import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv, VectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics, NormalizeReward
from loguru import logger
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import GSDEParams
from swarmbots.learn.algos.mat.wm.mat_nop_policy import MATNOPPolicy
from swarmbots.learn.algos.ppo.wm.ppo_wm import PPOWM
from swarmbots.learn.algos.ppo.ppo import AutomaticLearningRate, AutomaticLearningRateUpdateResult, \
    WholeEpisodesRolloutMode
from swarmbots.learn.env_wrappers.obs_normalization.feature_wise_obs_norm_wrapper import (
    FeatureWiseObsNormWrapper,
)
from swarmbots.learn.env_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.transition_obs_wrapper import TransitionObsWrapper
from swarmbots.learn.gsde_reset import GSDEProbabilityResetMode
from swarmbots.learn.summary_statistics import SummaryStatisticsFormat, SummaryStatistics
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.mj_env.float_or_dist_params import UniformDistParams
from swarmbots.mj_env.scenarios.obstacle_street_scenario import ObstacleStreetScenario
from swarmbots.mj_env.scenarios.scenario_presets import default_wall
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm, RandomLatticeUnitLocationsConfig
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


def make_env_fn(
    episode_length: int,
    scenario_kwargs: dict[str, Any] | None = None,
    render_mode: str | None = None,
) -> Callable[[], SwarmBotsEnv]:
    if scenario_kwargs is None:
        scenario_kwargs = {}
        
    def _init() -> SwarmBotsEnv:
        scenario = default_wall(
            swarm=HomogeneousSwarm(
                unit_start_locations=RandomLatticeUnitLocationsConfig(
                    num_units=5,
                    pairwise_distance=0.605,
                    max_radius=1.5,
                    num_unit_probs={
                        2: 0.25,
                        3: 0.25,
                        4: 0.25,
                        5: 0.25,
                    }
                ),
                randomize_unit_orientations=True,
            ),
            # unit_start_locations='8:hourglass',
            randomize_unit_orientations=True,
            first_wall_distance=2.0, # UniformDistParams(1.5, 2.5),
            **scenario_kwargs
        )
        return SwarmBotsEnv(
            scenario=scenario,
            episode_length=episode_length,
            render_mode=render_mode,
            camera=0
        )
    return _init


def wrap_vec_env(
        vector_env: SyncVectorEnv | AsyncVectorEnv,
        obs_indices: ObsIndices,
        gamma: float,
        rollout_device: torch.device
) -> SwarmBotsLearnEnvWrapper:

    vector_env = RecordEpisodeStatistics(vector_env)
    vector_env = FeatureWiseObsNormWrapper(
        vector_env,
        obs_key="local_obs",
        scalar_feature_indices=obs_indices.local_scalar_indices,
        quaternion_indices=obs_indices.local_quaternion_indices,
    )
    vector_env = FeatureWiseObsNormWrapper(
        vector_env,
        obs_key="global_obs",
        scalar_feature_indices=obs_indices.global_scalar_indices,
        quaternion_indices=obs_indices.global_quaternion_indices,
    )
    vector_env = FeatureWiseObsNormWrapper(
        vector_env,
        obs_key="hidden_vars",
        scalar_feature_indices=obs_indices.hidden_vars_scalar_indices,
        quaternion_indices=obs_indices.hidden_vars_quaternion_indices,
    )
    vector_env = TransitionObsWrapper(vector_env)
    vector_env = NormalizeReward(vector_env, gamma=gamma)

    env = SwarmBotsLearnEnvWrapper(vector_env, device=rollout_device)
    return env


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
    world_model_loss_coef = 0.1
    world_model_target_tau = None

    # =====  ID  =====
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ===== LOAD =====
    load_path: str | None = None
    # load_path = "runs/mat_nop_swarm_bots_obstacle_street/2026-02-06_16-54-32/models/model_9480192_steps_stopped.pt"
    std: float | None = None

    # ===== DEVICE =====
    use_cuda = True and torch.cuda.is_available()
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

    run_dir = f"runs/mat_nop_swarm_bots_obstacle_street/{run_id}/"
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

    obs_indices = build_obs_indices(
        env_settings=env_settings,
        local_obs_dim=local_obs_dim,
        global_obs_dim=global_obs_dim,
        hidden_vars_dim=hidden_vars_dim,
    )


    def make_record_env() -> SwarmBotsLearnEnvWrapper:
        record_env = SyncVectorEnv([
            make_env_fn(
                episode_length=episode_length,
                scenario_kwargs=scenario_kwargs,
                render_mode='rgb_array'
            )
        ])
        record_env = wrap_vec_env(
            vector_env=record_env,
            obs_indices=obs_indices,
            gamma=gamma,
            rollout_device=rollout_device,
        )
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

    gamma = 0.99
    
    print("Wrapping...")
    env = wrap_vec_env(
        vector_env=vector_env,
        obs_indices=obs_indices,
        gamma=gamma,
        rollout_device=rollout_device,
    )

    print(f"Environment initialized.")
    print(f"n_agents: {env.n_agents}")
    print(f"local_obs_dim: {env.local_obs_dim}")
    print(f"global_obs_dim: {env.global_obs_dim}")
    print(f"actuators_dim: {env.actuators_dim}")
    print(f"connectors_dim: {env.connectors_dim}")

    print("Initializing Policy...")
    policy = MATNOPPolicy(
        env=env,
        local_obs_encoder_hidden_dims=[192, 192],
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
        max_agents=20,
        # NOP
        wm_pre_transition_dims=[64],
        d_model_transition_model=64,
        nhead_transition_model=2,
        num_layers_transition_model=2,
        dim_feedforward_transition_model=128,
        transition_model_coembed_hidden_dims=[96],
        wm_pre_predictors_dims=[96, 96],
        wm_scalar_predictor_hidden_dims=[],
        wm_angle_predictor_hidden_dims=[],
        wm_rot6d_predictor_hidden_dims=[],
        wm_binary_predictor_hidden_dims=[],
        local_scalar_target_indices=obs_indices.local_scalar_indices,
        local_angle_target_indices=obs_indices.local_angle_indices,
        local_rot6d_target_indices=obs_indices.local_rot6d_indices,
        local_binary_target_indices=obs_indices.local_binary_indices,
        scalar_loss_fn='smooth_l1',
        scalar_loss_weight=1.0,
        angle_loss_weight=1.0,
        rot6d_loss_weight=1.0,
        binary_loss_weight=1.0,
    )
    print(policy)

    print("Initializing PPO Algorithm...")

    lr = 1e-4

    def auto_lr_updater(
            state: dict[str, Any],
            early_stop_kl_div: Optional[float],
            early_stop_epoch: Optional[int],
            metrics: dict[str, Any]
    ) -> AutomaticLearningRateUpdateResult:
        if early_stop_kl_div and early_stop_kl_div > 0.1:
            state['counter'] = 0
            decay_factor = np.clip(0.9 - early_stop_kl_div, 0.4, 0.8)
            return {'ratio': decay_factor, 'msg': f'kl={early_stop_kl_div:.3f}', 'event': 'max_kl_hit'}

        if early_stop_epoch is not None and early_stop_epoch < 2:
            state['counter'] = 0
            decay_factor = 0.9 if early_stop_epoch == 1 else 0.75
            return {'ratio': decay_factor, 'msg': f'epoch={early_stop_epoch}', 'event': 'min_epoch_hit'}

        clip_frac_stats: Optional[SummaryStatistics] = metrics.get('clip_frac', None)
        if clip_frac_stats and clip_frac_stats.mean > 0.25:
            clip_frac = clip_frac_stats.mean
            state['counter'] = 0
            decay_factor = np.clip(1.15 - clip_frac, 0.5, 0.9)
            return {'ratio': decay_factor, 'msg': f'{clip_frac=:.3f}', 'event': 'max_clip_frac_hit'}

        counter = state.get('counter', 0) + 1

        if counter >= 2:
            state['counter'] = 0
            return {'ratio': 1.3}

        state['counter'] = counter
        return {'ratio': None}

    auto_lr = AutomaticLearningRate(
        initial_lr=lr,
        max_lr=2e-4,
        updater=auto_lr_updater
    )
    ppo = PPOWM(
        policy=policy,
        env=env,
        learning_rate=auto_lr,
        rollout_mode=WholeEpisodesRolloutMode(6),
        max_episode_length=episode_length,
        batch_size=256,
        n_epochs=5,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.04,
        gsde_reset_mode=GSDEProbabilityResetMode(probability=1/6),
        ent_coef=0.003,
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
