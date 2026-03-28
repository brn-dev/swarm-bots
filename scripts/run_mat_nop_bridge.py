import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import torch
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics, NormalizeReward
from loguru import logger
from torch import nn

from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliConfig
from swarmbots.learn.action_dists.gsde_action_dist import GSDEConfig
from swarmbots.learn.algos.mat.mat_policy import MATPolicyConfig, MATCriticConfig
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.mat.mat_decoder import MATDecoderConfig
from swarmbots.learn.algos.mat.wm.mat_nop_policy import MATNOPPolicy
from swarmbots.learn.algos.mat.wm.mat_nop_policy import MATNOPPolicyConfig, MATNOPWorldModelConfig
from swarmbots.learn.algos.ppo.wm.ppo_wm import PPOWM
from swarmbots.learn.algos.ppo.ppo import AutomaticLearningRate, AutomaticLearningRateUpdateResult, StepsRolloutMode
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredConfig
from swarmbots.learn.env_wrappers.feature_wise_obs_norm_wrapper import (
    FeatureWiseObsNormWrapper,
)
from swarmbots.learn.env_wrappers.progress_guidance_ep_stats_wrapper import ProgressGuidanceEpisodeStatsWrapper
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.transition_obs_wrapper import TransitionObsWrapper
from swarmbots.learn.gsde_reset import GSDEProbabilityResetMode
from swarmbots.learn.summary_statistics import SummaryStatisticsFormat, SummaryStatistics
from swarmbots.learn.obs_indices import ObsIndices
from swarmbots.learn.swarmbots_obs_indices import build_obs_indices
from swarmbots.mj_env.scenarios.scenario_presets import default_bridge
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm, PoissonDiscUnitLocationsConfig
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


def make_env_fn(
    episode_length: int,
    scenario_kwargs: dict[str, Any] | None = None,
    render_mode: str | None = None,
    first_episode_length: int | None = None
) -> Callable[[], SwarmBotsEnv]:
    if scenario_kwargs is None:
        scenario_kwargs = {}
        
    def _init() -> SwarmBotsEnv:
        scenario = default_bridge(
            swarm=HomogeneousSwarm(
                unit_start_locations=PoissonDiscUnitLocationsConfig(
                    num_units=4,
                    max_radius=1.5,
                    num_unit_probs={
                        2: 0.25,
                        3: 0.25,
                        4: 0.25,
                        # 5: 0.25,
                    }
                ),
                randomize_unit_orientations=True,
            ),
            # unit_start_locations='8:hourglass',
            randomize_unit_orientations=True,
            bridge_x=0.0,
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
    vector_env = ProgressGuidanceEpisodeStatsWrapper(vector_env)
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
        obs_key="hidden_local_vars",
        scalar_feature_indices=obs_indices.hidden_local_vars_scalar_indices,
        quaternion_indices=obs_indices.hidden_local_vars_quaternion_indices,
    )
    vector_env = FeatureWiseObsNormWrapper(
        vector_env,
        obs_key="hidden_global_vars",
        scalar_feature_indices=obs_indices.hidden_global_vars_scalar_indices,
        quaternion_indices=obs_indices.hidden_global_vars_quaternion_indices,
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

    n_envs = 23
    # unit_start_locations = [
    #     (0.0, 0.0, 0.0),
    #     (-0.6, 0, 0),
    # ]
    episode_length = 512
    total_timesteps = 100_000_000
    save_interval = 500
    use_popart = False
    popart_beta = 3e-4
    popart_eps = 1e-5
    popart_min_std = 1e-4
    popart_init_sigma = 0.5
    world_model_num_next_steps = 3
    world_model_loss_coef = 0.1
    world_model_target_tau = None

    # =====  ID  =====
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ===== LOAD =====
    load_path: str | None = None
    # load_path = "../runs/mat_nop_swarm_bots_bridge/2026-02-12_14-14-10/models/model_55513104_steps_stopped.pt"

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

    run_dir = f"../runs/mat_nop_swarm_bots_bridge/{run_id}/"
    save_optimizer = True

    scenario_kwargs = {
    }

    env_fns = [
        make_env_fn(
            episode_length=episode_length,
            scenario_kwargs=scenario_kwargs,
            render_mode=None,
            first_episode_length=int(i * episode_length / n_envs)
        )
        for i in range(1, n_envs + 1)
    ]

    print("Creating dummy env for capturing settings...")
    dummy_env = env_fns[0]()
    env_settings = dummy_env.get_settings()
    local_obs_dim = int(dummy_env.observation_space["local_obs"].shape[-1])
    global_obs_dim = int(dummy_env.observation_space["global_obs"].shape[-1])
    hidden_local_vars_dim = int(dummy_env.observation_space["hidden_local_vars"].shape[-1])
    hidden_global_vars_dim = int(dummy_env.observation_space["hidden_global_vars"].shape[-1])
    dummy_env.close()
    del dummy_env
    print("Env settings captured.")

    obs_indices = build_obs_indices(
        env_settings=env_settings,
        local_obs_dim=local_obs_dim,
        global_obs_dim=global_obs_dim,
        hidden_local_vars_dim=hidden_local_vars_dim,
        hidden_global_vars_dim=hidden_global_vars_dim,
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
        config=MATNOPPolicyConfig(
            mat_policy_config=MATPolicyConfig(
                encoder_config=MATEncoderConfig(
                    d_model=128,
                    nhead=2,
                    num_layers=2,
                    dim_feedforward=256,
                    local_obs_encoder_hidden_dims=[256, 256],
                ),
                decoder_config=MATDecoderConfig(
                    d_model=64,
                    nhead=2,
                    num_layers=2,
                    dim_feedforward=128,
                    action_encoder_hidden_dims=[64],
                    cross_attn_first=True,
                ),
                critic_config=MATCriticConfig(
                    n_local_projection_hidden_layers=1,
                    n_value_regressor_hidden_layers=1,
                    use_popart=use_popart,
                    popart_config=PopArtConfig(
                        beta=popart_beta,
                        eps=popart_eps,
                        min_std=popart_min_std,
                        init_sigma=popart_init_sigma,
                    ),
                ),
                dropout=0.0,
                act_fn_cls=nn.GELU,
                continuous_config=GSDEConfig(
                    base_std=0.25,
                    latent_sde_dim=None,
                    std_learnable=True,
                    full_std=True,
                    sde_learn_features=False,
                    log_std_clamp_range=(-20.0, 2.0),
                    normalize_latent_sde_by_dim=True
                ),
                bernoulli_config=BernoulliConfig(initial_prob=0.75),
                max_agents=20,
            ),
            world_model_config=MATNOPWorldModelConfig(
                wm_pre_transition_dims=[128],
                d_model_transition_model=128,
                nhead_transition_model=2,
                num_layers_transition_model=2,
                dim_feedforward_transition_model=128,
                transition_model_coembed_hidden_dims=[128],
                wm_pre_predictors_dims=[128, 128],
                wm_scalar_predictor_hidden_dims=[],
                wm_angle_predictor_hidden_dims=[],
                wm_rot6d_predictor_hidden_dims=[],
                wm_binary_predictor_hidden_dims=[],
                scalar_loss_fn='smooth_l1',
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
        ),
    )
    print(policy)

    print("Initializing PPO Algorithm...")

    initial_lr = 1e-4

    def auto_lr_updater(
            old_lr: float,
            state: dict[str, Any],
            n_iterations: int,
            n_model_updates: int,
            n_timesteps: int,
            early_stop_kl_div: Optional[float],
            early_stop_epoch: Optional[int],
            metrics: dict[str, Any]
    ) -> AutomaticLearningRateUpdateResult:
        warmup_iterations: int = 250
        cold_lr = initial_lr / 50

        if early_stop_kl_div is not None and early_stop_kl_div > 0.1:
            state['counter'] = 0
            state['warmup'] = False
            decay_factor = np.clip(0.9 - early_stop_kl_div, 0.4, 0.8)
            return {
                'new_lr': old_lr * decay_factor,
                'msg': f'kl={early_stop_kl_div:.3f}',
                'event': 'max_kl_hit'
            }

        if early_stop_epoch is not None and early_stop_epoch < 2:
            state['counter'] = 0
            state['warmup'] = False
            decay_factor = 0.9 if early_stop_epoch == 1 else 0.75
            return {
                'new_lr': old_lr * decay_factor,
                'msg': f'epoch={early_stop_epoch}',
                'event': 'min_epoch_hit'
            }

        clip_frac_stats: Optional[SummaryStatistics] = metrics.get('clip_frac', None)
        if clip_frac_stats and clip_frac_stats.mean > 0.2:
            state['counter'] = 0
            state['warmup'] = False
            clip_frac = clip_frac_stats.mean
            decay_factor = np.clip(1.1 - clip_frac, 0.5, 0.9)
            return {
                'new_lr': old_lr * decay_factor,
                'msg': f'{clip_frac=:.3f}',
                'event': 'max_clip_frac_hit'
            }


        warmup: bool = state.get('warmup', warmup_iterations > 0) and n_iterations <= warmup_iterations
        state['warmup'] = warmup
        if warmup:
            new_lr = cold_lr + (initial_lr - cold_lr) * n_iterations / warmup_iterations
            return {
                'new_lr': new_lr,
                'msg': f'Warmup ({n_iterations}/{warmup_iterations})',
                'event': 'warmup'
            }

        counter = state.get('counter', 0) + 1

        if counter >= 2:
            state['counter'] = 0
            return {'new_lr': old_lr * 1.3}

        state['counter'] = counter
        return {'new_lr': None}

    auto_lr = AutomaticLearningRate(
        initial_lr=initial_lr,
        max_lr=2e-4,
        updater=auto_lr_updater
    )
    ppo = PPOWM(
        policy=policy,
        env=env,
        learning_rate=auto_lr,
        rollout_mode=StepsRolloutMode(256 * 16),
        max_episode_length=episode_length,
        batch_size=256,
        n_epochs=5,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        target_kl=0.04,
        max_grad_norm=10.0,
        gsde_reset_mode=GSDEProbabilityResetMode(probability=1/6),
        mc_ent_coef=1e-5,
        value_loss_fn=nn.SmoothL1Loss(),
        train_device=train_device,
        rollout_device=rollout_device,
        use_popart=use_popart,
        world_model_num_next_steps=world_model_num_next_steps,
        world_model_loss_coef=world_model_loss_coef,
        world_model_target_tau=world_model_target_tau,
    )

    if load_path:
        logger.info(f"Loading model from {load_path}")
        ppo.load(load_path, recover_best_return_ema=False)

    print("Starting training...")
    logging_console_keys: list[tuple[str, str | SummaryStatisticsFormat | None] | tuple[str, str | SummaryStatisticsFormat | None, str]] = [
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
        ('ep_len', SummaryStatisticsFormat(mean='3.0f', std='3.0f')),
        ('ep_rew', SummaryStatisticsFormat(mean=' .2f', std='.2f', max_value=' .2f', n='1')),
        ('ep_rew_ema', ' .3f'),
        ('best_ep_rew_ema', ' .3f', 'best_ema'),
        ('fps', None),
    ]
    if use_popart:
        logging_console_keys.insert(-5, ('popart_mu', '.3f', 'pa_mu'))
        logging_console_keys.insert(-5, ('popart_sigma', '.3f', 'pa_sigma'))

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
        logging_console_keys=logging_console_keys,
        make_record_env=make_record_env
    )
    
    print("Training Finished.")

    env.close()


if __name__ == "__main__":
    main()
