import sys
from datetime import datetime

import torch
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics, NormalizeReward
from loguru import logger
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import GSDEParams
from swarmbots.learn.algos.mat.mat_policy import MATPolicy
from swarmbots.learn.algos.ppo.ppo import PPO
from swarmbots.learn.env_wrappers.normalize_obs_wrapper import NormalizeLocalObsWrapper
from swarmbots.learn.env_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.recording import record_policy
from swarmbots.learn.summary_statistics import SummaryStatisticsFormat
from swarmbots.mj_env.scenarios.obstacle_street_scenario import ObstacleStreetScenario
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


def make_env_fn(
    episode_length,
    scenario_kwargs=None,
    render_mode=None
):
    if scenario_kwargs is None:
        scenario_kwargs = {}
        
    def _init():
        scenario = ObstacleStreetScenario.no_payload_no_opening_one_wall_easy(
            **scenario_kwargs
        )
        return SwarmBotsEnv(
            scenario=scenario,
            episode_length=episode_length,
            render_mode=render_mode,
            camera=0
        )
    return _init



def main():
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

    # =====  ID  =====
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # ===== LOAD =====
    load_path: str | None = None
    load_path = "runs/mat_swarm_bots/2026-01-14_14-28-07/models/model_18432000_steps.pt"
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

    run_dir = f"runs/mat_swarm_bots/{run_id}/"
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

    def make_record_env():
        record_env = SyncVectorEnv([
            make_env_fn(
                episode_length=episode_length,
                scenario_kwargs=scenario_kwargs,
                render_mode='rgb_array'
            )
        ])
        record_env = RecordEpisodeStatistics(record_env)
        record_env = NormalizeLocalObsWrapper(record_env)
        record_env = SwarmBotsLearnEnvWrapper(record_env, device=rollout_device)

        return record_env

    print("Creating dummy env for capturing settings...")
    dummy_env = env_fns[0]()
    env_settings = dummy_env.get_settings()
    dummy_env.close()
    del dummy_env
    print("Env settings captured.")

    print('Creating vector env...')
    vector_env = AsyncVectorEnv(env_fns)
    print(f"Created {type(vector_env)} with {n_envs} environments.")

    if isinstance(vector_env, SyncVectorEnv):
        for _ in range(10):
            logger.warning('USING SYNC VECTOR ENV')

    gamma = 0.987
    
    print("Wrapping with RecordEpisodeStatistics, NormalizeObservation, NormalizeReward...")
    vector_env = RecordEpisodeStatistics(vector_env)
    vector_env = NormalizeLocalObsWrapper(vector_env)
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
    policy = MATPolicy(
        env=env,
        d_model=96,
        d_model_decoder=64,
        nhead_encoder=3,
        nhead_decoder=2,
        num_layers_encoder=2,
        num_layers_decoder=2,
        dim_feedforward_encoder=128,
        dim_feedforward_decoder=96,
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
        bernoulli_initial_prob=0.7,
    )
    # policy = MATPolicy(
    #     env=env,
    #     d_model=16,
    #     d_model_decoder=16,
    #     nhead_encoder=1,
    #     nhead_decoder=1,
    #     num_layers_encoder=2,
    #     num_layers_decoder=2,
    #     dim_feedforward_encoder=24,
    #     dim_feedforward_decoder=24,
    #     dropout=0.0,
    #     n_critic_local_projection_hidden_layers=1,
    #     n_critic_value_regressor_hidden_layers=1,
    #     cross_attn_first=True,
    #     act_fn_cls=nn.GELU,
    #     continuous_config=GSDEParams(
    #         base_std=0.45,
    #         latent_sde_dim=None,
    #         std_learnable=True,
    #         full_std=True,
    #         sde_learn_features=False,
    #         log_std_clamp_range=(-20.0, 2.0),
    #         normalize_latent_sde_by_dim=True
    #     ),
    #     bernoulli_initial_prob=0.7,
    # )
    print(policy)

    lr = 1e-5
    # if load_path is not None:
    #     lr = 5e-6
    #     logger.warning(f'Setting {lr = :.2e}')

    print("Initializing PPO Algorithm...")
    ppo = PPO(
        policy=policy,
        env=env,
        learning_rate=lr,
        n_episodes_per_rollout=n_envs,
        max_episode_length=episode_length,
        batch_size=256,
        n_epochs=5,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        train_device=train_device,
        rollout_device=rollout_device,
        target_kl=0.04,
        gsde_sample_freq=6,
        ent_coef=0.01,
        agent_logprob_reduction=None,
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
            'env_settings': env_settings
        },
        logging_console_keys=[
            ('iteration', '5'),
            ('timesteps', '8'),
            ('tot_upd', '6'),
            ('act0', SummaryStatisticsFormat(histogram=10)),
            ('act1', SummaryStatisticsFormat(histogram=2)),
            ('std0', SummaryStatisticsFormat(mean='.3f', std='.3f', min_value='.3f', max_value='.3f')),
            ('upd', '3'),
            ('approx_kl', SummaryStatisticsFormat(mean='.3f', std='.3f', max_value='.3f')),
            ('clip_frac', None),
            ('ratio', SummaryStatisticsFormat(mean='.3f', std='.3f', min_value='.3f', max_value='.3f')),
            ('val_loss', None),
            ('expl_var', '.3f'),
            ('ep_rew', SummaryStatisticsFormat(mean=' .2f', std='.2f', max_value=' .2f')),
            ('ep_rew_ema', ' .3f'),
            ('best_ep_rew_ema', ' .3f'),
            ('fps', None),
        ],
        make_record_env=make_record_env
    )
    
    print("Training Finished.")

    env.close()


if __name__ == "__main__":
    main()
