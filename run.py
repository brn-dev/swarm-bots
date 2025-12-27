import sys
import argparse
from datetime import datetime

import torch
from loguru import logger
from gymnasium.vector import SyncVectorEnv, AsyncVectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics, NormalizeReward
from torch import nn

from swarmbots.learn.algos.mat.mat import MAT
from swarmbots.learn.algos.mat.mat_policy import MATPolicy
from swarmbots.learn.env_wrappers.normalize_obs_wrapper import NormalizeLocalObsWrapper
from swarmbots.learn.env_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.algos.ppo.ppo import PPO
from swarmbots.learn.algos.ppo.ppo_policy import PPOPolicy
from swarmbots.learn.recording import record_policy
from swarmbots.learn.torch_device import as_device
from swarmbots.mj_env.scenarios.obstacle_street_scenario import ObstacleStreetScenario
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv


def make_env_fn(
    unit_start_locations,
    episode_length,
    scenario_kwargs=None,
    render_mode=None
):
    if scenario_kwargs is None:
        scenario_kwargs = {}
        
    def _init():
        swarm = HomogeneousSwarm(
            unit_start_locations=unit_start_locations,
            randomize_unit_orientations=False
        )
        scenario = ObstacleStreetScenario(
            swarm=swarm,
            payload_type=None,
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Torch device for policy + learn env wrapper tensors.",
    )
    parser.add_argument(
        "--tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable TF32 matmul on CUDA for extra speed (slightly different numerics).",
    )
    args = parser.parse_args()

    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

    n_envs = 4
    unit_start_locations = [
        (0.0, 0.0, 0.0),
        (-0.6, 0, 0),
        # (0.6, 0, 0),
    ]
    episode_length = 512
    n_episodes_per_rollout = 4
    total_timesteps = 5_000_000
    save_interval = 1000
    run_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = f"runs/mat_swarm_bots/{run_id}/"
    load_path = None
    save_optimizer = True

    env_fns = [
        make_env_fn(
            unit_start_locations=unit_start_locations,
            episode_length=episode_length,
            scenario_kwargs={"num_walls": 1},
            render_mode=None
        )
        for _ in range(n_envs)
    ]

    vector_env = AsyncVectorEnv(env_fns)
    print(f"Created {type(vector_env)} with {n_envs} environments...")

    gamma = 0.95
    
    print("Wrapping with RecordEpisodeStatistics, NormalizeObservation, NormalizeReward...")
    vector_env = RecordEpisodeStatistics(vector_env)
    vector_env = NormalizeLocalObsWrapper(vector_env)
    vector_env = NormalizeReward(vector_env, gamma=gamma)

    print("Wrapping with SwarmBotsLearnEnvWrapper...")
    env = SwarmBotsLearnEnvWrapper(vector_env, device=args.device)
    
    print(f"Environment initialized.")
    print(f"n_agents: {env.n_agents}")
    print(f"local_obs_dim: {env.local_obs_dim}")
    print(f"global_obs_dim: {env.global_obs_dim}")
    print(f"actuators_dim: {env.actuators_dim}")
    print(f"connectors_dim: {env.connectors_dim}")

    print("Initializing Policy...")
    # policy = PPOPolicy(
    #     env=env,
    #     actor_hidden_dims=[256],
    #     latent_pi_dim_per_agent=256 // env.n_agents,
    #     critic_hidden_dims=[256, 256],
    #     act_fun_class=nn.Tanh
    # )
    policy = MATPolicy(
        env=env,
        d_model=64,
        nhead_encoder=2,
        nhead_decoder=2,
        num_layers_encoder=2,
        num_layers_decoder=2,
        dim_feedforward_encoder=96,
        dim_feedforward_decoder=96,
        dropout=0.0,
        latent_pi_dim_per_agent=64,
        base_std=1.0,
        n_critic_local_projection_hidden_layers=1,
        n_critic_value_regressor_hidden_layers=2,
    )
    print(policy)

    print("Initializing PPO Algorithm...")
    ppo = MAT(
        policy=policy,
        env=env,
        learning_rate=2e-5 if load_path is None else 1e-5,
        n_episodes_per_rollout=n_episodes_per_rollout,
        max_episode_length=episode_length,
        batch_size=64,
        n_epochs=10,
        gamma=gamma,
        gae_lambda=0.95,
        clip_range=0.2,
        device=args.device,
        target_kl=0.05
    )

    device = as_device(ppo.device)
    print(f"Using device: {device}")
    if device.type == "cuda":
        if args.tf32:
            torch.set_float32_matmul_precision("high")
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        print(f"CUDA device: {torch.cuda.get_device_name(device)}")

    if load_path:
        logger.info(f"Loading model from {load_path}")
        ppo.load(load_path)

    print("Starting training...")
    ppo.learn(
        total_timesteps=total_timesteps, 
        run_dir=run_dir,
        log_interval=1,
        save_interval=save_interval,
        save_optimizer=save_optimizer,
    )
    
    print("Training Finished.")
    
    print("Starting recording...")
    
    record_env_fn = make_env_fn(
        unit_start_locations=unit_start_locations,
        episode_length=episode_length,
        scenario_kwargs={"num_walls": 1, 'wall_height': 0.3},
        render_mode='rgb_array'
    )
    
    record_vector_env = SyncVectorEnv([record_env_fn])
    record_vector_env = RecordEpisodeStatistics(record_vector_env)
    
    record_norm_wrapper = NormalizeLocalObsWrapper(record_vector_env)
    
    training_norm_wrapper = env.env.env
    
    record_norm_wrapper.local_obs_rms.mean = training_norm_wrapper.local_obs_rms.mean.copy()
    record_norm_wrapper.local_obs_rms.var = training_norm_wrapper.local_obs_rms.var.copy()
    record_norm_wrapper.update_running_mean = False
    
    record_vector_env = record_norm_wrapper
    
    record_env = SwarmBotsLearnEnvWrapper(record_vector_env, device=str(device))
    
    record_policy(
        env=record_env,
        policy=policy,
        video_folder='videos',
        video_name_prefix='test_run',
        num_episodes=5,
        deterministic=True,
        device=device,
    )
    
    record_env.close()
    env.close()


if __name__ == "__main__":
    main()
