from loguru import logger
from gymnasium.vector import SyncVectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics
import torch
import sys

from swarmbots.learn.algos.mat.mat_policy import MATPolicy
from swarmbots.learn.env_wrappers.normalize_obs_wrapper import NormalizeLocalObsWrapper
from swarmbots.learn.env_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.checkpointing import (
    apply_env_state,
    extract_env_state,
    extract_policy_state_dict,
    freeze_env_normalization,
    load_checkpoint,
)
from swarmbots.learn.recording import record_policy
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
    logger.remove()
    logger.add(sys.stderr, format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <level>{message}</level>")

    # unit_start_locations = [
    #     (0.0, 0.0, 0.0),
    #     (-0.6, 0, 0),
    # ]
    unit_start_locations = [
        (0.0, 0.0, 0.0),
        (0.4, 0.4, 0),
        (0.4, -0.4, 0),
        (-0.4, 0.4, 0),
        (-0.4, -0.4, 0),
    ]
    episode_length = 512
    load_path = "runs/mat_swarm_bots/2025-12-28_17-45-07/models/model_18477056_steps.pt"
    rollout_device = torch.device("cpu")

    print("Creating env...")
    record_env_fn = make_env_fn(
        unit_start_locations=unit_start_locations,
        episode_length=episode_length,
        scenario_kwargs={"num_walls": 1, 'wall_height': 0.3},
        render_mode='rgb_array'
    )
    
    record_vector_env = SyncVectorEnv([record_env_fn])
    record_vector_env = RecordEpisodeStatistics(record_vector_env)
    
    record_norm_wrapper = NormalizeLocalObsWrapper(record_vector_env)
    record_vector_env = record_norm_wrapper
    
    record_env = SwarmBotsLearnEnvWrapper(record_vector_env, device=rollout_device)

    print("Initializing Policy...")
    # policy = PPOPolicy(
    #     env=env,
    #     actor_hidden_dims=[256],
    #     latent_pi_dim_per_agent=256 // env.n_agents,
    #     critic_hidden_dims=[256, 256],
    #     act_fun_class=nn.Tanh
    # )
    policy = MATPolicy(
        env=record_env,
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

    logger.info(f"Loading model from {load_path}")
    checkpoint = load_checkpoint(load_path)
    apply_env_state(record_env, extract_env_state(checkpoint))
    freeze_env_normalization(record_env)
    policy.load_state_dict(extract_policy_state_dict(checkpoint), strict=True)

    print("Starting recording...")
    record_policy(
        env=record_env,
        policy=policy,
        video_folder='videos',
        video_name_prefix='test_run',
        num_episodes=5,
        deterministic=True,
        device=rollout_device,
    )
    
    record_env.close()


if __name__ == "__main__":
    main()
