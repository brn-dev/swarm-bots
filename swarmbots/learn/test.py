from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from gymnasium.wrappers.vector import RecordEpisodeStatistics, NormalizeReward
from torch import nn

from swarmbots.learn.env_wrappers.normalize_obs_wrapper import NormalizeGlobalWithLocalObsWrapper, \
    NormalizeLocalObsWrapper
from swarmbots.learn.env_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.ppo.ppo import PPO
from swarmbots.learn.ppo.ppo_policy import PPOPolicy
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
    n_envs = 4
    unit_start_locations = [
        (0.0, 0.0, 0.0),
        (-0.65, 0, 0),
        (0.65, 0, 0),
    ]
    episode_length = 512
    n_episodes_per_rollout = 4
    total_timesteps = 5_000_000

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
    
    print("Wrapping with RecordEpisodeStatistics, NormalizeObservation, NormalizeReward...")
    vector_env = RecordEpisodeStatistics(vector_env)
    vector_env = NormalizeLocalObsWrapper(vector_env)
    vector_env = NormalizeReward(vector_env, gamma=0.99)

    print("Wrapping with SwarmBotsLearnEnvWrapper...")
    env = SwarmBotsLearnEnvWrapper(vector_env, device="cpu")
    
    print(f"Environment initialized.")
    print(f"n_agents: {env.n_agents}")
    print(f"local_obs_dim: {env.local_obs_dim}")
    print(f"global_obs_dim: {env.global_obs_dim}")
    print(f"actuators_dim: {env.actuators_dim}")
    print(f"connectors_dim: {env.connectors_dim}")

    print("Initializing PPO Policy...")
    policy = PPOPolicy(
        env=env,
        actor_hidden_dims=[256],
        latent_pi_dim=256,
        critic_hidden_dims=[256, 256],
        act_fun_class=nn.Tanh
    )

    print("Initializing PPO Algorithm...")
    ppo = PPO(
        policy=policy,
        env=env,
        learning_rate=2e-5,
        n_episodes_per_rollout=n_episodes_per_rollout,
        max_episode_length=episode_length,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        device='cpu',
        target_kl=0.05
    )

    print("Starting training...")
    ppo.learn(total_timesteps=total_timesteps, log_interval=1)
    
    print("Training Finished.")
    
    print("Starting recording...")
    
    record_env_fn = make_env_fn(
        unit_start_locations=unit_start_locations,
        episode_length=episode_length,
        scenario_kwargs={"num_walls": 1},
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
    
    record_env = SwarmBotsLearnEnvWrapper(record_vector_env, device="cpu")
    
    record_policy(
        env=record_env,
        policy=policy,
        video_folder='../../videos',
        video_name_prefix='test_run',
        num_episodes=1,
        deterministic=True
    )
    
    record_env.close()
    env.close()


if __name__ == "__main__":
    main()
