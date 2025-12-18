from gymnasium.vector import AsyncVectorEnv, SyncVectorEnv
from torch import nn

from swarmbots.learn.env_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.ppo.ppo import collect_rollout
from swarmbots.learn.ppo.ppo_policy import PPOPolicy
from swarmbots.learn.ppo.ppo_rollout_buffer import PPORolloutBuffer
from swarmbots.learn.testing_env import TestingSwarmBotsEnv


def make_env_fn(
    n_agents,
    n_local_obs,
    n_global_obs,
    actuators_dim,
    connectors_dim,
    max_steps
):
    def _init():
        return TestingSwarmBotsEnv(
            n_agents=n_agents,
            n_local_obs=n_local_obs,
            n_global_obs=n_global_obs,
            actuators_dim=actuators_dim,
            connectors_dim=connectors_dim,
            max_steps=max_steps,
        )
    return _init


def main():
    n_envs = 2
    n_agents = 2
    n_local_obs = 8
    n_global_obs = 4
    actuators_dim = 2
    connectors_dim = 1
    max_steps = 10
    n_episodes = 4

    # Create list of environment factories for AsyncVectorEnv
    env_fns = [
        make_env_fn(
            n_agents=n_agents,
            n_local_obs=n_local_obs,
            n_global_obs=n_global_obs,
            actuators_dim=actuators_dim,
            connectors_dim=connectors_dim,
            max_steps=max_steps,
        )
        for _ in range(n_envs)
    ]

    print(f"Creating AsyncVectorEnv with {n_envs} environments...")
    vector_env = SyncVectorEnv(env_fns)
    print(f"Vector Action Space: {vector_env.action_space}")
    print(f"Vector Action Space Type: {type(vector_env.action_space)}")

    print("Wrapping with SwarmBotsLearnVectorWrapper...")
    env = SwarmBotsLearnEnvWrapper(vector_env, device="cpu")

    print(f"Running {n_envs} environment(s) with max_steps={max_steps}...")

    # Initialize Policy
    print("Initializing PPO Policy...")
    policy = PPOPolicy(
        env=env,
        actor_hidden_dims=[64, 64],
        critic_hidden_dims=[64, 64],
        act_fun_class=nn.Tanh
    )

    print("Initializing PPORolloutBuffer...")
    buffer = PPORolloutBuffer(
        n_episodes=n_episodes,
        max_episode_length=max_steps,
        observation_space=env.observation_space,
        action_space=env.action_space,
        gamma=0.99,
        gae_lambda=0.95,
        storage_device='cpu', # Keep on CPU for test
        sampling_device='cpu'
    )

    for _ in range(2):
        print(f"Collecting {n_episodes} episodes...")
        episodes = collect_rollout(env, policy, buffer, device=env.device)

        print(f"Collected {len(episodes)} episodes.")
        for i, ep in enumerate(episodes):
            print(f"Episode {i+1}: length={ep.local_obs.shape[0]}")

    print("Finished.")
    env.close()


if __name__ == "__main__":
    main()
