from gymnasium.vector import AsyncVectorEnv
from swarmbots import TestingSwarmBotsEnv
from swarmbots import collect_rollout
from swarmbots import RolloutBuffer
from swarmbots import SwarmBotsLearnWrapper


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
    # Using shared_memory=False to avoid potential pickling/shared memory issues in simple tests on Windows
    # though AsyncVectorEnv defaults typically handle this. 
    # For Windows, we need to be careful with spawn.
    vector_env = AsyncVectorEnv(env_fns)

    print("Wrapping with SwarmBotsLearnVectorWrapper...")
    env = SwarmBotsLearnWrapper(vector_env, device="cpu")

    print(f"Running {n_envs} environment(s) with max_steps={max_steps}...")

    # Create dummy observation space for RolloutBuffer initialization
    # Note: mj_env.observation_space is already the correct Vector Dict space from the wrapper
    # But RolloutBuffer expects single observation space components structure but sized for vector?
    # No, RolloutBuffer usually takes the single observation space structure.
    # But let's look at how RolloutBuffer is implemented.
    # The previous code created a manual space. Let's stick to manual if unsure, 
    # but the wrapper exposes `single_observation_space` probably?
    # SwarmBotsLearnVectorWrapper exposes `observation_space` which is the single observation space (Dict).
    # Wait, SwarmBotsLearnVectorWrapper.__init__ sets self.observation_space to single_obs_space.
    
    buffer = RolloutBuffer(
        n_episodes=n_episodes,
        max_episode_length=max_steps,
        observation_space=env.observation_space, # This is the single observation space (Dict)
        action_space=env.action_space, # This is the HybridActionSpace
        gamma=0.99,
        gae_lambda=0.95,
        n_envs=n_envs,
    )

    for _ in range(2):
        print(f"Collecting {n_episodes} episodes...")
        episodes = collect_rollout(env, buffer, device=env.device)

        print(f"Collected {len(episodes)} episodes.")
        for i, ep in enumerate(episodes):
            print(f"Episode {i+1}: length={ep.local_obs.shape[0]}")

    print("Finished.")
    env.close()


if __name__ == "__main__":
    main()
