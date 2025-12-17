import torch
from swarmbots.learn.testing_env import TestingSwarmBotsLearnVectorEnv


def main():
    n_envs = 1
    n_agents = 2
    n_local_obs = 8
    n_global_obs = 4
    actuators_dim = 2
    connectors_dim = 1
    max_steps = 10

    env = TestingSwarmBotsLearnVectorEnv(
        n_envs=n_envs,
        n_agents=n_agents,
        n_local_obs=n_local_obs,
        n_global_obs=n_global_obs,
        actuators_dim=actuators_dim,
        connectors_dim=connectors_dim,
        max_steps=max_steps,
    )

    print(f"Running {n_envs} environment(s) with max_steps={max_steps}...")

    completed_episodes = 0
    target_episodes = 4

    obs, _ = env.reset()

    while completed_episodes < target_episodes:
        # Dummy actions
        actions = torch.zeros((n_envs, n_agents, actuators_dim + connectors_dim))

        obs, rewards, terminations, truncations, infos = env.step(actions)

        # Check for completions
        dones = torch.logical_or(terminations, truncations)
        if dones.any():
            num_finished = dones.sum().item()
            completed_episodes += int(num_finished)
            print(f"Episodes completed: {completed_episodes}/{target_episodes}")

    print("Finished.")


if __name__ == "__main__":
    main()

