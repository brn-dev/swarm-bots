import torch

from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.ppo.ppo_policy import PPOPolicy
from swarmbots.learn.ppo.ppo_rollout_buffer import PPOEpisode, PPORolloutBuffer


def collect_rollout(
        env: BaseLearnEnvWrapper,
        policy: PPOPolicy,
        buffer: PPORolloutBuffer,
        device: torch.device,
) -> list[PPOEpisode]:
    buffer.reset()
    obs, info = env.reset()
    is_final = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=device)

    while not buffer.is_ready():
        local_obs = obs['local_obs']
        global_obs = obs['global_obs']

        actions, log_probs, values = policy(local_obs, global_obs)

        new_obs, rewards, terminations, truncations, infos = env.step(actions)
        dones = torch.logical_or(terminations, truncations)

        buffer.add(
            local_obs=local_obs,
            global_obs=global_obs,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            is_final=is_final,
        )

        obs = new_obs
        is_final = dones

    return buffer.get_episodes_minimal()
