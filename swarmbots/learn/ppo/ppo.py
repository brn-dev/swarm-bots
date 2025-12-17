import torch

from swarmbots.learn.buffers.rollout_buffer import Episode, RolloutBuffer
from swarmbots.learn.ppo.ppo_policy import PPOPolicy
from swarmbots.learn.swarm_bots_learn_wrapper import SwarmBotsLearnVectorWrapper


def collect_rollout(
        env: SwarmBotsLearnVectorWrapper,
        policy: PPOPolicy,
        buffer: RolloutBuffer,
        device: torch.device,
) -> list[Episode]:
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
