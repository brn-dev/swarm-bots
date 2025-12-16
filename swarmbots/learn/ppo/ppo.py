import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from swarmbots.learn.buffers.rollout_buffer import Episode, RolloutBuffer
from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.learn.swarm_bots_learn_wrapper import SwarmBotsLearnVectorWrapper


def collect_rollout(
        env: SwarmBotsLearnVectorWrapper,
        buffer: RolloutBuffer,
        device: torch.device,
) -> list[Episode]:
    action_space: HybridActionSpace = env.action_space

    buffer.reset()
    obs, info = env.reset()

    dones = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=device)

    while not buffer.is_ready():
        actions = action_space.sample()   # todo use policy
        new_obs, rewards, terminations, truncations, infos = env.step(actions)

        buffer.add(
            local_obs=obs['local_obs'],
            global_obs=obs['global_obs'],
            actions=actions,
            rewards=rewards,
            log_probs=0,  # todo
            values=0,  # todo
            dones=dones,
        )

        dones = torch.logical_or(terminations, truncations)
        obs = new_obs

    return buffer.get_episodes_minimal()
