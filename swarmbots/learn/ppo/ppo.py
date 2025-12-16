import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv

from swarmbots.learn.buffers.rollout_buffer import Episode, RolloutBuffer
from swarmbots.learn.hybrid_action_space import HybridActionSpace


def collect_rollout(
        env: VectorEnv,
        buffer: RolloutBuffer,
        device: torch.device,
) -> list[Episode]:
    # noinspection PyTypeChecker
    action_space: HybridActionSpace = env.action_space

    buffer.reset()
    obs, info = env.reset()

    dones = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=device)

    while not buffer.is_ready():
        actions = env.action_space.sample()  # todo use policy
        obs, rewards, terminations, truncations, infos = env.step(actions)

        buffer.add(
            local_obs=obs['local_obs'],
            global_obs=obs['global_obs'],
            actions=action_space.concat_actions(actions),
            rewards=rewards,
            dones=dones,
            log_probs=0,  # todo
            values=0,  # todo
        )

