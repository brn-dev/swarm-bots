import numpy as np
import torch
from gymnasium.vector import AutoresetMode

from swarmbots.learn.buffers.rollout_buffer import Episode, RolloutBuffer
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.swarm_bots_learn_wrapper import SwarmBotsLearnVectorWrapper


def collect_rollout(
        env: SwarmBotsLearnVectorWrapper,
        buffer: RolloutBuffer,
        device: torch.device,
) -> list[Episode]:
    action_space: VectorHybridActionSpace = env.action_space

    buffer.reset()
    obs, info = env.reset()
    dones_prev = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=device)

    while not buffer.is_ready():
        # random actions (todo: use policy)
        actions_dict = action_space.sample()
        actions_np = action_space.concat_actions(actions_dict)
        actions = torch.as_tensor(actions_np, device=device, dtype=torch.float32)

        new_obs, rewards, terminations, truncations, infos = env.step(actions)
        dones = torch.logical_or(terminations, truncations)

        # todo: use policy for these
        log_probs = torch.zeros((buffer.n_envs, env.n_agents), device=device, dtype=torch.float32)
        values = torch.zeros((buffer.n_envs,), device=device, dtype=torch.float32)

        buffer.add(
            local_obs=obs['local_obs'],
            global_obs=obs['global_obs'],
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            dones=dones_prev,
        )

        obs = new_obs
        dones_prev = dones

    return buffer.get_episodes_minimal()
