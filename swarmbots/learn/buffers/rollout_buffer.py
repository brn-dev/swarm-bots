import torch
from gymnasium import spaces

from swarmbots.learn.buffers.base_buffer import BaseBuffer

class RolloutBuffer(BaseBuffer):

    def __init__(
        self,
        n_episodes: int,
        observation_space: spaces.Dict,
        action_space: spaces.Dict,
        storage_device: torch.device | str = "cpu",
        sampling_device: torch.device | str = "auto",
        n_envs: int = 1,
    ):
        super().__init__(
            n_episodes=n_episodes,
            observation_space=observation_space,
            action_space=action_space,
            storage_device=storage_device,
            sampling_device=sampling_device,
            n_envs=n_envs
        )

        self.local_obs = torch.zeros((self.n_episodes, ))