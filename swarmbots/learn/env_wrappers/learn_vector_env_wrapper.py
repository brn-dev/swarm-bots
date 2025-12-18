from __future__ import annotations

import abc
import itertools
from typing import Any, TypeAlias, TypeVar, Generic

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv, VectorWrapper

from swarmbots import VectorHybridActionSpace
from swarmbots import as_device

TorchObs: TypeAlias = dict[str, torch.Tensor]
NumpyObs: TypeAlias = dict[str, np.ndarray]

ActSpace = TypeVar('ActSpace', bound=VectorHybridActionSpace)


class LearnVectorEnvWrapper(VectorWrapper, Generic[ActSpace], abc.ABC):

    def __init__(
        self,
        env: VectorEnv,
        action_space: VectorHybridActionSpace,
        *,
        device: torch.device | str = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
    ):
        super().__init__(env)
        self.device = as_device(device)
        self.obs_dtype = obs_dtype
        self.reward_dtype = reward_dtype

        autoreset_mode = env.metadata.get("autoreset_mode", None)
        if not autoreset_mode == AutoresetMode.NEXT_STEP:
            raise ValueError(
                "SwarmBotsLearnVectorEnv requires VectorEnv autoreset_mode=NEXT_STEP "
                f"(got autoreset_mode={autoreset_mode!r}). The rollout buffer accumulator is designed for NEXT_STEP."
            )

        obs_space = env.observation_space
        if not isinstance(obs_space, spaces.Dict) or 'local_obs' not in obs_space or 'lobal_obs' not in obs_space:
            raise ValueError(f'Observations space must be a dict containing "local_obs" and "global_obs", got {obs_space}')
        self._observation_space: spaces.Dict = obs_space

        if not isinstance(action_space, VectorHybridActionSpace):
            raise ValueError(f'Action space must be a VectorHybridActionSpace, got {obs_space}')
        self._action_space: VectorHybridActionSpace = action_space


        self._n_envs: int = int(getattr(env, "num_envs"))

        for space in itertools.chain(obs_space.values(), action_space.values()):
            assert space.shape[0] == self._n_envs, space

        self.n_agents = obs_space['local_obs'].shape[1]

        for space in action_space.values():
            assert space.shape[1] == self.n_agents, space
            assert len(space.shape) == 3, space


    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space

    @property
    def action_space(self) -> VectorHybridActionSpace:
        return self._action_space

    def reset(self, **kwargs) -> tuple[TorchObs, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self._obs_to_torch(obs), info

    def step(
        self, actions: torch.Tensor
    ) -> tuple[TorchObs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        action_dict = self._actions_to_env_dict(actions)
        obs, rewards, terminations, truncations, infos = self.env.step(action_dict)

        obs_t = self._obs_to_torch(obs)
        rewards_t = torch.as_tensor(rewards, device=self.device, dtype=self.reward_dtype).reshape(self._n_envs)
        term_t = torch.as_tensor(terminations, device=self.device, dtype=torch.bool).reshape(self._n_envs)
        trunc_t = torch.as_tensor(truncations, device=self.device, dtype=torch.bool).reshape(self._n_envs)

        return obs_t, rewards_t, term_t, trunc_t, infos

    def close(self) -> None:
        self.env.close()

    def __getattr__(self, item: str) -> Any:
        return getattr(self.env, item)

    @property
    def unwrapped(self) -> Any:
        return getattr(self.env, "unwrapped", self.env)

    def _obs_to_torch(self, obs: NumpyObs) -> TorchObs:
        # local_obs: (n_envs, n_agents, n_local_obs)
        # global_obs: (n_envs, n_global_obs)
        return {
            "local_obs": torch.as_tensor(obs["local_obs"], device=self.device, dtype=self.obs_dtype),
            "global_obs": torch.as_tensor(obs["global_obs"], device=self.device, dtype=self.obs_dtype),
        }

    @abc.abstractmethod
    def _actions_to_env_dict(self, actions: torch.Tensor) -> dict[str, np.ndarray]:
        raise NotImplementedError()


