

import abc
import itertools
from typing import Any, TypeAlias, TypeVar, Generic

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv, VectorWrapper

from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.torch_device import as_device

TorchObs: TypeAlias = dict[str, torch.Tensor]
NumpyObs: TypeAlias = dict[str, np.ndarray]

ActSpace = TypeVar('ActSpace', bound=VectorHybridActionSpace)


class BaseLearnEnvWrapper(VectorWrapper, Generic[ActSpace], abc.ABC):

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
        if not isinstance(obs_space, spaces.Dict):
            raise ValueError(f'Observation space must be a gymnasium.spaces.Dict, got {obs_space}')

        required_obs_keys = {"local_obs", "global_obs", "hidden_vars"}
        missing_obs_keys = required_obs_keys.difference(obs_space.keys())
        if missing_obs_keys:
            raise ValueError(
                f'Observation space is missing required keys {sorted(missing_obs_keys)}, got keys {list(obs_space.keys())}'
            )

        self._observation_space: spaces.Dict = obs_space

        local_obs_shape = self._observation_space['local_obs'].shape
        if len(local_obs_shape) != 3:
            raise ValueError(f"Expected local_obs shape (n_envs, n_agents, n_local_obs), got {local_obs_shape}")
        global_obs_shape = self._observation_space['global_obs'].shape
        if len(global_obs_shape) != 2:
            raise ValueError(f"Expected global_obs shape (n_envs, n_global_obs), got {global_obs_shape}")
        hidden_vars_shape = self._observation_space['hidden_vars'].shape
        if len(hidden_vars_shape) != 2:
            raise ValueError(f"Expected hidden_vars shape (n_envs, n_hidden_vars), got {hidden_vars_shape}")

        self.local_obs_dim = local_obs_shape[2]
        self.global_obs_dim = global_obs_shape[1]
        self.hidden_vars_dim = hidden_vars_shape[1]
        self.has_agent_mask = "agent_mask" in self._observation_space.keys()
        if self.has_agent_mask:
            agent_mask_shape = self._observation_space["agent_mask"].shape
            if len(agent_mask_shape) != 2:
                raise ValueError(f"Expected agent_mask shape (n_envs, n_agents), got {agent_mask_shape}")

        if not isinstance(action_space, VectorHybridActionSpace):
            raise ValueError(f'Action space must be a VectorHybridActionSpace, got {action_space}')
        self._action_space: VectorHybridActionSpace = action_space


        self._n_envs: int = int(getattr(env, "num_envs"))

        for space in itertools.chain(obs_space.values(), action_space.values()):
            assert space.shape[0] == self._n_envs, space

        self.n_agents = obs_space['local_obs'].shape[1]

        if self.has_agent_mask:
            agent_mask_shape = self._observation_space["agent_mask"].shape
            if agent_mask_shape[1] != self.n_agents:
                raise ValueError(
                    f"Expected agent_mask second dim to match n_agents ({self.n_agents}), got {agent_mask_shape}"
                )

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
        if "hidden_vars" not in obs:
            raise ValueError("Expected observations to contain key 'hidden_vars'")
        obs_t: TorchObs = {
            "local_obs": torch.as_tensor(obs["local_obs"], device=self.device, dtype=self.obs_dtype),
            "global_obs": torch.as_tensor(obs["global_obs"], device=self.device, dtype=self.obs_dtype),
            "hidden_vars": torch.as_tensor(obs["hidden_vars"], device=self.device, dtype=self.obs_dtype),
        }
        if "agent_mask" in obs and obs["agent_mask"] is not None:
            obs_t["agent_mask"] = torch.as_tensor(obs["agent_mask"], device=self.device, dtype=torch.bool)
        return obs_t

    @abc.abstractmethod
    def _actions_to_env_dict(self, actions: torch.Tensor) -> dict[str, np.ndarray]:
        raise NotImplementedError()

