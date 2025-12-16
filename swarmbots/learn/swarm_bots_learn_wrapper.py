from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, TypeAlias

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv, VectorWrapper

from swarmbots.learn.hybrid_action_space import HybridActionSpace

TorchObs: TypeAlias = dict[str, torch.Tensor]
NumpyObs: TypeAlias = dict[str, np.ndarray]


def _as_device(device: torch.device | str) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def _require_dict_space(space: spaces.Space, *, what: str) -> spaces.Dict:
    if not isinstance(space, spaces.Dict):
        raise TypeError(f"Expected {what} to be gymnasium.spaces.Dict, got {type(space)}: {space}")
    return space


@dataclass(frozen=True)
class ActionSplitSpec:
    actuators_dim: int
    connectors_dim: int

    @property
    def total_dim(self) -> int:
        return self.actuators_dim + self.connectors_dim


class VectorSwarmBotsActionSpace(HybridActionSpace):
    def __init__(self, n_envs: int, n_agents: int, actuators_dim: int, connectors_dim: int):
        super().__init__(spaces={
            "actuators": spaces.Box(low=-1.0, high=1.0, shape=(n_envs, n_agents, actuators_dim), dtype=np.float32),
            "connectors": spaces.MultiBinary(shape=(n_envs, n_agents, connectors_dim)),
        })
        self.n_envs = int(n_envs)

    def sample(self, mask: dict[str, Any] | None = None, probability: dict[str, Any] | None = None) -> dict[str, Any]:
        if mask is not None or probability is not None:
            raise NotImplementedError("mask/probability sampling not supported")
        out: dict[str, Any] = {}
        for k, space in self.space_map.items():
            samples = [space.sample() for _ in range(self.n_envs)]
            out[k] = np.stack(samples, axis=0)
        return out


class SwarmBotsLearnVectorWrapper(VectorWrapper):

    def __init__(
        self,
        env: VectorEnv,
        *,
        device: torch.device | str = "cpu",
        obs_dtype: torch.dtype = torch.float32,
        reward_dtype: torch.dtype = torch.float32,
        connectors_threshold: float = 0.0,
    ):
        super().__init__(env)
        self.device = _as_device(device)
        self.obs_dtype = obs_dtype
        self.reward_dtype = reward_dtype
        self.connectors_threshold = float(connectors_threshold)

        autoreset_mode = env.metadata.get("autoreset_mode", None)
        if not autoreset_mode == AutoresetMode.NEXT_STEP:
            raise ValueError(
                "SwarmBotsLearnVectorEnv requires VectorEnv autoreset_mode=NEXT_STEP "
                f"(got autoreset_mode={autoreset_mode!r}). The rollout buffer accumulator is designed for NEXT_STEP."
            )

        single_obs_space = getattr(env, "single_observation_space", None) or env.observation_space
        single_act_space = getattr(env, "single_action_space", None) or env.action_space

        self.observation_space: spaces.Dict = _require_dict_space(single_obs_space, what="single_observation_space")
        act_dict = _require_dict_space(single_act_space, what="single_action_space")

        if "actuators" not in act_dict.spaces or "connectors" not in act_dict.spaces:
            raise KeyError(
                "Expected action space keys {'actuators', 'connectors'}, "
                f"got keys={list(act_dict.spaces.keys())}"
            )

        self._n_envs: int = int(getattr(env, "num_envs"))
        
        actuators_space = self.action_space["actuators"]
        connectors_space = self.action_space["connectors"]
        if getattr(actuators_space, "shape", None) is None or len(actuators_space.shape) != 2:
            raise ValueError(f"Expected actuators space shape (n_agents, dim), got {actuators_space}")
        if getattr(connectors_space, "shape", None) is None or len(connectors_space.shape) != 2:
            raise ValueError(f"Expected connectors space shape (n_agents, dim), got {connectors_space}")

        self._split_spec = ActionSplitSpec(
            actuators_dim=int(actuators_space.shape[1]),
            connectors_dim=int(connectors_space.shape[1]),
        )

        self.n_agents: int = actuators_space.shape[0]
        self.action_space: HybridActionSpace = VectorSwarmBotsActionSpace(
            n_envs=self._n_envs,
            n_agents=self.n_agents,
            actuators_dim=self._split_spec.actuators_dim,
            connectors_dim=self._split_spec.connectors_dim,
        )

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

    def _actions_to_env_dict(self, actions: torch.Tensor) -> dict[str, np.ndarray]:
        return self.action_space.concat_actions(actions)


