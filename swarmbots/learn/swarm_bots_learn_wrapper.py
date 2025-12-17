from __future__ import annotations

from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv, VectorWrapper

from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace

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


class VectorSwarmBotsActionSpace(VectorHybridActionSpace):
    def __init__(self, n_envs: int, n_agents: int, actuators_dim: int, connectors_dim: int):
        super().__init__(spaces={
            "actuators": spaces.Box(low=-1.0, high=1.0, shape=(n_envs, n_agents, actuators_dim), dtype=np.float32),
            "connectors": spaces.MultiBinary((n_envs, n_agents, connectors_dim)),
        })


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

        self._observation_space: spaces.Dict = _require_dict_space(single_obs_space, what="single_observation_space")
        act_dict = _require_dict_space(single_act_space, what="single_action_space")

        if "actuators" not in act_dict.spaces or "connectors" not in act_dict.spaces:
            raise KeyError(
                "Expected action space keys {'actuators', 'connectors'}, "
                f"got keys={list(act_dict.spaces.keys())}"
            )

        self._n_envs: int = int(getattr(env, "num_envs"))

        actuators_space = act_dict["actuators"]
        connectors_space = act_dict["connectors"]

        self._split_spec = ActionSplitSpec(
            actuators_dim=int(actuators_space.shape[1]),
            connectors_dim=int(connectors_space.shape[1]),
        )

        self.n_agents: int = actuators_space.shape[0]
        self._action_space: VectorHybridActionSpace = VectorSwarmBotsActionSpace(
            n_envs=self._n_envs,
            n_agents=self.n_agents,
            actuators_dim=self._split_spec.actuators_dim,
            connectors_dim=self._split_spec.connectors_dim,
        )

        self.local_obs_dim = self.observation_space['local_obs'].shape[-1]
        self.global_obs_dim = self.observation_space['global_obs'].shape[-1]


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

    def _actions_to_env_dict(self, actions: torch.Tensor) -> dict[str, np.ndarray]:
        if actions.ndim == 2:
            actions = actions.unsqueeze(0)
        act_dim = self._split_spec.actuators_dim
        actions = actions.detach().to("cpu")
        actuators = actions[..., :act_dim].numpy().astype(np.float32, copy=False)
        connectors = actions[..., act_dim:].numpy().astype(bool, copy=False)

        return {"actuators": actuators, "connectors": connectors}


