

# pyright: reportMissingImports=false

from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from gymnasium.logger import warn
from gymnasium.vector import AutoresetMode, VectorEnv, VectorWrapper


class TransitionObsWrapper(VectorWrapper, gym.utils.RecordConstructorArgs):
    """
    Stacks transition information into the observation.

    - local_obs becomes [previous_local_obs, previous_actions, new_local_obs]
    - global_obs (if present) becomes [previous_global_obs, new_global_obs]

    Designed for SwarmBots-style dict observations in a VectorEnv.
    """

    def __init__(self, env: VectorEnv, new_obs_first: bool = True):
        gym.utils.RecordConstructorArgs.__init__(self, new_obs_first=new_obs_first)
        VectorWrapper.__init__(self, env)

        self.new_obs_first = new_obs_first

        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` data. Assuming that the vector environment follows the "
                f"`SameStep` autoreset api. Read https://farama.org/Vector-Autoreset-Mode "
                "for more details."
            )
        else:
            assert self.env.metadata["autoreset_mode"] in {AutoresetMode.SAME_STEP}

        if not isinstance(self.env.single_observation_space, spaces.Dict):
            raise ValueError(f"Expected Dict observation space, got {type(self.env.single_observation_space)}")
        if "local_obs" not in self.env.single_observation_space.spaces:
            raise ValueError('Expected "local_obs" key in observation space')

        local_single: spaces.Box = self.env.single_observation_space["local_obs"]  # type: ignore[assignment]
        self._n_envs = int(getattr(self.env, "num_envs"))
        self._n_agents, self._n_local_obs = map(int, local_single.shape)

        self._has_global_obs = "global_obs" in self.env.single_observation_space.spaces
        self._n_action_features = self._infer_per_agent_action_dim(self.env.single_action_space)

        self._action_keys: tuple[str, ...] | None = None
        if isinstance(self.env.action_space, spaces.Dict):
            self._action_keys = tuple(self.env.action_space.spaces.keys())

        self._single_observation_space = self._with_transition_obs_space(self.env.single_observation_space)
        self._observation_space = self._with_transition_obs_space(self.env.observation_space, is_vector=True)

        self._prev_local = np.zeros((self._n_envs, self._n_agents, self._n_local_obs), dtype=np.float32)
        self._prev_global: np.ndarray | None = None
        if self._has_global_obs:
            global_single: spaces.Box = self.env.single_observation_space["global_obs"]  # type: ignore[assignment]
            self._prev_global = np.zeros((self._n_envs, int(global_single.shape[0])), dtype=np.float32)
        self._prev_dones = np.zeros(self._n_envs, dtype=bool)

    @property
    def single_observation_space(self) -> spaces.Dict:
        return self._single_observation_space

    @property
    def observation_space(self) -> spaces.Dict:
        return self._observation_space

    def reset(self, **kwargs) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        local = np.asarray(obs["local_obs"])
        self._prev_local = local.copy()
        self._prev_dones = np.zeros(self._n_envs, dtype=bool)

        if self._has_global_obs and "global_obs" in obs:
            global_obs = np.asarray(obs["global_obs"])
            self._prev_global = global_obs.copy()
            prev_global = np.zeros_like(global_obs)
        else:
            prev_global = None

        prev_local = np.zeros_like(local)
        prev_actions = np.zeros((self._n_envs, self._n_agents, self._n_action_features), dtype=local.dtype)
        return self._stack_obs(obs, prev_local=prev_local, prev_actions=prev_actions, prev_global=prev_global), info

    def step(
        self, actions: Any
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        prev_local = self._prev_local
        prev_actions = self._actions_to_features(actions, dtype=prev_local.dtype)
        if np.any(self._prev_dones):
            prev_actions = prev_actions.copy()
            prev_actions[self._prev_dones] = 0
        prev_global = self._prev_global if self._has_global_obs else None

        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        self._prev_local = np.asarray(obs["local_obs"]).copy()
        if self._has_global_obs and self._prev_global is not None and "global_obs" in obs:
            self._prev_global = np.asarray(obs["global_obs"]).copy()

        dones = np.logical_or(terminations, truncations)
        if np.any(dones):
            self._prev_local[dones] = 0
            if self._has_global_obs and self._prev_global is not None:
                self._prev_global[dones] = 0
        self._prev_dones = dones

        stacked_obs = self._stack_obs(obs, prev_local=prev_local, prev_actions=prev_actions, prev_global=prev_global)
        return stacked_obs, rewards, terminations, truncations, infos

    def _with_transition_obs_space(self, obs_space: spaces.Space, *, is_vector: bool = False) -> spaces.Dict:
        if not isinstance(obs_space, spaces.Dict):
            raise ValueError(f"Expected Dict observation space, got {type(obs_space)}")

        new_spaces: dict[str, spaces.Space] = dict(obs_space.spaces)

        local_space: spaces.Box = obs_space["local_obs"]  # type: ignore[assignment]
        local_shape = tuple(int(x) for x in local_space.shape)
        if is_vector:
            n_agents, n_local = local_shape[1], local_shape[2]
            new_spaces["local_obs"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(local_shape[0], n_agents, 2 * n_local + self._n_action_features),
                dtype=local_space.dtype,
            )
        else:
            n_agents, n_local = local_shape[0], local_shape[1]
            new_spaces["local_obs"] = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(n_agents, 2 * n_local + self._n_action_features),
                dtype=local_space.dtype,
            )

        if self._has_global_obs and "global_obs" in obs_space.keys():
            global_space: spaces.Box = obs_space["global_obs"]  # type: ignore[assignment]
            global_shape = tuple(int(x) for x in global_space.shape)
            if is_vector:
                new_spaces["global_obs"] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(global_shape[0], 2 * global_shape[1]),
                    dtype=global_space.dtype,
                )
            else:
                new_spaces["global_obs"] = spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(2 * global_shape[0],),
                    dtype=global_space.dtype,
                )

        return spaces.Dict(new_spaces)

    def _stack_obs(
        self,
        obs: dict[str, np.ndarray],
        *,
        prev_local: np.ndarray,
        prev_actions: np.ndarray,
        prev_global: np.ndarray | None,
    ) -> dict[str, np.ndarray]:
        stacked = dict(obs)
        local = np.asarray(obs["local_obs"])
        if self.new_obs_first:
            stacked["local_obs"] = np.concatenate((local, prev_actions, prev_local), axis=-1)
        else:
            stacked["local_obs"] = np.concatenate((prev_local, prev_actions, local), axis=-1)

        if self._has_global_obs and "global_obs" in obs:
            global_obs = np.asarray(obs["global_obs"])
            if prev_global is None:
                prev_global = np.zeros_like(global_obs)
            if self.new_obs_first:
                stacked["global_obs"] = np.concatenate((global_obs, prev_global), axis=-1)
            else:
                stacked["global_obs"] = np.concatenate((prev_global, global_obs), axis=-1)

        return stacked

    def _actions_to_features(self, actions: Any, *, dtype: np.dtype) -> np.ndarray:
        if isinstance(actions, dict) and self._action_keys is not None:
            parts = []
            for key in self._action_keys:
                part = np.asarray(actions[key])
                if part.ndim == 2:
                    part = part[..., None]
                parts.append(part.astype(dtype, copy=False))
            return np.concatenate(parts, axis=-1)

        arr = np.asarray(actions)
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr.astype(dtype, copy=False)

    def _infer_per_agent_action_dim(self, single_action_space: spaces.Space) -> int:
        if isinstance(single_action_space, spaces.Dict):
            dim = 0
            for space in single_action_space.spaces.values():
                shape = getattr(space, "shape", None)
                if shape is None or len(shape) < 2:
                    raise ValueError(f"Unsupported action subspace for transition wrapper: {space}")
                dim += int(shape[-1])
            return dim

        shape = getattr(single_action_space, "shape", None)
        if shape is None or len(shape) < 2:
            raise ValueError(f"Unsupported action space for transition wrapper: {single_action_space}")
        return int(shape[-1])
