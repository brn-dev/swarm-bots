from __future__ import annotations

from typing import Any

import gymnasium
import numpy as np
from gymnasium import spaces


class TestingSwarmBotsEnv(gymnasium.Env):
    def __init__(
        self,
        n_agents: int,
        n_local_obs: int,
        n_global_obs: int,
        actuators_dim: int,
        connectors_dim: int,
        max_steps: int = 100,
    ):
        self.n_agents = n_agents
        self.n_local_obs = n_local_obs
        self.n_global_obs = n_global_obs
        self.max_steps = max_steps

        self._step_count = 0

        self.observation_space = spaces.Dict({
            "local_obs": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(n_agents, n_local_obs),
                dtype=np.float32
            ),
            "global_obs": spaces.Box(
                low=-np.inf, high=np.inf,
                shape=(n_global_obs,),
                dtype=np.float32
            ),
        })

        self.action_space = spaces.Dict({
            "actuators": spaces.Box(
                low=-1.0, high=1.0,
                shape=(n_agents, actuators_dim),
                dtype=np.float32
            ),
            "connectors": spaces.MultiBinary((n_agents, connectors_dim)),
        })

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        return self._get_obs(), {}

    def step(
        self, action: dict[str, np.ndarray]
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self._step_count += 1
        
        truncated = self._step_count >= self.max_steps
        terminated = False
        reward = self._step_count
        
        obs = self._get_obs()
        
        return obs, reward, terminated, truncated, {}

    def _get_obs(self) -> dict[str, np.ndarray]:
        local_obs = np.full(
            (self.n_agents, self.n_local_obs), 
            self._step_count, 
            dtype=np.float32
        )
        global_obs = np.full(
            (self.n_global_obs,), 
            self._step_count, 
            dtype=np.float32
        )

        return {
            "local_obs": local_obs,
            "global_obs": global_obs,
        }

    def close(self) -> None:
        pass
