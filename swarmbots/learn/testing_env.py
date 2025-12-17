from __future__ import annotations

from typing import Any

import torch
from swarmbots.learn.swarm_bots_learn_wrapper import VectorSwarmBotsActionSpace


class TestingSwarmBotsLearnVectorEnv:
    def __init__(
        self,
        n_envs: int,
        n_agents: int,
        n_local_obs: int,
        n_global_obs: int,
        actuators_dim: int,
        connectors_dim: int,
        device: torch.device | str = "cpu",
        max_steps: int = 100,
    ):
        self._n_envs = n_envs
        self.n_agents = n_agents
        self.n_local_obs = n_local_obs
        self.n_global_obs = n_global_obs
        self.device = torch.device(device)
        self.max_steps = max_steps

        self.action_space = VectorSwarmBotsActionSpace(
            n_envs=n_envs,
            n_agents=n_agents,
            actuators_dim=actuators_dim,
            connectors_dim=connectors_dim,
        )

        self._step_count = torch.zeros(n_envs, dtype=torch.long, device=self.device)

    @property
    def n_envs(self) -> int:
        return self._n_envs

    def reset(self, **kwargs) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self._step_count.fill_(0)
        return self._get_obs(), {}

    def step(
        self, actions: torch.Tensor
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        self._step_count += 1

        truncations = self._step_count >= self.max_steps
        terminations = torch.zeros_like(truncations, dtype=torch.bool)

        rewards = torch.zeros(self._n_envs, dtype=torch.float32, device=self.device)

        # Handle auto-reset behavior (NEXT_STEP)
        dones = torch.logical_or(terminations, truncations)
        if dones.any():
            self._step_count[dones] = 0

        obs = self._get_obs()
        infos = {}

        return obs, rewards, terminations, truncations, infos

    def _get_obs(self) -> dict[str, torch.Tensor]:
        local_obs = self._step_count.view(-1, 1, 1).expand(
            self._n_envs, self.n_agents, self.n_local_obs
        ).float()

        global_obs = self._step_count.view(-1, 1).expand(
            self._n_envs, self.n_global_obs
        ).float()

        return {
            "local_obs": local_obs,
            "global_obs": global_obs,
        }

    def close(self) -> None:
        pass

