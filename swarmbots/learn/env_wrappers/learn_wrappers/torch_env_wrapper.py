from __future__ import annotations

from typing import Any

import torch

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper, TorchObs


class TorchEnvWrapper(BaseLearnEnvWrapper):
    def __init__(self, env: BaseLearnEnvWrapper) -> None:
        self.env = env
        self.device = env.device
        self.obs_dtype = env.obs_dtype
        self.reward_dtype = env.reward_dtype
        self._observation_space = env.observation_space
        self._action_space = env.action_space
        self._n_envs = int(getattr(env, "_n_envs", env.num_envs))
        self.local_obs_dim = env.local_obs_dim
        self.global_obs_dim = env.global_obs_dim
        self.hidden_local_vars_dim = env.hidden_local_vars_dim
        self.hidden_global_vars_dim = env.hidden_global_vars_dim
        self.has_agent_mask = env.has_agent_mask
        self.n_agents = env.n_agents

    def reset(self, **kwargs: Any) -> tuple[TorchObs, dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return self.observations(obs), info

    def step(
        self,
        actions: torch.Tensor,
    ) -> tuple[TorchObs, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        return self.observations(obs), self.rewards(rewards), terminations, truncations, infos

    def observations(self, observations: TorchObs) -> TorchObs:
        return observations

    def rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        return rewards

    def set_device(self, device: torch.device | str) -> None:
        self.env.set_device(device)
        self.device = self.env.device

    def _actions_to_env_dict(self, actions: torch.Tensor) -> dict[str, Any]:
        return self.env._actions_to_env_dict(actions)
