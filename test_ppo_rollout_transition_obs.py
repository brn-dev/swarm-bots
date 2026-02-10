from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv, VectorWrapper

from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout import collect_whole_episodes
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPORolloutBuffer
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.transition_obs_wrapper import TransitionObsWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv


@dataclass(frozen=True)
class _RolloutConfig:
    n_envs: int = 2
    n_agents: int = 2
    n_local_obs: int = 2
    n_global_obs: int = 1
    actuators_dim: int = 1
    connectors_dim: int = 1
    max_steps: int = 3
    hidden_vars_dim: int = 1
    gamma: float = 0.99
    gae_lambda: float = 0.95

    @property
    def action_dim(self) -> int:
        return self.actuators_dim + self.connectors_dim

    @property
    def stacked_local_dim(self) -> int:
        return 2 * self.n_local_obs + self.action_dim


class AddHiddenVarsWrapper(VectorWrapper):
    def __init__(self, env: SyncVectorEnv, hidden_vars_dim: int) -> None:
        super().__init__(env)
        self.hidden_vars_dim = hidden_vars_dim
        self.single_observation_space = self._add_space(env.single_observation_space, vector=False)
        self.observation_space = self._add_space(env.observation_space, vector=True)

    def reset(self, **kwargs: Any) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        obs["hidden_vars"] = np.zeros((self.num_envs, self.hidden_vars_dim), dtype=np.float32)
        return obs, info

    def step(
        self, actions: Any
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        obs, rewards, terminations, truncations, infos = self.env.step(actions)
        obs["hidden_vars"] = np.zeros((self.num_envs, self.hidden_vars_dim), dtype=np.float32)
        return obs, rewards, terminations, truncations, infos

    def _add_space(self, space: spaces.Space, *, vector: bool) -> spaces.Dict:
        if not isinstance(space, spaces.Dict):
            raise ValueError(f"Expected Dict space, got {type(space)}")
        spaces_dict = dict(space.spaces)
        shape = (self.num_envs, self.hidden_vars_dim) if vector else (self.hidden_vars_dim,)
        spaces_dict["hidden_vars"] = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=shape,
            dtype=np.float32,
        )
        return spaces.Dict(spaces_dict)


class DeterministicPolicy(BasePPOPolicy):
    def __init__(self, n_local_obs: int) -> None:
        super().__init__()
        self.n_local_obs = n_local_obs
        self.action_dist = type("ActionDist", (), {"has_gsde": False})()

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {}

    def forward(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        hidden_vars: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        new_local = local_obs[..., : self.n_local_obs]
        actuators = new_local[..., :1] + 0.5
        connectors = (new_local[..., 1:2] > 0).to(dtype=local_obs.dtype)
        actions = torch.cat((actuators, connectors), dim=-1)
        log_probs = -actions.sum(dim=-1)
        values = global_obs.sum(dim=-1)
        return actions, log_probs, values

    def evaluate_actions(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        actions: torch.Tensor,
        hidden_vars: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        log_probs = -actions.sum(dim=-1)
        values = global_obs.sum(dim=-1)
        return log_probs, None, values

    def act(
        self,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        hidden_vars: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
        deterministic: bool = False,
    ) -> torch.Tensor:
        return self.forward(local_obs, global_obs, hidden_vars, agent_mask, deterministic)[0]


def _make_env(config: _RolloutConfig) -> TestingSwarmBotsEnv:
    return TestingSwarmBotsEnv(
        n_agents=config.n_agents,
        n_local_obs=config.n_local_obs,
        n_global_obs=config.n_global_obs,
        actuators_dim=config.actuators_dim,
        connectors_dim=config.connectors_dim,
        max_steps=config.max_steps,
    )


def _build_expected_episode(config: _RolloutConfig) -> dict[str, torch.Tensor]:
    n_steps = config.max_steps
    local_obs = np.zeros((n_steps, config.n_agents, config.stacked_local_dim), dtype=np.float32)
    global_obs = np.zeros((n_steps, 2 * config.n_global_obs), dtype=np.float32)
    actions = np.zeros((n_steps, config.n_agents, config.action_dim), dtype=np.float32)
    log_probs = np.zeros((n_steps, config.n_agents), dtype=np.float32)
    values = np.zeros((n_steps,), dtype=np.float32)
    rewards = np.zeros((n_steps,), dtype=np.float32)

    prev_actions = np.zeros((config.n_agents, config.action_dim), dtype=np.float32)
    for step in range(n_steps):
        new_val = float(step)
        prev_local_val = float(max(step - 1, 0))
        prev_global_val = float(max(step - 1, 0))

        local_obs[step] = np.concatenate(
            (
                np.full((config.n_agents, config.n_local_obs), new_val, dtype=np.float32),
                prev_actions,
                np.full((config.n_agents, config.n_local_obs), prev_local_val, dtype=np.float32),
            ),
            axis=-1,
        )
        global_obs[step] = np.array([new_val, prev_global_val], dtype=np.float32)

        actuators = np.full((config.n_agents, 1), new_val + 0.5, dtype=np.float32)
        connectors = np.full((config.n_agents, 1), 1.0 if new_val > 0 else 0.0, dtype=np.float32)
        actions[step] = np.concatenate((actuators, connectors), axis=-1)
        log_probs[step] = -actions[step].sum(axis=-1)
        values[step] = global_obs[step].sum()

        rewards[step] = float(step + 1) if step < n_steps - 1 else -100.0
        prev_actions = actions[step]

    final_new_val = float(n_steps)
    final_prev_local_val = float(n_steps - 1)
    final_local_obs = np.concatenate(
        (
            np.full((config.n_agents, config.n_local_obs), final_new_val, dtype=np.float32),
            actions[-1],
            np.full((config.n_agents, config.n_local_obs), final_prev_local_val, dtype=np.float32),
        ),
        axis=-1,
    )
    final_global_obs = np.array([final_new_val, final_prev_local_val], dtype=np.float32)
    final_value = float(final_global_obs.sum())

    advantages, returns = _compute_gae(
        rewards=rewards,
        values=values,
        final_value=final_value,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
    )

    hidden_vars = torch.zeros((n_steps, config.hidden_vars_dim), dtype=torch.float32)
    final_hidden_vars = torch.zeros((config.hidden_vars_dim,), dtype=torch.float32)

    return {
        "local_obs": torch.tensor(local_obs),
        "global_obs": torch.tensor(global_obs),
        "hidden_vars": hidden_vars,
        "actions": torch.tensor(actions),
        "log_probs": torch.tensor(log_probs),
        "values": torch.tensor(values),
        "rewards": torch.tensor(rewards),
        "final_local_obs": torch.tensor(final_local_obs),
        "final_global_obs": torch.tensor(final_global_obs),
        "final_hidden_vars": final_hidden_vars,
        "final_value": torch.tensor(final_value),
        "advantages": torch.tensor(advantages),
        "returns": torch.tensor(returns),
    }


def _compute_gae(
    *,
    rewards: np.ndarray,
    values: np.ndarray,
    final_value: float,
    gamma: float,
    gae_lambda: float,
) -> tuple[np.ndarray, np.ndarray]:
    advantages = np.zeros_like(rewards, dtype=np.float32)
    last_gae_lam = 0.0
    for step in reversed(range(len(rewards))):
        next_val = final_value if step == len(rewards) - 1 else float(values[step + 1])
        delta = float(rewards[step]) + gamma * next_val - float(values[step])
        last_gae_lam = delta + gamma * gae_lambda * last_gae_lam
        advantages[step] = last_gae_lam
    returns = advantages + values
    return advantages, returns


def test_collect_whole_episodes_with_transition_wrapper() -> None:
    config = _RolloutConfig()
    def _thunk() -> TestingSwarmBotsEnv:
        return _make_env(config)

    vector_env = SyncVectorEnv(
        [_thunk for _ in range(config.n_envs)],
        autoreset_mode=AutoresetMode.NEXT_STEP,
    )
    vector_env = TransitionObsWrapper(vector_env)
    vector_env = AddHiddenVarsWrapper(vector_env, hidden_vars_dim=config.hidden_vars_dim)
    learn_env = SwarmBotsLearnEnvWrapper(vector_env, device="cpu")

    buffer = PPORolloutBuffer(
        n_episodes=config.n_envs * 2,
        max_episode_length=config.max_steps,
        observation_space=learn_env.observation_space,
        action_space=learn_env.action_space,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        rollout_device="cpu",
        train_device="cpu",
    )
    policy = DeterministicPolicy(n_local_obs=config.n_local_obs)

    episodes, episode_infos, metrics = collect_whole_episodes(learn_env, policy, buffer)
    expected = _build_expected_episode(config)

    assert len(episodes) == config.n_envs * 2
    assert episode_infos == []
    assert "env_reset_time" in metrics
    assert "buffer_get_whole_episodes_time" in metrics

    for episode in episodes:
        torch.testing.assert_close(episode.local_obs, expected["local_obs"])
        torch.testing.assert_close(episode.global_obs, expected["global_obs"])
        torch.testing.assert_close(episode.hidden_vars, expected["hidden_vars"])
        torch.testing.assert_close(episode.actions, expected["actions"])
        torch.testing.assert_close(episode.rewards, expected["rewards"])
        torch.testing.assert_close(episode.log_probs, expected["log_probs"])
        torch.testing.assert_close(episode.values, expected["values"])
        torch.testing.assert_close(episode.final_local_obs, expected["final_local_obs"])
        torch.testing.assert_close(episode.final_global_obs, expected["final_global_obs"])
        torch.testing.assert_close(episode.final_hidden_vars, expected["final_hidden_vars"])
        torch.testing.assert_close(episode.final_value, expected["final_value"])
        torch.testing.assert_close(episode.advantages, expected["advantages"])
        torch.testing.assert_close(episode.returns, expected["returns"])
        assert episode.agent_mask is None
        assert episode.final_agent_mask is None
        print('OK')

    learn_env.close()

if __name__ == '__main__':
    test_collect_whole_episodes_with_transition_wrapper()