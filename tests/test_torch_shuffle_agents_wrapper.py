from __future__ import annotations

from typing import Any
import unittest

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.torch_shuffle_agents_wrapper import TorchShuffleAgentsWrapper


class _AgentIdEnv(gymnasium.Env):
    def __init__(self, *, n_agents: int = 5, active_agents: int = 3, max_steps: int = 2) -> None:
        self.n_agents = int(n_agents)
        self.active_agents = int(active_agents)
        self.max_steps = int(max_steps)
        self._step_count = 0
        self.last_actuators: np.ndarray | None = None

        self.observation_space = spaces.Dict(
            {
                "local_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_agents, 1), dtype=np.float32),
                "global_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "hidden_local_vars": spaces.Box(
                    low=-np.inf,
                    high=np.inf,
                    shape=(self.n_agents, 1),
                    dtype=np.float32,
                ),
                "hidden_global_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
                "agent_mask": spaces.MultiBinary((self.n_agents,)),
            }
        )
        self.action_space = spaces.Dict(
            {
                "actuators": spaces.Box(low=-10.0, high=10.0, shape=(self.n_agents, 1), dtype=np.float32),
                "connectors": spaces.MultiBinary((self.n_agents, 1)),
            }
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        super().reset(seed=seed)
        self._step_count = 0
        return self._obs(), {}

    def step(
        self,
        action: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.last_actuators = np.asarray(action["actuators"], dtype=np.float32).copy()
        self._step_count += 1
        return self._obs(), 0.0, False, self._step_count >= self.max_steps, {}

    def _obs(self) -> dict[str, np.ndarray]:
        agent_ids = np.arange(self.n_agents, dtype=np.float32).reshape(self.n_agents, 1)
        mask = np.arange(self.n_agents) < self.active_agents
        return {
            "local_obs": agent_ids + (10.0 * self._step_count),
            "global_obs": np.asarray([self._step_count], dtype=np.float32),
            "hidden_local_vars": agent_ids + 100.0 + (10.0 * self._step_count),
            "hidden_global_vars": np.asarray([self._step_count + 100.0], dtype=np.float32),
            "agent_mask": mask,
        }


def _make_env(
    *,
    max_steps: int = 2,
    n_envs: int = 1,
    n_agents: int = 5,
    active_agents: int = 3,
    preserve_inactive_prefix_structure: bool = True,
    seed: int = 123,
) -> tuple[TorchShuffleAgentsWrapper, list[_AgentIdEnv]]:
    raw_envs = [
        _AgentIdEnv(n_agents=n_agents, active_agents=active_agents, max_steps=max_steps)
        for _ in range(n_envs)
    ]
    vector_env = SyncVectorEnv(
        [lambda raw_env=raw_env: raw_env for raw_env in raw_envs],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    learn_env = SwarmBotsLearnEnvWrapper(vector_env)
    return (
        TorchShuffleAgentsWrapper(
            learn_env,
            preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
            seed=seed,
        ),
        raw_envs,
    )


class TorchShuffleAgentsWrapperTests(unittest.TestCase):
    def test_preserve_inactive_prefix_structure_keeps_agent_mask_contiguous(self) -> None:
        env, _raw_envs = _make_env()
        try:
            obs, _info = env.reset()
            self.assertTrue(
                torch.equal(
                    obs["agent_mask"].cpu(),
                    torch.tensor([[True, True, True, False, False]]),
                )
            )
            self.assertEqual(set(obs["local_obs"][0, :3, 0].cpu().tolist()), {0.0, 1.0, 2.0})
            self.assertTrue(torch.equal(obs["local_obs"][0, 3:, 0].cpu(), torch.tensor([3.0, 4.0])))
        finally:
            env.close()

    def test_unrestricted_shuffle_can_move_inactive_agents(self) -> None:
        env, _raw_envs = _make_env(
            n_agents=8,
            active_agents=3,
            preserve_inactive_prefix_structure=False,
            seed=123,
        )
        try:
            obs, _info = env.reset()
            self.assertEqual(set(obs["local_obs"][0, :, 0].cpu().tolist()), set(range(8)))
            self.assertEqual(int(obs["agent_mask"].sum().item()), 3)
            self.assertFalse(
                torch.equal(
                    obs["agent_mask"].cpu(),
                    torch.tensor([[True, True, True, False, False, False, False, False]]),
                )
            )
        finally:
            env.close()

    def test_reset_with_same_seed_repeats_permutation(self) -> None:
        env, _raw_envs = _make_env(preserve_inactive_prefix_structure=False)
        try:
            obs_a, _info = env.reset(seed=456)
            obs_b, _info = env.reset(seed=456)
            self.assertTrue(torch.equal(obs_a["local_obs"], obs_b["local_obs"]))
            self.assertTrue(torch.equal(obs_a["agent_mask"], obs_b["agent_mask"]))
        finally:
            env.close()

    def test_partial_reset_keeps_unreset_env_permutation(self) -> None:
        env, _raw_envs = _make_env(
            n_envs=2,
            n_agents=8,
            active_agents=3,
            preserve_inactive_prefix_structure=False,
        )
        try:
            obs, _info = env.reset()
            old_permutation = env._agent_permutation.clone()

            reset_mask = torch.tensor([True, False])
            env._reset_agent_permutations(reset_mask=reset_mask, agent_mask=obs["agent_mask"])

            self.assertEqual(set(env._agent_permutation[0].cpu().tolist()), set(range(8)))
            self.assertTrue(torch.equal(env._agent_permutation[1], old_permutation[1]))
        finally:
            env.close()

    def test_actions_are_unshuffled_before_reaching_env(self) -> None:
        env, raw_envs = _make_env()
        try:
            obs, _info = env.reset()
            actions = torch.cat(
                (
                    obs["local_obs"][..., :1],
                    torch.zeros((1, env.n_agents, 1), dtype=obs["local_obs"].dtype),
                ),
                dim=-1,
            )

            env.step(actions)

            raw_env = raw_envs[0]
            self.assertIsNotNone(raw_env.last_actuators)
            np.testing.assert_allclose(
                raw_env.last_actuators[:, 0],
                np.arange(env.n_agents, dtype=np.float32),
            )
        finally:
            env.close()

    def test_two_dimensional_single_env_actions_are_unshuffled(self) -> None:
        env, raw_envs = _make_env()
        try:
            obs, _info = env.reset()
            actions = torch.cat(
                (
                    obs["local_obs"][0, :, :1],
                    torch.zeros((env.n_agents, 1), dtype=obs["local_obs"].dtype),
                ),
                dim=-1,
            )

            env.step(actions)

            raw_env = raw_envs[0]
            self.assertIsNotNone(raw_env.last_actuators)
            np.testing.assert_allclose(
                raw_env.last_actuators[:, 0],
                np.arange(env.n_agents, dtype=np.float32),
            )
        finally:
            env.close()

    def test_final_obs_uses_terminal_episode_permutation(self) -> None:
        env, _raw_envs = _make_env(max_steps=1)
        try:
            obs, _info = env.reset()
            actions = torch.zeros((1, env.n_agents, env.action_space.total_agent_action_dim), dtype=torch.float32)

            _next_obs, _rewards, _terminations, _truncations, infos = env.step(actions)

            final_obs = infos["final_obs"][0]
            self.assertTrue(torch.equal(final_obs["local_obs"][:, 0] - 10.0, obs["local_obs"][0, :, 0]))
            self.assertTrue(torch.equal(final_obs["hidden_local_vars"][:, 0] - 10.0, obs["hidden_local_vars"][0, :, 0]))
        finally:
            env.close()

    def test_batched_final_obs_dict_uses_supplied_terminal_permutation(self) -> None:
        env, _raw_envs = _make_env(n_envs=2, n_agents=6, active_agents=4)
        try:
            env.reset()
            permutation = env._agent_permutation.clone()
            local_obs = torch.arange(12, dtype=torch.float32).reshape(2, 6, 1)
            hidden_local_vars = local_obs + 100.0
            final_obs = {
                "local_obs": local_obs,
                "global_obs": torch.tensor([[1.0], [2.0]]),
                "hidden_local_vars": hidden_local_vars,
                "hidden_global_vars": torch.tensor([[11.0], [12.0]]),
                "agent_mask": torch.tensor(
                    [
                        [True, True, True, True, False, False],
                        [True, True, True, True, False, False],
                    ]
                ),
            }

            transformed_infos = env._shuffle_final_obs(
                infos={"_final_obs": torch.tensor([True, True]), "final_obs": final_obs},
                permutation=permutation,
            )

            expected_local = local_obs.gather(dim=1, index=permutation.unsqueeze(-1))
            expected_hidden_local = hidden_local_vars.gather(dim=1, index=permutation.unsqueeze(-1))
            self.assertTrue(torch.equal(transformed_infos["final_obs"]["local_obs"], expected_local))
            self.assertTrue(torch.equal(transformed_infos["final_obs"]["hidden_local_vars"], expected_hidden_local))
            self.assertTrue(torch.equal(transformed_infos["final_obs"]["global_obs"], final_obs["global_obs"]))
        finally:
            env.close()

    def test_preserve_inactive_prefix_structure_requires_agent_mask(self) -> None:
        env, _raw_envs = _make_env()
        try:
            with self.assertRaisesRegex(ValueError, "requires agent_mask"):
                env._reset_agent_permutations(
                    reset_mask=torch.tensor([True]),
                    agent_mask=None,
                )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
