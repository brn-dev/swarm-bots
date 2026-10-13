from __future__ import annotations

from typing import Any
import unittest

import gymnasium
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.algos.off_policy.off_policy_rollout import collect_off_policy_steps
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBuffer
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.multi_scenario_vector_env import MultiScenarioVectorEnv
from swarmbots.learn.env_wrappers.torch_shuffle_agents_wrapper import TorchShuffleAgentsWrapper
from swarmbots.learn.env_wrappers.torch_transition_obs_wrapper import TorchTransitionObsWrapper


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


class _FirstEpisodeLengthAgentIdEnv(_AgentIdEnv):
    def __init__(self, *, first_episode_length: int, later_episode_length: int, **kwargs: Any) -> None:
        super().__init__(max_steps=first_episode_length, **kwargs)
        self._first_episode_length = int(first_episode_length)
        self._later_episode_length = int(later_episode_length)
        self._reset_count = 0

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        self.max_steps = self._first_episode_length if self._reset_count == 0 else self._later_episode_length
        self._reset_count += 1
        return super().reset(seed=seed, options=options)


def _make_env(
    *,
    max_steps: int = 2,
    n_envs: int = 1,
    n_agents: int = 5,
    active_agents: int = 3,
    preserve_inactive_prefix_structure: bool = True,
    seed: int = 123,
    multi_scenario: bool = False,
) -> tuple[TorchShuffleAgentsWrapper, list[_AgentIdEnv]]:
    raw_envs = [
        _AgentIdEnv(n_agents=n_agents, active_agents=active_agents, max_steps=max_steps)
        for _ in range(n_envs)
    ]
    vector_env = SyncVectorEnv(
        [lambda raw_env=raw_env: raw_env for raw_env in raw_envs],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    if multi_scenario:
        vector_env = MultiScenarioVectorEnv({"agents": vector_env})
    learn_env = SwarmBotsLearnEnvWrapper(vector_env)
    return (
        TorchShuffleAgentsWrapper(
            learn_env,
            preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
            seed=seed,
        ),
        raw_envs,
    )


def _transition_action_from_current_obs(obs: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        (
            obs["local_obs"][..., :1],
            torch.zeros((*obs["local_obs"].shape[:2], 1), dtype=obs["local_obs"].dtype, device=obs["local_obs"].device),
        ),
        dim=-1,
    )


def _assert_transition_parts(
    test_case: unittest.TestCase,
    obs: dict[str, torch.Tensor],
    *,
    current_local: torch.Tensor,
    prev_actions: torch.Tensor,
    prev_local: torch.Tensor,
    current_global: torch.Tensor,
    prev_global: torch.Tensor,
    new_obs_first: bool = True,
) -> None:
    if new_obs_first:
        actual_current_local = obs["local_obs"][..., :1]
        actual_prev_actions = obs["local_obs"][..., 1:3]
        actual_prev_local = obs["local_obs"][..., 3:4]
        actual_current_global = obs["global_obs"][..., :1]
        actual_prev_global = obs["global_obs"][..., 1:2]
    else:
        actual_prev_local = obs["local_obs"][..., :1]
        actual_prev_actions = obs["local_obs"][..., 1:3]
        actual_current_local = obs["local_obs"][..., 3:4]
        actual_prev_global = obs["global_obs"][..., :1]
        actual_current_global = obs["global_obs"][..., 1:2]

    test_case.assertTrue(torch.equal(actual_current_local, current_local))
    test_case.assertTrue(torch.equal(actual_prev_actions, prev_actions))
    test_case.assertTrue(torch.equal(actual_prev_local, prev_local))
    test_case.assertTrue(torch.equal(actual_current_global, current_global))
    test_case.assertTrue(torch.equal(actual_prev_global, prev_global))


@pytest.mark.parametrize("seed", [123, 456])
@pytest.mark.parametrize("preserve_inactive_prefix_structure", [False, True])
def test_off_policy_collection_preserves_shuffle_stream_on_same_device_calls(
    seed: int, preserve_inactive_prefix_structure: bool,
) -> None:
    reference_env, _raw_envs = _make_env(
        max_steps=1, n_envs=2, n_agents=8,
        preserve_inactive_prefix_structure=preserve_inactive_prefix_structure, seed=seed,
    )
    chunked_env, _raw_envs = _make_env(
        max_steps=1, n_envs=2, n_agents=8,
        preserve_inactive_prefix_structure=preserve_inactive_prefix_structure, seed=seed,
    )
    try:
        def make_replay(env: TorchShuffleAgentsWrapper) -> OffPolicyReplayBuffer:
            return OffPolicyReplayBuffer(
                capacity_per_env=8, observation_space=env.observation_space,
                action_space=env.action_space, train_device="cpu",
            )

        reference_replay = make_replay(reference_env)
        chunked_replay = make_replay(chunked_env)
        collect_off_policy_steps(reference_env, reference_replay, n_steps=8, random_actions=True)
        generator = chunked_env._rng
        state = None
        for _ in range(4):
            _infos, _metrics, state = collect_off_policy_steps(
                chunked_env, chunked_replay, n_steps=2, random_actions=True,
                rollout_state=state, rollout_device="cpu:0",
            )

        assert chunked_env._rng is generator
        reference = reference_replay.get_all()
        chunked = chunked_replay.get_all()
        for name in ("local_obs", "next_local_obs", "hidden_local_vars", "next_hidden_local_vars", "agent_mask", "next_agent_mask"):
            torch.testing.assert_close(getattr(chunked, name), getattr(reference, name), rtol=0, atol=0)
        assert torch.unique(chunked.local_obs[..., 0], dim=0).shape[0] > 1
    finally:
        reference_env.close()
        chunked_env.close()


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_shuffle_device_moves_retain_reset_seed() -> None:
    env, _raw_envs = _make_env(n_agents=8, preserve_inactive_prefix_structure=False, seed=123)
    try:
        env.reset(seed=456)
        for device in (torch.device("cuda", torch.cuda.current_device()), torch.device("cpu")):
            env.set_device(device)
            assert env._rng.initial_seed() == 456
            expected_rng = torch.Generator(device=device).manual_seed(456)
            expected = torch.rand((1, 8), generator=expected_rng, device=device).argsort(dim=1)
            env._reset_agent_permutations(
                reset_mask=torch.ones(1, device=device, dtype=torch.bool), agent_mask=None,
            )
            torch.testing.assert_close(env._agent_permutation, expected)
            generator = env._rng
            state = generator.get_state().clone()
            env.set_device(device)
            assert env._rng is generator
            torch.testing.assert_close(env._rng.get_state(), state)
    finally:
        env.close()


@pytest.mark.parametrize("preserve_inactive_prefix_structure", [False, True])
@pytest.mark.parametrize("wrapper_seed", [123, 2**64 - 1])
def test_list_reset_seeds_repeat_permutations_and_keep_lanes_independent(
    preserve_inactive_prefix_structure: bool,
    wrapper_seed: int,
) -> None:
    env, _ = _make_env(
        n_envs=2, n_agents=8, active_agents=5, max_steps=1,
        preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
        seed=wrapper_seed,
    )
    reference, _ = _make_env(
        n_envs=2, n_agents=8, active_agents=5, max_steps=1,
        preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
        seed=wrapper_seed,
    )
    try:
        initial, _ = env.reset(seed=[123, 456])
        repeated, _ = env.reset(seed=[123, 456])
        torch.testing.assert_close(initial["local_obs"], repeated["local_obs"], rtol=0, atol=0)
        reference_obs, _ = reference.reset(seed=[123, 999])
        torch.testing.assert_close(repeated["local_obs"][0], reference_obs["local_obs"][0], rtol=0, atol=0)

        partial, _ = env.reset(seed=[888, 789], options={"reset_mask": np.array([False, True])})
        torch.testing.assert_close(partial["local_obs"][0], repeated["local_obs"][0], rtol=0, atol=0)
        actions = torch.zeros(2, env.n_agents, env.action_space.total_agent_action_dim)
        for _ in range(4):
            observed, *_ = env.step(actions)
            expected, *_ = reference.step(actions)
            torch.testing.assert_close(observed["local_obs"][0], expected["local_obs"][0], rtol=0, atol=0)
    finally:
        env.close()
        reference.close()


@pytest.mark.parametrize("preserve_inactive_prefix_structure", [False, True])
@pytest.mark.parametrize("initial_seed, partial_seed", [
    (123, [888, 789]),
    (123, 789),
    ([123, 456], 789),
])
@pytest.mark.parametrize("reset_mask", [[False, True], [True, False], [False, False]])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_partial_seeded_resets_preserve_untouched_world_shuffle_streams(
    preserve_inactive_prefix_structure: bool,
    initial_seed: int | list[int],
    partial_seed: int | list[int],
    reset_mask: list[bool],
    device: str,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    env, _ = _make_env(
        n_envs=2, n_agents=8, active_agents=5, max_steps=1,
        preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
        multi_scenario=True,
    )
    reference, _ = _make_env(
        n_envs=2, n_agents=8, active_agents=5, max_steps=1,
        preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
        multi_scenario=True,
    )
    try:
        env.set_device(device)
        reference.set_device(device)
        initial, _ = env.reset(seed=initial_seed)
        reference.reset(seed=initial_seed)
        untouched = ~torch.tensor(reset_mask, device=device)
        observed, _ = env.reset(seed=partial_seed, options={"reset_mask": np.array(reset_mask)})
        reference.reset(options={"reset_mask": np.array(reset_mask)})
        torch.testing.assert_close(observed["local_obs"][untouched], initial["local_obs"][untouched], rtol=0, atol=0)
        actions = torch.zeros(2, env.n_agents, env.action_space.total_agent_action_dim, device=device)
        for _ in range(4):
            observed, *_ = env.step(actions)
            expected, *_ = reference.step(actions)
            torch.testing.assert_close(observed["local_obs"][untouched], expected["local_obs"][untouched], rtol=0, atol=0)
    finally:
        env.close()
        reference.close()


@pytest.mark.parametrize("preserve_inactive_prefix_structure", [False, True])
def test_full_integer_reset_restores_the_original_shuffle_sequence(
    preserve_inactive_prefix_structure: bool,
) -> None:
    env, _ = _make_env(
        n_envs=2, n_agents=8, active_agents=5, max_steps=1,
        preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
    )
    reference, _ = _make_env(
        n_envs=2, n_agents=8, active_agents=5, max_steps=1,
        preserve_inactive_prefix_structure=preserve_inactive_prefix_structure,
    )
    try:
        env.reset(seed=[123, 456])
        observed, _ = env.reset(seed=789)
        expected, _ = reference.reset(seed=789)
        torch.testing.assert_close(observed["local_obs"], expected["local_obs"], rtol=0, atol=0)
        actions = torch.zeros(2, env.n_agents, env.action_space.total_agent_action_dim)
        for _ in range(4):
            observed, *_ = env.step(actions)
            expected, *_ = reference.step(actions)
            torch.testing.assert_close(observed["local_obs"], expected["local_obs"], rtol=0, atol=0)
    finally:
        env.close()
        reference.close()


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

    def test_transition_obs_zeroes_reset_obs_state_after_shuffle_resamples_done_envs(self) -> None:
        shuffled_env, _raw_envs = _make_env(max_steps=1)
        env = TorchTransitionObsWrapper(shuffled_env)
        try:
            obs, _info = env.reset()
            actions = torch.cat(
                (
                    obs["local_obs"][..., :1],
                    torch.zeros((1, env.n_agents, 1), dtype=obs["local_obs"].dtype),
                ),
                dim=-1,
            )

            next_obs, _rewards, _terminations, truncations, infos = env.step(actions)

            self.assertTrue(bool(truncations[0].item()))
            self.assertTrue(torch.equal(next_obs["local_obs"][0, :, 1:3], torch.zeros((env.n_agents, 2))))
            self.assertTrue(torch.equal(next_obs["local_obs"][0, :, 3], torch.zeros((env.n_agents,))))
            self.assertTrue(torch.equal(infos["final_obs"][0]["local_obs"][:, 1], obs["local_obs"][0, :, 0]))
            self.assertTrue(torch.equal(infos["final_obs"][0]["local_obs"][:, 3], obs["local_obs"][0, :, 0]))
        finally:
            env.close()

    def test_transition_obs_carries_history_on_non_done_step_with_shuffle(self) -> None:
        shuffled_env, _raw_envs = _make_env(max_steps=3)
        env = TorchTransitionObsWrapper(shuffled_env)
        try:
            obs, _info = env.reset()
            actions = _transition_action_from_current_obs(obs)

            next_obs, _rewards, _terminations, truncations, infos = env.step(actions)

            self.assertFalse(bool(truncations[0].item()))
            if "_final_obs" in infos:
                self.assertFalse(bool(torch.as_tensor(infos["_final_obs"])[0].item()))
            _assert_transition_parts(
                self,
                next_obs,
                current_local=obs["local_obs"][..., :1] + 10.0,
                prev_actions=actions,
                prev_local=obs["local_obs"][..., :1],
                current_global=obs["global_obs"][..., :1] + 1.0,
                prev_global=obs["global_obs"][..., :1],
            )
        finally:
            env.close()

    def test_transition_obs_false_new_obs_first_layout_handles_same_step_reset(self) -> None:
        shuffled_env, _raw_envs = _make_env(max_steps=1)
        env = TorchTransitionObsWrapper(shuffled_env, new_obs_first=False)
        try:
            obs, _info = env.reset()
            current_local = obs["local_obs"][..., 3:4]
            actions = torch.cat(
                (
                    current_local,
                    torch.zeros((1, env.n_agents, 1), dtype=obs["local_obs"].dtype),
                ),
                dim=-1,
            )

            reset_obs, _rewards, _terminations, truncations, infos = env.step(actions)

            self.assertTrue(bool(truncations[0].item()))
            _assert_transition_parts(
                self,
                reset_obs,
                current_local=reset_obs["local_obs"][..., 3:4],
                prev_actions=torch.zeros((1, env.n_agents, 2)),
                prev_local=torch.zeros((1, env.n_agents, 1)),
                current_global=reset_obs["global_obs"][..., 1:2],
                prev_global=torch.zeros((1, 1)),
                new_obs_first=False,
            )
            _assert_transition_parts(
                self,
                {
                    "local_obs": infos["final_obs"][0]["local_obs"].unsqueeze(0),
                    "global_obs": infos["final_obs"][0]["global_obs"].unsqueeze(0),
                },
                current_local=current_local + 10.0,
                prev_actions=actions,
                prev_local=current_local,
                current_global=obs["global_obs"][..., 1:2] + 1.0,
                prev_global=obs["global_obs"][..., 1:2],
                new_obs_first=False,
            )
        finally:
            env.close()

    def test_transition_obs_normalizes_binary_prev_actions_in_final_obs_but_zeroes_reset_obs(self) -> None:
        shuffled_env, _raw_envs = _make_env(max_steps=1)
        env = TorchTransitionObsWrapper(shuffled_env, normalize_prev_binary_actions=True)
        try:
            obs, _info = env.reset()
            actions = torch.cat(
                (
                    obs["local_obs"][..., :1] * 0.25,
                    torch.zeros((1, env.n_agents, 1), dtype=obs["local_obs"].dtype),
                ),
                dim=-1,
            )
            actions[..., 1] = torch.tensor([[1.0, 0.0, 1.0, 0.0, 1.0]])

            reset_obs, _rewards, _terminations, truncations, infos = env.step(actions)

            self.assertTrue(bool(truncations[0].item()))
            self.assertTrue(torch.equal(reset_obs["local_obs"][0, :, 1:3], torch.zeros((env.n_agents, 2))))
            expected_final_prev_actions = actions.clone()
            expected_final_prev_actions[..., 1] = expected_final_prev_actions[..., 1] * 2.0 - 1.0
            self.assertTrue(torch.equal(infos["final_obs"][0]["local_obs"][:, 1:3], expected_final_prev_actions[0]))
        finally:
            env.close()

    def test_transition_obs_accepts_two_dimensional_single_env_actions(self) -> None:
        shuffled_env, _raw_envs = _make_env(max_steps=3)
        env = TorchTransitionObsWrapper(shuffled_env)
        try:
            obs, _info = env.reset()
            actions = _transition_action_from_current_obs(obs)[0]

            next_obs, _rewards, _terminations, truncations, _infos = env.step(actions)

            self.assertFalse(bool(truncations[0].item()))
            self.assertTrue(torch.equal(next_obs["local_obs"][0, :, 1:3], actions))
            self.assertTrue(torch.equal(next_obs["local_obs"][0, :, 3], obs["local_obs"][0, :, 0]))
        finally:
            env.close()

    def test_transition_obs_simultaneous_same_step_dones_reset_all_returned_histories(self) -> None:
        shuffled_env, _raw_envs = _make_env(max_steps=1, n_envs=3)
        env = TorchTransitionObsWrapper(shuffled_env)
        try:
            obs, _info = env.reset()
            actions = _transition_action_from_current_obs(obs)

            reset_obs, _rewards, _terminations, truncations, infos = env.step(actions)

            self.assertTrue(torch.equal(truncations.cpu(), torch.tensor([True, True, True])))
            self.assertTrue(torch.equal(reset_obs["local_obs"][..., 1:3], torch.zeros((3, env.n_agents, 2))))
            self.assertTrue(torch.equal(reset_obs["local_obs"][..., 3], torch.zeros((3, env.n_agents))))
            self.assertTrue(torch.equal(reset_obs["global_obs"][..., 1], torch.zeros((3,))))
            final_obs_entries = infos["final_obs"]
            for env_idx in range(3):
                self.assertTrue(
                    torch.equal(final_obs_entries[env_idx]["local_obs"][:, 1:3], actions[env_idx])
                )
                self.assertTrue(
                    torch.equal(final_obs_entries[env_idx]["local_obs"][:, 3], obs["local_obs"][env_idx, :, 0])
                )
        finally:
            env.close()

    def test_transition_obs_unrestricted_shuffle_same_step_reset_keeps_final_old_order_and_reset_zeroed(self) -> None:
        shuffled_env, _raw_envs = _make_env(
            max_steps=1,
            n_agents=8,
            active_agents=3,
            preserve_inactive_prefix_structure=False,
            seed=123,
        )
        env = TorchTransitionObsWrapper(shuffled_env)
        try:
            obs, _info = env.reset()
            actions = _transition_action_from_current_obs(obs)

            reset_obs, _rewards, _terminations, truncations, infos = env.step(actions)

            self.assertTrue(bool(truncations[0].item()))
            self.assertFalse(torch.equal(reset_obs["agent_mask"], obs["agent_mask"]))
            self.assertTrue(torch.equal(reset_obs["local_obs"][0, :, 1:3], torch.zeros((env.n_agents, 2))))
            self.assertTrue(torch.equal(reset_obs["local_obs"][0, :, 3], torch.zeros((env.n_agents,))))
            self.assertTrue(torch.equal(infos["final_obs"][0]["local_obs"][:, 1:3], actions[0]))
            self.assertTrue(torch.equal(infos["final_obs"][0]["local_obs"][:, 3], obs["local_obs"][0, :, 0]))
        finally:
            env.close()

    def test_transition_obs_repeated_one_step_episodes_zero_each_reset_but_chain_next_episode_state(self) -> None:
        raw_env = _FirstEpisodeLengthAgentIdEnv(first_episode_length=1, later_episode_length=1)
        vector_env = SyncVectorEnv([lambda: raw_env], autoreset_mode=AutoresetMode.SAME_STEP)
        shuffled_env = TorchShuffleAgentsWrapper(
            SwarmBotsLearnEnvWrapper(vector_env),
            preserve_inactive_prefix_structure=True,
            seed=123,
        )
        env = TorchTransitionObsWrapper(shuffled_env)
        try:
            obs, _info = env.reset()
            for _ in range(3):
                actions = _transition_action_from_current_obs(obs)
                reset_obs, _rewards, _terminations, truncations, infos = env.step(actions)

                self.assertTrue(bool(truncations[0].item()))
                self.assertTrue(torch.equal(reset_obs["local_obs"][0, :, 1:3], torch.zeros((env.n_agents, 2))))
                self.assertTrue(torch.equal(reset_obs["local_obs"][0, :, 3], torch.zeros((env.n_agents,))))
                self.assertTrue(torch.equal(infos["final_obs"][0]["local_obs"][:, 1:3], actions[0]))
                self.assertTrue(torch.equal(infos["final_obs"][0]["local_obs"][:, 3], obs["local_obs"][0, :, 0]))
                obs = reset_obs
        finally:
            env.close()

    def test_transition_obs_zeroes_only_done_envs_after_partial_shuffle_reset(self) -> None:
        raw_envs = [
            _AgentIdEnv(max_steps=1),
            _AgentIdEnv(max_steps=3),
        ]
        vector_env = SyncVectorEnv(
            [lambda raw_env=raw_env: raw_env for raw_env in raw_envs],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        shuffled_env = TorchShuffleAgentsWrapper(
            SwarmBotsLearnEnvWrapper(vector_env),
            preserve_inactive_prefix_structure=True,
            seed=123,
        )
        env = TorchTransitionObsWrapper(shuffled_env)
        try:
            obs, _info = env.reset()
            actions = torch.cat(
                (
                    obs["local_obs"][..., :1],
                    torch.zeros((2, env.n_agents, 1), dtype=obs["local_obs"].dtype),
                ),
                dim=-1,
            )

            next_obs, _rewards, _terminations, truncations, infos = env.step(actions)

            self.assertTrue(torch.equal(truncations.cpu(), torch.tensor([True, False])))
            self.assertTrue(torch.equal(next_obs["local_obs"][0, :, 1:3], torch.zeros((env.n_agents, 2))))
            self.assertTrue(torch.equal(next_obs["local_obs"][0, :, 3], torch.zeros((env.n_agents,))))
            self.assertTrue(torch.equal(next_obs["global_obs"][0, 1:], torch.zeros((1,))))
            self.assertTrue(torch.equal(next_obs["local_obs"][1, :, 1], obs["local_obs"][1, :, 0]))
            self.assertTrue(torch.equal(next_obs["local_obs"][1, :, 3], obs["local_obs"][1, :, 0]))
            self.assertTrue(torch.equal(next_obs["global_obs"][1, 1:], obs["global_obs"][1, :1]))
            self.assertTrue(torch.equal(infos["final_obs"][0]["local_obs"][:, 1], obs["local_obs"][0, :, 0]))
            self.assertTrue(torch.equal(infos["final_obs"][0]["local_obs"][:, 3], obs["local_obs"][0, :, 0]))
        finally:
            env.close()

    def test_transition_obs_keeps_reset_obs_as_state_for_step_after_same_step_reset(self) -> None:
        raw_env = _FirstEpisodeLengthAgentIdEnv(first_episode_length=1, later_episode_length=3)
        vector_env = SyncVectorEnv([lambda: raw_env], autoreset_mode=AutoresetMode.SAME_STEP)
        shuffled_env = TorchShuffleAgentsWrapper(
            SwarmBotsLearnEnvWrapper(vector_env),
            preserve_inactive_prefix_structure=True,
            seed=123,
        )
        env = TorchTransitionObsWrapper(shuffled_env)
        try:
            obs, _info = env.reset()
            terminal_actions = torch.zeros((1, env.n_agents, env.action_space.total_agent_action_dim))
            reset_obs, _rewards, _terminations, truncations, _infos = env.step(terminal_actions)
            self.assertTrue(bool(truncations[0].item()))
            self.assertTrue(torch.equal(reset_obs["local_obs"][0, :, 1:3], torch.zeros((env.n_agents, 2))))
            self.assertTrue(torch.equal(reset_obs["local_obs"][0, :, 3], torch.zeros((env.n_agents,))))

            reset_current_local = reset_obs["local_obs"][..., :1]
            first_new_episode_actions = torch.cat(
                (
                    reset_current_local,
                    torch.zeros((1, env.n_agents, 1), dtype=reset_obs["local_obs"].dtype),
                ),
                dim=-1,
            )
            next_obs, _rewards, _terminations, truncations, _infos = env.step(first_new_episode_actions)

            self.assertFalse(bool(truncations[0].item()))
            self.assertTrue(torch.equal(next_obs["local_obs"][0, :, 1], reset_obs["local_obs"][0, :, 0]))
            self.assertTrue(torch.equal(next_obs["local_obs"][0, :, 3], reset_obs["local_obs"][0, :, 0]))
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
