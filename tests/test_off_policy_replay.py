import unittest
from typing import Any
from unittest.mock import patch

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

import swarmbots.learn.algos.off_policy.off_policy_rollout as off_policy_rollout
from swarmbots.learn.algos.off_policy import OffPolicyReplayBuffer, collect_off_policy_steps
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEIntervalResetMode


class _ScriptedOffPolicyEnv(gymnasium.Env):
    metadata = {}

    def __init__(
            self,
            *,
            done_steps: tuple[int, ...] = (),
            terminate_steps: tuple[int, ...] = (),
            env_id: int = 0,
            include_agent_mask: bool = False,
    ) -> None:
        super().__init__()
        self.done_steps = done_steps
        self.terminate_steps = terminate_steps
        self.env_id = env_id
        self.include_agent_mask = include_agent_mask
        self.episode_id = 0
        self.step_count = 0
        self.episode_return = 0.0
        observation_spaces: dict[str, spaces.Space] = {
            "local_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 3), dtype=np.float32),
            "global_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            "hidden_local_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(2, 1), dtype=np.float32),
            "hidden_global_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32),
        }
        if include_agent_mask:
            observation_spaces["agent_mask"] = spaces.MultiBinary(2)
        self.observation_space = spaces.Dict(observation_spaces)
        self.action_space = spaces.Dict({
            "actuators": spaces.Box(low=-1.0, high=1.0, shape=(2, 1), dtype=np.float32),
            "connectors": spaces.MultiBinary((2, 1)),
        })

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _ = options
        super().reset(seed=seed)
        self.episode_id += 1
        self.step_count = 0
        self.episode_return = 0.0
        return self._obs(), {}

    def step(
            self,
            action: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        _ = action
        self.step_count += 1
        reward = float(self.step_count)
        self.episode_return += reward
        terminated = self.step_count in self.terminate_steps
        truncated = self.step_count in self.done_steps
        info: dict[str, Any] = {}
        if terminated or truncated:
            info["episode"] = {"r": np.float64(self.episode_return), "success": np.bool_(terminated)}
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> dict[str, np.ndarray]:
        value = float(self.env_id * 1000 + self.episode_id * 100 + self.step_count)
        obs = {
            "local_obs": np.full((2, 3), value, dtype=np.float32),
            "global_obs": np.full((2,), value + 0.5, dtype=np.float32),
            "hidden_local_vars": np.full((2, 1), value + 1.0, dtype=np.float32),
            "hidden_global_vars": np.full((1,), value + 2.0, dtype=np.float32),
        }
        if self.include_agent_mask:
            obs["agent_mask"] = np.asarray(
                [True, (self.episode_id + self.step_count) % 2 == 0],
                dtype=np.bool_,
            )
        return obs


def _make_env(
        *,
        done_steps: tuple[int, ...] = (),
        terminate_steps: tuple[int, ...] = (),
) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [lambda: _ScriptedOffPolicyEnv(done_steps=done_steps, terminate_steps=terminate_steps)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_agent_mask_env(*, done_steps: tuple[int, ...] = ()) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [lambda: _ScriptedOffPolicyEnv(done_steps=done_steps, include_agent_mask=True)],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_multi_env(*done_steps_per_env: tuple[int, ...]) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [
            (lambda env_id=env_id, done_steps=done_steps: _ScriptedOffPolicyEnv(
                done_steps=done_steps,
                env_id=env_id,
            ))
            for env_id, done_steps in enumerate(done_steps_per_env)
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def _make_buffer(
        env: SwarmBotsLearnEnvWrapper,
        *,
        capacity_per_env: int,
        store_previous_actions: bool = False,
        temporal_state_store_interval: int | None = None,
) -> OffPolicyReplayBuffer:
    return OffPolicyReplayBuffer(
        capacity_per_env=capacity_per_env,
        observation_space=env.observation_space,
        action_space=env.action_space,
        store_previous_actions=store_previous_actions,
        temporal_state_store_interval=temporal_state_store_interval,
        storage_device="cpu",
        train_device="cpu",
    )


def _obs(value: float) -> dict[str, torch.Tensor]:
    return {
        "local_obs": torch.full((1, 2, 3), value),
        "global_obs": torch.full((1, 2), value + 0.5),
        "hidden_local_vars": torch.full((1, 2, 1), value + 1.0),
        "hidden_global_vars": torch.full((1, 1), value + 2.0),
    }


def _multi_obs(values: tuple[float, ...]) -> dict[str, torch.Tensor]:
    value_tensor = torch.tensor(values, dtype=torch.float32)
    return {
        "local_obs": value_tensor.view(-1, 1, 1).expand(-1, 2, 3).clone(),
        "global_obs": (value_tensor + 0.5).view(-1, 1).expand(-1, 2).clone(),
        "hidden_local_vars": (value_tensor + 1.0).view(-1, 1, 1).expand(-1, 2, 1).clone(),
        "hidden_global_vars": (value_tensor + 2.0).view(-1, 1).clone(),
    }


def _actions(value: float) -> torch.Tensor:
    return torch.full((1, 2, 2), value)


def _temporal_state(value: float) -> torch.Tensor:
    return torch.full((1, 2, 1), value)


class _PreviousActionPolicy(BasePolicy):
    @property
    def gsde_enabled(self) -> bool:
        return False

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {}

    def get_grad_norms(self) -> dict[str, float]:
        return {}

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        _ = local_obs
        _ = global_obs
        _ = hidden_local_vars
        _ = hidden_global_vars
        _ = agent_mask
        _ = deterministic
        if previous_actions is None:
            raise AssertionError("previous_actions must be passed to policies that require them")
        return previous_actions + 1.0

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unknown loss weights: {sorted(weights)}")

    def requires_previous_actions(self) -> bool:
        return True


class _NoPreviousActionPolicy(BasePolicy):
    @property
    def gsde_enabled(self) -> bool:
        return False

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {}

    def get_grad_norms(self) -> dict[str, float]:
        return {}

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        _ = global_obs
        _ = hidden_local_vars
        _ = hidden_global_vars
        _ = agent_mask
        _ = deterministic
        if previous_actions is not None:
            raise AssertionError("previous_actions must not be passed to policies that do not require them")
        return torch.zeros((*local_obs.shape[:2], 2), dtype=local_obs.dtype, device=local_obs.device)

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unknown loss weights: {sorted(weights)}")

    def requires_previous_actions(self) -> bool:
        return False


class _TemporalPolicy(BasePolicy):
    def __init__(self) -> None:
        super().__init__()
        self.episode_start_masks: list[torch.Tensor] = []

    @property
    def gsde_enabled(self) -> bool:
        return False

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {}

    def get_grad_norms(self) -> dict[str, float]:
        return {}

    def initial_temporal_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> torch.Tensor:
        return torch.zeros((batch_size, n_agents, 1), dtype=dtype, device=device)

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        _ = local_obs
        _ = global_obs
        _ = hidden_local_vars
        _ = hidden_global_vars
        _ = agent_mask
        _ = previous_actions
        _ = deterministic
        raise AssertionError("off-policy rollout should use act_with_temporal_state()")

    def act_with_temporal_state(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            *,
            temporal_state: Any = None,
            episode_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = global_obs
        _ = hidden_local_vars
        _ = hidden_global_vars
        _ = agent_mask
        _ = previous_actions
        _ = deterministic
        if not isinstance(temporal_state, torch.Tensor):
            raise AssertionError("temporal_state must be passed")
        if episode_start_mask is None:
            raise AssertionError("episode_start_mask must be passed")
        self.episode_start_masks.append(episode_start_mask.detach().cpu().clone())
        reset_mask = episode_start_mask.view(-1, 1, 1)
        current_state = temporal_state.masked_fill(reset_mask, 0.0)
        action_value = current_state + 1.0
        actions = action_value.expand(*local_obs.shape[:2], 2).clone()
        return actions, action_value

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unknown loss weights: {sorted(weights)}")

    def requires_previous_actions(self) -> bool:
        return False


class _SpyGSDEActionDist:
    def __init__(self) -> None:
        self.episode_start_masks: list[torch.Tensor] = []
        self.step_resets: list[tuple[torch.Tensor | None, tuple[int, ...] | None]] = []

    def reset_temporal_correlations_on_ep_start(self, mask: torch.Tensor) -> None:
        self.episode_start_masks.append(mask.detach().cpu().clone())

    def reset_temporal_correlations_on_step(
            self,
            mask: torch.Tensor | None = None,
            batch_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.step_resets.append((None if mask is None else mask.detach().cpu().clone(), batch_shape))


class _GSDEPolicy(_NoPreviousActionPolicy):
    def __init__(self) -> None:
        super().__init__()
        self.action_dist = _SpyGSDEActionDist()

    @property
    def gsde_enabled(self) -> bool:
        return True


class OffPolicyReplayTests(unittest.TestCase):
    def assertNoNextPreviousActions(self, batch: object) -> None:
        self.assertFalse(hasattr(batch, "next_previous_actions"))

    def test_storage_pin_memory_requires_cpu_storage(self) -> None:
        env = _make_env()
        try:
            with self.assertRaisesRegex(ValueError, "storage_pin_memory"):
                OffPolicyReplayBuffer(
                    capacity_per_env=2,
                    observation_space=env.observation_space,
                    action_space=env.action_space,
                    storage_device="cuda",
                    storage_pin_memory=True,
                    train_device="cpu",
                )
        finally:
            env.close()

    def test_non_blocking_train_transfer_can_be_enabled_explicitly(self) -> None:
        env = _make_env()
        try:
            buffer = OffPolicyReplayBuffer(
                capacity_per_env=2,
                observation_space=env.observation_space,
                action_space=env.action_space,
                storage_device="cpu",
                train_device="cpu",
                non_blocking_train_transfer=True,
            )

            self.assertTrue(buffer.non_blocking_train_transfer)
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required for pinned-memory replay transfer")
    def test_storage_pin_memory_allocates_pinned_cpu_storage(self) -> None:
        env = _make_env()
        try:
            buffer = OffPolicyReplayBuffer(
                capacity_per_env=2,
                observation_space=env.observation_space,
                action_space=env.action_space,
                storage_device="cpu",
                storage_pin_memory=True,
                train_device="cuda",
            )

            self.assertTrue(buffer.local_obs.is_pinned())
            self.assertTrue(buffer.actions.is_pinned())
            self.assertTrue(buffer.rewards.is_pinned())
            self.assertTrue(buffer.terminations.is_pinned())
            self.assertTrue(buffer.truncations.is_pinned())
            self.assertTrue(buffer.non_blocking_train_transfer)
        finally:
            env.close()

    def test_ring_observation_slots_survive_wraparound(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            for step in range(5):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [2.0, 3.0, 4.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [3.0, 4.0, 5.0])
        finally:
            env.close()

    def test_sample_defaults_to_replacement_and_allows_oversized_batches(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            for step in range(2):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                )

            batch = buffer.sample(5)
            self.assertEqual(batch.actions.shape[0], 5)
            self.assertEqual(batch.local_obs.shape[0], 5)
        finally:
            env.close()

    def test_default_buffer_does_not_allocate_recurrent_storage(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)

            self.assertIsNone(buffer.episode_starts)
            self.assertIsNone(buffer.temporal_states)
            self.assertIsNone(buffer._temporal_state_available)
            self.assertIsNone(buffer._current_episode_start_mask)

            buffer.add(
                obs=_obs(0.0),
                actions=_actions(0.0),
                rewards=torch.tensor([0.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([False]),
                next_obs=_obs(1.0),
            )

            self.assertIsNone(buffer.get_all().episode_start_mask)
        finally:
            env.close()

    def test_sample_without_replacement_rejects_oversized_batches(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            buffer.add(
                obs=_obs(0.0),
                actions=_actions(0.0),
                rewards=torch.tensor([0.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([False]),
                next_obs=_obs(1.0),
            )

            with self.assertRaisesRegex(ValueError, "without replacement"):
                buffer.sample(2, replacement=False)
        finally:
            env.close()

    def test_replacement_sample_keeps_wrapped_ring_fields_aligned(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            for step in range(5):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step + 10)),
                    rewards=torch.tensor([float(step + 20)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                )

            with patch("torch.randint", return_value=torch.tensor([0, 2])):
                batch = buffer.sample(2, replacement=True)

            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [2.0, 4.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [3.0, 5.0])
            self.assertEqual(batch.actions[:, 0, 0].tolist(), [12.0, 14.0])
            self.assertEqual(batch.rewards.tolist(), [22.0, 24.0])
        finally:
            env.close()

    def test_replacement_sample_replaces_duplicate_done_next_obs_with_terminal_obs(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            buffer.add(
                obs=_obs(0.0),
                actions=_actions(10.0),
                rewards=torch.tensor([1.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([True]),
                next_obs=_obs(100.0),
                terminal_obs=_obs(1.0),
            )

            with patch("torch.randint", return_value=torch.tensor([0, 0])):
                batch = buffer.sample(2, replacement=True)

            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [0.0, 0.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [1.0, 1.0])
            self.assertEqual(batch.truncations.tolist(), [True, True])
        finally:
            env.close()

    def test_done_heavy_observation_stream_survives_wraparound(self) -> None:
        env = _make_env(done_steps=(1,))
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=5,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [400.0, 500.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [401.0, 501.0])
            self.assertEqual(batch.truncations.tolist(), [True, True])
        finally:
            env.close()

    def test_direct_done_add_uses_streamed_obs_slots_after_first_insert(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            for step in range(3):
                buffer.add(
                    obs=_obs(float(step * 100)),
                    actions=_actions(float(step)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([True]),
                    next_obs=_obs(float((step + 1) * 100)),
                    terminal_obs=_obs(float(step * 100 + 1)),
                )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [100.0, 200.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [101.0, 201.0])
            self.assertEqual(batch.truncations.tolist(), [True, True])
        finally:
            env.close()

    def test_direct_done_add_accepts_full_vector_terminal_obs(self) -> None:
        env = _make_multi_env((), ())
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            buffer.add(
                obs=_multi_obs((0.0, 100.0)),
                actions=torch.full((2, 2, 2), 1.0),
                rewards=torch.tensor([0.0, 1.0]),
                terminations=torch.tensor([False, False]),
                truncations=torch.tensor([False, True]),
                next_obs=_multi_obs((10.0, 200.0)),
                terminal_obs=_multi_obs((999.0, 1001.0)),
            )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [0.0, 100.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [10.0, 1001.0])
            self.assertEqual(batch.truncations.tolist(), [False, True])
        finally:
            env.close()

    def test_direct_done_add_accepts_packed_terminal_obs_for_sparse_lanes(self) -> None:
        env = _make_multi_env((), (), ())
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            buffer.add(
                obs=_multi_obs((0.0, 100.0, 200.0)),
                actions=torch.full((3, 2, 2), 1.0),
                rewards=torch.tensor([0.0, 1.0, 2.0]),
                terminations=torch.tensor([True, False, False]),
                truncations=torch.tensor([False, False, True]),
                next_obs=_multi_obs((10.0, 110.0, 210.0)),
                terminal_obs=_multi_obs((1.0, 201.0)),
            )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [0.0, 100.0, 200.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [1.0, 110.0, 201.0])
            self.assertEqual(batch.terminations.tolist(), [True, False, False])
            self.assertEqual(batch.truncations.tolist(), [False, False, True])
        finally:
            env.close()

    def test_direct_done_add_rejects_terminal_obs_with_wrong_leading_dimension(self) -> None:
        env = _make_multi_env((), (), ())
        try:
            buffer = _make_buffer(env, capacity_per_env=2)

            with self.assertRaisesRegex(ValueError, "leading dimension"):
                buffer.add(
                    obs=_multi_obs((0.0, 100.0, 200.0)),
                    actions=torch.full((3, 2, 2), 1.0),
                    rewards=torch.tensor([0.0, 1.0, 2.0]),
                    terminations=torch.tensor([False, False, False]),
                    truncations=torch.tensor([False, True, False]),
                    next_obs=_multi_obs((10.0, 110.0, 210.0)),
                    terminal_obs=_multi_obs((999.0, 1001.0)),
                )
        finally:
            env.close()

    def test_terminal_obs_for_reused_slot_does_not_leak_to_new_transition(self) -> None:
        env = _make_multi_env((), ())
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            buffer.add(
                obs=_multi_obs((0.0, 100.0)),
                actions=torch.full((2, 2, 2), 0.0),
                rewards=torch.tensor([0.0, 0.0]),
                terminations=torch.tensor([False, False]),
                truncations=torch.tensor([True, False]),
                next_obs=_multi_obs((10.0, 110.0)),
                terminal_obs=_multi_obs((1.0,)),
            )
            buffer.add(
                obs=_multi_obs((999.0, 999.0)),
                actions=torch.full((2, 2, 2), 1.0),
                rewards=torch.tensor([1.0, 1.0]),
                terminations=torch.tensor([False, False]),
                truncations=torch.tensor([False, False]),
                next_obs=_multi_obs((20.0, 120.0)),
            )
            buffer.add(
                obs=_multi_obs((999.0, 999.0)),
                actions=torch.full((2, 2, 2), 2.0),
                rewards=torch.tensor([2.0, 2.0]),
                terminations=torch.tensor([False, False]),
                truncations=torch.tensor([False, True]),
                next_obs=_multi_obs((30.0, 130.0)),
                terminal_obs=_multi_obs((131.0,)),
            )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [10.0, 20.0, 110.0, 120.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [20.0, 30.0, 120.0, 131.0])
            self.assertEqual(batch.truncations.tolist(), [False, False, False, True])
            self.assertEqual(set(buffer._terminal_obs_by_env_slot.keys()), {(1, 0)})
        finally:
            env.close()

    def test_copy_current_obs_true_after_stream_start_is_rejected(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            buffer.add(
                obs=_obs(0.0),
                actions=_actions(0.0),
                rewards=torch.tensor([0.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([False]),
                next_obs=_obs(1.0),
            )

            with self.assertRaisesRegex(ValueError, "copy_current_obs=True"):
                buffer.add(
                    obs=_obs(1.0),
                    actions=_actions(1.0),
                    rewards=torch.tensor([1.0]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(2.0),
                    copy_current_obs=True,
                )
        finally:
            env.close()

    def test_collect_uses_final_obs_for_done_transition_and_reset_obs_afterward(self) -> None:
        env = _make_env(done_steps=(2,))
        try:
            buffer = _make_buffer(env, capacity_per_env=4, store_previous_actions=True)
            with patch(
                    "swarmbots.learn.algos.off_policy.rollout.sample_random_actions",
                    side_effect=[_actions(1.0), _actions(2.0), _actions(3.0)],
            ):
                episode_infos, metrics, rollout_state = collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=3,
                    random_actions=True,
                )

            batch = buffer.get_all()
            self.assertEqual(len(buffer), 3)
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [100.0, 101.0, 200.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [101.0, 102.0, 201.0])
            self.assertEqual(batch.truncations.tolist(), [False, True, False])
            self.assertEqual(batch.previous_actions[:, 0, 0].tolist(), [0.0, 1.0, 0.0])
            self.assertEqual(batch.actions[:, 0, 0].tolist(), [1.0, 2.0, 3.0])
            self.assertNoNextPreviousActions(batch)
            self.assertEqual(len(episode_infos), 1)
            self.assertEqual(metrics["transitions_collected"], 3)
            self.assertTrue(torch.equal(rollout_state.episode_start_mask.cpu(), torch.tensor([False])))
            self.assertEqual(rollout_state.previous_actions[:, 0, 0].tolist(), [3.0])
        finally:
            env.close()

    def test_collect_stores_termination_flag_and_terminal_next_obs(self) -> None:
        env = _make_env(terminate_steps=(2,))
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=3,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [100.0, 101.0, 200.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [101.0, 102.0, 201.0])
            self.assertEqual(batch.terminations.tolist(), [False, True, False])
            self.assertEqual(batch.truncations.tolist(), [False, False, False])
            self.assertEqual(batch.episode_ends.tolist(), [False, True, False])
            self.assertEqual(batch.terminal_mask.tolist(), [False, True, False])
        finally:
            env.close()

    def test_collect_resumes_with_rollout_state_in_same_buffer(self) -> None:
        env = _make_env(done_steps=(3,))
        try:
            buffer = _make_buffer(env, capacity_per_env=8)
            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=2,
                random_actions=True,
            )
            episode_infos, metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=2,
                rollout_state=rollout_state,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [100.0, 101.0, 102.0, 200.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [101.0, 102.0, 103.0, 201.0])
            self.assertEqual(batch.truncations.tolist(), [False, False, True, False])
            self.assertEqual(len(episode_infos), 1)
            self.assertEqual(metrics["transitions_collected"], 2)
            self.assertEqual(rollout_state.rollout_step_idx, 4)
            self.assertEqual(float(rollout_state.obs["local_obs"][0, 0, 0].item()), 201.0)
        finally:
            env.close()

    def test_collect_only_snapshots_current_obs_when_buffer_needs_it(self) -> None:
        env = _make_env(done_steps=(4,))
        try:
            buffer = _make_buffer(env, capacity_per_env=8)
            with patch(
                    "swarmbots.learn.algos.off_policy.rollout.snapshot_obs",
                    wraps=off_policy_rollout.snapshot_obs,
            ) as snapshot_mock:
                _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=3,
                    random_actions=True,
                )

            self.assertEqual(snapshot_mock.call_count, 1)

            with patch(
                    "swarmbots.learn.algos.off_policy.rollout.snapshot_obs",
                    wraps=off_policy_rollout.snapshot_obs,
            ) as snapshot_mock:
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=2,
                    rollout_state=rollout_state,
                    random_actions=True,
                )

            self.assertEqual(snapshot_mock.call_count, 0)
        finally:
            env.close()

    def test_collect_applies_explicit_rollout_device_to_env(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            with patch.object(env, "set_device", wraps=env.set_device) as set_device_mock:
                _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=1,
                    random_actions=True,
                    rollout_device="cpu",
                )

            set_device_mock.assert_called_once_with(torch.device("cpu"))
            self.assertEqual(rollout_state.obs["local_obs"].device, torch.device("cpu"))
        finally:
            env.close()

    def test_collect_preserves_agent_mask_for_terminal_and_reset_obs(self) -> None:
        env = _make_agent_mask_env(done_steps=(2,))
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=3,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(
                batch.agent_mask.tolist(),
                [[True, False], [True, True], [True, True]],
            )
            self.assertEqual(
                batch.next_agent_mask.tolist(),
                [[True, True], [True, False], [True, False]],
            )
        finally:
            env.close()

    def test_policy_rollout_stores_previous_action_history(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4, store_previous_actions=True)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=3,
                policy=_PreviousActionPolicy(),
            )

            batch = buffer.get_all()
            self.assertEqual(batch.previous_actions[:, 0, 0].tolist(), [0.0, 1.0, 2.0])
            self.assertEqual(batch.actions[:, 0, 0].tolist(), [1.0, 2.0, 3.0])
            self.assertNoNextPreviousActions(batch)
        finally:
            env.close()

    def test_policy_rollout_uses_temporal_state_api(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            policy = _TemporalPolicy()
            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=3,
                policy=policy,
            )

            batch = buffer.get_all()
            self.assertEqual(batch.actions[:, 0, 0].tolist(), [1.0, 2.0, 3.0])
            self.assertEqual(
                [mask.tolist() for mask in policy.episode_start_masks],
                [[True], [False], [False]],
            )
            self.assertIsInstance(rollout_state.temporal_state, torch.Tensor)
            self.assertEqual(rollout_state.temporal_state[:, 0, 0].tolist(), [3.0])
        finally:
            env.close()

    def test_temporal_state_resets_on_episode_start_mask(self) -> None:
        env = _make_env(done_steps=(2,))
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            policy = _TemporalPolicy()
            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=3,
                policy=policy,
            )

            batch = buffer.get_all()
            self.assertEqual(batch.actions[:, 0, 0].tolist(), [1.0, 2.0, 1.0])
            self.assertEqual(
                [mask.tolist() for mask in policy.episode_start_masks],
                [[True], [False], [True]],
            )
            self.assertIsInstance(rollout_state.temporal_state, torch.Tensor)
            self.assertEqual(rollout_state.temporal_state[:, 0, 0].tolist(), [1.0])
        finally:
            env.close()

    def test_temporal_state_initializes_when_policy_starts_after_random_rollout(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=1,
                random_actions=True,
            )
            policy = _TemporalPolicy()
            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=2,
                policy=policy,
                rollout_state=rollout_state,
            )

            self.assertEqual(
                [mask.tolist() for mask in policy.episode_start_masks],
                [[False], [False]],
            )
            self.assertIsInstance(rollout_state.temporal_state, torch.Tensor)
            self.assertEqual(rollout_state.temporal_state[:, 0, 0].tolist(), [2.0])
        finally:
            env.close()

    def test_episode_segment_sampling_uses_temporal_checkpoints_and_burn_in_mask(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=6, temporal_state_store_interval=2)
            for step in range(6):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step + 10)),
                    rewards=torch.tensor([float(step + 20)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                    temporal_state=_temporal_state(float(step)),
                    next_temporal_state=_temporal_state(float(step + 1)),
                )

            with patch("torch.randint", return_value=torch.tensor([1])):
                batch = buffer.sample_episode_segments(
                    1,
                    segment_length=2,
                    burn_in_steps=1,
                )

            self.assertEqual(batch.local_obs[0, :, 0, 0].tolist(), [2.0, 3.0, 4.0])
            self.assertEqual(batch.actions[0, :, 0, 0].tolist(), [12.0, 13.0, 14.0])
            self.assertEqual(batch.rewards[0].tolist(), [22.0, 23.0, 24.0])
            self.assertEqual(batch.train_mask.tolist(), [[False, True, True]])
            self.assertEqual(batch.burn_in_mask.tolist(), [[True, False, False]])
            self.assertEqual(batch.episode_start_mask.tolist(), [[False, False, False]])
            self.assertEqual(batch.burn_in_steps, 1)
            self.assertEqual(batch.segment_length, 2)
            self.assertIsInstance(batch.initial_temporal_state, torch.Tensor)
            self.assertEqual(batch.initial_temporal_state[:, 0, 0].tolist(), [2.0])
        finally:
            env.close()

    def test_episode_segment_sampling_rejects_cross_episode_windows(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4, temporal_state_store_interval=1)
            truncations = [False, True, False, False]
            for step, truncated in enumerate(truncations):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([truncated]),
                    next_obs=_obs(float(step + 1)),
                    terminal_obs=_obs(float(step + 1)) if truncated else None,
                    temporal_state=_temporal_state(float(step)),
                    next_temporal_state=_temporal_state(float(step + 1)),
                )

            batch = buffer.sample_episode_segments(
                2,
                segment_length=2,
                replacement=False,
            )

            segments_by_start = {
                float(batch.local_obs[batch_idx, 0, 0, 0].item()): batch.episode_ends[batch_idx].tolist()
                for batch_idx in range(2)
            }
            self.assertEqual(segments_by_start, {0.0: [False, True], 2.0: [False, False]})
        finally:
            env.close()

    def test_episode_segment_sampling_can_skip_initial_temporal_state_requirement(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            for step in range(3):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                )

            with patch("torch.randint", return_value=torch.tensor([1])):
                batch = buffer.sample_episode_segments(
                    1,
                    segment_length=2,
                    require_initial_temporal_state=False,
                )

            self.assertEqual(batch.local_obs[0, :, 0, 0].tolist(), [1.0, 2.0])
            self.assertEqual(batch.actions[0, :, 0, 0].tolist(), [1.0, 2.0])
            self.assertEqual(batch.train_mask.tolist(), [[True, True]])
            self.assertIsNone(batch.episode_start_mask)
            self.assertIsNone(batch.initial_temporal_state)
        finally:
            env.close()

    def test_episode_segment_sampling_clears_stale_temporal_checkpoint_on_obs_wraparound(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3, temporal_state_store_interval=10)
            for step in range(5):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                    temporal_state=_temporal_state(float(step)),
                    next_temporal_state=_temporal_state(float(step + 1)),
                )

            with self.assertRaisesRegex(ValueError, "no contiguous replay windows"):
                buffer.sample_episode_segments(1, segment_length=1)
        finally:
            env.close()

    def test_replay_episode_start_mask_marks_initial_and_post_done_transitions(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4, temporal_state_store_interval=1)
            truncations = [False, True, False]
            for step, truncated in enumerate(truncations):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([truncated]),
                    next_obs=_obs(float(step + 1)),
                    terminal_obs=_obs(float(step + 1)) if truncated else None,
                )

            batch = buffer.get_all()
            self.assertEqual(batch.episode_start_mask.tolist(), [True, False, True])
            self.assertEqual(batch.truncations.tolist(), truncations)
        finally:
            env.close()

    def test_policy_rollout_stores_temporal_state_checkpoints_for_segments(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=5, temporal_state_store_interval=2)
            policy = _TemporalPolicy()
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=5,
                policy=policy,
            )

            with patch("torch.randint", return_value=torch.tensor([1])):
                batch = buffer.sample_episode_segments(
                    1,
                    segment_length=2,
                    burn_in_steps=1,
                )

            self.assertEqual(batch.local_obs[0, :, 0, 0].tolist(), [102.0, 103.0, 104.0])
            self.assertIsInstance(batch.initial_temporal_state, torch.Tensor)
            self.assertEqual(batch.initial_temporal_state[:, 0, 0].tolist(), [2.0])
            self.assertEqual(batch.actions[0, :, 0, 0].tolist(), [3.0, 4.0, 5.0])
            self.assertEqual(batch.train_mask.tolist(), [[False, True, True]])
        finally:
            env.close()

    def test_episode_segment_sampling_requires_temporal_state_storage_by_default(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            buffer.add(
                obs=_obs(0.0),
                actions=_actions(0.0),
                rewards=torch.tensor([0.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([False]),
                next_obs=_obs(1.0),
            )

            with self.assertRaisesRegex(ValueError, "temporal_state_store_interval"):
                buffer.sample_episode_segments(1, segment_length=1)
        finally:
            env.close()

    def test_gsde_interval_resets_are_applied_before_policy_forward(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            policy = _GSDEPolicy()
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=3,
                policy=policy,
                gsde_reset_mode=GSDEIntervalResetMode(interval=2),
            )

            self.assertEqual(
                [mask.tolist() for mask in policy.action_dist.episode_start_masks],
                [[True], [False], [False]],
            )
            self.assertEqual(len(policy.action_dist.step_resets), 2)
            self.assertEqual(policy.action_dist.step_resets[0], (None, (1, 2)))
            self.assertEqual(policy.action_dist.step_resets[1], (None, (1, 2)))
        finally:
            env.close()

    def test_gsde_interval_noise_initializes_after_random_warmup(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=1,
                random_actions=True,
            )
            self.assertFalse(rollout_state.gsde_noise_initialized)

            policy = _GSDEPolicy()
            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=1,
                policy=policy,
                rollout_state=rollout_state,
                gsde_reset_mode=GSDEIntervalResetMode(interval=3),
            )

            self.assertTrue(rollout_state.gsde_noise_initialized)
            self.assertEqual(policy.action_dist.step_resets, [(None, (1, 2))])

            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=1,
                policy=policy,
                rollout_state=rollout_state,
                gsde_reset_mode=GSDEIntervalResetMode(interval=3),
            )

            self.assertEqual(policy.action_dist.step_resets, [(None, (1, 2))])
        finally:
            env.close()

    def test_gsde_policy_requires_reset_mode(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            with self.assertRaisesRegex(RuntimeError, "gsde_reset_mode"):
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=1,
                    policy=_GSDEPolicy(),
                )
        finally:
            env.close()

    def test_sampled_actions_are_available_for_next_previous_action_conditioning(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3, store_previous_actions=True)
            for step in range(3):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step + 10)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                    previous_actions=_actions(float(step + 20)),
                )

            with patch("torch.randint", return_value=torch.tensor([2, 0])):
                batch = buffer.sample(2, replacement=True)

            self.assertEqual(batch.actions[:, 0, 0].tolist(), [12.0, 10.0])
            self.assertEqual(batch.previous_actions[:, 0, 0].tolist(), [22.0, 20.0])
            self.assertNoNextPreviousActions(batch)
        finally:
            env.close()

    def test_direct_add_defaults_missing_previous_actions_to_zero(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3, store_previous_actions=True)
            buffer.add(
                obs=_obs(0.0),
                actions=_actions(7.0),
                rewards=torch.tensor([1.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([False]),
                next_obs=_obs(1.0),
            )

            batch = buffer.get_all()
            self.assertEqual(batch.previous_actions[:, 0, 0].tolist(), [0.0])
            self.assertEqual(batch.actions[:, 0, 0].tolist(), [7.0])
            self.assertNoNextPreviousActions(batch)
        finally:
            env.close()

    def test_batch_does_not_expose_next_previous_actions(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3, store_previous_actions=True)
            buffer.add(
                obs=_obs(0.0),
                actions=_actions(7.0),
                rewards=torch.tensor([1.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([False]),
                next_obs=_obs(1.0),
                previous_actions=_actions(3.0),
            )

            self.assertNoNextPreviousActions(buffer.get_all())
            self.assertNoNextPreviousActions(buffer.sample(1))
        finally:
            env.close()

    def test_policy_requiring_previous_actions_requires_replay_storage(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            with self.assertRaisesRegex(ValueError, "store_previous_actions=True"):
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=2,
                    policy=_PreviousActionPolicy(),
                )
        finally:
            env.close()

    def test_collect_rejects_random_actions_with_policy(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            with self.assertRaisesRegex(ValueError, "either a policy or random_actions"):
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=2,
                    policy=_NoPreviousActionPolicy(),
                    random_actions=True,
                )
        finally:
            env.close()

    def test_policy_without_previous_actions_does_not_receive_tracked_history(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4, store_previous_actions=True)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=2,
                policy=_NoPreviousActionPolicy(),
            )

            batch = buffer.get_all()
            self.assertEqual(batch.previous_actions[:, 0, 0].tolist(), [0.0, 0.0])
            self.assertEqual(batch.actions[:, 0, 0].tolist(), [0.0, 0.0])
            self.assertNoNextPreviousActions(batch)
        finally:
            env.close()

    def test_previous_action_storage_is_optional(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=2,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertIsNone(batch.previous_actions)
            self.assertNoNextPreviousActions(batch)
        finally:
            env.close()

    def test_collect_keeps_terminal_obs_per_done_lane(self) -> None:
        env = _make_multi_env((), (2,))
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=6,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(batch.local_obs[:, 0, 0].tolist(), [100.0, 101.0, 102.0, 1100.0, 1101.0, 1200.0])
            self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), [101.0, 102.0, 103.0, 1101.0, 1102.0, 1201.0])
            self.assertEqual(batch.truncations.tolist(), [False, False, False, False, True, False])
        finally:
            env.close()

    def test_multi_env_asymmetric_done_lanes_survive_wraparound(self) -> None:
        env = _make_multi_env((), (1,))
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=10,
                random_actions=True,
            )

            batch = buffer.get_all()
            self.assertEqual(
                batch.local_obs[:, 0, 0].tolist(),
                [102.0, 103.0, 104.0, 1300.0, 1400.0, 1500.0],
            )
            self.assertEqual(
                batch.next_local_obs[:, 0, 0].tolist(),
                [103.0, 104.0, 105.0, 1301.0, 1401.0, 1501.0],
            )
            self.assertEqual(batch.truncations.tolist(), [False, False, False, True, True, True])
        finally:
            env.close()

    def test_collect_rejects_non_multiple_transition_count(self) -> None:
        env = _make_multi_env((), ())
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            with self.assertRaisesRegex(ValueError, "multiple"):
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=3,
                    random_actions=True,
                )
        finally:
            env.close()

    def test_collect_requires_rollout_state_for_non_empty_buffer(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=1,
                random_actions=True,
            )

            with self.assertRaisesRegex(ValueError, "rollout_state"):
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=1,
                    random_actions=True,
                )
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
