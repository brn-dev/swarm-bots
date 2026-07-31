import unittest
from dataclasses import fields
from typing import Any
from unittest.mock import patch

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.algos.off_policy import (
    NoEpisodeSegmentCandidatesError,
    OffPolicyReplayBuffer,
    collect_off_policy_steps,
    off_policy_rollout,
    replay_buffer_tensor_ops,
    warmup_off_policy_steps,
)
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import (
    SwarmBotsLearnEnvWrapper,
)
from swarmbots.learn.gsde_reset import GSDEIntervalResetMode, GSDEProbabilityResetMode


class _ScriptedOffPolicyEnv(gymnasium.Env):
    metadata = {}

    def __init__(
            self,
            *,
            done_steps: tuple[int, ...] = (),
            terminate_steps: tuple[int, ...] = (),
            env_id: int = 0,
            include_agent_mask: bool = False,
            include_scenario_id: bool = False,
    ) -> None:
        super().__init__()
        self.done_steps = done_steps
        self.terminate_steps = terminate_steps
        self.env_id = env_id
        self.include_agent_mask = include_agent_mask
        self.include_scenario_id = include_scenario_id
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
        if include_scenario_id:
            observation_spaces["scenario_id"] = spaces.Discrete(3)
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
        if self.include_scenario_id:
            obs["scenario_id"] = np.int64(self.episode_id % 3)
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


def _make_scenario_env(*, done_steps: tuple[int, ...] = ()) -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [
            lambda: _ScriptedOffPolicyEnv(
                done_steps=done_steps,
                include_scenario_id=True,
            )
        ],
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
        temporal_state_storage_dtype: torch.dtype | None = None,
        storage_device: str = "cpu",
        train_device: str = "cpu",
        compile_tensor_operations: bool | None = None,
) -> OffPolicyReplayBuffer:
    return OffPolicyReplayBuffer(
        capacity_per_env=capacity_per_env,
        observation_space=env.observation_space,
        action_space=env.action_space,
        store_previous_actions=store_previous_actions,
        temporal_state_store_interval=temporal_state_store_interval,
        temporal_state_storage_dtype=temporal_state_storage_dtype,
        storage_device=storage_device,
        train_device=train_device,
        compile_tensor_operations=compile_tensor_operations,
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


def _add_direct_step(
        buffer: OffPolicyReplayBuffer,
        *,
        obs_values: tuple[float, ...],
        next_obs_values: tuple[float, ...],
        action_value: float,
        terminations: tuple[bool, ...] | None = None,
        truncations: tuple[bool, ...] | None = None,
        terminal_obs_values: tuple[float, ...] | None = None,
        previous_action_value: float | None = None,
        episode_start_mask: tuple[bool, ...] | None = None,
        temporal_state: Any = None,
        next_temporal_state: Any = None,
) -> None:
    n_envs = len(obs_values)
    if len(next_obs_values) != n_envs:
        raise ValueError("obs_values and next_obs_values must have equal lengths")
    terminations = (False,) * n_envs if terminations is None else terminations
    truncations = (False,) * n_envs if truncations is None else truncations
    has_done = any(terminations) or any(truncations)
    if has_done and terminal_obs_values is None:
        raise ValueError("terminal_obs_values is required for done transitions")

    make_obs = _obs if n_envs == 1 else _multi_obs
    obs_argument = make_obs(obs_values[0]) if n_envs == 1 else make_obs(obs_values)
    next_obs_argument = make_obs(next_obs_values[0]) if n_envs == 1 else make_obs(next_obs_values)
    terminal_obs = None
    if terminal_obs_values is not None:
        terminal_obs = (
            make_obs(terminal_obs_values[0])
            if n_envs == 1
            else make_obs(terminal_obs_values)
        )
    previous_actions = None
    if previous_action_value is not None:
        previous_actions = torch.full((n_envs, 2, 2), previous_action_value)

    buffer.add(
        obs=obs_argument,
        actions=torch.full((n_envs, 2, 2), action_value),
        rewards=torch.full((n_envs,), action_value + 0.25),
        terminations=torch.tensor(terminations, dtype=torch.bool),
        truncations=torch.tensor(truncations, dtype=torch.bool),
        next_obs=next_obs_argument,
        terminal_obs=terminal_obs,
        previous_actions=previous_actions,
        episode_start_mask=(
            None if episode_start_mask is None else torch.tensor(episode_start_mask, dtype=torch.bool)
        ),
        temporal_state=temporal_state,
        next_temporal_state=next_temporal_state,
    )


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


class _ObsEncodingPolicy(BasePolicy):
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
        _ = previous_actions
        _ = deterministic
        action_values = local_obs[..., :1] + 0.25
        return action_values.expand(*local_obs.shape[:2], 2).clone()

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unknown loss weights: {sorted(weights)}")

    def requires_previous_actions(self) -> bool:
        return False


class _ScenarioEncodingPolicy(_NoPreviousActionPolicy):
    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        _ = (
            global_obs,
            hidden_local_vars,
            hidden_global_vars,
            agent_mask,
            previous_actions,
            deterministic,
        )
        if scenario_ids is None:
            raise AssertionError("scenario_ids must be passed for scenario observations")
        return scenario_ids.to(dtype=local_obs.dtype).view(-1, 1, 1).expand(
            -1,
            local_obs.shape[1],
            2,
        )


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
        self.call_order: list[str] = []
        self.noise_state: torch.Tensor | None = None

    def get_temporal_correlation_state(self) -> torch.Tensor | None:
        return self.noise_state

    def set_temporal_correlation_state(self, state: torch.Tensor | None) -> None:
        self.noise_state = state

    def reset_temporal_correlations_on_ep_start(self, mask: torch.Tensor) -> None:
        self.call_order.append("ep_start")
        self.episode_start_masks.append(mask.detach().cpu().clone())

    def reset_temporal_correlations_on_step(
            self,
            mask: torch.Tensor | None = None,
            batch_shape: tuple[int, ...] | None = None,
    ) -> None:
        self.call_order.append("step")
        self.step_resets.append((None if mask is None else mask.detach().cpu().clone(), batch_shape))
        if batch_shape is not None:
            self.noise_state = torch.zeros((*batch_shape, 1))
        elif self.noise_state is None and mask is not None:
            self.noise_state = torch.zeros((*mask.shape, 1))


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

    def assertFlatTransitionScalars(
            self,
            batch: Any,
            *,
            local_obs: list[float],
            next_local_obs: list[float],
            actions: list[float],
            rewards: list[float],
            terminations: list[bool] | None = None,
            truncations: list[bool] | None = None,
            previous_actions: list[float] | None = None,
            episode_start_mask: list[bool] | None = None,
    ) -> None:
        self.assertEqual(batch.local_obs[:, 0, 0].tolist(), local_obs)
        self.assertEqual(batch.local_obs[:, 1, 2].tolist(), local_obs)
        self.assertEqual(batch.global_obs[:, 1].tolist(), [value + 0.5 for value in local_obs])
        self.assertEqual(batch.hidden_local_vars[:, 1, 0].tolist(), [value + 1.0 for value in local_obs])
        self.assertEqual(batch.hidden_global_vars[:, 0].tolist(), [value + 2.0 for value in local_obs])
        self.assertEqual(batch.next_local_obs[:, 0, 0].tolist(), next_local_obs)
        self.assertEqual(batch.next_local_obs[:, 1, 2].tolist(), next_local_obs)
        self.assertEqual(batch.next_global_obs[:, 1].tolist(), [value + 0.5 for value in next_local_obs])
        self.assertEqual(batch.next_hidden_local_vars[:, 1, 0].tolist(), [value + 1.0 for value in next_local_obs])
        self.assertEqual(batch.next_hidden_global_vars[:, 0].tolist(), [value + 2.0 for value in next_local_obs])
        self.assertEqual(batch.actions[:, 0, 0].tolist(), actions)
        self.assertEqual(batch.actions[:, 1, 1].tolist(), actions)
        self.assertEqual(batch.rewards.tolist(), rewards)

        if terminations is None:
            terminations = [False] * len(local_obs)
        if truncations is None:
            truncations = [False] * len(local_obs)
        self.assertEqual(batch.terminations.tolist(), terminations)
        self.assertEqual(batch.truncations.tolist(), truncations)

        if previous_actions is not None:
            self.assertIsNotNone(batch.previous_actions)
            assert batch.previous_actions is not None
            self.assertEqual(batch.previous_actions[:, 0, 0].tolist(), previous_actions)
            self.assertEqual(batch.previous_actions[:, 1, 1].tolist(), previous_actions)

        if episode_start_mask is not None:
            self.assertIsNotNone(batch.episode_start_mask)
            assert batch.episode_start_mask is not None
            self.assertEqual(batch.episode_start_mask.tolist(), episode_start_mask)

    def assertSegmentTransitionScalars(
            self,
            batch: Any,
            batch_idx: int,
            *,
            local_obs: list[float],
            next_local_obs: list[float],
            actions: list[float],
            rewards: list[float],
            terminations: list[bool] | None = None,
            truncations: list[bool] | None = None,
    ) -> None:
        self.assertEqual(batch.local_obs[batch_idx, :, 0, 0].tolist(), local_obs)
        self.assertEqual(batch.local_obs[batch_idx, :, 1, 2].tolist(), local_obs)
        self.assertEqual(batch.global_obs[batch_idx, :, 1].tolist(), [value + 0.5 for value in local_obs])
        self.assertEqual(batch.hidden_local_vars[batch_idx, :, 1, 0].tolist(), [value + 1.0 for value in local_obs])
        self.assertEqual(batch.hidden_global_vars[batch_idx, :, 0].tolist(), [value + 2.0 for value in local_obs])
        self.assertEqual(batch.next_local_obs[batch_idx, :, 0, 0].tolist(), next_local_obs)
        self.assertEqual(batch.next_local_obs[batch_idx, :, 1, 2].tolist(), next_local_obs)
        self.assertEqual(batch.next_global_obs[batch_idx, :, 1].tolist(), [value + 0.5 for value in next_local_obs])
        self.assertEqual(
            batch.next_hidden_local_vars[batch_idx, :, 1, 0].tolist(),
            [value + 1.0 for value in next_local_obs],
        )
        self.assertEqual(
            batch.next_hidden_global_vars[batch_idx, :, 0].tolist(),
            [value + 2.0 for value in next_local_obs],
        )
        self.assertEqual(batch.actions[batch_idx, :, 0, 0].tolist(), actions)
        self.assertEqual(batch.actions[batch_idx, :, 1, 1].tolist(), actions)
        self.assertEqual(batch.rewards[batch_idx].tolist(), rewards)

        if terminations is None:
            terminations = [False] * len(local_obs)
        if truncations is None:
            truncations = [False] * len(local_obs)
        self.assertEqual(batch.terminations[batch_idx].tolist(), terminations)
        self.assertEqual(batch.truncations[batch_idx].tolist(), truncations)

    def test_constructor_rejects_invalid_capacity_observation_space_and_checkpoint_interval(self) -> None:
        env = _make_env()
        try:
            for capacity_per_env in (0, -1):
                with self.subTest(capacity_per_env=capacity_per_env):
                    with self.assertRaisesRegex(ValueError, "capacity_per_env must be > 0"):
                        _make_buffer(env, capacity_per_env=capacity_per_env)

            with self.assertRaisesRegex(ValueError, "observation_space must be"):
                OffPolicyReplayBuffer(
                    capacity_per_env=2,
                    observation_space=spaces.Box(low=-1.0, high=1.0, shape=(1,)),
                    action_space=env.action_space,
                    storage_device="cpu",
                    train_device="cpu",
                )

            for store_interval in (0, -2):
                with self.subTest(store_interval=store_interval):
                    with self.assertRaisesRegex(ValueError, "temporal_state_store_interval must be > 0"):
                        _make_buffer(
                            env,
                            capacity_per_env=2,
                            temporal_state_store_interval=store_interval,
                        )
        finally:
            env.close()

    def test_sampling_apis_reject_empty_buffers_and_invalid_sizes(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            with self.assertRaisesRegex(ValueError, "empty replay buffer"):
                buffer.sample(1)
            with self.assertRaisesRegex(ValueError, "empty replay buffer"):
                buffer.get_all()
            with self.assertRaisesRegex(ValueError, "num_next_steps must be > 0"):
                buffer.sample_episode_windows(1, num_next_steps=0)
            with self.assertRaisesRegex(ValueError, "batch_size must be > 0"):
                buffer.sample_episode_segments(0, segment_length=1)
            with self.assertRaisesRegex(ValueError, "segment_length must be > 0"):
                buffer.sample_episode_segments(1, segment_length=0)
            with self.assertRaisesRegex(ValueError, "burn_in_steps must be >= 0"):
                buffer.sample_episode_segments(1, segment_length=1, burn_in_steps=-1)

            _add_direct_step(
                buffer,
                obs_values=(0.0,),
                next_obs_values=(1.0,),
                action_value=0.0,
            )
            with self.assertRaisesRegex(ValueError, "batch_size must be > 0"):
                buffer.sample(0)
        finally:
            env.close()

    def test_cpu_replay_defaults_to_eager_tensor_operations(self) -> None:
        env = _make_env()
        try:
            with patch.object(replay_buffer_tensor_ops.torch, "compile") as compile_mock:
                buffer = _make_buffer(env, capacity_per_env=2)

            self.assertFalse(buffer.compile_tensor_operations)
            compile_mock.assert_not_called()
        finally:
            env.close()

    def test_compiled_tensor_operations_cover_add_fetch_windows_and_temporal_slots(self) -> None:
        env = _make_env()
        try:
            with patch.object(
                    replay_buffer_tensor_ops.torch,
                    "compile",
                    side_effect=lambda function, **_kwargs: function,
            ) as compile_mock:
                buffer = _make_buffer(
                    env,
                    capacity_per_env=3,
                    temporal_state_store_interval=1,
                    compile_tensor_operations=True,
                )
                _add_direct_step(
                    buffer,
                    obs_values=(0.0,),
                    next_obs_values=(1.0,),
                    action_value=0.0,
                    temporal_state=_temporal_state(10.0),
                    next_temporal_state=_temporal_state(11.0),
                )
                _add_direct_step(
                    buffer,
                    obs_values=(1.0,),
                    next_obs_values=(2.0,),
                    action_value=1.0,
                    temporal_state=_temporal_state(11.0),
                    next_temporal_state=_temporal_state(12.0),
                )
                flat_batch = buffer.get_all()
                window_batch = buffer.sample_episode_windows(
                    1,
                    num_next_steps=2,
                    generator=torch.Generator().manual_seed(0),
                )
                segment_batch = buffer.sample_episode_segments(
                    1,
                    segment_length=1,
                    generator=torch.Generator().manual_seed(0),
                )

            self.assertTrue(buffer.compile_tensor_operations)
            self.assertEqual(compile_mock.call_count, 6)
            self.assertEqual(
                {compile_call.args[0].__name__ for compile_call in compile_mock.call_args_list},
                {
                    "_gather_replay_storage",
                    "_episode_window_indices",
                    "_episode_segment_indices",
                    "_episode_segment_candidate_mask",
                    "_release_temporal_state_slots",
                    "_allocate_temporal_state_slots",
                },
            )
            compile_kwargs_by_operation = {
                compile_call.args[0].__name__: compile_call.kwargs
                for compile_call in compile_mock.call_args_list
            }
            self.assertEqual(compile_kwargs_by_operation, {
                "_gather_replay_storage": {"fullgraph": True},
                "_episode_window_indices": {"fullgraph": True},
                "_episode_segment_indices": {"fullgraph": True},
                "_episode_segment_candidate_mask": {"fullgraph": True},
                "_release_temporal_state_slots": {"fullgraph": True, "dynamic": False},
                "_allocate_temporal_state_slots": {"fullgraph": True, "dynamic": False},
            })
            self.assertEqual(flat_batch.local_obs[:, 0, 0].tolist(), [0.0, 1.0])
            self.assertEqual(window_batch.actions.shape[1], 2)
            self.assertIsNotNone(segment_batch.initial_temporal_state)
        finally:
            env.close()

    def test_get_all_is_env_major_and_chronological_after_wraparound(self) -> None:
        env = _make_multi_env((), ())
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            for step in range(5):
                _add_direct_step(
                    buffer,
                    obs_values=(float(step), float(100 + step)),
                    next_obs_values=(float(step + 1), float(101 + step)),
                    action_value=float(step),
                )

            batch = buffer.get_all()

            self.assertEqual(
                batch.local_obs[:, 0, 0].tolist(),
                [2.0, 3.0, 4.0, 102.0, 103.0, 104.0],
            )
            self.assertEqual(
                batch.next_local_obs[:, 0, 0].tolist(),
                [3.0, 4.0, 5.0, 103.0, 104.0, 105.0],
            )
            self.assertEqual(batch.actions[:, 0, 0].tolist(), [2.0, 3.0, 4.0] * 2)
        finally:
            env.close()

    def test_seeded_replacement_sampling_is_reproducible_across_every_field(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4, store_previous_actions=True)
            for step in range(6):
                _add_direct_step(
                    buffer,
                    obs_values=(float(step),),
                    next_obs_values=(float(step + 1),),
                    action_value=float(step),
                    previous_action_value=float(step - 1),
                )

            first = buffer.sample(20, generator=torch.Generator().manual_seed(1234))
            second = buffer.sample(20, generator=torch.Generator().manual_seed(1234))

            for field in fields(first):
                first_value = getattr(first, field.name)
                second_value = getattr(second, field.name)
                if first_value is None:
                    self.assertIsNone(second_value)
                else:
                    torch.testing.assert_close(first_value, second_value)
        finally:
            env.close()

    def test_reset_discards_ring_terminal_and_temporal_state_and_accepts_a_fresh_stream(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(
                env,
                capacity_per_env=2,
                store_previous_actions=True,
                temporal_state_store_interval=1,
            )
            _add_direct_step(
                buffer,
                obs_values=(0.0,),
                next_obs_values=(100.0,),
                action_value=1.0,
                truncations=(True,),
                terminal_obs_values=(9.0,),
                previous_action_value=-1.0,
                temporal_state=_temporal_state(20.0),
                next_temporal_state=_temporal_state(21.0),
            )

            buffer.reset()

            self.assertEqual(len(buffer), 0)
            self.assertEqual(buffer.total_transitions_added, 0)
            self.assertFalse(buffer.has_current_obs)

            _add_direct_step(
                buffer,
                obs_values=(50.0,),
                next_obs_values=(51.0,),
                action_value=5.0,
                previous_action_value=4.0,
                temporal_state=_temporal_state(30.0),
                next_temporal_state=_temporal_state(31.0),
            )
            batch = buffer.get_all()
            self.assertFlatTransitionScalars(
                batch,
                local_obs=[50.0],
                next_local_obs=[51.0],
                actions=[5.0],
                rewards=[5.25],
                previous_actions=[4.0],
                episode_start_mask=[True],
            )
            segment = buffer.sample_episode_segments(1, segment_length=1)
            self.assertIsInstance(segment.initial_temporal_state, torch.Tensor)
            self.assertEqual(segment.initial_temporal_state[:, 0, 0].tolist(), [30.0])
        finally:
            env.close()

    def test_nested_temporal_state_round_trips_with_float_conversion_and_integer_preservation(self) -> None:
        env = _make_env()
        try:
            buffer = OffPolicyReplayBuffer(
                capacity_per_env=1,
                observation_space=env.observation_space,
                action_space=env.action_space,
                temporal_state_store_interval=1,
                temporal_state_storage_dtype=torch.float16,
                storage_device="cpu",
                train_device="cpu",
                train_dtype=torch.float64,
            )
            temporal_state = {
                "actor": (
                    torch.full((1, 2, 1), 3.5, dtype=torch.float32),
                    [torch.full((1, 2, 1), 7, dtype=torch.int64)],
                ),
            }
            next_temporal_state = {
                "actor": (
                    torch.full((1, 2, 1), 4.5, dtype=torch.float32),
                    [torch.full((1, 2, 1), 8, dtype=torch.int64)],
                ),
            }
            _add_direct_step(
                buffer,
                obs_values=(0.0,),
                next_obs_values=(1.0,),
                action_value=0.0,
                temporal_state=temporal_state,
                next_temporal_state=next_temporal_state,
            )

            batch = buffer.sample_episode_segments(1, segment_length=1)
            restored_state = batch.initial_temporal_state

            self.assertIsInstance(restored_state, dict)
            self.assertEqual(restored_state["actor"][0].dtype, torch.float64)
            self.assertEqual(restored_state["actor"][1][0].dtype, torch.int64)
            torch.testing.assert_close(
                restored_state["actor"][0],
                torch.full((1, 2, 1), 3.5, dtype=torch.float64),
            )
            torch.testing.assert_close(
                restored_state["actor"][1][0],
                torch.full((1, 2, 1), 7, dtype=torch.int64),
            )
        finally:
            env.close()

    def test_storage_and_training_dtypes_are_applied_to_all_sampled_float_fields(self) -> None:
        env = _make_env()
        try:
            buffer = OffPolicyReplayBuffer(
                capacity_per_env=1,
                observation_space=env.observation_space,
                action_space=env.action_space,
                store_previous_actions=True,
                storage_device="cpu",
                storage_dtype=torch.float16,
                train_device="cpu",
                train_dtype=torch.float64,
            )
            _add_direct_step(
                buffer,
                obs_values=(0.0,),
                next_obs_values=(1.0,),
                action_value=2.0,
                previous_action_value=1.0,
            )

            batch = buffer.get_all()

            self.assertEqual(buffer.local_obs.dtype, torch.float16)
            self.assertEqual(buffer.actions.dtype, torch.float16)
            for field_name in (
                    "local_obs",
                    "global_obs",
                    "hidden_local_vars",
                    "hidden_global_vars",
                    "actions",
                    "rewards",
                    "previous_actions",
                    "next_local_obs",
                    "next_global_obs",
                    "next_hidden_local_vars",
                    "next_hidden_global_vars",
            ):
                self.assertEqual(getattr(batch, field_name).dtype, torch.float64)
            self.assertEqual(batch.terminations.dtype, torch.bool)
            self.assertEqual(batch.truncations.dtype, torch.bool)
        finally:
            env.close()

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

    def test_terminal_mask_marks_terminations_not_time_limit_truncations(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            buffer.add(
                obs=_obs(0.0),
                actions=_actions(10.0),
                rewards=torch.tensor([1.0]),
                terminations=torch.tensor([True]),
                truncations=torch.tensor([False]),
                next_obs=_obs(100.0),
                terminal_obs=_obs(1.0),
            )
            buffer.add(
                obs=_obs(100.0),
                actions=_actions(20.0),
                rewards=torch.tensor([2.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([True]),
                next_obs=_obs(200.0),
                terminal_obs=_obs(101.0),
            )

            batch = buffer.get_all()

            self.assertEqual(batch.episode_ends.tolist(), [True, True])
            self.assertEqual(batch.terminal_mask.tolist(), [True, False])
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA replay storage")
    def test_cpu_generator_samples_cuda_replay(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(
                env,
                capacity_per_env=2,
                temporal_state_store_interval=1,
                storage_device="cuda",
                train_device="cpu",
            )
            buffer.add(
                obs=_obs(0.0),
                actions=_actions(0.0),
                rewards=torch.tensor([0.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([False]),
                next_obs=_obs(1.0),
                temporal_state=_temporal_state(0.0),
                next_temporal_state=_temporal_state(1.0),
            )
            generator = torch.Generator(device="cpu").manual_seed(7)

            batch = buffer.sample(1, generator=generator)
            window_batch = buffer.sample_episode_windows(
                1,
                num_next_steps=2,
                generator=generator,
            )
            segment_batch = buffer.sample_episode_segments(
                1,
                segment_length=1,
                generator=generator,
            )

            self.assertEqual(batch.actions.device.type, "cpu")
            self.assertEqual(window_batch.actions.device.type, "cpu")
            self.assertEqual(window_batch.train_mask.tolist(), [[True, False]])
            self.assertEqual(segment_batch.actions.device.type, "cpu")
            self.assertEqual(segment_batch.initial_temporal_state.device.type, "cpu")
        finally:
            env.close()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA replay storage")
    def test_compiled_cuda_replay_matches_eager_after_wraparound_and_episode_boundaries(self) -> None:
        env = _make_env()
        try:
            buffers = [
                _make_buffer(
                    env,
                    capacity_per_env=4,
                    store_previous_actions=True,
                    temporal_state_store_interval=2,
                    storage_device="cuda",
                    train_device="cpu",
                    compile_tensor_operations=compile_tensor_operations,
                )
                for compile_tensor_operations in (False, True)
            ]
            for step in range(7):
                termination = step == 2
                truncation = step == 5
                for buffer in buffers:
                    _add_direct_step(
                        buffer,
                        obs_values=(float(step),),
                        next_obs_values=(float(step + 1),),
                        action_value=float(step),
                        terminations=(termination,),
                        truncations=(truncation,),
                        terminal_obs_values=(float(100 + step),) if termination or truncation else None,
                        previous_action_value=float(step - 1),
                        temporal_state=_temporal_state(float(10 + step)),
                        next_temporal_state=_temporal_state(float(11 + step)),
                    )

            def assert_batches_equal(first: Any, second: Any) -> None:
                self.assertIs(type(first), type(second))
                for field in fields(first):
                    first_value = getattr(first, field.name)
                    second_value = getattr(second, field.name)
                    if first_value is None:
                        self.assertIsNone(second_value)
                    elif torch.is_tensor(first_value):
                        torch.testing.assert_close(first_value, second_value)
                    else:
                        self.assertEqual(first_value, second_value)

            eager_buffer, compiled_buffer = buffers
            assert_batches_equal(eager_buffer.get_all(), compiled_buffer.get_all())

            all_indices = torch.arange(len(eager_buffer), device="cuda")
            assert_batches_equal(
                eager_buffer._fetch_episode_windows(all_indices, num_next_steps=4),
                compiled_buffer._fetch_episode_windows(all_indices, num_next_steps=4),
            )

            eager_candidates = eager_buffer._replay_segment_candidates(
                total_sequence_length=3,
                require_initial_temporal_state=True,
                allow_episode_boundaries=True,
            )
            compiled_candidates = compiled_buffer._replay_segment_candidates(
                total_sequence_length=3,
                require_initial_temporal_state=True,
                allow_episode_boundaries=True,
            )
            for eager_indices, compiled_indices in zip(eager_candidates, compiled_candidates, strict=True):
                torch.testing.assert_close(eager_indices.cpu(), compiled_indices.cpu())

            segment_kwargs = {
                "batch_size": int(eager_candidates[0].numel()),
                "segment_length": 2,
                "burn_in_steps": 1,
                "replacement": False,
                "allow_episode_boundaries": True,
            }
            eager_segment = eager_buffer.sample_episode_segments(
                **segment_kwargs,
                generator=torch.Generator().manual_seed(123),
            )
            compiled_segment = compiled_buffer.sample_episode_segments(
                **segment_kwargs,
                generator=torch.Generator().manual_seed(123),
            )
            assert_batches_equal(eager_segment, compiled_segment)
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

    def test_direct_stream_keeps_full_transition_tuple_aligned_after_multiple_wraparounds(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4, store_previous_actions=True)
            for step in range(11):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(1000 + step)),
                    rewards=torch.tensor([float(2000 + step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                    previous_actions=_actions(float(3000 + step)),
                )

            self.assertFlatTransitionScalars(
                buffer.get_all(),
                local_obs=[7.0, 8.0, 9.0, 10.0],
                next_local_obs=[8.0, 9.0, 10.0, 11.0],
                actions=[1007.0, 1008.0, 1009.0, 1010.0],
                rewards=[2007.0, 2008.0, 2009.0, 2010.0],
                previous_actions=[3007.0, 3008.0, 3009.0, 3010.0],
            )
        finally:
            env.close()

    def test_without_replacement_sample_keeps_arbitrary_wrapped_rows_aligned(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=5)
            for step in range(9):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(10 + step)),
                    rewards=torch.tensor([float(20 + step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([False]),
                    next_obs=_obs(float(step + 1)),
                )

            with patch("torch.randperm", return_value=torch.tensor([4, 0, 2, 1, 3])):
                batch = buffer.sample(3, replacement=False)

            self.assertFlatTransitionScalars(
                batch,
                local_obs=[8.0, 4.0, 6.0],
                next_local_obs=[9.0, 5.0, 7.0],
                actions=[18.0, 14.0, 16.0],
                rewards=[28.0, 24.0, 26.0],
            )
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
            self.assertIsNone(buffer._temporal_state_indices)
            self.assertIsNone(buffer._temporal_state_slots_in_use)
            self.assertEqual(buffer.temporal_state_capacity_per_env, 0)
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

    def test_temporal_state_storage_dtype_must_be_floating_point(self) -> None:
        env = _make_env()
        try:
            with self.assertRaisesRegex(ValueError, "floating-point dtype"):
                _make_buffer(
                    env,
                    capacity_per_env=2,
                    temporal_state_store_interval=1,
                    temporal_state_storage_dtype=torch.int8,
                )
        finally:
            env.close()

    def test_temporal_state_uses_storage_dtype_and_restores_train_dtype(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(
                env,
                capacity_per_env=2,
                temporal_state_store_interval=1,
                temporal_state_storage_dtype=torch.float16,
            )
            buffer.add(
                obs=_obs(0.0),
                actions=_actions(0.0),
                rewards=torch.tensor([0.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([False]),
                next_obs=_obs(1.0),
                temporal_state=_temporal_state(1.25),
                next_temporal_state=_temporal_state(2.5),
            )

            self.assertIsInstance(buffer.temporal_states, torch.Tensor)
            self.assertEqual(buffer.temporal_states.dtype, torch.float16)
            self.assertEqual(buffer.temporal_states.shape[1], 3)
            batch = buffer.sample_episode_segments(1, segment_length=1)
            self.assertIsInstance(batch.initial_temporal_state, torch.Tensor)
            self.assertEqual(batch.initial_temporal_state.dtype, torch.float32)
            torch.testing.assert_close(
                batch.initial_temporal_state,
                _temporal_state(1.25),
            )
        finally:
            env.close()

    def test_temporal_state_storage_is_compact_and_reuses_slots_after_ring_wrap(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(
                env,
                capacity_per_env=5,
                temporal_state_store_interval=2,
            )
            for step in range(12):
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

            self.assertEqual(buffer.temporal_state_capacity_per_env, 3)
            self.assertIsInstance(buffer.temporal_states, torch.Tensor)
            self.assertEqual(buffer.temporal_states.shape[1], 3)

            batch = buffer.sample_episode_segments(
                2,
                segment_length=1,
                replacement=False,
                allow_episode_boundaries=True,
            )
            self.assertIsInstance(batch.initial_temporal_state, torch.Tensor)
            self.assertEqual(
                set(batch.initial_temporal_state[:, 0, 0].tolist()),
                {8.0, 10.0},
            )
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

    def test_direct_done_stream_keeps_terminal_and_reset_obs_associated_after_wraparound(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=5)
            next_values = [1.0, 100.0, 101.0, 200.0, 201.0, 202.0, 300.0, 301.0]
            terminal_values = {
                1: 2.0,
                3: 102.0,
                6: 203.0,
            }
            for step, next_value in enumerate(next_values):
                is_termination = step == 3
                is_truncation = step in (1, 6)
                buffer.add(
                    obs=_obs(float(-1000 - step)),
                    actions=_actions(float(1000 + step)),
                    rewards=torch.tensor([float(2000 + step)]),
                    terminations=torch.tensor([is_termination]),
                    truncations=torch.tensor([is_truncation]),
                    next_obs=_obs(next_value),
                    terminal_obs=_obs(terminal_values[step]) if step in terminal_values else None,
                )

            self.assertFlatTransitionScalars(
                buffer.get_all(),
                local_obs=[101.0, 200.0, 201.0, 202.0, 300.0],
                next_local_obs=[102.0, 201.0, 202.0, 203.0, 301.0],
                actions=[1003.0, 1004.0, 1005.0, 1006.0, 1007.0],
                rewards=[2003.0, 2004.0, 2005.0, 2006.0, 2007.0],
                terminations=[True, False, False, False, False],
                truncations=[False, False, False, True, False],
            )
        finally:
            env.close()

    def test_replacement_sample_keeps_wrapped_done_rows_aligned_with_terminal_obs(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=5)
            next_values = [1.0, 100.0, 101.0, 200.0, 201.0, 202.0, 300.0, 301.0]
            terminal_values = {
                1: 2.0,
                3: 102.0,
                6: 203.0,
            }
            for step, next_value in enumerate(next_values):
                is_termination = step == 3
                is_truncation = step in (1, 6)
                buffer.add(
                    obs=_obs(float(-1000 - step)),
                    actions=_actions(float(1000 + step)),
                    rewards=torch.tensor([float(2000 + step)]),
                    terminations=torch.tensor([is_termination]),
                    truncations=torch.tensor([is_truncation]),
                    next_obs=_obs(next_value),
                    terminal_obs=_obs(terminal_values[step]) if step in terminal_values else None,
                )

            with patch("torch.randint", return_value=torch.tensor([0, 3, 0])):
                batch = buffer.sample(3, replacement=True)

            self.assertFlatTransitionScalars(
                batch,
                local_obs=[101.0, 202.0, 101.0],
                next_local_obs=[102.0, 203.0, 102.0],
                actions=[1003.0, 1006.0, 1003.0],
                rewards=[2003.0, 2006.0, 2003.0],
                terminations=[True, False, True],
                truncations=[False, True, False],
            )
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
                    "swarmbots.learn.algos.off_policy.off_policy_rollout.sample_random_actions",
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

    def test_collect_with_obs_encoded_policy_keeps_actions_tied_to_current_obs_after_wraparound(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=9,
                policy=_ObsEncodingPolicy(),
            )

            self.assertFlatTransitionScalars(
                buffer.get_all(),
                local_obs=[105.0, 106.0, 107.0, 108.0],
                next_local_obs=[106.0, 107.0, 108.0, 109.0],
                actions=[105.25, 106.25, 107.25, 108.25],
                rewards=[6.0, 7.0, 8.0, 9.0],
            )
        finally:
            env.close()

    def test_collect_with_done_wraparound_keeps_action_terminal_and_reset_obs_alignment(self) -> None:
        env = _make_env(done_steps=(3,))
        try:
            buffer = _make_buffer(env, capacity_per_env=5)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=10,
                policy=_ObsEncodingPolicy(),
            )

            self.assertFlatTransitionScalars(
                buffer.get_all(),
                local_obs=[202.0, 300.0, 301.0, 302.0, 400.0],
                next_local_obs=[203.0, 301.0, 302.0, 303.0, 401.0],
                actions=[202.25, 300.25, 301.25, 302.25, 400.25],
                rewards=[3.0, 1.0, 2.0, 3.0, 1.0],
                truncations=[True, False, False, True, False],
            )
        finally:
            env.close()

    def test_scenario_ids_follow_policy_actions_terminal_obs_and_reset_stream_after_wraparound(self) -> None:
        env = _make_scenario_env(done_steps=(1,))
        try:
            buffer = _make_buffer(env, capacity_per_env=3)

            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=4,
                policy=_ScenarioEncodingPolicy(),
            )

            retained = buffer.get_all()
            assert retained.scenario_ids is not None
            assert retained.next_scenario_ids is not None
            self.assertEqual(retained.scenario_ids.tolist(), [2, 0, 1])
            self.assertEqual(retained.next_scenario_ids.tolist(), [2, 0, 1])
            self.assertEqual(retained.actions[:, 0, 0].tolist(), [2.0, 0.0, 1.0])
            self.assertEqual(retained.local_obs[:, 0, 0].tolist(), [200.0, 300.0, 400.0])
            self.assertEqual(retained.next_local_obs[:, 0, 0].tolist(), [201.0, 301.0, 401.0])
            self.assertTrue(retained.truncations.all())

            sampled = buffer.sample(20, generator=torch.Generator().manual_seed(7))
            assert sampled.scenario_ids is not None
            assert sampled.next_scenario_ids is not None
            torch.testing.assert_close(sampled.next_scenario_ids, sampled.scenario_ids)
            torch.testing.assert_close(sampled.actions[:, 0, 0], sampled.scenario_ids.float())
            torch.testing.assert_close(
                sampled.next_local_obs[:, 0, 0],
                sampled.local_obs[:, 0, 0] + 1.0,
            )
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

    def test_discarded_warmup_preserves_same_step_stream_for_first_replay_write(self) -> None:
        env = _make_env(done_steps=(3,))
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            policy = _ObsEncodingPolicy()

            rollout_state = warmup_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=4,
                policy=policy,
            )

            self.assertEqual(len(buffer), 0)
            self.assertFalse(buffer.has_current_obs)
            self.assertEqual(rollout_state.rollout_step_idx, 4)
            self.assertEqual(float(rollout_state.obs["local_obs"][0, 0, 0]), 201.0)

            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=2,
                policy=policy,
                rollout_state=rollout_state,
            )

            self.assertFlatTransitionScalars(
                buffer.get_all(),
                local_obs=[201.0, 202.0],
                next_local_obs=[202.0, 203.0],
                actions=[201.25, 202.25],
                rewards=[2.0, 3.0],
                truncations=[False, True],
            )
            self.assertEqual(rollout_state.rollout_step_idx, 6)
            self.assertEqual(float(rollout_state.obs["local_obs"][0, 0, 0]), 300.0)
        finally:
            env.close()

    def test_collect_only_snapshots_current_obs_when_buffer_needs_it(self) -> None:
        env = _make_env(done_steps=(4,))
        try:
            buffer = _make_buffer(env, capacity_per_env=8)
            with patch(
                    "swarmbots.learn.algos.off_policy.off_policy_rollout.snapshot_obs",
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
                    "swarmbots.learn.algos.off_policy.off_policy_rollout.snapshot_obs",
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

    def test_collect_previous_actions_stay_aligned_through_done_and_wraparound(self) -> None:
        env = _make_env(done_steps=(3,))
        try:
            buffer = _make_buffer(env, capacity_per_env=5, store_previous_actions=True)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=10,
                policy=_ObsEncodingPolicy(),
            )

            batch = buffer.get_all()
            self.assertFlatTransitionScalars(
                batch,
                local_obs=[202.0, 300.0, 301.0, 302.0, 400.0],
                next_local_obs=[203.0, 301.0, 302.0, 303.0, 401.0],
                actions=[202.25, 300.25, 301.25, 302.25, 400.25],
                rewards=[3.0, 1.0, 2.0, 3.0, 1.0],
                truncations=[True, False, False, True, False],
                previous_actions=[201.25, 0.0, 300.25, 301.25, 0.0],
            )
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

    def test_recurrent_policy_cannot_start_mid_episode_after_untracked_random_rollout(self) -> None:
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
            with self.assertRaisesRegex(ValueError, "Pass the policy during random collection"):
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=2,
                    policy=policy,
                    rollout_state=rollout_state,
                )
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

    def test_episode_window_sampling_keeps_origins_uniform_and_masks_unavailable_future_steps(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            transitions = [
                (0.0, 10.0, False, 1.0, None),
                (1.0, 11.0, True, 100.0, 2.0),
                (100.0, 12.0, False, 101.0, None),
                (101.0, 13.0, False, 102.0, None),
            ]
            for obs_value, action_value, truncated, next_obs_value, terminal_obs_value in transitions:
                buffer.add(
                    obs=_obs(obs_value),
                    actions=_actions(action_value),
                    rewards=torch.tensor([action_value + 10.0]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([truncated]),
                    next_obs=_obs(next_obs_value),
                    terminal_obs=None if terminal_obs_value is None else _obs(terminal_obs_value),
                )

            with patch("torch.randint", return_value=torch.tensor([0, 1, 2, 3])):
                windows = buffer.sample_episode_windows(4, num_next_steps=3)

            self.assertEqual(windows.origin_batch.local_obs[:, 0, 0].tolist(), [0.0, 1.0, 100.0, 101.0])
            self.assertEqual(windows.origin_batch.actions[:, 0, 0].tolist(), [10.0, 11.0, 12.0, 13.0])
            self.assertEqual(
                windows.train_mask.tolist(),
                [
                    [True, True, False],
                    [True, False, False],
                    [True, True, False],
                    [True, False, False],
                ],
            )
            self.assertEqual(
                windows.actions[:, :, 0, 0].tolist(),
                [
                    [10.0, 11.0, 0.0],
                    [11.0, 0.0, 0.0],
                    [12.0, 13.0, 0.0],
                    [13.0, 0.0, 0.0],
                ],
            )
            self.assertEqual(
                windows.next_local_obs[:, :, 0, 0].tolist(),
                [
                    [1.0, 2.0, 0.0],
                    [2.0, 0.0, 0.0],
                    [101.0, 102.0, 0.0],
                    [102.0, 0.0, 0.0],
                ],
            )
        finally:
            env.close()

    def test_episode_window_padding_uses_attention_safe_agent_masks(self) -> None:
        env = _make_agent_mask_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=2)
            obs = _obs(0.0)
            obs["agent_mask"] = torch.tensor([[True, False]])
            next_obs = _obs(1.0)
            next_obs["agent_mask"] = torch.tensor([[False, True]])
            buffer.add(
                obs=obs,
                actions=_actions(10.0),
                rewards=torch.tensor([20.0]),
                terminations=torch.tensor([False]),
                truncations=torch.tensor([False]),
                next_obs=next_obs,
            )

            with patch("torch.randint", return_value=torch.tensor([0])):
                windows = buffer.sample_episode_windows(1, num_next_steps=3)

            self.assertEqual(windows.train_mask.tolist(), [[True, False, False]])
            assert windows.agent_mask is not None
            assert windows.next_agent_mask is not None
            self.assertEqual(
                windows.agent_mask.tolist(),
                [[[True, False], [True, True], [True, True]]],
            )
            self.assertEqual(
                windows.next_agent_mask.tolist(),
                [[[False, True], [True, True], [True, True]]],
            )
        finally:
            env.close()

    def test_episode_segment_sampling_keeps_wrapped_transition_tuples_aligned(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=5, temporal_state_store_interval=1)
            for step in range(8):
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(100 + step)),
                    rewards=torch.tensor([float(200 + step)]),
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

            self.assertSegmentTransitionScalars(
                batch,
                0,
                local_obs=[4.0, 5.0, 6.0],
                next_local_obs=[5.0, 6.0, 7.0],
                actions=[104.0, 105.0, 106.0],
                rewards=[204.0, 205.0, 206.0],
            )
            self.assertEqual(batch.train_mask.tolist(), [[False, True, True]])
            self.assertIsInstance(batch.initial_temporal_state, torch.Tensor)
            self.assertEqual(batch.initial_temporal_state[:, 0, 0].tolist(), [4.0])
        finally:
            env.close()

    def test_episode_segment_sampling_allows_final_done_transition_after_wraparound(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=5, temporal_state_store_interval=1)
            for step in range(8):
                truncated = step == 6
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(100 + step)),
                    rewards=torch.tensor([float(200 + step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([truncated]),
                    next_obs=_obs(100.0 if truncated else float(step + 1)),
                    terminal_obs=_obs(7.0) if truncated else None,
                    temporal_state=_temporal_state(float(step)),
                    next_temporal_state=_temporal_state(float(step + 1)),
                )

            with patch("torch.randint", return_value=torch.tensor([2])):
                batch = buffer.sample_episode_segments(
                    1,
                    segment_length=2,
                )

            self.assertSegmentTransitionScalars(
                batch,
                0,
                local_obs=[5.0, 6.0],
                next_local_obs=[6.0, 7.0],
                actions=[105.0, 106.0],
                rewards=[205.0, 206.0],
                truncations=[False, True],
            )
            self.assertEqual(batch.episode_ends.tolist(), [[False, True]])
            self.assertIsInstance(batch.initial_temporal_state, torch.Tensor)
            self.assertEqual(batch.initial_temporal_state[:, 0, 0].tolist(), [5.0])
        finally:
            env.close()

    def test_episode_segment_sampling_allows_cross_episode_windows(self) -> None:
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

            episode_batch = buffer.sample_episode_segments(
                2,
                segment_length=2,
                replacement=False,
            )
            self.assertEqual(
                sorted(episode_batch.local_obs[:, 0, 0, 0].tolist()),
                [0.0, 2.0],
            )
            self.assertIsInstance(episode_batch.initial_temporal_state, torch.Tensor)
            initial_states_by_start = {
                float(episode_batch.local_obs[batch_idx, 0, 0, 0].item()): float(
                    episode_batch.initial_temporal_state[batch_idx, 0, 0].item()
                )
                for batch_idx in range(2)
            }
            self.assertEqual(initial_states_by_start[2.0], 2.0)

            batch = buffer.sample_episode_segments(
                3,
                segment_length=2,
                replacement=False,
                allow_episode_boundaries=True,
            )

            segments_by_start = {
                float(batch.local_obs[batch_idx, 0, 0, 0].item()): batch.episode_ends[batch_idx].tolist()
                for batch_idx in range(3)
            }
            self.assertEqual(segments_by_start, {
                0.0: [False, True],
                1.0: [True, False],
                2.0: [False, False],
            })
        finally:
            env.close()

    def test_episode_segment_sampling_exhaustively_respects_wrapped_episode_boundaries(self) -> None:
        env = _make_env()
        try:
            capacity = 4
            for done_bits in range(1 << capacity):
                with self.subTest(done_bits=f"{done_bits:0{capacity}b}"):
                    buffer = _make_buffer(env, capacity_per_env=capacity)
                    retained_episode_ends = [bool(done_bits & (1 << idx)) for idx in range(capacity)]
                    all_episode_ends = [False, False, *retained_episode_ends]
                    for step, episode_end in enumerate(all_episode_ends):
                        _add_direct_step(
                            buffer,
                            obs_values=(float(step),),
                            next_obs_values=(float(step + 1),),
                            action_value=float(step),
                            terminations=(episode_end and step % 2 == 0,),
                            truncations=(episode_end and step % 2 == 1,),
                            terminal_obs_values=(float(step + 0.5),) if episode_end else None,
                        )

                    for sequence_length in range(1, capacity + 1):
                        expected_starts = [
                            start
                            for start in range(capacity - sequence_length + 1)
                            if not any(retained_episode_ends[start:start + sequence_length - 1])
                        ]
                        if expected_starts:
                            sampled = buffer.sample_episode_segments(
                                len(expected_starts),
                                segment_length=sequence_length,
                                replacement=False,
                                require_initial_temporal_state=False,
                            )
                            actual_starts = sorted(sampled.local_obs[:, 0, 0, 0].tolist())
                            self.assertEqual(actual_starts, [float(start + 2) for start in expected_starts])
                        else:
                            with self.assertRaises(NoEpisodeSegmentCandidatesError):
                                buffer.sample_episode_segments(
                                    1,
                                    segment_length=sequence_length,
                                    require_initial_temporal_state=False,
                                )

                        cross_boundary_count = capacity - sequence_length + 1
                        cross_boundary_sample = buffer.sample_episode_segments(
                            cross_boundary_count,
                            segment_length=sequence_length,
                            replacement=False,
                            require_initial_temporal_state=False,
                            allow_episode_boundaries=True,
                        )
                        self.assertEqual(
                            sorted(cross_boundary_sample.local_obs[:, 0, 0, 0].tolist()),
                            [float(start + 2) for start in range(cross_boundary_count)],
                        )
        finally:
            env.close()

    def test_wrapped_segment_candidates_only_start_at_live_temporal_checkpoints(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(
                env,
                capacity_per_env=5,
                temporal_state_store_interval=2,
            )
            for step in range(8):
                _add_direct_step(
                    buffer,
                    obs_values=(float(step),),
                    next_obs_values=(float(step + 1),),
                    action_value=float(step),
                    temporal_state=_temporal_state(float(step)),
                    next_temporal_state=_temporal_state(float(step + 1)),
                )

            batch = buffer.sample_episode_segments(
                2,
                segment_length=2,
                replacement=False,
            )

            self.assertEqual(sorted(batch.local_obs[:, 0, 0, 0].tolist()), [4.0, 6.0])
            self.assertIsInstance(batch.initial_temporal_state, torch.Tensor)
            self.assertEqual(
                sorted(batch.initial_temporal_state[:, 0, 0].tolist()),
                [4.0, 6.0],
            )
        finally:
            env.close()

    def test_multi_env_segment_candidates_keep_checkpoint_rows_and_boundaries_aligned(self) -> None:
        env = _make_multi_env((), ())
        try:
            buffer = _make_buffer(
                env,
                capacity_per_env=4,
                temporal_state_store_interval=2,
            )
            for step in range(6):
                env_values = (float(step), float(100 + step))
                next_env_values = (float(step + 1), float(101 + step))
                temporal_state = torch.tensor(env_values).view(2, 1, 1).expand(2, 2, 1).clone()
                next_temporal_state = (
                    torch.tensor(next_env_values).view(2, 1, 1).expand(2, 2, 1).clone()
                )
                _add_direct_step(
                    buffer,
                    obs_values=env_values,
                    next_obs_values=next_env_values,
                    action_value=float(step),
                    terminations=(step == 2, False),
                    terminal_obs_values=next_env_values if step == 2 else None,
                    temporal_state=temporal_state,
                    next_temporal_state=next_temporal_state,
                )

            episode_contained = buffer.sample_episode_segments(
                3,
                segment_length=2,
                replacement=False,
            )
            episode_contained_starts = episode_contained.local_obs[:, 0, 0, 0]
            self.assertEqual(sorted(episode_contained_starts.tolist()), [4.0, 102.0, 104.0])
            self.assertIsInstance(episode_contained.initial_temporal_state, torch.Tensor)
            torch.testing.assert_close(
                episode_contained.initial_temporal_state[:, 0, 0],
                episode_contained_starts,
            )

            cross_episode = buffer.sample_episode_segments(
                4,
                segment_length=2,
                replacement=False,
                allow_episode_boundaries=True,
            )
            cross_episode_starts = cross_episode.local_obs[:, 0, 0, 0]
            self.assertEqual(sorted(cross_episode_starts.tolist()), [2.0, 4.0, 102.0, 104.0])
            self.assertIsInstance(cross_episode.initial_temporal_state, torch.Tensor)
            torch.testing.assert_close(
                cross_episode.initial_temporal_state[:, 0, 0],
                cross_episode_starts,
            )
            env_zero_start_two = torch.nonzero(cross_episode_starts == 2.0, as_tuple=False).item()
            self.assertEqual(
                cross_episode.episode_ends[env_zero_start_two].tolist(),
                [True, False],
            )
        finally:
            env.close()

    def test_multi_env_wrapped_segments_keep_checkpoints_and_terminal_transitions_aligned(self) -> None:
        env = _make_multi_env((), ())
        try:
            buffer = _make_buffer(
                env,
                capacity_per_env=5,
                temporal_state_store_interval=1,
            )
            env0_obs_values = [0.0, 1.0, 2.0, 3.0, 4.0, 50.0, 51.0]
            for step, env0_obs_value in enumerate(env0_obs_values):
                env1_obs_value = float(100 + step)
                truncated = step == 4
                next_env0_obs_value = 50.0 if truncated else env0_obs_value + 1.0
                next_obs_values = (next_env0_obs_value, env1_obs_value + 1.0)
                current_state = torch.tensor(
                    [env0_obs_value, env1_obs_value],
                    dtype=torch.float32,
                ).view(2, 1, 1).expand(-1, 2, 1).clone()
                next_state = torch.tensor(
                    next_obs_values,
                    dtype=torch.float32,
                ).view(2, 1, 1).expand(-1, 2, 1).clone()
                _add_direct_step(
                    buffer,
                    obs_values=(env0_obs_value, env1_obs_value),
                    next_obs_values=next_obs_values,
                    action_value=float(step),
                    truncations=(truncated, False),
                    terminal_obs_values=(5.0, env1_obs_value + 1.0) if truncated else None,
                    temporal_state=current_state,
                    next_temporal_state=next_state,
                )

            batch = buffer.sample_episode_segments(
                6,
                segment_length=2,
                burn_in_steps=1,
                replacement=False,
                allow_episode_boundaries=True,
            )

            starts = batch.local_obs[:, 0, 0, 0]
            self.assertEqual(
                sorted(starts.tolist()),
                [2.0, 3.0, 4.0, 102.0, 103.0, 104.0],
            )
            self.assertIsInstance(batch.initial_temporal_state, torch.Tensor)
            torch.testing.assert_close(
                batch.initial_temporal_state[:, 0, 0],
                starts,
            )
            self.assertEqual(
                batch.train_mask.tolist(),
                [[False, True, True]] * 6,
            )

            crossing_row = int(torch.nonzero(starts == 4.0, as_tuple=False).item())
            self.assertSegmentTransitionScalars(
                batch,
                crossing_row,
                local_obs=[4.0, 50.0, 51.0],
                next_local_obs=[5.0, 51.0, 52.0],
                actions=[4.0, 5.0, 6.0],
                rewards=[4.25, 5.25, 6.25],
                truncations=[True, False, False],
            )
            self.assertEqual(
                batch.episode_start_mask[crossing_row].tolist(),
                [False, True, False],
            )
        finally:
            env.close()

    def test_episode_segment_sampling_without_temporal_state_can_cross_every_boundary(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
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

            with self.assertRaisesRegex(NoEpisodeSegmentCandidatesError, "no contiguous replay windows"):
                buffer.sample_episode_segments(
                    1,
                    segment_length=2,
                    require_initial_temporal_state=False,
                )

            batch = buffer.sample_episode_segments(
                2,
                segment_length=2,
                replacement=False,
                require_initial_temporal_state=False,
                allow_episode_boundaries=True,
            )

            segments_by_start = {
                float(batch.local_obs[batch_idx, 0, 0, 0].item()): batch.episode_ends[batch_idx].tolist()
                for batch_idx in range(2)
            }
            self.assertEqual(segments_by_start, {
                0.0: [True, True],
                100.0: [True, True],
            })
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

            with self.assertRaisesRegex(NoEpisodeSegmentCandidatesError, "no contiguous replay windows"):
                buffer.sample_episode_segments(1, segment_length=1)
        finally:
            env.close()

    def test_episode_start_without_temporal_checkpoint_is_not_a_segment_start(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4, temporal_state_store_interval=10)
            for step in range(4):
                truncated = step == 1
                buffer.add(
                    obs=_obs(float(step)),
                    actions=_actions(float(step)),
                    rewards=torch.tensor([float(step)]),
                    terminations=torch.tensor([False]),
                    truncations=torch.tensor([truncated]),
                    next_obs=_obs(100.0 if truncated else float(step + 1)),
                    terminal_obs=_obs(2.0) if truncated else None,
                    temporal_state=_temporal_state(float(step + 5)),
                    next_temporal_state=_temporal_state(float(step + 6)),
                )

            batch = buffer.sample_episode_segments(
                1,
                segment_length=2,
                replacement=False,
                allow_episode_boundaries=True,
            )

            self.assertEqual(batch.local_obs[0, :, 0, 0].tolist(), [0.0, 1.0])
            self.assertEqual(batch.episode_start_mask.tolist(), [[True, False]])
            self.assertIsInstance(batch.initial_temporal_state, torch.Tensor)
            self.assertEqual(batch.initial_temporal_state[:, 0, 0].tolist(), [5.0])
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
            self.assertIsNone(rollout_state.gsde_noise_state)

            policy = _GSDEPolicy()
            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=1,
                policy=policy,
                rollout_state=rollout_state,
                gsde_reset_mode=GSDEIntervalResetMode(interval=3),
            )

            self.assertIsNotNone(rollout_state.gsde_noise_state)
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

    def test_gsde_probability_noise_force_reset_after_random_warmup(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=1,
                random_actions=True,
            )
            self.assertIsNone(rollout_state.gsde_noise_state)

            policy = _GSDEPolicy()
            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=1,
                policy=policy,
                rollout_state=rollout_state,
                gsde_reset_mode=GSDEProbabilityResetMode(probability=0.5),
            )

            self.assertIsNotNone(rollout_state.gsde_noise_state)
            self.assertEqual(policy.action_dist.call_order, ["step", "ep_start"])
            self.assertEqual(policy.action_dist.step_resets, [(None, (1, 2))])
            self.assertEqual([mask.tolist() for mask in policy.action_dist.episode_start_masks], [[False]])
        finally:
            env.close()

    def test_gsde_policy_requires_reset_mode(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            with self.assertRaisesRegex(ValueError, "gsde_reset_mode"):
                collect_off_policy_steps(
                    env=env,
                    replay_buffer=buffer,
                    n_steps=1,
                    policy=_GSDEPolicy(),
                )
        finally:
            env.close()

    def test_gsde_reset_mode_rejects_invalid_parameters(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            for reset_mode in (
                    GSDEIntervalResetMode(interval=0),
                    GSDEProbabilityResetMode(probability=1.0),
            ):
                with self.subTest(reset_mode=reset_mode):
                    with self.assertRaises(ValueError):
                        collect_off_policy_steps(
                            env=env,
                            replay_buffer=buffer,
                            n_steps=1,
                            policy=_GSDEPolicy(),
                            gsde_reset_mode=reset_mode,
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

    def test_random_collection_advances_recurrent_policy_state(self) -> None:
        env = _make_env()
        try:
            buffer = _make_buffer(env, capacity_per_env=4)
            policy = _TemporalPolicy()

            _episode_infos, _metrics, rollout_state = collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=2,
                policy=policy,
                random_actions=True,
            )

            self.assertEqual(
                [mask.tolist() for mask in policy.episode_start_masks],
                [[True], [False]],
            )
            self.assertIsInstance(rollout_state.temporal_state, torch.Tensor)
            self.assertEqual(rollout_state.temporal_state[:, 0, 0].tolist(), [2.0])
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

    def test_multi_env_obs_encoded_policy_keeps_each_lane_aligned_after_wraparound(self) -> None:
        env = _make_multi_env((), (2,))
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=16,
                policy=_ObsEncodingPolicy(),
            )

            self.assertFlatTransitionScalars(
                buffer.get_all(),
                local_obs=[105.0, 106.0, 107.0, 1301.0, 1400.0, 1401.0],
                next_local_obs=[106.0, 107.0, 108.0, 1302.0, 1401.0, 1402.0],
                actions=[105.25, 106.25, 107.25, 1301.25, 1400.25, 1401.25],
                rewards=[6.0, 7.0, 8.0, 2.0, 1.0, 2.0],
                truncations=[False, False, False, True, False, True],
            )
        finally:
            env.close()

    def test_multi_env_sample_keeps_flat_env_major_indices_aligned_after_wraparound(self) -> None:
        env = _make_multi_env((), (2,))
        try:
            buffer = _make_buffer(env, capacity_per_env=3)
            collect_off_policy_steps(
                env=env,
                replay_buffer=buffer,
                n_steps=16,
                policy=_ObsEncodingPolicy(),
            )

            with patch("torch.randint", return_value=torch.tensor([0, 4, 5, 2])):
                batch = buffer.sample(4, replacement=True)

            self.assertFlatTransitionScalars(
                batch,
                local_obs=[105.0, 1400.0, 1401.0, 107.0],
                next_local_obs=[106.0, 1401.0, 1402.0, 108.0],
                actions=[105.25, 1400.25, 1401.25, 107.25],
                rewards=[6.0, 1.0, 2.0, 8.0],
                truncations=[False, False, True, False],
            )
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
