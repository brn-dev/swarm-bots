from typing import Any

import gymnasium
import numpy as np
import pytest
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv
from gymnasium.vector.utils import batch_space

from swarmbots.learn.algos.off_policy import OffPolicyReplayBuffer, collect_off_policy_steps
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import (
    SwarmBotsLearnEnvWrapper,
)
from swarmbots.learn.env_wrappers.learn_wrappers.torch_env_wrapper import TorchEnvWrapper
from swarmbots.learn.env_wrappers.multi_scenario_vector_env import MultiScenarioVectorEnv


class _ScenarioEnv(gymnasium.Env):
    def __init__(
            self,
            *,
            global_obs_dim: int,
            hidden_local_vars_dim: int,
            hidden_global_vars_dim: int,
            terminates: bool = True,
            emit_marker: bool = False,
    ) -> None:
        super().__init__()
        self.global_obs_dim = global_obs_dim
        self.hidden_local_vars_dim = hidden_local_vars_dim
        self.hidden_global_vars_dim = hidden_global_vars_dim
        self.terminates = terminates
        self.emit_marker = emit_marker
        self.reset_calls = 0
        self.reset_seeds: list[int | None] = []
        self.last_action: dict[str, np.ndarray] | None = None
        self.was_closed = False
        self.observation_space = spaces.Dict({
            "local_obs": spaces.Box(-np.inf, np.inf, shape=(2, 3), dtype=np.float32),
            "global_obs": spaces.Box(-np.inf, np.inf, shape=(global_obs_dim,), dtype=np.float32),
            "hidden_local_vars": spaces.Box(
                -np.inf,
                np.inf,
                shape=(2, hidden_local_vars_dim),
                dtype=np.float32,
            ),
            "hidden_global_vars": spaces.Box(
                -np.inf,
                np.inf,
                shape=(hidden_global_vars_dim,),
                dtype=np.float32,
            ),
            "agent_mask": spaces.MultiBinary(2),
        })
        self.action_space = spaces.Dict({
            "actuators": spaces.Box(-1.0, 1.0, shape=(2, 1), dtype=np.float32),
            "connectors": spaces.Box(-1.0, 1.0, shape=(2, 1), dtype=np.float32),
        })

    def reset(
            self,
            *,
            seed: int | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        _ = options
        super().reset(seed=seed)
        self.reset_calls += 1
        self.reset_seeds.append(seed)
        return self._obs(), {}

    def step(
            self,
            action: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], float, bool, bool, dict[str, Any]]:
        self.last_action = action
        infos: dict[str, Any] = {"success": True}
        if self.emit_marker:
            infos["marker"] = 7
        return self._obs(), 1.0, self.terminates, False, infos

    def _obs(self) -> dict[str, np.ndarray]:
        return {
            "local_obs": np.ones((2, 3), dtype=np.float32),
            "global_obs": np.ones((self.global_obs_dim,), dtype=np.float32),
            "hidden_local_vars": np.ones(
                (2, self.hidden_local_vars_dim),
                dtype=np.float32,
            ),
            "hidden_global_vars": np.ones(
                (self.hidden_global_vars_dim,),
                dtype=np.float32,
            ),
            "agent_mask": np.ones((2,), dtype=np.bool_),
        }

    def close(self) -> None:
        self.was_closed = True


class _DirectVectorEnv:
    def __init__(
            self,
            *,
            num_envs: int,
            global_obs_dim: int,
            hidden_local_vars_dim: int,
            hidden_global_vars_dim: int,
            backend: str = "numpy",
            autoreset_mode: AutoresetMode = AutoresetMode.SAME_STEP,
            final_obs_mask: tuple[bool, ...] | None = None,
            emit_final_obs: bool = True,
            pool_size: int = 8,
    ) -> None:
        self.num_envs = num_envs
        self.action_backend = backend
        self.metadata = {"autoreset_mode": autoreset_mode}
        self.pool_size = pool_size
        self.active_pool_size = pool_size
        self.was_closed = False
        self.close_calls = 0
        self.close_kwargs: list[dict[str, Any]] = []
        self.reset_arguments: list[tuple[int | list[int] | None, dict[str, Any] | None]] = []
        self.last_actions: dict[str, np.ndarray | torch.Tensor] | None = None
        self.final_obs_mask = (
            (False,) * num_envs
            if final_obs_mask is None
            else final_obs_mask
        )
        self.emit_final_obs = emit_final_obs
        self.single_observation_space = spaces.Dict({
            "local_obs": spaces.Box(-np.inf, np.inf, shape=(2, 3), dtype=np.float32),
            "global_obs": spaces.Box(-np.inf, np.inf, shape=(global_obs_dim,), dtype=np.float32),
            "hidden_local_vars": spaces.Box(
                -np.inf,
                np.inf,
                shape=(2, hidden_local_vars_dim),
                dtype=np.float32,
            ),
            "hidden_global_vars": spaces.Box(
                -np.inf,
                np.inf,
                shape=(hidden_global_vars_dim,),
                dtype=np.float32,
            ),
            "agent_mask": spaces.MultiBinary(2),
        })
        self.observation_space = batch_space(self.single_observation_space, n=num_envs)
        self.single_action_space = spaces.Dict({
            "actuators": spaces.Box(-1.0, 1.0, shape=(2, 1), dtype=np.float32),
            "connectors": spaces.Box(-1.0, 1.0, shape=(2, 1), dtype=np.float32),
        })
        self.action_space = batch_space(self.single_action_space, n=num_envs)

    def reset(
            self,
            *,
            seed: int | list[int] | None = None,
            options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray | torch.Tensor], dict[str, Any]]:
        self.reset_arguments.append((seed, options))
        return self._observations(), {}

    def step(
            self,
            actions: dict[str, np.ndarray | torch.Tensor],
    ) -> tuple[
        dict[str, np.ndarray | torch.Tensor],
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        dict[str, Any],
    ]:
        self.last_actions = actions
        observations = self._observations()
        final_obs = {
            key: value.clone() if isinstance(value, torch.Tensor) else value.copy()
            for key, value in observations.items()
        }
        bool_dtype = torch.bool if self.action_backend == "torch" else np.bool_
        float_dtype = torch.float32 if self.action_backend == "torch" else np.float32
        infos: dict[str, Any] = {
            "_final_obs": self._full((self.num_envs,), self.final_obs_mask, dtype=bool_dtype),
        }
        if self.emit_final_obs:
            infos["final_obs"] = final_obs
        return (
            observations,
            self._full((self.num_envs,), 1.0, dtype=float_dtype),
            self._full((self.num_envs,), self.final_obs_mask, dtype=bool_dtype),
            self._full((self.num_envs,), False, dtype=bool_dtype),
            infos,
        )

    def close(self, **kwargs: Any) -> None:
        self.close_calls += 1
        self.close_kwargs.append(kwargs)
        self.was_closed = True

    def get_settings(self) -> dict[str, Any]:
        return {"pool_size": self.pool_size}

    def get_swarm_pool_size(self) -> int:
        return self.pool_size

    def get_active_swarm_pool_size(self) -> int:
        return self.active_pool_size

    def set_active_swarm_pool_size(self, active_pool_size: int) -> int:
        self.active_pool_size = min(active_pool_size, self.pool_size)
        return self.active_pool_size

    def _observations(self) -> dict[str, np.ndarray | torch.Tensor]:
        return {
            key: self._full(space.shape, 1.0, dtype=(
                torch.bool if self.action_backend == "torch" else np.bool_
            ) if key == "agent_mask" else (
                torch.float32 if self.action_backend == "torch" else np.float32
            ))
            for key, space in self.observation_space.items()
        }

    def _full(
            self,
            shape: tuple[int, ...],
            value: Any,
            *,
            dtype: torch.dtype | type[np.generic],
    ) -> np.ndarray | torch.Tensor:
        if self.action_backend == "torch":
            assert isinstance(dtype, torch.dtype)
            if isinstance(value, tuple):
                return torch.tensor(value, dtype=dtype)
            return torch.full(shape, value, dtype=dtype)
        if isinstance(value, tuple):
            return np.asarray(value, dtype=dtype)
        return np.full(shape, value, dtype=dtype)


class _DeferredDirectVectorEnv(_DirectVectorEnv):
    supports_deferred_step = True

    def __init__(
            self,
            *,
            step_events: list[str],
            scenario_name: str,
            fail_begin: bool = False,
            **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.step_events = step_events
        self.scenario_name = scenario_name
        self.fail_begin = fail_begin
        self.pending_actions: dict[str, np.ndarray | torch.Tensor] | None = None

    @property
    def has_pending_step(self) -> bool:
        return self.pending_actions is not None

    def begin_step(self, actions: dict[str, np.ndarray | torch.Tensor]) -> None:
        if self.pending_actions is not None:
            raise RuntimeError("step already pending")
        self.step_events.append(f"begin:{self.scenario_name}")
        if self.fail_begin:
            raise RuntimeError("begin failed")
        self.pending_actions = actions

    def end_step(
            self,
    ) -> tuple[
        dict[str, np.ndarray | torch.Tensor],
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        np.ndarray | torch.Tensor,
        dict[str, Any],
    ]:
        if self.pending_actions is None:
            raise RuntimeError("no pending step")
        self.step_events.append(f"end:{self.scenario_name}")
        actions = self.pending_actions
        self.pending_actions = None
        return super().step(actions)


class _ScenarioIdPolicy(BasePolicy):
    def __init__(self, *, n_agents: int, action_dim: int) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.seen_scenario_ids: list[torch.Tensor] = []

    @property
    def gsde_enabled(self) -> bool:
        return False

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
            raise AssertionError("scenario_ids must be forwarded to the policy")
        self.seen_scenario_ids.append(scenario_ids.detach().clone())
        return local_obs.new_zeros(
            local_obs.shape[0],
            self.n_agents,
            self.action_dim,
        )

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {}

    def get_grad_norms(self) -> dict[str, float]:
        return {}

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unknown weights given: {weights}")

    def requires_previous_actions(self) -> bool:
        return False


def _vector_env(
        *,
        global_obs_dim: int,
        hidden_local_vars_dim: int,
        hidden_global_vars_dim: int,
        terminates: bool = True,
) -> SyncVectorEnv:
    return SyncVectorEnv(
        [lambda: _ScenarioEnv(
            global_obs_dim=global_obs_dim,
            hidden_local_vars_dim=hidden_local_vars_dim,
            hidden_global_vars_dim=hidden_global_vars_dim,
            terminates=terminates,
        )],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


def _vector_env_with_markers(markers: tuple[bool, ...]) -> SyncVectorEnv:
    return SyncVectorEnv(
        [
            lambda emit_marker=emit_marker: _ScenarioEnv(
                global_obs_dim=1,
                hidden_local_vars_dim=1,
                hidden_global_vars_dim=1,
                terminates=False,
                emit_marker=emit_marker,
            )
            for emit_marker in markers
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )


def test_multi_scenario_vector_env_pads_observations_and_preserves_terminal_ids() -> None:
    env = MultiScenarioVectorEnv({
        "wall": _vector_env(
            global_obs_dim=0,
            hidden_local_vars_dim=2,
            hidden_global_vars_dim=1,
        ),
        "payload": _vector_env(
            global_obs_dim=4,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=3,
        ),
    })
    try:
        obs, infos = env.reset(seed=1)

        assert obs["global_obs"].shape == (2, 4)
        assert obs["hidden_local_vars"].shape == (2, 2, 2)
        assert obs["hidden_global_vars"].shape == (2, 3)
        assert np.array_equal(obs["global_obs"][0], np.zeros(4, dtype=np.float32))
        assert np.array_equal(
            obs["hidden_local_vars"][1],
            np.zeros((2, 2), dtype=np.float32),
        )
        assert np.array_equal(obs["scenario_id"], np.asarray([0, 1]))
        assert np.array_equal(infos["scenario_name"], np.asarray(["wall", "payload"], dtype=object))

        actions = {
            "actuators": np.zeros((2, 2, 1), dtype=np.float32),
            "connectors": np.zeros((2, 2, 1), dtype=np.float32),
        }
        next_obs, _rewards, terminations, _truncations, step_infos = env.step(actions)

        assert np.array_equal(terminations, np.asarray([True, True]))
        assert np.array_equal(next_obs["scenario_id"], np.asarray([0, 1]))
        assert np.array_equal(step_infos["_final_obs"], np.asarray([True, True]))
        assert np.array_equal(
            np.asarray([
                final_obs["scenario_id"]
                for final_obs in step_infos["final_obs"]
            ]),
            np.asarray([0, 1]),
        )
    finally:
        env.close()


def test_multi_scenario_vector_env_handles_asynchronous_scenario_terminations() -> None:
    env = MultiScenarioVectorEnv({
        "terminating": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
        ),
        "continuing": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            terminates=False,
        ),
    })
    try:
        env.reset(seed=1)
        actions = {
            "actuators": np.zeros((2, 2, 1), dtype=np.float32),
            "connectors": np.zeros((2, 2, 1), dtype=np.float32),
        }

        _obs, _rewards, terminations, _truncations, infos = env.step(actions)

        assert np.array_equal(terminations, np.asarray([True, False]))
        assert np.array_equal(infos["_final_obs"], np.asarray([True, False]))
        assert infos["final_obs"][0]["scenario_id"] == 0
        assert infos["final_obs"][1] is None
    finally:
        env.close()


def test_multi_scenario_vector_env_rejects_missing_reported_final_observations() -> None:
    env = MultiScenarioVectorEnv({
        "broken": _DirectVectorEnv(
            num_envs=1,
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            final_obs_mask=(True,),
            emit_final_obs=False,
        ),
        "valid": _DirectVectorEnv(
            num_envs=1,
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
        ),
    })
    try:
        env.reset()
        actions = {
            "actuators": np.zeros((2, 2, 1), dtype=np.float32),
            "connectors": np.zeros((2, 2, 1), dtype=np.float32),
        }

        with pytest.raises(ValueError, match="did not provide 'final_obs'"):
            env.step(actions)
    finally:
        env.close()


def test_multi_scenario_vector_env_skips_children_with_empty_partial_reset_masks() -> None:
    first_child = _vector_env(
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
        terminates=False,
    )
    second_child = _vector_env(
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
        terminates=False,
    )
    env = MultiScenarioVectorEnv({
        "first": first_child,
        "second": second_child,
    })
    try:
        env.reset(seed=1)

        observations, infos = env.reset(options={
            "reset_mask": np.asarray([True, False], dtype=np.bool_),
        })

        assert first_child.envs[0].reset_calls == 2
        assert second_child.envs[0].reset_calls == 1
        assert np.array_equal(observations["scenario_id"], np.asarray([0, 1]))
        assert np.array_equal(
            infos["scenario_name"],
            np.asarray(["first", "second"], dtype=object),
        )
    finally:
        env.close()


def test_multi_scenario_vector_env_preserves_child_info_masks() -> None:
    env = MultiScenarioVectorEnv({
        "partially_present": _vector_env_with_markers((True, False)),
        "missing": _vector_env_with_markers((False,)),
    })
    try:
        env.reset(seed=1)
        actions = {
            "actuators": np.zeros((3, 2, 1), dtype=np.float32),
            "connectors": np.zeros((3, 2, 1), dtype=np.float32),
        }

        _obs, _rewards, _terminations, _truncations, infos = env.step(actions)

        assert np.array_equal(infos["marker"], np.asarray([7, 0, 0]))
        assert np.array_equal(infos["_marker"], np.asarray([True, False, False]))
    finally:
        env.close()


def test_scenario_ids_round_trip_through_off_policy_replay() -> None:
    vector_env = MultiScenarioVectorEnv({
        "wall": _vector_env(
            global_obs_dim=0,
            hidden_local_vars_dim=2,
            hidden_global_vars_dim=1,
        ),
        "payload": _vector_env(
            global_obs_dim=4,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=3,
        ),
    })
    env = SwarmBotsLearnEnvWrapper(vector_env)
    replay_buffer = OffPolicyReplayBuffer(
        capacity_per_env=2,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    try:
        collect_off_policy_steps(
            env=env,
            replay_buffer=replay_buffer,
            n_steps=2,
            random_actions=True,
        )
        batch = replay_buffer.get_all()

        assert batch.scenario_ids is not None
        assert batch.next_scenario_ids is not None
        assert torch.equal(batch.scenario_ids.cpu(), torch.tensor([0, 1]))
        assert torch.equal(batch.next_scenario_ids.cpu(), torch.tensor([0, 1]))
    finally:
        env.close()


def test_replay_uses_terminal_scenario_id_instead_of_reset_scenario_id() -> None:
    vector_env = MultiScenarioVectorEnv({
        "wall": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            terminates=False,
        ),
        "payload": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            terminates=False,
        ),
    })
    env = SwarmBotsLearnEnvWrapper(vector_env)
    replay_buffer = OffPolicyReplayBuffer(
        capacity_per_env=2,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    try:
        obs, _infos = env.reset()
        next_obs = {
            key: value.clone()
            for key, value in obs.items()
        }
        next_obs["scenario_id"].zero_()
        terminal_obs = {
            key: value[:1].clone()
            for key, value in obs.items()
        }
        terminal_obs["scenario_id"][0] = 1

        replay_buffer.add(
            obs=obs,
            actions=torch.zeros(2, 2, 2),
            rewards=torch.zeros(2),
            terminations=torch.tensor([True, False]),
            truncations=torch.zeros(2, dtype=torch.bool),
            next_obs=next_obs,
            terminal_obs=terminal_obs,
        )
        batch = replay_buffer.get_all()

        assert batch.next_scenario_ids is not None
        assert batch.next_scenario_ids[0].item() == 1
        assert batch.next_scenario_ids[1].item() == 0
    finally:
        env.close()


def test_scenario_ids_survive_replay_window_sampling() -> None:
    vector_env = MultiScenarioVectorEnv({
        "wall": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            terminates=False,
        ),
        "payload": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            terminates=False,
        ),
    })
    env = SwarmBotsLearnEnvWrapper(vector_env)
    replay_buffer = OffPolicyReplayBuffer(
        capacity_per_env=4,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    try:
        collect_off_policy_steps(
            env=env,
            replay_buffer=replay_buffer,
            n_steps=8,
            random_actions=True,
        )

        windows = replay_buffer.sample_episode_windows(
            batch_size=8,
            num_next_steps=2,
            replacement=True,
            generator=torch.Generator().manual_seed(5),
        )

        assert windows.scenario_ids is not None
        assert windows.next_scenario_ids is not None
        assert windows.scenario_ids.dtype == torch.long
        assert windows.next_scenario_ids.dtype == torch.long
        expected_ids = windows.scenario_ids[:, :1].expand_as(windows.scenario_ids)
        assert torch.equal(
            windows.scenario_ids[windows.train_mask],
            expected_ids[windows.train_mask],
        )
        assert torch.equal(
            windows.scenario_ids[windows.train_mask],
            windows.next_scenario_ids[windows.train_mask],
        )
    finally:
        env.close()


def test_off_policy_rollout_forwards_scenario_ids_to_policy() -> None:
    vector_env = MultiScenarioVectorEnv({
        "wall": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            terminates=False,
        ),
        "payload": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            terminates=False,
        ),
    })
    env = SwarmBotsLearnEnvWrapper(vector_env)
    replay_buffer = OffPolicyReplayBuffer(
        capacity_per_env=2,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    policy = _ScenarioIdPolicy(
        n_agents=env.n_agents,
        action_dim=env.action_space.total_agent_action_dim,
    )
    try:
        collect_off_policy_steps(
            env=env,
            replay_buffer=replay_buffer,
            n_steps=4,
            policy=policy,
            random_actions=False,
        )

        assert len(policy.seen_scenario_ids) == 2
        for seen_ids in policy.seen_scenario_ids:
            assert torch.equal(seen_ids.cpu(), torch.tensor([0, 1]))
    finally:
        env.close()


def test_scenario_replay_rejects_observations_without_ids() -> None:
    vector_env = MultiScenarioVectorEnv({
        "wall": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            terminates=False,
        ),
        "payload": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            terminates=False,
        ),
    })
    env = SwarmBotsLearnEnvWrapper(vector_env)
    replay_buffer = OffPolicyReplayBuffer(
        capacity_per_env=2,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    try:
        obs, _infos = env.reset()
        obs_without_ids = dict(obs)
        del obs_without_ids["scenario_id"]

        with pytest.raises(ValueError, match="obs must include scenario_id"):
            replay_buffer.add(
                obs=obs_without_ids,
                actions=torch.zeros(2, 2, 2),
                rewards=torch.zeros(2),
                terminations=torch.zeros(2, dtype=torch.bool),
                truncations=torch.zeros(2, dtype=torch.bool),
                next_obs=obs,
            )
    finally:
        env.close()


def test_scenario_replay_rejects_terminal_observations_without_ids() -> None:
    vector_env = MultiScenarioVectorEnv({
        "wall": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            terminates=False,
        ),
        "payload": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=1,
            terminates=False,
        ),
    })
    env = SwarmBotsLearnEnvWrapper(vector_env)
    replay_buffer = OffPolicyReplayBuffer(
        capacity_per_env=2,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    try:
        obs, _infos = env.reset()
        terminal_obs = dict(obs)
        del terminal_obs["scenario_id"]

        with pytest.raises(ValueError, match="terminal_obs must include scenario_id"):
            replay_buffer.add(
                obs=obs,
                actions=torch.zeros(2, 2, 2),
                rewards=torch.zeros(2),
                terminations=torch.tensor([True, False]),
                truncations=torch.zeros(2, dtype=torch.bool),
                next_obs=obs,
                terminal_obs=terminal_obs,
            )
    finally:
        env.close()


def test_multi_scenario_vector_env_requires_at_least_one_child() -> None:
    with pytest.raises(ValueError, match="at least one"):
        MultiScenarioVectorEnv({})


def test_multi_scenario_vector_env_requires_each_child_to_have_a_lane() -> None:
    child = _DirectVectorEnv(
        num_envs=0,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )

    with pytest.raises(ValueError, match="must contain at least one lane"):
        MultiScenarioVectorEnv({"empty": child})


def test_multi_scenario_vector_env_exposes_combined_spaces_and_metadata() -> None:
    env = MultiScenarioVectorEnv({
        "small": _DirectVectorEnv(
            num_envs=2,
            global_obs_dim=1,
            hidden_local_vars_dim=2,
            hidden_global_vars_dim=1,
        ),
        "large": _DirectVectorEnv(
            num_envs=3,
            global_obs_dim=4,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=3,
        ),
    })

    assert env.num_envs == 5
    assert env.scenario_names == ("small", "large")
    assert env.scenario_ids_by_name == {"small": 0, "large": 1}
    assert env.single_observation_space["global_obs"].shape == (4,)
    assert env.single_observation_space["hidden_local_vars"].shape == (2, 2)
    assert env.single_observation_space["hidden_global_vars"].shape == (3,)
    assert env.observation_space["scenario_id"].shape == (5,)
    assert env.scenario_observation_dims == {
        "small": {
            "global_obs": 1,
            "hidden_local_vars": 2,
            "hidden_global_vars": 1,
        },
        "large": {
            "global_obs": 4,
            "hidden_local_vars": 1,
            "hidden_global_vars": 3,
        },
    }


def test_scenario_metadata_survives_learning_wrapper_stack() -> None:
    vector_env = MultiScenarioVectorEnv({
        "small": _vector_env(
            global_obs_dim=1,
            hidden_local_vars_dim=2,
            hidden_global_vars_dim=1,
        ),
        "large": _vector_env(
            global_obs_dim=4,
            hidden_local_vars_dim=1,
            hidden_global_vars_dim=3,
        ),
    })
    learn_env = SwarmBotsLearnEnvWrapper(vector_env)
    torch_env = TorchEnvWrapper(learn_env)
    try:
        expected_dims = {
            "small": {
                "global_obs": 1,
                "hidden_local_vars": 2,
                "hidden_global_vars": 1,
            },
            "large": {
                "global_obs": 4,
                "hidden_local_vars": 1,
                "hidden_global_vars": 3,
            },
        }
        for wrapped_env in (learn_env, torch_env):
            assert wrapped_env.scenario_names == ("small", "large")
            assert wrapped_env.scenario_observation_dims == expected_dims
    finally:
        torch_env.close()


def test_multi_scenario_vector_env_offsets_integer_seeds_by_lane_group() -> None:
    first = _DirectVectorEnv(
        num_envs=2,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DirectVectorEnv(
        num_envs=3,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})

    env.reset(seed=100)

    assert first.reset_arguments == [(100, None)]
    assert second.reset_arguments == [(102, None)]


def test_multi_scenario_vector_env_slices_explicit_seed_lists() -> None:
    first = _DirectVectorEnv(
        num_envs=2,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})

    env.reset(seed=[11, 12, 13])

    assert first.reset_arguments == [([11, 12], None)]
    assert second.reset_arguments == [([13], None)]


def test_multi_scenario_vector_env_rejects_wrong_number_of_explicit_seeds() -> None:
    first = _DirectVectorEnv(
        num_envs=2,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})

    try:
        with pytest.raises(ValueError, match="Expected 3 reset seeds, got 2"):
            env.reset(seed=[10, 11])
        assert first.reset_arguments == []
        assert second.reset_arguments == []
    finally:
        env.close()


def test_multi_scenario_vector_env_rejects_seed_lists_before_resetting_unsupported_children() -> None:
    supported = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    unsupported = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    unsupported.supports_per_env_reset_seeds = False
    env = MultiScenarioVectorEnv({
        "supported": supported,
        "unsupported": unsupported,
    })

    try:
        with pytest.raises(
                ValueError,
                match="Explicit per-environment reset seeds are not supported.*unsupported",
        ):
            env.reset(seed=[10, 11])
        assert supported.reset_arguments == []
        assert unsupported.reset_arguments == []
    finally:
        env.close()


@pytest.mark.parametrize(
    "reset_mask, expected_error",
    [
        (np.asarray([True], dtype=np.bool_), "Expected reset_mask shape"),
        (np.asarray([1, 0], dtype=np.int64), "Expected reset_mask dtype"),
        ([True, False], "must be a numpy array or torch tensor"),
    ],
)
def test_multi_scenario_vector_env_validates_reset_mask(
        reset_mask: Any,
        expected_error: str,
) -> None:
    child = _DirectVectorEnv(
        num_envs=2,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"only": child})

    try:
        with pytest.raises(ValueError, match=expected_error):
            env.reset(options={"reset_mask": reset_mask})
        assert child.reset_arguments == []
    finally:
        env.close()


def test_multi_scenario_vector_env_slices_actions_without_copying_lane_groups() -> None:
    first = _DirectVectorEnv(
        num_envs=2,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})
    env.reset()
    actions = {
        "actuators": np.arange(6, dtype=np.float32).reshape(3, 2, 1),
        "connectors": np.arange(6, 12, dtype=np.float32).reshape(3, 2, 1),
    }

    env.step(actions)

    assert first.last_actions is not None
    assert second.last_actions is not None
    assert np.shares_memory(first.last_actions["actuators"], actions["actuators"])
    assert np.shares_memory(second.last_actions["connectors"], actions["connectors"])
    assert np.array_equal(first.last_actions["actuators"], actions["actuators"][:2])
    assert np.array_equal(second.last_actions["actuators"], actions["actuators"][2:])


def test_multi_scenario_vector_env_begins_all_deferred_steps_before_ending_them() -> None:
    step_events: list[str] = []
    first = _DeferredDirectVectorEnv(
        step_events=step_events,
        scenario_name="first",
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DeferredDirectVectorEnv(
        step_events=step_events,
        scenario_name="second",
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})
    actions = {
        "actuators": np.zeros((2, 2, 1), dtype=np.float32),
        "connectors": np.zeros((2, 2, 1), dtype=np.float32),
    }

    observations, rewards, terminations, truncations, _infos = env.step(actions)

    assert step_events == ["begin:first", "begin:second", "end:first", "end:second"]
    assert observations["scenario_id"].tolist() == [0, 1]
    assert rewards.tolist() == [1.0, 1.0]
    assert not np.any(terminations)
    assert not np.any(truncations)


def test_multi_scenario_vector_env_uses_sequential_steps_for_mixed_children() -> None:
    step_events: list[str] = []
    deferred = _DeferredDirectVectorEnv(
        step_events=step_events,
        scenario_name="deferred",
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    ordinary = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"deferred": deferred, "ordinary": ordinary})
    actions = {
        "actuators": np.zeros((2, 2, 1), dtype=np.float32),
        "connectors": np.zeros((2, 2, 1), dtype=np.float32),
    }

    env.step(actions)

    assert step_events == []
    assert deferred.last_actions is not None
    assert ordinary.last_actions is not None


def test_multi_scenario_vector_env_drains_started_steps_when_a_later_begin_fails() -> None:
    step_events: list[str] = []
    first = _DeferredDirectVectorEnv(
        step_events=step_events,
        scenario_name="first",
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DeferredDirectVectorEnv(
        step_events=step_events,
        scenario_name="second",
        fail_begin=True,
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})
    actions = {
        "actuators": np.zeros((2, 2, 1), dtype=np.float32),
        "connectors": np.zeros((2, 2, 1), dtype=np.float32),
    }

    with pytest.raises(RuntimeError, match="begin failed"):
        env.step(actions)

    assert step_events == ["begin:first", "begin:second", "end:first"]
    assert not first.has_pending_step


def test_multi_scenario_vector_env_supports_torch_observations_actions_and_infos() -> None:
    first = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=2,
        hidden_global_vars_dim=1,
        backend="torch",
        final_obs_mask=(True,),
    )
    second = _DirectVectorEnv(
        num_envs=2,
        global_obs_dim=3,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=2,
        backend="torch",
        final_obs_mask=(False, True),
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})
    obs, _infos = env.reset()
    actions = {
        "actuators": torch.zeros(3, 2, 1),
        "connectors": torch.zeros(3, 2, 1),
    }

    next_obs, rewards, terminations, truncations, infos = env.step(actions)

    assert all(isinstance(value, torch.Tensor) for value in obs.values())
    assert all(isinstance(value, torch.Tensor) for value in next_obs.values())
    assert isinstance(rewards, torch.Tensor)
    assert isinstance(terminations, torch.Tensor)
    assert isinstance(truncations, torch.Tensor)
    assert isinstance(infos["final_obs"], dict)
    assert torch.equal(infos["_final_obs"], torch.tensor([True, False, True]))
    assert torch.equal(infos["final_obs"]["scenario_id"], torch.tensor([0, 1, 1]))
    assert first.last_actions is not None
    assert torch.equal(first.last_actions["actuators"], actions["actuators"][:1])


def test_multi_scenario_vector_env_rejects_empty_partial_reset_before_cache_exists() -> None:
    child = _DirectVectorEnv(
        num_envs=2,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"only": child})

    with pytest.raises(ValueError, match="before that scenario has been reset"):
        env.reset(options={"reset_mask": np.zeros(2, dtype=np.bool_)})


def test_multi_scenario_vector_env_accepts_torch_partial_reset_masks() -> None:
    first = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DirectVectorEnv(
        num_envs=2,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})
    env.reset()

    obs, _infos = env.reset(options={
        "reset_mask": torch.tensor([False, True, False]),
        "custom": "kept",
    })

    assert len(first.reset_arguments) == 1
    assert len(second.reset_arguments) == 2
    second_options = second.reset_arguments[-1][1]
    assert second_options is not None
    assert np.array_equal(
        second_options["reset_mask"],
        np.asarray([True, False], dtype=np.bool_),
    )
    assert second_options["custom"] == "kept"
    assert np.array_equal(obs["scenario_id"], np.asarray([0, 1, 1]))


def test_multi_scenario_vector_env_normalizes_torch_reset_masks_for_sync_children() -> None:
    first_child = _vector_env(
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
        terminates=False,
    )
    second_child = _vector_env(
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
        terminates=False,
    )
    env = MultiScenarioVectorEnv({
        "first": first_child,
        "second": second_child,
    })
    try:
        env.reset()

        observations, _infos = env.reset(options={
            "reset_mask": torch.tensor([True, False]),
        })

        assert first_child.envs[0].reset_calls == 2
        assert second_child.envs[0].reset_calls == 1
        assert np.array_equal(observations["scenario_id"], np.asarray([0, 1]))
    finally:
        env.close()


def test_multi_scenario_vector_env_reports_settings_and_delegates_pool_controls() -> None:
    first = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DirectVectorEnv(
        num_envs=2,
        global_obs_dim=2,
        hidden_local_vars_dim=2,
        hidden_global_vars_dim=2,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})

    settings = env.get_settings()

    assert settings["scenario_ids"] == {"first": 0, "second": 1}
    assert settings["scenario_num_envs"] == {"first": 1, "second": 2}
    assert settings["scenarios"] == {
        "first": {"pool_size": 8},
        "second": {"pool_size": 8},
    }
    assert env.get_swarm_pool_size() == 8
    assert env.get_active_swarm_pool_size() == 8
    assert env.set_active_swarm_pool_size(5) == 5
    assert first.active_pool_size == 5
    assert second.active_pool_size == 5


def test_multi_scenario_vector_env_rejects_disagreeing_child_pool_values() -> None:
    first = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
        pool_size=8,
    )
    second = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
        pool_size=9,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})

    with pytest.raises(ValueError, match="different get_swarm_pool_size"):
        env.get_swarm_pool_size()


def test_multi_scenario_vector_env_closes_every_child() -> None:
    first = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})

    env.close()

    assert first.was_closed
    assert second.was_closed


def test_multi_scenario_vector_env_forwards_close_options() -> None:
    first = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})

    env.close(terminate=True)

    assert first.close_kwargs == [{"terminate": True}]
    assert second.close_kwargs == [{"terminate": True}]


def test_multi_scenario_vector_env_close_is_idempotent() -> None:
    first = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    env = MultiScenarioVectorEnv({"first": first, "second": second})

    env.close()
    env.close()

    assert env.closed
    assert first.close_calls == 1
    assert second.close_calls == 1


def test_multi_scenario_vector_env_requires_same_step_children() -> None:
    child = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
        autoreset_mode=AutoresetMode.NEXT_STEP,
    )

    with pytest.raises(ValueError, match="must use SAME_STEP"):
        MultiScenarioVectorEnv({"invalid": child})


def test_multi_scenario_vector_env_rejects_mixed_action_backends() -> None:
    numpy_child = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    torch_child = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
        backend="torch",
    )

    with pytest.raises(ValueError, match="same action backend"):
        MultiScenarioVectorEnv({"numpy": numpy_child, "torch": torch_child})


def test_multi_scenario_vector_env_rejects_reserved_child_scenario_id() -> None:
    child = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    child.single_observation_space = spaces.Dict({
        **child.single_observation_space.spaces,
        "scenario_id": spaces.Discrete(1),
    })
    child.observation_space = batch_space(child.single_observation_space, n=1)

    with pytest.raises(ValueError, match="scenario_id is reserved"):
        MultiScenarioVectorEnv({"invalid": child})


def test_multi_scenario_vector_env_rejects_different_local_observation_spaces() -> None:
    first = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second = _DirectVectorEnv(
        num_envs=1,
        global_obs_dim=1,
        hidden_local_vars_dim=1,
        hidden_global_vars_dim=1,
    )
    second.single_observation_space = spaces.Dict({
        **second.single_observation_space.spaces,
        "local_obs": spaces.Box(-np.inf, np.inf, shape=(2, 4), dtype=np.float32),
    })
    second.observation_space = batch_space(second.single_observation_space, n=1)

    with pytest.raises(ValueError, match="same local observation space"):
        MultiScenarioVectorEnv({"first": first, "second": second})
