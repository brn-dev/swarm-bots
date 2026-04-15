from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, VectorEnv

from swarmbots.mjx_env.mjx_swarm_bots_env import MjxBatchedSwarmBotsEnv, MjxSwarmBotsEnv
from swarmbots.mjx_env.types import MjxEnvState, MjxObsDict


class MjxGymVectorEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}

    def __init__(
        self,
        env: MjxSwarmBotsEnv,
        num_envs: int,
        *,
        first_episode_lengths: np.ndarray | None = None,
        seed: int = 0,
    ) -> None:
        super().__init__()
        self.single_env = env
        self.batched_env: MjxBatchedSwarmBotsEnv = env.batched(num_envs)
        self.num_envs = int(num_envs)
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.observation_space = self.batched_env.observation_space
        self.action_space = self.batched_env.action_space
        self._batched_get_obs = jax.jit(jax.vmap(env.scenario.get_obs))
        self.metadata = {"autoreset_mode": AutoresetMode.NEXT_STEP}
        self.render_mode = None
        self.closed = False
        self._key = jax.random.PRNGKey(seed)
        self._states: MjxEnvState | None = None
        self._autoreset_envs = np.zeros((self.num_envs,), dtype=bool)

        if first_episode_lengths is None:
            self._first_step_offsets = np.zeros((self.num_envs,), dtype=np.int32)
        else:
            first_episode_lengths = np.asarray(first_episode_lengths, dtype=np.int32)
            if first_episode_lengths.shape != (self.num_envs,):
                raise ValueError(
                    f"Expected first_episode_lengths shape ({self.num_envs},), got {first_episode_lengths.shape}"
                )
            if (first_episode_lengths <= 0).any() or (first_episode_lengths > env.episode_length).any():
                raise ValueError("first_episode_lengths must be in [1, episode_length]")
            self._first_step_offsets = env.episode_length - first_episode_lengths
        self._first_reset_pending = True

    def reset(
        self,
        *,
        seed: int | list[int | None] | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        reset_mask = _normalize_reset_mask(options=options, num_envs=self.num_envs)
        obs = self._reset_states(reset_mask=reset_mask, seed=seed)
        return _obs_to_numpy(obs), {}

    def step(
        self,
        actions: dict[str, np.ndarray],
    ) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
        if self._states is None:
            raise RuntimeError("reset() must be called before step().")

        reset_mask = self._autoreset_envs.copy()
        if reset_mask.any():
            self._reset_states(reset_mask=reset_mask)

        action_jax = {
            "actuators": jnp.asarray(actions["actuators"], dtype=jnp.float32),
            "connectors": jnp.asarray(actions["connectors"]).astype(bool),
        }
        self._states, obs, rewards, terminations, truncations, info = self.batched_env.step_masked(
            self._states,
            action_jax,
            jnp.asarray(reset_mask),
        )
        rewards_np = np.array(jax.device_get(rewards), dtype=np.float32, copy=True)
        term_np = np.array(jax.device_get(terminations), dtype=bool, copy=True)
        trunc_np = np.array(jax.device_get(truncations), dtype=bool, copy=True)
        self._autoreset_envs = np.logical_or(term_np, trunc_np)
        infos = {
            "progress_reward": np.array(jax.device_get(info.progress_reward), dtype=np.float32, copy=True),
            "_progress_reward": np.ones((self.num_envs,), dtype=bool),
            "guidance_reward": np.array(jax.device_get(info.guidance_reward), dtype=np.float32, copy=True),
            "_guidance_reward": np.ones((self.num_envs,), dtype=bool),
            "simulation_unstable": np.array(jax.device_get(info.simulation_unstable), dtype=bool, copy=True),
            "_simulation_unstable": np.ones((self.num_envs,), dtype=bool),
        }
        return _obs_to_numpy(obs), rewards_np, term_np, trunc_np, infos

    def close_extras(self, **kwargs: Any) -> None:
        _ = kwargs
        self._states = None

    def _next_keys(self) -> jax.Array:
        self._key, subkey = jax.random.split(self._key)
        return jax.random.split(subkey, self.num_envs)

    def _keys_from_seed(self, seed: int | list[int | None]) -> jax.Array:
        if isinstance(seed, int):
            return jax.random.split(jax.random.PRNGKey(seed), self.num_envs)
        seed_values = np.asarray([0 if value is None else int(value) for value in seed], dtype=np.uint32)
        if seed_values.shape != (self.num_envs,):
            raise ValueError(f"Expected {self.num_envs} seeds, got {seed_values.shape[0]}")
        return jax.vmap(jax.random.PRNGKey)(jnp.asarray(seed_values))

    def _reset_states(
        self,
        *,
        reset_mask: np.ndarray,
        seed: int | list[int | None] | None = None,
    ) -> MjxObsDict:
        if not reset_mask.any():
            if self._states is None:
                raise RuntimeError("reset() must be called before step().")
            return self._batched_get_obs(self._states)

        keys = self._keys_from_seed(seed) if seed is not None else self._next_keys()
        if seed is not None:
            self._key = jax.random.fold_in(keys[0], 1)
        new_states, new_obs = self.batched_env.reset(keys)
        if self._first_reset_pending:
            new_states = new_states._replace(current_step=jnp.asarray(self._first_step_offsets, dtype=jnp.int32))
            new_obs = self._batched_get_obs(new_states)
            self._first_reset_pending = False

        if self._states is None or reset_mask.all():
            self._states = new_states
            obs = new_obs
        else:
            mask = jnp.asarray(reset_mask)
            self._states = jax.tree.map(_select_reset(mask), self._states, new_states)
            obs = self._batched_get_obs(self._states)

        self._autoreset_envs[reset_mask] = False
        return obs


def _obs_to_numpy(obs: MjxObsDict) -> dict[str, np.ndarray]:
    return {key: np.array(jax.device_get(value), copy=True) for key, value in obs.items()}


def _normalize_reset_mask(options: dict[str, Any] | None, num_envs: int) -> np.ndarray:
    reset_mask = None if options is None else options.get("reset_mask")
    if reset_mask is None:
        return np.ones((num_envs,), dtype=bool)
    reset_mask = np.asarray(reset_mask, dtype=bool)
    if reset_mask.shape != (num_envs,):
        raise ValueError(f"Expected reset_mask shape ({num_envs},), got {reset_mask.shape}")
    return reset_mask


def _select_reset(mask: jax.Array):
    def select(old_value: jax.Array, new_value: jax.Array) -> jax.Array:
        if not hasattr(old_value, "shape") or old_value.shape == ():
            return new_value
        reshape = (mask.shape[0],) + (1,) * (old_value.ndim - 1)
        return jnp.where(mask.reshape(reshape), new_value, old_value)

    return select
