from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from gymnasium import spaces
import mujoco.mjx as mjx

from swarmbots.mjx_env.scenarios.mjx_base_scenario import MjxBaseScenario
from swarmbots.mjx_env.types import MjxActDict, MjxEnvState, MjxObsDict


class MjxStepInfo(NamedTuple):
    progress_reward: jax.Array
    guidance_reward: jax.Array
    simulation_unstable: jax.Array


class MjxSwarmBotsEnv:
    metadata = {"render_modes": []}

    def __init__(
        self,
        scenario: MjxBaseScenario,
        episode_length: int = 500,
        action_repeat: int = 15,
        simulation_unstable_reward: float = -1.0,
    ) -> None:
        if episode_length <= 0:
            raise ValueError(f"Expected episode_length > 0, got {episode_length}")
        if action_repeat <= 0:
            raise ValueError(f"Expected action_repeat > 0, got {action_repeat}")
        self.scenario = scenario
        self.episode_length = int(episode_length)
        self.action_repeat = int(action_repeat)
        self.simulation_unstable_reward = float(simulation_unstable_reward)
        self.observation_space: spaces.Dict = scenario.get_obs_space()
        self.action_space: spaces.Dict = scenario.get_action_space()
        self.jit_reset = jax.jit(self.reset)
        self.jit_step = jax.jit(self.step)

    def get_settings(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.get_settings(),
            "episode_length": self.episode_length,
            "action_repeat": self.action_repeat,
            "simulation_unstable_reward": self.simulation_unstable_reward,
        }

    def reset(self, key: jax.Array) -> tuple[MjxEnvState, MjxObsDict]:
        state = self.scenario.reset(key)
        return state, self.scenario.get_obs(state)

    def step(
        self,
        state: MjxEnvState,
        action: MjxActDict,
    ) -> tuple[MjxEnvState, MjxObsDict, jax.Array, jax.Array, jax.Array, MjxStepInfo]:
        state = self.scenario.apply_action(state, action)
        data = jax.lax.fori_loop(
            0,
            self.action_repeat,
            lambda _, loop_data: mjx.step(self.scenario.mjx_model, loop_data),
            state.data,
        )
        state = state._replace(data=data)
        state, reward, terminated = self.scenario.evaluate_step(state, action)
        current_step = state.current_step + 1
        truncated = current_step >= self.episode_length
        state = state._replace(current_step=current_step)

        obs = self.scenario.get_obs(state)
        finite = jnp.isfinite(state.data.qpos).all() & jnp.isfinite(state.data.qvel).all()
        reward = jnp.where(finite, reward, self.simulation_unstable_reward)
        terminated = terminated | (~finite)
        obs = _where_obs(finite, obs, self._unstable_obs(state))
        info = MjxStepInfo(
            progress_reward=state.weighted_progress_reward,
            guidance_reward=state.weighted_guidance_reward,
            simulation_unstable=~finite,
        )
        return state, obs, reward, terminated, truncated, info

    def step_masked(
        self,
        state: MjxEnvState,
        action: MjxActDict,
        skip_step: jax.Array,
    ) -> tuple[MjxEnvState, MjxObsDict, jax.Array, jax.Array, jax.Array, MjxStepInfo]:
        def do_skip(_: None) -> tuple[MjxEnvState, MjxObsDict, jax.Array, jax.Array, jax.Array, MjxStepInfo]:
            zero_float = jnp.array(0.0, dtype=jnp.float32)
            false = jnp.array(False)
            return (
                state,
                self.scenario.get_obs(state),
                zero_float,
                false,
                false,
                MjxStepInfo(
                    progress_reward=zero_float,
                    guidance_reward=zero_float,
                    simulation_unstable=false,
                ),
            )

        return jax.lax.cond(skip_step, do_skip, lambda _: self.step(state, action), operand=None)

    def batched(self, num_envs: int) -> "MjxBatchedSwarmBotsEnv":
        return MjxBatchedSwarmBotsEnv(self, num_envs=num_envs)

    def _unstable_obs(self, state: MjxEnvState) -> MjxObsDict:
        zero_obs: MjxObsDict = {
            key: jnp.zeros(space.shape, dtype=space.dtype)
            for key, space in self.observation_space.items()
            if key != "agent_mask"
        }
        if "agent_mask" in self.observation_space.spaces:
            zero_obs["agent_mask"] = state.units_active_mask
        return zero_obs


class MjxBatchedSwarmBotsEnv:
    def __init__(self, env: MjxSwarmBotsEnv, num_envs: int) -> None:
        if num_envs <= 0:
            raise ValueError(f"Expected num_envs > 0, got {num_envs}")
        self.single_env = env
        self.num_envs = int(num_envs)
        self.single_observation_space = env.observation_space
        self.single_action_space = env.action_space
        self.observation_space = spaces.Dict({key: _batch_space(space, self.num_envs) for key, space in env.observation_space.items()})
        self.action_space = spaces.Dict({key: _batch_space(space, self.num_envs) for key, space in env.action_space.items()})
        self.reset = jax.jit(jax.vmap(env.reset))
        self.step = jax.jit(jax.vmap(env.step, in_axes=(0, {"actuators": 0, "connectors": 0})))
        self.step_masked = jax.jit(jax.vmap(env.step_masked, in_axes=(0, {"actuators": 0, "connectors": 0}, 0)))


def _where_obs(mask: jax.Array, obs: MjxObsDict, fallback: MjxObsDict) -> MjxObsDict:
    return {
        key: jnp.where(mask, value, fallback[key])
        for key, value in obs.items()
    }


def _batch_space(space: spaces.Space, n: int) -> spaces.Space:
    if isinstance(space, spaces.Box):
        shape = (n, *space.shape)
        return spaces.Box(
            low=np.broadcast_to(space.low, shape),
            high=np.broadcast_to(space.high, shape),
            shape=shape,
            dtype=space.dtype,
        )
    if isinstance(space, spaces.MultiBinary):
        return spaces.MultiBinary((n, *space.shape))
    raise TypeError(f"Unsupported space type for batching: {type(space).__name__}")
