from typing import Any, SupportsFloat, Literal, Iterable

import gymnasium
import mujoco
import numpy as np
from gymnasium.core import RenderFrame
from mujoco import MjvOption

from swarmbots.mj_env.scenarios.base_scenario import BaseScenario, SwarmActDict, SwarmObsDict
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections


class SwarmBotsEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        scenario: BaseScenario,
        episode_length: int = 500,
        physics_steps_per_step: int = 1,
        actuator_strength: float = 3.0,
        action_repeat: int = 15,
        render_mode: str | None = None,
        width: int = 640,
        height: int = 480,
        camera: int | list[int] | Literal['all'] = 'all',
        scene_option: MjvOption = None,
        simulation_unstable_reward: float = -0.1,
    ):
        self.action_repeat = action_repeat
        self.episode_length = episode_length
        self.physics_steps_per_step = physics_steps_per_step
        self.action_scale = actuator_strength
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.cameras = list(camera) if isinstance(camera, list) else [camera]
        self.scene_option = scene_option
        self.simulation_unstable_reward = simulation_unstable_reward

        self.current_step = 0

        self.scenario = scenario
        self.model, self.data = self.scenario.build()

        self.scenario_state: dict | None = None
        self.swarm_connections: SwarmConnections | None = None

        self.observation_space = scenario.get_obs_space()
        self.action_space = scenario.get_action_space()

        if isinstance(camera, int):
            self.cameras = [camera]
        if camera == 'all':
            self.cameras = list(range(self.scenario.swarm.config.num_units))
        elif isinstance(camera, Iterable):
            self.cameras = list(camera)

        self._renderer = None

        self._zeros_obs: SwarmObsDict = {
            'global_obs': np.zeros(self.observation_space['global_obs'].shape),
            'local_obs': np.zeros(self.observation_space['local_obs'].shape)
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[SwarmObsDict, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.scenario_state, self.swarm_connections = self.scenario.reset_scenario(self.model, self.data)

        return self.scenario.get_obs(self.model, self.data, self.scenario_state, self.swarm_connections), {}

    def step(
        self, action: SwarmActDict
    ) -> tuple[SwarmObsDict, SupportsFloat, bool, bool, dict[str, Any]]:
        self.scenario.apply_action(
            model=self.model,
            data=self.data,
            action=action,
            action_scale=self.action_scale,
            state=self.scenario_state,
            connections=self.swarm_connections,
        )

        for _ in range(self.physics_steps_per_step):
            mujoco.mj_step(self.model, self.data, self.action_repeat)

        mj_warning: mujoco.MjWarningStat = self.data.warning[mujoco.mjtWarning.mjWARN_BADQACC]
        if mj_warning.number > 0 or np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            print('Simulation Unstable!')
            return (
                self._zeros_obs.copy(),
                self.simulation_unstable_reward,
                True,
                False,
                {"error": "simulation_unstable", "mj_warning.lastinfo": mj_warning.lastinfo},
            )

        reward, terminated = self.scenario.evaluate_step(
            action, self.model, self.data, self.scenario_state, self.swarm_connections
        )

        self.current_step += 1
        truncated = self.current_step >= self.episode_length

        if self.render_mode == "human":
            self.render()

        obs = self.scenario.get_obs(self.model, self.data, self.scenario_state, self.swarm_connections).copy()

        return obs, reward, terminated, truncated, self.scenario_state

    def render(
            self,
            scene_option: mujoco.MjvOption = None
    ) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )

        scene_option = scene_option or self.scene_option

        frames = []

        if self.render_mode == "depth_array":
            self._renderer.enable_depth_rendering()

        for camera in self.cameras:
            self._renderer.update_scene(self.data, camera=camera, scene_option=scene_option)
            frames.append(self._renderer.render())

        if self.render_mode == "depth_array":
                self._renderer.disable_depth_rendering()

        return np.concatenate(frames, axis=1)

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
