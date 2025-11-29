from typing import Any, SupportsFloat

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.core import ObsType, RenderFrame
from mujoco import MjvOption

from swarmbots.scenarios.base_scenario import BaseScenario
from swarmbots.swarm.swarm_connections import SwarmConnections


class SwarmBotsEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        scenario: BaseScenario,
        episode_length: int = 500,
        physics_steps_per_step: int = 1,
        action_scale: float = 1.0,
        action_repeat: int = 15,
        render_mode: str | None = None,
        width: int = 640,
        height: int = 480,
        camera: int = 0,
        scene_option: MjvOption = None,
    ):
        self.action_repeat = action_repeat
        self.episode_length = episode_length
        self.physics_steps_per_step = physics_steps_per_step
        self.action_scale = action_scale
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.camera = camera
        self.scene_option = scene_option

        self.current_step = 0

        self.scenario = scenario
        self.model, self.data = self.scenario.model, self.scenario.data

        self.scenario_state: dict | None = None
        self.swarm_connections: SwarmConnections | None = None

        self.observation_space = scenario.get_obs_space()
        self.action_space = scenario.get_action_space()

        self._renderer = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)
        self.current_step = 0
        self.scenario_state, self.swarm_connections = self.scenario.reset_scenario(self.model, self.data)

        return self.scenario.get_obs(self.model, self.data, self.scenario_state, self.swarm_connections), {}

    def step(
        self, action: dict[str, Any]
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
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
            return (
                np.zeros_like(self.scenario.get_obs(self.model, self.data, self.scenario_state, self.swarm_connections)),
                -100.0,
                True,
                False,
                {"error": "simulation_unstable", "mj_warning.lastinfo": mj_warning.lastinfo},
            )

        reward, terminated = self.scenario.evaluate_step(
            action, self.model, self.data, self.scenario_state
        )

        self.current_step += 1
        truncated = self.current_step >= self.episode_length

        if self.render_mode == "human":
            self.render()

        obs = self.scenario.get_obs(self.model, self.data, self.scenario_state, self.swarm_connections).copy()

        return obs, reward, terminated, truncated, {}

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
        self._renderer.update_scene(self.data, camera=self.camera, scene_option=scene_option)

        if self.render_mode in ("human", "rgb_array"):
            return self._renderer.render()
        elif self.render_mode == "depth_array":
            self._renderer.enable_depth_rendering()
            depth = self._renderer.render()
            self._renderer.disable_depth_rendering()
            return depth

        return None

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
