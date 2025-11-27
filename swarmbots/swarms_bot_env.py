from typing import Any, SupportsFloat, TypeVar

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
from mujoco import MjvOption

from swarmbots.scenarios.base_scenario import BaseScenario


class SwarmBotsEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        scenario: BaseScenario,
        duration: float = 10.0,
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
        self.duration = duration
        self.physics_steps_per_step = physics_steps_per_step
        self.action_scale = action_scale
        self.render_mode = render_mode
        self.width = width
        self.height = height
        self.camera = camera
        self.scene_option = scene_option

        self.scenario = scenario
        self.model, self.data = self.scenario.model, self.scenario.data

        self.scenario_state: dict | None = None

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=scenario.get_obs_shape(), dtype=np.float32
        )

        self.action_space = spaces.Box(
            low=-1, high=1, shape=scenario.get_action_shape(), dtype=np.float32
        )

        self._renderer = None

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.scenario_state = self.scenario.reset_scenario(self.model, self.data)

        return self.scenario.get_obs(self.model, self.data, self.scenario_state), {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.array(action)
        self.scenario.apply_action(
            self.model,
            self.data,
            action,
            self.action_scale,
            self.scenario_state
        )

        for _ in range(self.physics_steps_per_step):
            mujoco.mj_step(self.model, self.data, self.action_repeat)

        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return (
                np.zeros_like(self.scenario.get_obs(self.model, self.data, self.scenario_state)),
                -100.0,
                True,
                False,
                {"error": "simulation_unstable"},
            )

        reward, terminated = self.scenario.evaluate_step(
            action, self.model, self.data, self.scenario_state
        )

        truncated = self.data.time >= self.duration

        if self.render_mode == "human":
            self.render()

        return self.scenario.get_obs(self.model, self.data, self.scenario_state), reward, terminated, truncated, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )

        self._renderer.update_scene(self.data, camera=self.camera, scene_option=self.scene_option)

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
