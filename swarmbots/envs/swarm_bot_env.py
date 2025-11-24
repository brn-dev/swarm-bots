from typing import Any, SupportsFloat, TypeVar

import gymnasium
import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType, ObsType, RenderFrame
from mujoco import MjsBody

from swarmbots.scenarios.base_scenario import BaseScenario
from swarmbots.swarm.base_swarm import BaseSwarm

TState = TypeVar("TState")


class SwarmBotEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array", "depth_array"]}

    def __init__(
        self,
        swarm: BaseSwarm,
        scenario: BaseScenario[TState],
        duration: float = 5.0,
        physics_steps_per_step: int = 1,
        action_scale: float = 1.0,
        render_mode: str | None = None,
        width: int = 640,
        height: int = 480,
    ):
        self.swarm = swarm
        self.scenario = scenario
        self.duration = duration
        self.physics_steps_per_step = physics_steps_per_step
        self.action_scale = action_scale
        self.render_mode = render_mode
        self.width = width
        self.height = height

        spec = mujoco.MjSpec()
        worldbody: MjsBody = spec.worldbody


        swarm_site = worldbody.add_site(pos=scenario.get_start_location(), name='swarm_site')
        spec.attach(swarm.build_swarm_spec(), '', site=swarm_site)

        scenario_site = worldbody.add_site(pos=[0, 0, 0], name='scenario_site')
        spec.attach(scenario.build_scenario_spec(), '', site=scenario_site)

        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.agent_prefixes = swarm.get_unit_prefixes()
        self.n_agents = len(self.agent_prefixes)

        self.scenario_state: TState | None = None

        dummy_obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=dummy_obs.shape, dtype=np.float32
        )

        action_dim = self.swarm.get_n_actions_per_agent(self.model)
        self.action_space = spaces.Box(
            low=-1, high=1, shape=(self.n_agents, action_dim), dtype=np.float32
        )

        self._renderer = None

    def _get_obs(self) -> np.ndarray:
        return self.scenario.modify_obs(
            self.model,
            self.data,
            self.scenario_state,
            self.swarm.get_obs(self.model, self.data)
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[ObsType, dict[str, Any]]:
        super().reset(seed=seed)

        mujoco.mj_resetData(self.model, self.data)
        self.scenario_state = self.scenario.reset_scenario(self.model, self.data)
        self.swarm.reset_swarm(self.model, self.data)

        mujoco.mj_forward(self.model, self.data)

        return self._get_obs(), {}

    def step(
        self, action: ActType
    ) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        action = np.array(action)
        self.swarm.apply_action(self.model, self.data, action, self.action_scale)

        for _ in range(self.physics_steps_per_step):
            mujoco.mj_step(self.model, self.data)

        if np.isnan(self.data.qpos).any() or np.isnan(self.data.qvel).any():
            return (
                np.zeros_like(self._get_obs()),
                -100.0,
                True,
                False,
                {"error": "simulation_unstable"},
            )

        self.scenario_state, reward, terminated = self.scenario.scenario_step(
            self.model, self.data, self.scenario_state
        )

        truncated = self.data.time >= self.duration

        if self.render_mode == "human":
            self.render()

        return self._get_obs(), reward, terminated, truncated, {}

    def render(self) -> RenderFrame | list[RenderFrame] | None:
        if self.render_mode is None:
            return None

        if self._renderer is None:
            self._renderer = mujoco.Renderer(
                self.model, height=self.height, width=self.width
            )

        self._renderer.update_scene(self.data)

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
