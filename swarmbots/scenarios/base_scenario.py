import abc
from typing import Generic, TypeVar, Optional

import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.swarm.base_swarm import BaseSwarm
import swarmbots.mujoco_utils as mj_utils

class BaseScenario(abc.ABC):

    def __init__(self, swarm: BaseSwarm):
        self.swarm = swarm
        self.rng = swarm.rng

        self.spec, self.model, self.data = self.build_scenario()

        self._unit_prefixes = self.swarm.get_unit_prefixes()
        self._qpos_indices = np.array(
            [mj_utils.qpos_indices_for_prefix(self.model, prefix) for prefix in self._unit_prefixes],
            dtype=int,
        )
        self._qvel_indices = np.array(
            [mj_utils.dof_indices_for_prefix(self.model, prefix) for prefix in self._unit_prefixes],
            dtype=int,
        )
        self._ctrl_indices = np.array(
            [mj_utils.ctrl_indices_for_prefix(self.model, prefix) for prefix in self._unit_prefixes],
            dtype=int,
        )

        self._dummy_state = self.reset_scenario(self.model, self.data)

    @abc.abstractmethod
    def create_scenario_spec(self) -> mujoco.MjSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate_step(
            self,
            action: np.ndarray,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
    ) -> tuple[float, bool]:
        """
        :return: (new_state, reward for step, done)
        """
        raise NotImplementedError()

    def build_scenario(self) -> tuple[mujoco.MjSpec, mujoco.MjModel, mujoco.MjData]:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody

        swarm_site = worldbody.add_site(pos=self.get_swarm_start_location(), name='swarm_site')
        spec.attach(self.swarm.create_swarm_spec(), '', site=swarm_site)

        scenario_site = worldbody.add_site(pos=[0, 0, 0], name='scenario_site')
        spec.attach(self.create_scenario_spec(), '', site=scenario_site)

        model = spec.compile()
        data = mujoco.MjData(model)

        return spec, model, data

    def get_swarm_start_location(self):
        return np.array([0.0, 0.0, 1.0])

    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
        return self.swarm.reset_swarm(model, data)

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict
    ) -> np.ndarray:
        qpos = data.qpos[self._qpos_indices]
        qvel = data.qvel[self._qvel_indices]
        return np.concatenate([qpos, qvel], axis=1)

    def apply_action(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            action: np.ndarray,
            action_scale: float,
            state: dict
    ) -> None:
        data.ctrl[self._ctrl_indices] = action * action_scale

    def get_obs_shape(self) -> tuple[int, ...]:
        return self.get_obs(self.model, self.data, self._dummy_state).shape

    def get_action_shape(self):
        return self._ctrl_indices.shape

