import abc
from typing import Generic, TypeVar, Optional, Any

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.core import ActType
from mujoco import MjsBody

from swarmbots.swarm.base_swarm import BaseSwarm
import swarmbots.mujoco_utils as mj_utils
from swarmbots.swarm.swarm_connections import SwarmConnections


class BaseScenario(abc.ABC):

    def __init__(self, swarm: BaseSwarm, seed: int | None):
        self.swarm = swarm
        self.rng = np.random.default_rng(seed)

        self.spec, self.model, self.data = self.build_scenario()

        unit_prefixes = self.swarm.config.unit_prefixes
        self._qpos_indices = np.array(
            [mj_utils.qpos_indices_for_prefix(self.model, prefix) for prefix in unit_prefixes],
            dtype=int,
        )
        self._qvel_indices = np.array(
            [mj_utils.dof_indices_for_prefix(self.model, prefix) for prefix in unit_prefixes],
            dtype=int,
        )
        self._ctrl_indices = np.array(
            [mj_utils.ctrl_indices_for_prefix(self.model, prefix) for prefix in unit_prefixes],
            dtype=int,
        )

        num_units = self.swarm.config.num_units
        num_connectors = self.swarm.config.limbs_per_unit
        self._connector_body_indices = np.zeros((num_units, num_connectors), dtype=int)
        for u in range(num_units):
            for c in range(num_connectors):
                conn_body_name = self.swarm.config.get_connector_name(u, c)
                self._connector_body_indices[u, c] = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, conn_body_name
                )

        self._eq_indices = np.full((num_units, num_connectors, num_units, num_connectors), -1, dtype=int)

        for u1 in range(num_units - 1):
            for u2 in range(u1 + 1, num_units):
                for c1 in range(num_connectors):
                    for c2 in range(num_connectors):
                        eq_name = self.swarm.config.get_eq_name(u1, c1, u2, c2)
                        eq_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
                        self._eq_indices[u1, c1, u2, c2] = eq_id
                        self._eq_indices[u2, c2, u1, c1] = eq_id

        self._dummy_state, self._dummy_connections = self.reset_scenario(self.model, self.data)

    @abc.abstractmethod
    def create_scenario_spec(self) -> mujoco.MjSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate_step(
            self,
            action: dict[str, Any],
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

        swarm_spec = self.swarm.create_swarm_spec()

        swarm_site = worldbody.add_site(pos=self.get_swarm_start_location(), name='swarm_site')
        spec.attach(swarm_spec, '', site=swarm_site)

        scenario_site = worldbody.add_site(pos=[0, 0, 0], name='scenario_site')
        spec.attach(self.create_scenario_spec(), '', site=scenario_site)

        model = spec.compile()
        data = mujoco.MjData(model)

        return spec, model, data

    def get_swarm_start_location(self):
        return np.array([0.0, 0.0, 1.0])

    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict, SwarmConnections]:
        mujoco.mj_resetData(self.model, self.data)

        state = dict()
        connections = self.swarm.reset_swarm(model, data, self.rng)

        data.eq_active[:] = 0
        # TODO activate eq constraints

        return state, connections

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> np.ndarray:
        qpos = data.qpos[self._qpos_indices]
        qvel = data.qvel[self._qvel_indices]
        return np.concatenate([qpos, qvel], axis=1)

    def apply_action(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            action: dict[str, Any],
            action_scale: float,
            state: dict,
            connections: SwarmConnections
    ) -> None:
        data.ctrl[self._ctrl_indices] = action['actuators'] * action_scale

        connector_action = np.asarray(action['connectors'], dtype=bool)

        currently_active, active_edges, active_angles = connections.get_active_connections()

        newly_activated = np.logical_and(connector_action, np.logical_not(currently_active))

        # TODO


        newly_deactivated = np.logical_and(np.logical_not(connector_action), currently_active)
        for deactivated_unit, deactivated_connector in np.stack(newly_deactivated).T:
            self.disconnect(data, connections, deactivated_unit, deactivated_connector)


    def get_obs_shape(self) -> tuple[int, ...]:
        return self.get_obs(self.model, self.data, self._dummy_state, self._dummy_connections).shape

    def get_obs_space(self):
        return spaces.Box(
            low=-np.inf, high=np.inf, shape=self.get_obs_shape(), dtype=np.float32
        )

    def get_actuator_action_shape(self):
        return self._ctrl_indices.shape

    def get_connector_action_shape(self):
        return self.swarm.config.num_units, self.swarm.config.limbs_per_unit

    def get_action_space(self):
        return spaces.Dict({
            'actuators': spaces.Box(
                low=-1, high=1, shape=self.get_actuator_action_shape(), dtype=np.float32
            ),
            'connectors': spaces.MultiBinary(self.get_connector_action_shape())
        })

    def disconnect(
            self,
            data: mujoco.MjData,
            connections: SwarmConnections,
            unit: int,
            connector: int,
    ):
        other_unit, other_connector = connections.disconnect(unit, connector)
        data.eq_active[self._eq_indices[unit, connector, other_unit, other_connector]] = False
