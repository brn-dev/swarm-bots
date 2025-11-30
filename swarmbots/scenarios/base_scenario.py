import abc
from typing import Any

import mujoco
import numpy as np
from gymnasium import spaces
from mujoco import MjsBody

import swarmbots.mujoco_utils as mj_utils
from swarmbots.swarm.base_swarm import BaseSwarm
from swarmbots.swarm.swarm_connections import SwarmConnections


class BaseScenario(abc.ABC):

    def __init__(
        self,
        swarm: BaseSwarm,
        seed: int | None,
    ):
        self.rng = np.random.default_rng(seed)

        self.swarm = swarm
        self.connection_dist_threshold = swarm.config.connection_dist_threshold
        self.connection_angle_threshold = swarm.config.connection_angle_threshold

        self.spec = self.create_scenario_spec()
        self.dummy_model, self.dummy_data = self.build()

        unit_prefixes = self.swarm.config.unit_prefixes
        self._qpos_indices = np.array(
            [mj_utils.qpos_indices_for_prefix(self.dummy_model, prefix) for prefix in unit_prefixes],
            dtype=int,
        )
        self._qvel_indices = np.array(
            [mj_utils.dof_indices_for_prefix(self.dummy_model, prefix) for prefix in unit_prefixes],
            dtype=int,
        )
        self._ctrl_indices = np.array(
            [mj_utils.ctrl_indices_for_prefix(self.dummy_model, prefix) for prefix in unit_prefixes],
            dtype=int,
        )

        num_units = self.swarm.config.num_units
        num_connectors = self.swarm.config.limbs_per_unit
        self._connector_body_indices = np.zeros((num_units, num_connectors), dtype=int)
        for u in range(num_units):
            for c in range(num_connectors):
                conn_body_name = self.swarm.config.get_connector_name(u, c)
                self._connector_body_indices[u, c] = mujoco.mj_name2id(
                    self.dummy_model, mujoco.mjtObj.mjOBJ_BODY, conn_body_name
                )

        self._eq_indices = np.full((num_units, num_connectors, num_units, num_connectors), -1, dtype=int)

        for u1 in range(num_units - 1):
            for u2 in range(u1 + 1, num_units):
                for c1 in range(num_connectors):
                    for c2 in range(num_connectors):
                        eq_name = self.swarm.config.get_eq_name(u1, c1, u2, c2)
                        eq_id = mujoco.mj_name2id(self.dummy_model, mujoco.mjtObj.mjOBJ_EQUALITY, eq_name)
                        self._eq_indices[u1, c1, u2, c2] = eq_id
                        self._eq_indices[u2, c2, u1, c1] = eq_id

        self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)

    @abc.abstractmethod
    def _create_scenario_spec(self) -> mujoco.MjSpec:
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

    def create_scenario_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: MjsBody = spec.worldbody

        swarm_spec = self.swarm.create_swarm_spec()

        swarm_site = worldbody.add_site(pos=self.get_swarm_start_location(), name='swarm_site')
        spec.attach(swarm_spec, '', site=swarm_site)

        scenario_site = worldbody.add_site(pos=[0, 0, 0], name='scenario_site')
        spec.attach(self._create_scenario_spec(), '', site=scenario_site)

        return spec

    def build(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        model = self.spec.compile()
        data = mujoco.MjData(model)
        return model, data

    def get_swarm_start_location(self):
        return np.array([0.0, 0.0, 1.0])

    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict, SwarmConnections]:
        mujoco.mj_resetData(model, data)

        state = dict()
        connections = self.swarm.reset_swarm(model, data, self.rng)

        for (u1, c1, u2, c2), angle in zip(*connections.get_active_connections()):
            self._activate_equality_constraint(model, data, u1, c1, u2, c2, angle)

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
        # todo connectors
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

        currently_active = connections.get_is_active_mask()

        newly_activated = np.logical_and(connector_action, np.logical_not(currently_active))
        newly_deactivated = np.logical_and(np.logical_not(connector_action), currently_active)

        num_connectors_successfully_connected, num_connectors_unsuccessfully_connected = self.try_connect(
            model,
            data,
            connections,
            newly_activated
        )
        state['num_connectors_successfully_connected'] = num_connectors_successfully_connected
        state['num_connectors_unsuccessfully_connected'] = num_connectors_unsuccessfully_connected

        num_connectors_disconnected = self.disconnect(data, connections, newly_deactivated)
        state['num_connectors_disconnected'] = num_connectors_disconnected


    def get_obs_shape(self) -> tuple[int, ...]:
        return self.get_obs(self.dummy_model, self.dummy_data, self._dummy_state, self._dummy_connections).shape

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

    def try_connect(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            connections: SwarmConnections,
            newly_activated: np.ndarray
    ):
        activated_indices = np.stack(np.where(newly_activated)).T
        activated_indices = np.concatenate((
            np.arange(len(activated_indices))[:, np.newaxis],
            activated_indices
        ), axis=-1)
        unused = np.ones(len(activated_indices), dtype=bool)

        num_successful = 0

        for i, unit1, conn1 in activated_indices:
            if not unused[i]:
                continue

            available_connectors = activated_indices[i + 1:][unused[i + 1:]]
            available_connectors = available_connectors[available_connectors[:, 1] != unit1]

            if len(available_connectors) == 0:
                continue

            body_id1 = self._connector_body_indices[unit1, conn1]

            available_xpos = data.xpos[self._connector_body_indices[
                available_connectors[:, 1], available_connectors[:, 2]
            ]]

            pos1 = data.xpos[body_id1]

            available_dist = np.linalg.norm(available_xpos - pos1, axis=1)
            available_close_enough_indices = np.arange(len(available_connectors))[
                available_dist < self.connection_dist_threshold
            ]

            if len(available_close_enough_indices) == 0:
                continue

            available_connectors = available_connectors[available_close_enough_indices]
            available_xpos = available_xpos[available_close_enough_indices]
            available_dist = available_dist[available_close_enough_indices]

            closest_to_farthest_order = np.argsort(available_dist)

            available_connectors = available_connectors[closest_to_farthest_order]
            available_xpos = available_xpos[closest_to_farthest_order]

            for (j, unit2, conn2), pos2 in zip(available_connectors, available_xpos):
                body_id2 = self._connector_body_indices[unit2, conn2]

                mat1 = data.xmat[body_id1].reshape(3, 3)
                mat2 = data.xmat[body_id2].reshape(3, 3)

                z1 = mat1[:, 2]
                z2 = mat2[:, 2]

                # Orientation Check (Anti-aligned)
                if np.dot(z1, z2) > self.connection_angle_threshold:
                    continue

                # "In Front" Check
                rel_pos = pos2 - pos1
                if np.dot(rel_pos, z1) < 0:
                    continue

                x1 = mat1[:, 0]
                y1 = mat1[:, 1]
                x2 = mat2[:, 0]

                twist = np.arctan2(np.dot(x2, y1), np.dot(x2, x1))
                
                connections.connect(unit1, conn1, unit2, conn2, twist)

                self._activate_equality_constraint(model, data, unit1, conn1, unit2, conn2, twist)

                unused[j] = False
                num_successful += 1
                break

        num_connectors_successful = num_successful * 2
        num_connectors_unsuccessful = len(activated_indices) - num_connectors_successful

        return num_connectors_successful, num_connectors_unsuccessful

    def disconnect(
            self,
            data: mujoco.MjData,
            connections: SwarmConnections,
            newly_deactivated: np.ndarray
    ):
        num_disconnected = 0
        already_disconnected = np.zeros((self.swarm.config.num_units, self.swarm.config.limbs_per_unit), dtype=bool)

        disconnect_indices = np.stack(np.where(newly_deactivated)).T
        for unit, connector in disconnect_indices:
            if already_disconnected[unit, connector]:
                continue

            other_unit, other_connector = connections.disconnect(unit, connector)

            already_disconnected[unit, connector] = True
            already_disconnected[other_unit, other_connector] = True

            self._deactivate_equality_constraint(data, unit, connector, other_unit, other_connector)

            num_disconnected += 2

        return num_disconnected

    def _activate_equality_constraint(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            unit1: int,
            conn1: int,
            unit2: int,
            conn2: int,
            twist: float
    ):
        eq_idx = self._eq_indices[unit1, conn1, unit2, conn2]
        if eq_idx == -1:
            raise ValueError(f'Equality constraint {unit1}-{conn1}_{unit2}-{conn2} not found')
        data.eq_active[eq_idx] = 1
        mj_utils.apply_twist(model, eq_idx, twist)

    def _deactivate_equality_constraint(
            self,
            data: mujoco.MjData,
            unit1: int,
            conn1: int,
            unit2: int,
            conn2: int
    ):
        eq_idx = self._eq_indices[unit1, conn1, unit2, conn2]
        if eq_idx == -1:
            raise ValueError(f'Equality constraint {unit1}-{conn1}_{unit2}-{conn2} not found')
        data.eq_active[eq_idx] = 0

