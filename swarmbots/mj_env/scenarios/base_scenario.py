import abc
from typing import Any, TypedDict, Iterable

import mujoco
import numpy as np
from gymnasium import spaces
from mujoco import MjsBody

import swarmbots.mj_env.mujoco_utils as mj_utils
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections

class SwarmObsDict(TypedDict):
    local_obs: np.ndarray  # shape (n_unit, n_obs_per_unit)
    global_obs: np.ndarray  # shape (n_global_features,)

class SwarmActDict(TypedDict):
    actuators: np.ndarray  # shape (n_unit, n_actuators_per_unit), type float
    connectors: np.ndarray  # shape (n_unit, n_connectors_per_unit), type bool

DEFAULT_GEOM_FRICTION: tuple[float, float, float] = (1.0, 0.005, 0.0001)
HIGH_FRICTION_SLIDING_THRESHOLD: float = 2.0


def _validate_geom_friction(
        friction: float | Iterable[float] | None,
) -> tuple[float, float, float] | None:
    if friction is None:
        return None
    if isinstance(friction, (int, float)):
        return (float(friction), DEFAULT_GEOM_FRICTION[1], DEFAULT_GEOM_FRICTION[2])
    friction_tuple = tuple(float(x) for x in friction)
    if len(friction_tuple) != 3:
        raise ValueError(f"Expected friction to be a float or 3-tuple, got {friction_tuple!r}")
    return friction_tuple


class BaseScenario(abc.ABC):

    def __init__(
        self,
            swarm: BaseSwarm,
            actuators_activation_reward_weight: float,
            units_without_connections_reward_weight: float,
            movement_reward_weight: float,
            connectors_stayed_active_reward_weight: float,
            connectors_successfully_activated_reward_weight: float,
            connectors_unsuccessfully_activated_reward_weight: float,
            connectors_deactivated_reward_weight: float,
            average_connectors_reward: bool,
            include_connectors_xpos_in_obs: bool,
            include_connectors_xquat_in_obs: bool,
            friction: float | Iterable[float] | None,
            force_elliptic_cone: bool,
            seed: int | None,
            _reset_in_init: bool = True,
    ):
        self.seed = seed
        self.rng = np.random.default_rng(seed)

        self.swarm = swarm
        self.num_units = swarm.config.num_units
        self.limbs_per_unit = swarm.config.limbs_per_unit
        self.num_connectors = self.num_units * self.limbs_per_unit
        self.connection_dist_threshold = swarm.config.connection_dist_threshold
        self.connection_angle_threshold = swarm.config.connection_angle_threshold

        self.actuators_activation_reward_weight = actuators_activation_reward_weight
        self.units_without_connections_reward_weight = units_without_connections_reward_weight
        self.movement_reward_weight = movement_reward_weight
        self.connectors_stayed_active_reward_weight = connectors_stayed_active_reward_weight
        self.connectors_successfully_activated_reward_weight = connectors_successfully_activated_reward_weight
        self.connectors_unsuccessfully_activated_reward_weight = connectors_unsuccessfully_activated_reward_weight
        self.connectors_deactivated_reward_weight = connectors_deactivated_reward_weight
        self.average_connectors_reward = average_connectors_reward
        self.include_connectors_xpos_in_obs = include_connectors_xpos_in_obs
        self.include_connectors_xquat_in_obs = include_connectors_xquat_in_obs
        self.friction = _validate_geom_friction(friction)
        self.force_elliptic_cone = force_elliptic_cone
        
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

        num_units = self.num_units
        num_connectors = self.limbs_per_unit
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

        if _reset_in_init:
            self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)
        else:
            self._dummy_state, self._dummy_connections = None, None

    def get_settings(self):
        return {
            'swarm': self.swarm.get_settings(),
            'actuators_activation_reward_weight': self.actuators_activation_reward_weight,
            'units_without_connections_reward_weight': self.units_without_connections_reward_weight,
            'movement_reward_weight': self.movement_reward_weight,
            'connectors_stayed_active_reward_weight': self.connectors_stayed_active_reward_weight,
            'connectors_successfully_activated_reward_weight': self.connectors_successfully_activated_reward_weight,
            'connectors_unsuccessfully_activated_reward_weight': self.connectors_unsuccessfully_activated_reward_weight,
            'connectors_deactivated_reward_weight': self.connectors_deactivated_reward_weight,
            'average_connectors_reward': self.average_connectors_reward,
            'include_connectors_xpos_in_obs': self.include_connectors_xpos_in_obs,
            'include_connectors_xquat_in_obs': self.include_connectors_xquat_in_obs,
            'friction': self.friction,
            'force_elliptic_cone': self.force_elliptic_cone,
            'seed': self.seed,
        }

    @abc.abstractmethod
    def _create_scenario_spec(self) -> mujoco.MjSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate_step(
            self,
            action: SwarmActDict,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
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

        self.add_cameras(spec)

        return spec

    def add_cameras(self, spec: mujoco.MjSpec):
        for unit_prefix in self.swarm.config.unit_prefixes:
            unit1_body = spec.body(unit_prefix + '-main_body')
            unit1_body.add_camera(
                pos=[5, 0, 3], euler=[0, np.pi / 3, np.pi / 2],
                mode=mujoco.mjtCamLight.mjCAMLIGHT_TRACK)
            
    def build(self) -> tuple[mujoco.MjModel, mujoco.MjData]:
        model = self.spec.compile()
        if self.friction is not None:
            model.geom_friction[:] = np.asarray(self.friction, dtype=float)

            # Elliptic cone is more stable for high friction forces but slower to compute.
            # If a high friction is set while using pyramidal cone is used, geoms can fall through the plane.
            sliding = float(self.friction[0])
            if self.force_elliptic_cone or (
                sliding >= HIGH_FRICTION_SLIDING_THRESHOLD
                and int(model.opt.cone) == int(mujoco.mjtCone.mjCONE_PYRAMIDAL)
            ):
                model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC
        data = mujoco.MjData(model)
        return model, data

    def get_swarm_start_location(self):
        return np.array([0.0, 0.0, 1.0])

    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict, SwarmConnections]:
        mujoco.mj_resetData(model, data)

        state = dict()
        connections = self.swarm.reset_swarm(model, data, self.rng, self.get_swarm_start_location())

        for (u1, c1, u2, c2), angle in zip(*connections.get_active_connections()):
            self._activate_equality_constraint(model, data, u1, c1, u2, c2, angle)

        state['unit_positions'] = data.qpos[self._qpos_indices[:, :3]].copy()

        return state, connections

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> SwarmObsDict:
        """
        Local observations contain for each unit:

        * qpos - hinge angles are encoded via sin/cos pairs
        * qvel
        * connector obs - containing is-connected status, twist angles and disconnect potentials
        * [Optionally] xpos of connectors
        * [Optionally] xquat of connectors
        """
        qpos = data.qpos[self._qpos_indices]
        qvel = data.qvel[self._qvel_indices]

        # encoding hinge angles to sin and cos
        free_joint_qpos = qpos[:, :7]
        hinge_qpos = qpos[:, 7:]
        hinge_sin = np.sin(hinge_qpos)
        hinge_cos = np.cos(hinge_qpos)
        hinge_obs = np.stack([hinge_sin, hinge_cos], axis=-1).reshape(self.num_units, -1)
        qpos_obs = np.concatenate([free_joint_qpos, hinge_obs], axis=1)

        # connector obs
        is_active = connections.get_is_active_mask()
        active_indices = np.where(is_active)
        non_active_indices = np.where(np.logical_not(is_active))
        twist_angles = connections.twist_angles
        disconnect_potentials = connections.disconnect_potentials

        connector_obs = np.zeros((self.num_units, self.limbs_per_unit, 4), dtype=float)
        connector_obs[non_active_indices[0], non_active_indices[1], 0] = 1
        connector_obs[active_indices[0], active_indices[1], 1] = 1
        connector_obs[active_indices[0], active_indices[1], 2] = twist_angles[is_active]
        connector_obs[active_indices[0], active_indices[1], 3] = disconnect_potentials[is_active]
        connector_obs = connector_obs.reshape((self.num_units, -1))

        obs_list = [qpos_obs, qvel, connector_obs]

        if self.include_connectors_xpos_in_obs:
            conn_xpos = data.xpos[self._connector_body_indices]
            conn_xpos = conn_xpos.reshape((self.num_units, -1))
            obs_list.append(conn_xpos)

        if self.include_connectors_xquat_in_obs:
            conn_xquat = data.xquat[self._connector_body_indices]
            conn_xquat = conn_xquat.reshape((self.num_units, -1))
            obs_list.append(conn_xquat)

        return {
            'local_obs': np.concatenate(obs_list, axis=1),
            'global_obs': np.empty(0, dtype=float)
        }

    def apply_action(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            action: SwarmActDict,
            action_scale: float,
            state: dict,
            connections: SwarmConnections
    ) -> None:
        data.ctrl[self._ctrl_indices] = action['actuators'] * action_scale

        connectors_action = np.asarray(action['connectors'], dtype=bool)

        currently_active_mask = connections.get_is_active_mask()

        stayed_active_mask = np.logical_and(connectors_action, currently_active_mask)
        newly_activated_mask = np.logical_and(connectors_action, np.logical_not(currently_active_mask))
        newly_deactivated_mask = np.logical_and(np.logical_not(connectors_action), currently_active_mask)

        state['num_connectors_stayed_active'] = stayed_active_mask.sum()

        num_connectors_successfully_activated, num_connectors_unsuccessfully_activated = self.try_connect(
            model,
            data,
            connections,
            newly_activated_mask
        )
        state['num_connectors_successfully_activated'] = num_connectors_successfully_activated
        state['num_connectors_unsuccessfully_activated'] = num_connectors_unsuccessfully_activated

        deactivation_mask = connections.update_disconnect_potentials(currently_active_mask, newly_deactivated_mask)
        num_connectors_deactivated = self.disconnect(data, connections, deactivation_mask)
        state['num_connectors_deactivated'] = num_connectors_deactivated


    def get_obs_space(self):
        obs = self.get_obs(self.dummy_model, self.dummy_data, self._dummy_state, self._dummy_connections)
        return spaces.Dict({
            'local_obs': spaces.Box(
                low=-np.inf, high=np.inf, shape=obs['local_obs'].shape, dtype=np.float32
            ),
            'global_obs': spaces.Box(
                low=-np.inf, high=np.inf, shape=obs['global_obs'].shape, dtype=np.float32
            )
        })

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
            newly_activated_mask: np.ndarray
    ) -> tuple[int, int]:
        """
        Considers newly activated connectors (those that are not connected but are activated by the policy) and connects
        the closest pairs if the following criteria are fulfilled:
        * They don't belong to the same unit
        * They are below a certain threshold in distance
        * Their z-axis is anti-aligned up to a threshold - meaning they face each other
        * They are in front of each other (positive z-distance)
        :return: num_connectors_successful, num_connectors_unsuccessful
        """
        activated_indices = np.stack(np.where(newly_activated_mask)).T
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
            deactivation_mask: np.ndarray,
    ):
        num_disconnected = 0
        already_disconnected = np.zeros((self.num_units, self.limbs_per_unit), dtype=bool)

        disconnect_indices = np.stack(np.where(deactivation_mask)).T
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

    def compute_guidance_reward(
            self,
            data: mujoco.MjData,
            action: dict[str, Any],
            state: dict,
            connections: SwarmConnections
    ):
        reward = 0.0

        actuator_activation = np.square(action['actuators']).mean()
        reward += actuator_activation * self.actuators_activation_reward_weight

        num_units_without_connections = np.logical_not(connections.get_is_active_mask()).all(axis=1).sum()
        state['num_units_without_connections'] = num_units_without_connections
        units_without_connections_ratio = num_units_without_connections / self.num_units
        reward += units_without_connections_ratio * self.units_without_connections_reward_weight

        prev_unit_positions = state['unit_positions']
        unit_positions = data.qpos[self._qpos_indices[:, :3]].copy()
        state['unit_positions'] = unit_positions
        avg_movement = np.linalg.norm(unit_positions - prev_unit_positions, axis=1).mean()
        state['avg_movement'] = avg_movement
        reward += avg_movement * self.movement_reward_weight

        connectors_reward = 0.0
        connectors_reward += state['num_connectors_stayed_active'] * self.connectors_stayed_active_reward_weight
        connectors_reward += state['num_connectors_successfully_activated'] * self.connectors_successfully_activated_reward_weight
        connectors_reward += state['num_connectors_unsuccessfully_activated'] * self.connectors_unsuccessfully_activated_reward_weight
        connectors_reward += state['num_connectors_deactivated'] * self.connectors_deactivated_reward_weight

        if self.average_connectors_reward:
            connectors_reward /= self.num_connectors

        reward += connectors_reward

        return reward

