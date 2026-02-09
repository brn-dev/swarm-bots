from typing import Any, Iterable

import mujoco
import numpy as np

from swarmbots.mj_env.float_or_dist_params import (
    BoundedDistParams,
    FloatOrBoundedDistParams,
    eval_fodp,
)
from swarmbots.mj_env.scenarios.base_scenario import SwarmActDict, SwarmObsDict
from swarmbots.mj_env.scenarios.payload_scenario import PayloadScenario
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections


class BridgeScenario(PayloadScenario):
    def __init__(
            self,
            swarm: BaseSwarm,
            payload_type: None | str,
            payload_size: Iterable[float] = (0.2, 0.2, 0.2),
            payload_mass: float = 5.0,
            payload_start_location_offset: Iterable[float] = (0, 1, 0),
            street_width: float = 6.0,
            bridge_width: float = 1.0,
            bridge_length: float = 4.0,
            bridge_x: FloatOrBoundedDistParams = 0.0,
            platform_length: float = 4.0,
            platform_height: float = 0.2,
            fall_z_threshold: float = -1.0,
            fell_off_bridge_reward: float = -1.0,
            actuator_strength: float = 8.0,
            connection_dist_threshold: float = 0.1,
            connection_angle_threshold: float = -0.5,
            disconnect_potential_threshold: float = 5.0,
            friction: float | Iterable[float] | None = None,
            force_elliptic_cone: bool = False,
            progress_reward_weight: float = 1.0,
            guidance_reward_weight: float = 1.0,
            actuators_activation_reward_weight: float = 0.0,
            actuators_activation_reward_power: int = 8,
            units_without_connections_reward_weight: float = 0.0,
            movement_reward_weight: float = 0.0,
            height_reward_weight: float = 0.0,
            connectors_stayed_active_reward_weight: float = 0.0,
            connectors_successfully_activated_reward_weight: float = 0.0,
            connectors_unsuccessfully_activated_reward_weight: float = 0.0,
            connectors_deactivated_reward_weight: float = 0.0,
            average_connectors_reward: bool = True,
            include_connectors_xpos_in_obs: bool = True,
            include_connectors_xquat_in_obs: bool = False,
            quat_rot6d_representation: bool = True,
            seed: int | None = None,
    ) -> None:
        self.street_width = float(street_width)
        self.bridge_width = float(bridge_width)
        self.bridge_length = float(bridge_length)
        self.platform_length = float(platform_length)
        self.platform_height = float(platform_height)
        self.fall_z_threshold = float(fall_z_threshold)
        if self.street_width <= 0:
            raise ValueError(f"Expected street_width > 0, got {self.street_width}")
        if self.bridge_width <= 0:
            raise ValueError(f"Expected bridge_width > 0, got {self.bridge_width}")
        if self.bridge_width >= self.street_width:
            raise ValueError(f"Expected bridge_width < street_width, got {self.bridge_width} >= {self.street_width}")
        if self.bridge_length <= 0:
            raise ValueError(f"Expected bridge_length > 0, got {self.bridge_length}")
        if self.platform_length <= 0:
            raise ValueError(f"Expected platform_length > 0, got {self.platform_length}")
        if self.platform_height <= 0:
            raise ValueError(f"Expected platform_height > 0, got {self.platform_height}")

        self.side_wall_x = self.street_width / 2.0
        self.bridge_x_param = bridge_x
        self.bridge_x = 0.0

        self.platform1_min_y = -50.0
        self.platform1_max_y = self.platform_length / 2.0
        self.platform1_center_y = (self.platform1_min_y + self.platform1_max_y) / 2.0
        self.platform1_half_length = (self.platform1_max_y - self.platform1_min_y) / 2.0
        self.platform2_center_y = self.platform_length + self.bridge_length
        self.bridge_center_y = (self.platform_length / 2.0) + (self.bridge_length / 2.0)
        self.bridge_y_min = self.platform_length / 2.0
        self.bridge_y_max = self.bridge_y_min + self.bridge_length

        self.fell_off_bridge_reward = fell_off_bridge_reward

        self._validate_bridge_x_param()

        super().__init__(
            swarm=swarm,
            payload_type=payload_type,
            payload_size=payload_size,
            payload_mass=payload_mass,
            payload_start_location_offset=payload_start_location_offset,
            actuator_strength=actuator_strength,
            progress_reward_weight=progress_reward_weight,
            guidance_reward_weight=guidance_reward_weight,
            actuators_activation_reward_weight=actuators_activation_reward_weight,
            actuators_activation_reward_power=actuators_activation_reward_power,
            units_without_connections_reward_weight=units_without_connections_reward_weight,
            movement_reward_weight=movement_reward_weight,
            height_reward_weight=height_reward_weight,
            connectors_stayed_active_reward_weight=connectors_stayed_active_reward_weight,
            connectors_successfully_activated_reward_weight=connectors_successfully_activated_reward_weight,
            connectors_unsuccessfully_activated_reward_weight=connectors_unsuccessfully_activated_reward_weight,
            connectors_deactivated_reward_weight=connectors_deactivated_reward_weight,
            average_connectors_reward=average_connectors_reward,
            seed=seed,
            include_connectors_xpos_in_obs=include_connectors_xpos_in_obs,
            include_connectors_xquat_in_obs=include_connectors_xquat_in_obs,
            quat_rot6d_representation=quat_rot6d_representation,
            friction=friction,
            connection_dist_threshold=connection_dist_threshold,
            connection_angle_threshold=connection_angle_threshold,
            disconnect_potential_threshold=disconnect_potential_threshold,
            force_elliptic_cone=force_elliptic_cone,
            _reset_in_init=False,
        )

    def _validate_bridge_x_param(self) -> None:
        min_x = -self.side_wall_x + (self.bridge_width / 2.0)
        max_x = self.side_wall_x - (self.bridge_width / 2.0)

        if isinstance(self.bridge_x_param, BoundedDistParams):
            if self.bridge_x_param.low < min_x or self.bridge_x_param.high > max_x:
                raise ValueError(
                    "Expected bridge_x bounds within "
                    f"[{min_x}, {max_x}], got {self.bridge_x_param}"
                )
            return
        if not isinstance(self.bridge_x_param, (int, float)):
            raise TypeError(f"Expected bridge_x to be float or BoundedDistParams, got {self.bridge_x_param!r}")
        bridge_x_value = float(self.bridge_x_param)
        if bridge_x_value < min_x or bridge_x_value > max_x:
            raise ValueError(f"Expected bridge_x within [{min_x}, {max_x}], got {bridge_x_value}")

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update({
            "payload_type": self.payload_type,
            "payload_size": self.payload_size,
            "payload_mass": self.payload_mass,
            "payload_start_location_offset": self.payload_start_location_offset,
            "street_width": self.street_width,
            "bridge_width": self.bridge_width,
            "bridge_length": self.bridge_length,
            "bridge_x": self.bridge_x_param,
            "platform_length": self.platform_length,
            "platform_height": self.platform_height,
            "fall_z_threshold": self.fall_z_threshold,
        })
        return settings

    def _create_scenario_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: mujoco.MjsBody = spec.worldbody

        worldbody.add_light(pos=[0, 0, 100], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 100, 100], dir=[-1, -1, -1])

        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[self.side_wall_x, 0, 0],
        )
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[-self.side_wall_x, 0, 0],
        )

        platform1_size = [self.street_width / 2.0, self.platform1_half_length, self.platform_height / 2.0]
        platform2_size = [self.street_width / 2.0, self.platform_length / 2.0, self.platform_height / 2.0]
        platform_rgba = [0.45, 0.45, 0.5, 1]
        platform_top_z = 0.0
        platform_center_z = platform_top_z - (self.platform_height / 2.0)
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=platform1_size,
            rgba=platform_rgba,
            pos=[0, self.platform1_center_y, platform_center_z],
        )
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=platform2_size,
            rgba=platform_rgba,
            pos=[0, self.platform2_center_y, platform_center_z],
        )

        bridge_body = worldbody.add_body(name="Bridge", mocap=True, pos=[0, 0, 0])
        bridge_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.bridge_width / 2.0, self.bridge_length / 2.0, self.platform_height / 2.0],
            rgba=[0.7, 0.6, 0.3, 1],
        )

        self._maybe_add_payload_spec(spec)

        return spec

    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data)

        self.bridge_x = float(eval_fodp(self.bridge_x_param, self.rng))

        bridge_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "Bridge")
        if bridge_body_id == -1:
            raise RuntimeError("Bridge body not found in model")
        mocap_id = model.body_mocapid[bridge_body_id]
        data.mocap_pos[mocap_id] = [
            self.bridge_x,
            self.bridge_center_y,
            -(self.platform_height / 2.0),
        ]

        mujoco.mj_forward(model, data)

        state["progress"] = self.compute_progress(data, state.get("units_active_mask"))
        state["hidden_vars"] = np.array([self.bridge_x], dtype=float)
        state["fell_off_bridge"] = False

        return state, connections

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections,
    ) -> SwarmObsDict:
        obs = super().get_obs(model, data, state, connections)
        obs["hidden_vars"] = state["hidden_vars"].copy()
        return obs

    def compute_progress(
            self,
            data: mujoco.MjData,
            units_active_mask: np.ndarray | None,
    ) -> float:
        if self.payload_type is None:
            unit_positions = data.qpos[self._qpos_indices[:, 1]]
            if units_active_mask is None:
                return float(unit_positions.mean())
            active_units_mask = np.asarray(units_active_mask, dtype=bool)
            if not active_units_mask.any():
                return 0.0
            return float(unit_positions[active_units_mask].mean())
        return float(data.xpos[self.payload_body_id, 1])

    def evaluate_step(
            self,
            action: SwarmActDict,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections,
    ) -> tuple[float, bool]:
        reward, terminated = super().evaluate_step(action, model, data, state, connections)
        if terminated:
            return reward, True

        fell_off_bridge = self._is_any_body_below_threshold(data, state.get("units_active_mask"))
        state["fell_off_bridge"] = fell_off_bridge
        if fell_off_bridge:
            reward += self.fell_off_bridge_reward
            return reward, True

        return reward, False

    def _is_any_body_below_threshold(
            self,
            data: mujoco.MjData,
            units_active_mask: np.ndarray | None,
    ) -> bool:
        units_pos_z = data.qpos[self._qpos_indices[:, 2]]
        if units_active_mask is not None:
            active_units_mask = np.asarray(units_active_mask, dtype=bool)
            units_pos_z = units_pos_z[active_units_mask]

        if self._z_pos_below_threshold(units_pos_z):
            return True

        if self.payload_type is None:
            return False

        payload_pos_z = data.xpos[self.payload_body_id, 2]
        return self._z_pos_below_threshold(np.atleast_1d(payload_pos_z))

    def _z_pos_below_threshold(self, z_pos: np.ndarray) -> bool:
        if z_pos.size == 0:
            return False
        return bool(np.any(z_pos < self.fall_z_threshold))

    @staticmethod
    def default(
            seed: int | None = None,
            swarm: BaseSwarm | None = None,
            unit_start_locations: list[tuple[float, float, float]] | str | None = None,
            randomize_unit_orientations: bool = False,
            **kwargs
    ) -> 'BridgeScenario':
        assert swarm is None or unit_start_locations is None

        if swarm is None:
            if unit_start_locations is None:
                unit_start_locations = '4:diamond'
            swarm = HomogeneousSwarm(
                unit_start_locations=unit_start_locations,
                randomize_unit_orientations=randomize_unit_orientations
            )

        scenario_kwargs = {
            'friction': [2, 1e-2, 2e-4],
            'force_elliptic_cone': True,
            'actuators_activation_reward_weight': -5e-3,
            'units_without_connections_reward_weight': -2e-3,
            'movement_reward_weight': 0e-1,
            'height_reward_weight': 0e-4,
            'connectors_stayed_active_reward_weight': 0e-5,
            'connectors_successfully_activated_reward_weight': 0e-3,
            'connectors_unsuccessfully_activated_reward_weight': -1e-4,
            'connectors_deactivated_reward_weight': 0e-3,
        }
        scenario_kwargs.update(kwargs)
        return BridgeScenario(
            swarm=swarm,
            payload_type=None,
            **scenario_kwargs,
            seed=seed,
        )
