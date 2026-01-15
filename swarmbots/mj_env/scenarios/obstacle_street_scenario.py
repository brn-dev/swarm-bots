from typing import Any, Iterable, Self

import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.mj_env.scenarios.base_scenario import BaseScenario, SwarmObsDict
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections


class ObstacleStreetScenario(BaseScenario):

    def __init__(
            self,
            swarm: BaseSwarm,
            payload_type: None | str,
            payload_size: Iterable[float] = (0.2, 0.2, 0.2),
            payload_mass: float = 5.0,
            payload_start_location_offset: Iterable[float] = (0, 1, 0),
            num_walls: int = 3,
            wall_height: float | list[float] = 0.5,
            wall_distance: float = 4.0,
            first_wall_distance: float = 2.0,
            opening_width: float | list[float] = 2.0,
            unusable_opening_offset: float = 2.0,
            street_width: float = 10.0,
            no_initial_ramp: bool = True,
            actuator_strength: float = 3.0,
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
            include_connectors_xpos_in_obs: bool = False,
            include_connectors_xquat_in_obs: bool = False,
            seed: int = None,
    ):
        self.payload_type = payload_type
        self.payload_size = payload_size
        self.payload_mass = payload_mass
        self.payload_start_location_offset = payload_start_location_offset

        self.num_walls = num_walls
        self.no_initial_ramp = no_initial_ramp

        self.street_width = street_width
        self.side_wall_x = street_width / 2
        self.wall_fixed_width = 25.0
        self.wall_heights = wall_height if isinstance(wall_height, list) else [wall_height] * num_walls
        self.wall_distance = wall_distance
        self.first_wall_distance = first_wall_distance

        self.opening_widths = opening_width if isinstance(opening_width, list) else [opening_width] * num_walls
        self.unusable_opening_offset = unusable_opening_offset
        self.ramp_length = wall_distance - 1
        self.ramp_range_x = 8.5
        self.ramp_angles = [np.asin(wh / self.ramp_length) for wh in self.wall_heights]
        self.ramp_distances_to_wall = [self.ramp_length * np.cos(ra) for ra in self.ramp_angles]

        super().__init__(
            swarm=swarm,
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
            friction=friction,
            connection_dist_threshold=connection_dist_threshold,
            connection_angle_threshold=connection_angle_threshold,
            disconnect_potential_threshold=disconnect_potential_threshold,
            force_elliptic_cone=force_elliptic_cone,
            _reset_in_init=False
        )
        
        self.payload_body_id = mujoco.mj_name2id(self.dummy_model, mujoco.mjtObj.mjOBJ_BODY, 'Payload')

        self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)

    def get_settings(self):
        settings =  super().get_settings()
        settings.update({
            'payload_type': self.payload_type,
            'payload_size': self.payload_size,
            'payload_mass': self.payload_mass,
            'payload_start_location_offset': self.payload_start_location_offset,
            'num_walls': self.num_walls,
            'wall_heights': self.wall_heights,
            'wall_distance': self.wall_distance,
            'first_wall_distance': self.first_wall_distance,
            'opening_widths': self.opening_widths,
            'unusable_opening_offset': self.unusable_opening_offset,
            'street_width': self.street_width,
            'no_initial_ramp': self.no_initial_ramp,
        })
        return settings


    def _create_scenario_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody: mujoco.MjsBody = spec.worldbody

        # base stuff
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[100, 100, 0.1],
            rgba=[0.2, 0.3, 0.4, 1],
            pos=[0, 0, 0]
        )

        worldbody.add_light(pos=[0, 0, 100], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 100, 100], dir=[-1, -1, -1])

        # side walls
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[self.side_wall_x, 0, 0]
        )
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[0.1, 100, 5],
            rgba=[0.3, 0.4, 0.5, 0.1],
            pos=[-self.side_wall_x, 0, 0]
        )

        for i in range(self.num_walls):
            y = self.first_wall_distance + self.wall_distance * i

            body_left = worldbody.add_body(name=f'Wall_{i}_Left', mocap=True, pos=[0, y, 0])
            body_left.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.wall_fixed_width / 2, 0.1, self.wall_heights[i]],
                rgba=[0.5, 0.5, 0.6, 1],
            )

            body_right = worldbody.add_body(name=f'Wall_{i}_Right', mocap=True, pos=[0, y, 0])
            body_right.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.wall_fixed_width / 2, 0.1, self.wall_heights[i]],
                rgba=[0.5, 0.5, 0.6, 1],
            )

            if i > 0 or not self.no_initial_ramp:
                body_ramp = worldbody.add_body(
                    name=f'Ramp_{i}', mocap=True,
                    pos=[0, y - self.ramp_distances_to_wall[i]/2, self.wall_heights[i]/2],
                    euler=[self.ramp_angles[i], 0, 0]
                )
                body_ramp.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[1, self.ramp_length * 1.2 / 2, 0.1],
                    rgba=[0.5, 0.5, 0.6, 1],
                )

        if self.payload_type is not None:
            payload_start_position = (
                    np.array(self.get_swarm_start_location()) + np.array(self.payload_start_location_offset)
            )
            payload_body: MjsBody = worldbody.add_body(name='Payload', pos=payload_start_position)

            payload_geom_type = {
                'sphere': mujoco.mjtGeom.mjGEOM_SPHERE,
                'box': mujoco.mjtGeom.mjGEOM_BOX,
                'cylinder': mujoco.mjtGeom.mjGEOM_CYLINDER,
                'capsule': mujoco.mjtGeom.mjGEOM_CAPSULE,
                'ellipsoid': mujoco.mjtGeom.mjGEOM_ELLIPSOID
            }[self.payload_type]

            payload_rgba = [0.8, 0.3, 0.3, 0.9]
            payload_body.add_geom(
                type=payload_geom_type,
                size=self.payload_size,
                mass=self.payload_mass,
                rgba=payload_rgba
            )
            payload_body.add_joint(type=mujoco.mjtJoint.mjJNT_FREE)

        return spec

    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data)

        rng = self.rng

        for i in range(self.num_walls):
            y = self.first_wall_distance + self.wall_distance * i
            
            opening_x = (rng.random() - 0.5) * 2 * (
                self.side_wall_x 
                - self.opening_widths[i] / 2 
                + self.unusable_opening_offset)  # small chance that there is no usable opening
                                                # -> must use ramp to continue
            
            first_wall_end_x = opening_x - self.opening_widths[i] / 2
            wall_left_pos_x = first_wall_end_x - (self.wall_fixed_width / 2)
            
            wall_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Wall_{i}_Left')
            mocap_id = model.body_mocapid[wall_left_id]
            data.mocap_pos[mocap_id] = [wall_left_pos_x, y, 0]

            second_wall_start_x = opening_x + self.opening_widths[i] / 2

            wall_right_pos_x = second_wall_start_x + (self.wall_fixed_width / 2)

            wall_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Wall_{i}_Right')
            mocap_id = model.body_mocapid[wall_right_id]
            data.mocap_pos[mocap_id] = [wall_right_pos_x, y, 0]

            if i > 0 or not self.no_initial_ramp:
                ramp_x = (rng.random() - 0.5) * 2 * self.ramp_range_x

                ramp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Ramp_{i}')
                mocap_id = model.body_mocapid[ramp_id]
                data.mocap_pos[mocap_id] = [ramp_x, y - self.ramp_distances_to_wall[i] / 2, self.wall_heights[i] / 2]

        mujoco.mj_forward(model, data)

        state['progress'] = self._compute_progress(data)

        return state, connections

    def evaluate_step(
            self,
            action: dict[str, Any],
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> tuple[float, bool]:
        """
        :return: (reward, done)
        """

        old_progress = state['progress']
        new_progress = self._compute_progress(data)
        state['progress'] = new_progress

        progress_reward = new_progress - old_progress
        state['progress_reward'] = progress_reward

        guidance_reward = self.compute_guidance_reward(data, action, state, connections)
        state['guidance_reward'] = guidance_reward

        weighted_progress_reward = progress_reward * self.reward_weights['progress_reward_weight']
        weighted_guidance_reward = guidance_reward * self.reward_weights['guidance_reward_weight']
        state['weighted_progress_reward'] = weighted_progress_reward
        state['weighted_guidance_reward'] = weighted_guidance_reward

        return weighted_progress_reward + weighted_guidance_reward, False

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> SwarmObsDict:
        obs = super().get_obs(model, data, state, connections)

        if self.payload_type is not None:
            obs['global_obs'] = np.concatenate([
                data.xpos[self.payload_body_id],
                data.xquat[self.payload_body_id]
            ])

        return obs

    def _compute_progress(
            self,
            data: mujoco.MjData
    ):
        if self.payload_type is None:
            return data.qpos[self._qpos_indices[:, 1]].mean()  # avg y pos of the unit bodies

        return data.xpos[self.payload_body_id, 1]

    @staticmethod
    def no_payload_no_opening_one_wall_easy(
            seed: int | None = None,
            swarm: BaseSwarm | None = None,
            unit_start_locations: list[tuple[float, float, float]] | str | None = None,
            randomize_unit_orientations: bool = False,
            **kwargs
    ) -> 'ObstacleStreetScenario':
        assert swarm is None or unit_start_locations is None

        if swarm is None:
            if unit_start_locations is None:
                unit_start_locations = '4:diamond'
            swarm = HomogeneousSwarm(
                unit_start_locations=unit_start_locations,
                randomize_unit_orientations=randomize_unit_orientations
            )

        scenario_kwargs = {
            'wall_height': 0.2,
            'friction': [2, 1e-2, 2e-4],
            'force_elliptic_cone': True,
            'actuator_strength': 5.0,
            'actuators_activation_reward_weight': -1.75e-2,
            'units_without_connections_reward_weight': -6e-3,
            'movement_reward_weight':  0e-1,
            'height_reward_weight':  3e-3,
            'connectors_stayed_active_reward_weight':  0e-5,
            'connectors_successfully_activated_reward_weight':  0e-3,
            'connectors_unsuccessfully_activated_reward_weight': -1e-4,
            'connectors_deactivated_reward_weight': 0e-3,
        }
        scenario_kwargs.update(kwargs)
        return ObstacleStreetScenario(
            swarm=swarm,
            payload_type=None,
            num_walls=1,
            opening_width=0.01,
            **scenario_kwargs,
            seed=seed,
        )
