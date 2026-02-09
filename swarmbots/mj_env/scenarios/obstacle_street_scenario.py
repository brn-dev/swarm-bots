from dataclasses import dataclass
from typing import Any, Iterable, Literal

import mujoco
import numpy as np
from mujoco import MjsBody

from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams, eval_fodp, fodp_low, FloatOrBoundedDistParams
from swarmbots.mj_env.scenarios.base_scenario import BaseScenario, SwarmActDict, SwarmObsDict
from swarmbots.mj_env.quat_rot6d import quat_to_rot6d
from swarmbots.mj_env.scenarios.payload_scenario import PayloadScenario
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections


@dataclass(frozen=True, slots=True)
class PoleParams:
    x: FloatOrDistParams
    y: FloatOrDistParams


TruncatedMultivariateNormal2DSamplingMode = Literal["clamp", "rejection"]


@dataclass(frozen=True, slots=True)
class CorrelatedPoleParams:
    mean: tuple[float, float]
    cov: tuple[tuple[float, float], tuple[float, float]]
    low: tuple[float, float] | None = None
    high: tuple[float, float] | None = None
    sampling_mode: TruncatedMultivariateNormal2DSamplingMode = "clamp"

    @staticmethod
    def from_std_and_rho(
            *,
            mean: tuple[float, float],
            std_x: float,
            std_y: float,
            rho: float,
            low: tuple[float, float] | None = None,
            high: tuple[float, float] | None = None,
            sampling_mode: TruncatedMultivariateNormal2DSamplingMode = "clamp",
    ) -> "CorrelatedPoleParams":
        if std_x < 0 or std_y < 0:
            raise ValueError(f"Expected std_x/std_y >= 0, got {std_x=} {std_y=}")
        if not (-1.0 <= rho <= 1.0):
            raise ValueError(f"Expected rho in [-1, 1], got {rho}")
        if (low is None) != (high is None):
            raise ValueError(f"Expected low/high to be both set or both None, got {low=} {high=}")
        cov_xy = float(rho) * float(std_x) * float(std_y)
        cov = ((float(std_x) ** 2, cov_xy), (cov_xy, float(std_y) ** 2))
        return CorrelatedPoleParams(mean=mean, cov=cov, low=low, high=high, sampling_mode=sampling_mode)


PoleSpec = PoleParams | CorrelatedPoleParams


class ObstacleStreetScenario(PayloadScenario):

    def __init__(
            self,
            swarm: BaseSwarm,
            payload_type: None | str,
            payload_size: Iterable[float] = (0.2, 0.2, 0.2),
            payload_mass: float = 5.0,
            payload_start_location_offset: Iterable[float] = (0, 1, 0),
            poles: Iterable[PoleSpec | tuple[FloatOrDistParams, FloatOrDistParams]] = (),
            pole_radius: float = 0.1,
            pole_height: float = 1.0,
            num_walls: int = 3,
            wall_height: float | list[float] = 0.5,
            inter_wall_distance: FloatOrBoundedDistParams = 4.0,
            first_wall_distance: FloatOrDistParams = 2.0,
            opening_width: FloatOrDistParams | list[FloatOrDistParams] = 2.0,
            unusable_opening_offset: FloatOrDistParams = 2.0,
            street_width: float = 10.0,
            no_initial_ramp: bool = True,
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
            units_with_double_connection_reward_weight: float = 0.0,
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
    ):
        self.poles: list[PoleSpec] = []
        for p in poles:
            if isinstance(p, tuple):
                if len(p) != 2:
                    raise ValueError(f"Expected pole tuple (x, y), got {p!r}")
                self.poles.append(PoleParams(x=p[0], y=p[1]))
            else:
                self.poles.append(p)
        self.pole_radius = float(pole_radius)
        self.pole_height = float(pole_height)

        self.num_walls = num_walls
        self.no_initial_ramp = no_initial_ramp

        self.street_width = street_width
        self.side_wall_x = street_width / 2
        self.wall_fixed_width = 25.0
        self.wall_heights = wall_height if isinstance(wall_height, list) else [wall_height] * num_walls
        self.inter_wall_distance = inter_wall_distance
        self.first_wall_distance = first_wall_distance

        self.opening_widths = opening_width if isinstance(opening_width, list) else [opening_width] * num_walls
        self.unusable_opening_offset = unusable_opening_offset
        min_inter_wall_distance = fodp_low(self.inter_wall_distance)
        if min_inter_wall_distance <= 1.0:
            raise ValueError(f"Expected inter_wall_distance.low > 1.0, got {min_inter_wall_distance}")
        self.ramp_length = min_inter_wall_distance - 1.0
        self.ramp_range_x = self.side_wall_x - 1.5
        self.ramp_angles = [np.asin(wh / self.ramp_length) + np.pi/64 for wh in self.wall_heights]
        self.ramp_distances_to_wall = [self.ramp_length * np.cos(ra) for ra in self.ramp_angles]

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
            units_with_double_connection_reward_weight=units_with_double_connection_reward_weight,
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
            _reset_in_init=False
        )

        self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update({
            'payload_type': self.payload_type,
            'payload_size': self.payload_size,
            'payload_mass': self.payload_mass,
            'payload_start_location_offset': self.payload_start_location_offset,
            'poles': self.poles,
            'pole_radius': self.pole_radius,
            'pole_height': self.pole_height,
            'num_walls': self.num_walls,
            'wall_heights': self.wall_heights,
            'inter_wall_distance': self.inter_wall_distance,
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

        for i, _ in enumerate(self.poles):
            body_pole = worldbody.add_body(name=f'Pole_{i}', mocap=True, pos=[0, 0, 0])
            body_pole.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CYLINDER,
                fromto=[0, 0, 0, 0, 0, self.pole_height],
                size=[self.pole_radius, 0, 0],
                rgba=[0.15, 0.8, 0.95, 1],
            )

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
            body_left = worldbody.add_body(name=f'Wall_{i}_Left', mocap=True, pos=[0, 0, 0])
            body_left.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.wall_fixed_width / 2, 0.1, self.wall_heights[i]],
                rgba=[0.5, 0.5, 0.6, 1],
            )

            body_right = worldbody.add_body(name=f'Wall_{i}_Right', mocap=True, pos=[0, 0, 0])
            body_right.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[self.wall_fixed_width / 2, 0.1, self.wall_heights[i]],
                rgba=[0.5, 0.5, 0.6, 1],
            )

            if i > 0 or not self.no_initial_ramp:
                body_ramp = worldbody.add_body(
                    name=f'Ramp_{i}', mocap=True,
                    pos=[0, 0, 0],
                    euler=[self.ramp_angles[i], 0, 0],
                )
                body_ramp.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[1, self.ramp_length * 1.2 / 2, 0.1],
                    rgba=[0.5, 0.5, 0.6, 1],
                )

        self._maybe_add_payload_spec(spec)

        return spec

    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data)

        hidden_vars: list[float] = []
        self.reset_walls_and_ramps(data, model, hidden_vars)
        self.reset_poles(data, model, hidden_vars)

        mujoco.mj_forward(model, data)

        state['progress'] = self.compute_progress(data, state.get("units_active_mask"))
        state['hidden_vars'] = np.array(hidden_vars)

        return state, connections

    def reset_poles(self, data: mujoco.MjData, model: mujoco.MjModel, hidden_vars: list[float]) -> None:
        rng = self.rng
        for i, pole in enumerate(self.poles):
            pole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Pole_{i}')
            mocap_id = model.body_mocapid[pole_id]
            x, y = sample_pole_xy(pole, rng)
            hidden_vars.extend([x, y])
            data.mocap_pos[mocap_id] = [x, y, 0.0]

    def reset_walls_and_ramps(self, data: mujoco.MjData, model: mujoco.MjModel, hidden_vars: list[float]):
        rng = self.rng

        unusable_opening_offset = eval_fodp(self.unusable_opening_offset, rng)

        wall_y = eval_fodp(self.first_wall_distance, rng)
        for i in range(self.num_walls):
            if i != 0:
                wall_y += eval_fodp(self.inter_wall_distance, rng)
            hidden_vars.append(wall_y)

            opening_width = eval_fodp(self.opening_widths[i], rng)
            hidden_vars.append(opening_width)

            opening_x = (rng.random() - 0.5) * 2 * (
                    self.side_wall_x
                    - opening_width / 2
                    + unusable_opening_offset)  # small chance that there is no usable opening
                                                # -> must use ramp to continue
            hidden_vars.append(opening_x)

            first_wall_end_x = opening_x - opening_width / 2
            wall_left_pos_x = first_wall_end_x - (self.wall_fixed_width / 2)

            wall_left_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Wall_{i}_Left')
            mocap_id = model.body_mocapid[wall_left_id]
            data.mocap_pos[mocap_id] = [wall_left_pos_x, wall_y, 0.0]

            second_wall_start_x = opening_x + opening_width / 2

            wall_right_pos_x = second_wall_start_x + (self.wall_fixed_width / 2)

            wall_right_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Wall_{i}_Right')
            mocap_id = model.body_mocapid[wall_right_id]
            data.mocap_pos[mocap_id] = [wall_right_pos_x, wall_y, 0.0]

            if i > 0 or not self.no_initial_ramp:
                ramp_x = (rng.random() - 0.5) * 2 * self.ramp_range_x
                hidden_vars.append(ramp_x)

                ramp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Ramp_{i}')
                mocap_id = model.body_mocapid[ramp_id]
                data.mocap_pos[mocap_id] = [
                    ramp_x,
                    wall_y - self.ramp_distances_to_wall[i] / 2,
                    self.wall_heights[i] / 2 - 0.05,
                ]

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> SwarmObsDict:
        obs = super().get_obs(model, data, state, connections)

        obs['hidden_vars'] = state['hidden_vars'].copy()

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

def sample_pole_xy(pole: PoleSpec, rng: np.random.Generator) -> tuple[float, float]:
    if isinstance(pole, PoleParams):
        return eval_fodp(pole.x, rng), eval_fodp(pole.y, rng)

    if not isinstance(pole, CorrelatedPoleParams):
        raise TypeError(f"Unsupported pole spec: {type(pole).__name__}")

    mean = np.asarray(pole.mean, dtype=float)
    cov = np.asarray(pole.cov, dtype=float)
    if mean.shape != (2,) or cov.shape != (2, 2):
        raise ValueError(f"Expected mean shape (2,) and cov shape (2,2), got {mean.shape=} {cov.shape=}")

    if (pole.low is None) != (pole.high is None):
        raise ValueError(f"Expected low/high to be both set or both None, got {pole.low=} {pole.high=}")

    if pole.low is None:
        x, y = rng.multivariate_normal(mean=mean, cov=cov)
        return float(x), float(y)

    low = np.asarray(pole.low, dtype=float)
    high = np.asarray(pole.high, dtype=float)
    if low.shape != (2,) or high.shape != (2,):
        raise ValueError(f"Expected low/high shape (2,), got {low.shape=} {high.shape=}")
    if np.any(low > high):
        raise ValueError(f"Expected low <= high, got {pole.low=} {pole.high=}")

    if pole.sampling_mode == "clamp":
        xy = rng.multivariate_normal(mean=mean, cov=cov)
        xy = np.clip(xy, low, high)
        return float(xy[0]), float(xy[1])

    if pole.sampling_mode == "rejection":
        for _ in range(10_000):
            xy = rng.multivariate_normal(mean=mean, cov=cov)
            if np.all((xy >= low) & (xy <= high)):
                return float(xy[0]), float(xy[1])
        raise RuntimeError(f"Failed to sample pole position within bounds after many retries: {pole=}")

    raise ValueError(f"Unknown sampling_mode: {pole.sampling_mode!r}")
