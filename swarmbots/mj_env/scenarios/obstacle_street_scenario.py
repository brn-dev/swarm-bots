from dataclasses import dataclass
from typing import Any, Iterable, Literal

import mujoco
import numpy as np

from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams, eval_fodp, fodp_low, FloatOrBoundedDistParams
from swarmbots.mj_env.scenarios.base_scenario import (
    BaseScenario,
    SwarmActDict,
    SwarmObsDict,
)
from swarmbots.mj_env.swarm.base_swarm import BaseSwarm
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


class ObstacleStreetScenario(BaseScenario):

    def __init__(
            self,
            swarm: BaseSwarm,
            timestep: float = 0.002,
            action_repeat: int = 15,
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
            forward_reward_weight: float = 1.0,
            forward_reward_max_y: float | None = None,
            guidance_reward_weight: float = 1.0,
            wall_pass_reward_weight: float = 0.0,
            wall_pass_thresholds: list[float] | None = None,
            units_without_connections_reward_weight: float = 0.0,
            include_connectors_xpos_in_obs: bool = True,
            include_connectors_xquat_in_obs: bool = False,
            quat_rot6d_representation: bool = True,
            reset_settle_time: float = 0,
            reset_settle_timestep_scale: float = 1.0,
            swarm_start_x: FloatOrDistParams = 0.0,
            swarm_start_y: FloatOrDistParams = 0.0,
            randomize_initial_swarm_z_rotation: bool = False,
            seed: int | None = None,
    ) -> None:
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
        self.forward_reward_weight = float(forward_reward_weight)
        self.forward_reward_max_y = None if forward_reward_max_y is None else float(forward_reward_max_y)
        self.wall_pass_reward_weight = float(wall_pass_reward_weight)
        if wall_pass_thresholds is None:
            wall_pass_thresholds = [0.0]
        self.wall_pass_thresholds = np.sort(wall_pass_thresholds)
        min_inter_wall_distance = fodp_low(self.inter_wall_distance)
        if min_inter_wall_distance <= 1.0:
            raise ValueError(f"Expected inter_wall_distance.low > 1.0, got {min_inter_wall_distance}")
        self.ramp_length = min_inter_wall_distance - 1.0
        self.ramp_range_x = self.side_wall_x - 1.5
        self.ramp_angles = [np.asin(wh / self.ramp_length) + np.pi/64 for wh in self.wall_heights]
        self.ramp_distances_to_wall = [self.ramp_length * np.cos(ra) for ra in self.ramp_angles]

        super().__init__(
            swarm=swarm,
            timestep=timestep,
            action_repeat=action_repeat,
            actuator_strength=actuator_strength,
            progress_reward_weight=progress_reward_weight,
            guidance_reward_weight=guidance_reward_weight,
            units_without_connections_reward_weight=units_without_connections_reward_weight,
            seed=seed,
            include_connectors_xpos_in_obs=include_connectors_xpos_in_obs,
            include_connectors_xquat_in_obs=include_connectors_xquat_in_obs,
            quat_rot6d_representation=quat_rot6d_representation,
            friction=friction,
            connection_dist_threshold=connection_dist_threshold,
            connection_angle_threshold=connection_angle_threshold,
            disconnect_potential_threshold=disconnect_potential_threshold,
            force_elliptic_cone=force_elliptic_cone,
            reset_settle_time=reset_settle_time,
            reset_settle_timestep_scale=reset_settle_timestep_scale,
            swarm_start_x=swarm_start_x,
            swarm_start_y=swarm_start_y,
            randomize_initial_swarm_z_rotation=randomize_initial_swarm_z_rotation,
            inactive_area_location=[street_width * 2, 0, 0.1],
            _reset_in_init=False
        )

        self._dummy_state, self._dummy_connections = self.reset_scenario(self.dummy_model, self.dummy_data)

    def get_settings(self) -> dict[str, Any]:
        settings = super().get_settings()
        settings.update({
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
            'forward_reward_weight': self.forward_reward_weight,
            'forward_reward_max_y': self.forward_reward_max_y,
            'wall_pass_reward_weight': self.wall_pass_reward_weight,
            'wall_pass_thresholds': self.wall_pass_thresholds.tolist(),
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

        return spec

    def reset_scenario(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            settle: bool = True,
    ) -> tuple[dict, SwarmConnections]:
        state, connections = super().reset_scenario(model, data, settle=False)

        hidden_global_vars: list[float] = []
        wall_y = self.reset_walls_and_ramps(data, model, hidden_global_vars)
        self.reset_poles(data, model, hidden_global_vars)

        mujoco.mj_forward(model, data)
        if settle:
            self.settle_reset(model, data, state)

        state['progress'] = self._compute_forward_progress_baseline(data, state.get("units_active_mask"))
        state['hidden_global_vars'] = np.array(hidden_global_vars, dtype=float)
        state['wall_y'] = wall_y
        wall_pass_absolute_thresholds = self._compute_wall_pass_thresholds(wall_y)
        state['wall_pass_absolute_thresholds'] = wall_pass_absolute_thresholds
        unit_y = np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
        passed_thresholds_mask = (
            unit_y[:, np.newaxis] > wall_pass_absolute_thresholds[np.newaxis, :]
            if wall_pass_absolute_thresholds.size > 0
            else np.zeros((self.num_units, 0), dtype=bool)
        )
        state['passed_thresholds_mask'] = passed_thresholds_mask
        state['next_threshold_for_unit'] = passed_thresholds_mask.sum(axis=1).astype(int)
        state['num_walls_passed'] = 0
        state['walls_passed_reward'] = 0.0

        return state, connections

    def reset_poles(self, data: mujoco.MjData, model: mujoco.MjModel, hidden_global_vars: list[float]) -> None:
        rng = self.rng
        for i, pole in enumerate(self.poles):
            pole_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Pole_{i}')
            mocap_id = model.body_mocapid[pole_id]
            x, y = sample_pole_xy(pole, rng)
            hidden_global_vars.extend([x, y])
            data.mocap_pos[mocap_id] = [x, y, 0.0]

    def reset_walls_and_ramps(
            self,
            data: mujoco.MjData,
            model: mujoco.MjModel,
            hidden_global_vars: list[float]
    ) -> np.ndarray:
        rng = self.rng
        wall_y_values = np.zeros(self.num_walls, dtype=float)

        unusable_opening_offset = eval_fodp(self.unusable_opening_offset, rng)

        wall_y = eval_fodp(self.first_wall_distance, rng)
        for i in range(self.num_walls):
            if i != 0:
                wall_y += eval_fodp(self.inter_wall_distance, rng)
            hidden_global_vars.append(wall_y)
            wall_y_values[i] = wall_y

            opening_width = eval_fodp(self.opening_widths[i], rng)
            hidden_global_vars.append(opening_width)

            opening_x = (rng.random() - 0.5) * 2 * (
                    self.side_wall_x
                    - opening_width / 2
                    + unusable_opening_offset)  # small chance that there is no usable opening
                                                # -> must use ramp to continue
            hidden_global_vars.append(opening_x)

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
                hidden_global_vars.append(ramp_x)

                ramp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f'Ramp_{i}')
                mocap_id = model.body_mocapid[ramp_id]
                data.mocap_pos[mocap_id] = [
                    ramp_x,
                    wall_y - self.ramp_distances_to_wall[i] / 2,
                    self.wall_heights[i] / 2 - 0.05,
                ]

        return wall_y_values

    def get_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> SwarmObsDict:
        obs = super().get_obs(model, data, state, connections)

        obs['hidden_global_vars'] = state['hidden_global_vars'].copy()
        obs['hidden_local_vars'] = np.asarray(state['passed_thresholds_mask'], dtype=float).copy()

        return obs

    def _compute_wall_pass_reward(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> float:
        wall_pass_absolute_thresholds = np.asarray(state.get("wall_pass_absolute_thresholds"), dtype=float)
        next_threshold_for_unit = np.asarray(state.get("next_threshold_for_unit"), dtype=int)
        if wall_pass_absolute_thresholds.size == 0 or next_threshold_for_unit.shape != (self.num_units,):
            state["num_walls_passed"] = 0
            state["walls_passed_reward"] = 0.0
            return 0.0

        unit_y = np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
        units_active_mask = state.get("units_active_mask")
        active_units_mask = None if units_active_mask is None else np.asarray(units_active_mask, dtype=bool)
        active_units_count = self.num_units if active_units_mask is None else int(active_units_mask.sum())
        thresholds_per_wall = int(self.wall_pass_thresholds.size)

        num_walls_passed = 0
        for unit_idx in range(self.num_units):
            if active_units_mask is not None and not active_units_mask[unit_idx]:
                continue
            next_threshold_idx = int(next_threshold_for_unit[unit_idx])
            while (next_threshold_idx < wall_pass_absolute_thresholds.size
                   and unit_y[unit_idx] > wall_pass_absolute_thresholds[next_threshold_idx]):
                next_threshold_idx += 1
                num_walls_passed += 1
            next_threshold_for_unit[unit_idx] = next_threshold_idx

        if active_units_count > 0 and thresholds_per_wall > 0:
            walls_passed_reward = (
                num_walls_passed / (active_units_count * thresholds_per_wall)
            ) * self.wall_pass_reward_weight
        else:
            walls_passed_reward = 0.0
        state["next_threshold_for_unit"] = next_threshold_for_unit
        if wall_pass_absolute_thresholds.size > 0:
            unit_threshold_indices = np.arange(wall_pass_absolute_thresholds.size, dtype=int)
            state["passed_thresholds_mask"] = (
                unit_threshold_indices[np.newaxis, :] < next_threshold_for_unit[:, np.newaxis]
            )
        else:
            state["passed_thresholds_mask"] = np.zeros((self.num_units, 0), dtype=bool)
        state["num_walls_passed"] = num_walls_passed
        state["walls_passed_reward"] = walls_passed_reward

        return walls_passed_reward

    def compute_progress_reward(
            self,
            data: mujoco.MjData,
            state: dict,
    ) -> float:
        old_progress = state["progress"]
        new_progress = self._compute_forward_progress_baseline(data, state.get("units_active_mask"))
        state["progress"] = new_progress

        forward_reward = new_progress - old_progress
        wall_pass_reward = self._compute_wall_pass_reward(data, state)
        progress_reward = forward_reward * self.forward_reward_weight + wall_pass_reward

        state['forward_reward'] = forward_reward
        state['progress_reward'] = progress_reward
        state['wall_pass_reward'] = wall_pass_reward
        return progress_reward

    def evaluate_step(
            self,
            action: SwarmActDict,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: dict,
            connections: SwarmConnections
    ) -> tuple[float, bool]:
        units_active_mask = state.get("units_active_mask")
        if units_active_mask is not None:
            self._enforce_inactive_units_state(model, data, units_active_mask)

        self.compute_progress_reward(data, state)
        forward_reward = state['forward_reward']
        wall_pass_reward = state['wall_pass_reward']
        progress_reward = state['progress_reward']

        guidance_reward = super().compute_guidance_reward(data, action, state, connections)
        state['guidance_reward'] = guidance_reward

        progress_reward_weight = self.reward_weights['progress_reward_weight']
        weighted_progress_reward = progress_reward * progress_reward_weight
        weighted_forward_reward = forward_reward * self.forward_reward_weight * progress_reward_weight
        weighted_wall_pass_reward = wall_pass_reward * progress_reward_weight
        weighted_guidance_reward = guidance_reward * self.reward_weights['guidance_reward_weight']
        state['weighted_progress_reward'] = weighted_progress_reward
        state['weighted_forward_reward'] = weighted_forward_reward
        state['weighted_forward_progress_reward'] = weighted_forward_reward
        state['weighted_wall_pass_reward'] = weighted_wall_pass_reward
        state['weighted_guidance_reward'] = weighted_guidance_reward
        state['reward_terms'] = {
            'forward': weighted_forward_reward,
            'wall': weighted_wall_pass_reward,
            'guidance': weighted_guidance_reward,
        }

        return weighted_progress_reward + weighted_guidance_reward, False

    def _compute_wall_pass_thresholds(self, wall_y: np.ndarray) -> np.ndarray:
        if wall_y.size == 0:
            return np.asarray([], dtype=float)
        wall_thresholds = wall_y[:, np.newaxis] + self.wall_pass_thresholds[np.newaxis, :]
        return np.sort(wall_thresholds.reshape(-1))

    def _compute_forward_progress_baseline(
            self,
            data: mujoco.MjData,
            units_active_mask: np.ndarray | None,
    ) -> float:
        unit_y = np.asarray(data.qpos[self._qpos_indices[:, 1]], dtype=float)
        if self.forward_reward_max_y is not None:
            unit_y = np.minimum(unit_y, self.forward_reward_max_y)
        if units_active_mask is None:
            return float(unit_y.mean())
        active_units_mask = np.asarray(units_active_mask, dtype=bool)
        if not active_units_mask.any():
            return 0.0
        return float(unit_y[active_units_mask].mean())

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
