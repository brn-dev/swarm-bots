from dataclasses import dataclass
from typing import Iterable, Literal

import jax
import jax.numpy as jnp
import mujoco
import numpy as np

from swarmbots.mjx_env.mjx_float_or_dist_params import (
    MjxFloatOrBoundedDistParams,
    MjxFloatOrDistParams,
    mjx_eval_fodp,
    mjx_fodp_low,
)
from swarmbots.mjx_env.scenarios.mjx_base_scenario import MjxActuatorsActivationRewardType, MjxImplConfig, _ResetSample
from swarmbots.mjx_env.scenarios.mjx_payload_scenario import MjxPayloadScenario
from swarmbots.mjx_env.swarm.mjx_base_swarm import MjxBaseSwarm
from swarmbots.mjx_env.types import MjxActDict, MjxEnvState, MjxObsDict


@dataclass(frozen=True, slots=True)
class MjxPoleParams:
    x: MjxFloatOrDistParams
    y: MjxFloatOrDistParams


MjxTruncatedMultivariateNormal2DSamplingMode = Literal["clamp", "rejection"]


@dataclass(frozen=True, slots=True)
class MjxCorrelatedPoleParams:
    mean: tuple[float, float]
    cov: tuple[tuple[float, float], tuple[float, float]]
    low: tuple[float, float] | None = None
    high: tuple[float, float] | None = None
    sampling_mode: MjxTruncatedMultivariateNormal2DSamplingMode = "clamp"


MjxPoleSpec = MjxPoleParams | MjxCorrelatedPoleParams


class MjxObstacleStreetScenario(MjxPayloadScenario):
    def __init__(
        self,
        swarm: MjxBaseSwarm,
        payload_type: None | str,
        payload_size: Iterable[float] = (0.2, 0.2, 0.2),
        payload_mass: float = 5.0,
        payload_start_location_offset: Iterable[float] = (0, 1, 0),
        poles: Iterable[MjxPoleSpec | tuple[MjxFloatOrDistParams, MjxFloatOrDistParams]] = (),
        pole_radius: float = 0.1,
        pole_height: float = 1.0,
        num_walls: int = 3,
        wall_height: float | list[float] = 0.5,
        inter_wall_distance: MjxFloatOrBoundedDistParams = 4.0,
        first_wall_distance: MjxFloatOrDistParams = 2.0,
        opening_width: MjxFloatOrDistParams | list[MjxFloatOrDistParams] = 2.0,
        unusable_opening_offset: MjxFloatOrDistParams = 2.0,
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
        wall_pass_reward_weight: float = 0.0,
        wall_pass_thresholds: list[float] | None = None,
        hinge_qvel_magnitude_reward_weight: float = 0.0,
        hinge_qvel_magnitude_reward_threshold: float = 0.0,
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
        reset_settle_time: float = 0,
        reset_settle_timestep_scale: float = 1.0,
        swarm_start_x: MjxFloatOrDistParams = 0.0,
        swarm_start_y: MjxFloatOrDistParams = 0.0,
        seed: int | None = None,
        reset_pool_size: int = 256,
        mjx_impl: MjxImplConfig = None,
    ) -> None:
        self.poles: list[MjxPoleSpec] = []
        for pole in poles:
            if isinstance(pole, tuple):
                self.poles.append(MjxPoleParams(x=pole[0], y=pole[1]))
            else:
                self.poles.append(pole)
        self.pole_radius = float(pole_radius)
        self.pole_height = float(pole_height)
        self.num_walls = int(num_walls)
        self.no_initial_ramp = bool(no_initial_ramp)
        self.street_width = float(street_width)
        self.side_wall_x = self.street_width / 2.0
        self.wall_fixed_width = 25.0
        self.wall_heights = wall_height if isinstance(wall_height, list) else [float(wall_height)] * self.num_walls
        self.inter_wall_distance = inter_wall_distance
        self.first_wall_distance = first_wall_distance
        self.opening_widths = opening_width if isinstance(opening_width, list) else [opening_width] * self.num_walls
        self.unusable_opening_offset = unusable_opening_offset
        self.wall_pass_reward_weight = float(wall_pass_reward_weight)
        self.wall_pass_thresholds = np.sort(np.asarray([0.0] if wall_pass_thresholds is None else wall_pass_thresholds, dtype=float))

        min_inter_wall_distance = mjx_fodp_low(self.inter_wall_distance)
        if min_inter_wall_distance <= 1.0:
            raise ValueError(f"Expected inter_wall_distance.low > 1.0, got {min_inter_wall_distance}")
        self.ramp_length = min_inter_wall_distance - 1.0
        self.ramp_range_x = self.side_wall_x - 1.5
        self.ramp_angles = [np.asin(height / self.ramp_length) + np.pi / 64 for height in self.wall_heights]
        self.ramp_distances_to_wall = [self.ramp_length * np.cos(angle) for angle in self.ramp_angles]

        super().__init__(
            swarm=swarm,
            payload_type=payload_type,
            payload_size=payload_size,
            payload_mass=payload_mass,
            payload_start_location_offset=payload_start_location_offset,
            actuator_strength=actuator_strength,
            progress_reward_weight=progress_reward_weight,
            guidance_reward_weight=guidance_reward_weight,
            hinge_qvel_magnitude_reward_weight=hinge_qvel_magnitude_reward_weight,
            hinge_qvel_magnitude_reward_threshold=hinge_qvel_magnitude_reward_threshold,
            units_without_connections_reward_weight=units_without_connections_reward_weight,
            units_with_double_connection_reward_weight=units_with_double_connection_reward_weight,
            movement_reward_weight=movement_reward_weight,
            height_reward_weight=height_reward_weight,
            connectors_stayed_active_reward_weight=connectors_stayed_active_reward_weight,
            connectors_successfully_activated_reward_weight=connectors_successfully_activated_reward_weight,
            connectors_unsuccessfully_activated_reward_weight=connectors_unsuccessfully_activated_reward_weight,
            connectors_deactivated_reward_weight=connectors_deactivated_reward_weight,
            average_connectors_reward=average_connectors_reward,
            include_connectors_xpos_in_obs=include_connectors_xpos_in_obs,
            include_connectors_xquat_in_obs=include_connectors_xquat_in_obs,
            quat_rot6d_representation=quat_rot6d_representation,
            connection_dist_threshold=connection_dist_threshold,
            connection_angle_threshold=connection_angle_threshold,
            disconnect_potential_threshold=disconnect_potential_threshold,
            friction=friction,
            force_elliptic_cone=force_elliptic_cone,
            reset_settle_time=reset_settle_time,
            reset_settle_timestep_scale=reset_settle_timestep_scale,
            swarm_start_x=swarm_start_x,
            swarm_start_y=swarm_start_y,
            inactive_area_location=[street_width * 2, 0, 0.1],
            seed=seed,
            reset_pool_size=reset_pool_size,
            mjx_impl=mjx_impl,
        )

    def _create_scenario_spec(self) -> mujoco.MjSpec:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody = spec.worldbody
        worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[100, 100, 0.1], rgba=[0.2, 0.3, 0.4, 1], pos=[0, 0, 0])
        worldbody.add_light(pos=[0, 0, 100], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 100, 100], dir=[-1, -1, -1])

        for pole_idx, _ in enumerate(self.poles):
            body = worldbody.add_body(name=f"Pole_{pole_idx}", mocap=True, pos=[0, 0, 0])
            body.add_geom(
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                fromto=[0, 0, 0, 0, 0, self.pole_height],
                size=[self.pole_radius, 0, 0],
                rgba=[0.15, 0.8, 0.95, 1],
            )

        for x in (self.side_wall_x, -self.side_wall_x):
            worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.1, 100, 5],
                rgba=[0.3, 0.4, 0.5, 0.1],
                pos=[x, 0, 0],
                contype=0,
                conaffinity=0,
            )

        for wall_idx in range(self.num_walls):
            for side in ("Left", "Right"):
                body = worldbody.add_body(name=f"Wall_{wall_idx}_{side}", mocap=True, pos=[0, 0, 0])
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[self.wall_fixed_width / 2, 0.1, self.wall_heights[wall_idx]],
                    rgba=[0.5, 0.5, 0.6, 1],
                )
            if wall_idx > 0 or not self.no_initial_ramp:
                ramp = worldbody.add_body(
                    name=f"Ramp_{wall_idx}",
                    mocap=True,
                    pos=[0, 0, 0],
                    euler=[self.ramp_angles[wall_idx], 0, 0],
                )
                ramp.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_BOX,
                    size=[1, self.ramp_length * 1.2 / 2, 0.1],
                    rgba=[0.5, 0.5, 0.6, 1],
                )

        self._maybe_add_payload_spec(spec)
        return spec

    def _augment_reset_sample(self, sample: _ResetSample, swarm_start_location: np.ndarray) -> _ResetSample:
        sample = super()._augment_reset_sample(sample, swarm_start_location)
        hidden: list[float] = []
        mocap_pos = sample.mocap_pos.copy()
        wall_y = self._reset_walls_and_ramps(mocap_pos, hidden)
        self._reset_poles(mocap_pos, hidden)
        thresholds = self._compute_wall_pass_thresholds_np(wall_y)
        return sample._replace(
            mocap_pos=mocap_pos,
            hidden_global_vars=np.asarray(hidden, dtype=np.float32),
            passed_thresholds_mask=np.zeros((self.num_units, thresholds.shape[0]), dtype=bool),
            next_threshold_for_unit=np.zeros((self.num_units,), dtype=np.int32),
            wall_pass_absolute_thresholds=thresholds.astype(np.float32),
        )

    def _reset_poles(self, mocap_pos: np.ndarray, hidden: list[float]) -> None:
        for pole_idx, pole in enumerate(self.poles):
            x, y = _sample_pole_xy(pole, self.rng)
            hidden.extend([x, y])
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"Pole_{pole_idx}")
            mocap_pos[self.model.body_mocapid[body_id]] = [x, y, 0.0]

    def _reset_walls_and_ramps(self, mocap_pos: np.ndarray, hidden: list[float]) -> np.ndarray:
        wall_y_values = np.zeros((self.num_walls,), dtype=float)
        unusable_opening_offset = mjx_eval_fodp(self.unusable_opening_offset, self.rng)
        wall_y = mjx_eval_fodp(self.first_wall_distance, self.rng)
        for wall_idx in range(self.num_walls):
            if wall_idx != 0:
                wall_y += mjx_eval_fodp(self.inter_wall_distance, self.rng)
            hidden.append(wall_y)
            wall_y_values[wall_idx] = wall_y
            opening_width = mjx_eval_fodp(self.opening_widths[wall_idx], self.rng)
            hidden.append(opening_width)
            opening_x = (self.rng.random() - 0.5) * 2.0 * (self.side_wall_x - opening_width / 2.0 + unusable_opening_offset)
            hidden.append(opening_x)
            left_x = opening_x - opening_width / 2.0 - self.wall_fixed_width / 2.0
            right_x = opening_x + opening_width / 2.0 + self.wall_fixed_width / 2.0
            for side, x_pos in (("Left", left_x), ("Right", right_x)):
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"Wall_{wall_idx}_{side}")
                mocap_pos[self.model.body_mocapid[body_id]] = [x_pos, wall_y, 0.0]
            if wall_idx > 0 or not self.no_initial_ramp:
                ramp_x = (self.rng.random() - 0.5) * 2.0 * self.ramp_range_x
                hidden.append(ramp_x)
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"Ramp_{wall_idx}")
                mocap_pos[self.model.body_mocapid[body_id]] = [
                    ramp_x,
                    wall_y - self.ramp_distances_to_wall[wall_idx] / 2.0,
                    self.wall_heights[wall_idx] / 2.0 - 0.05,
                ]
        return wall_y_values

    def get_obs(self, state: MjxEnvState) -> MjxObsDict:
        obs = super().get_obs(state)
        obs["hidden_global_vars"] = state.hidden_global_vars
        obs["hidden_local_vars"] = state.passed_thresholds_mask.astype(jnp.float32)
        return obs

    def _hidden_local_vars_shape(self) -> tuple[int, int]:
        return self.num_units, self.num_walls * int(self.wall_pass_thresholds.size)

    def _hidden_global_vars_size(self) -> int:
        ramp_count = self.num_walls if not self.no_initial_ramp else max(self.num_walls - 1, 0)
        return len(self.poles) * 2 + self.num_walls * 3 + ramp_count

    def compute_progress(self, state: MjxEnvState) -> jax.Array:
        if self.payload_type is None:
            unit_y = state.data.qpos[self.indices.qpos_indices[:, 1]]
            active = state.units_active_mask
            return jnp.sum(jnp.where(active, unit_y, 0.0)) / jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
        return state.data.xpos[self.payload_body_id, 1]

    def finalize_reset_state(self, state: MjxEnvState) -> MjxEnvState:
        state = super().finalize_reset_state(state)
        unit_y = state.data.qpos[self.indices.qpos_indices[:, 1]]
        passed = unit_y[:, None] > state.wall_pass_absolute_thresholds[None, :]
        return state._replace(
            passed_thresholds_mask=passed,
            next_threshold_for_unit=jnp.sum(passed.astype(jnp.int32), axis=1),
        )

    def evaluate_step(self, state: MjxEnvState, action: MjxActDict) -> tuple[MjxEnvState, jax.Array, jax.Array]:
        state = self.enforce_inactive_units_state(state)
        old_progress = state.progress
        new_progress = self.compute_progress(state)
        progress_reward = new_progress - old_progress
        wall_pass_reward, state = self._compute_wall_pass_reward(state)
        guidance_reward, state = self.compute_guidance_reward(state, action)
        weighted_progress_reward = progress_reward * self.reward_weights["progress_reward_weight"] + wall_pass_reward
        weighted_guidance_reward = guidance_reward * self.reward_weights["guidance_reward_weight"]
        state = state._replace(
            progress=new_progress,
            weighted_progress_reward=weighted_progress_reward,
            weighted_guidance_reward=weighted_guidance_reward,
        )
        return state, weighted_progress_reward + weighted_guidance_reward, jnp.array(False)

    def _compute_wall_pass_reward(self, state: MjxEnvState) -> tuple[jax.Array, MjxEnvState]:
        unit_y = state.data.qpos[self.indices.qpos_indices[:, 1]]
        active = state.units_active_mask
        passed_now = unit_y[:, None] > state.wall_pass_absolute_thresholds[None, :]
        new_passed = passed_now & (~state.passed_thresholds_mask) & active[:, None]
        num_passed = jnp.sum(new_passed.astype(jnp.int32))
        active_count = jnp.maximum(jnp.sum(active.astype(jnp.float32)), 1.0)
        thresholds_per_wall = max(int(self.wall_pass_thresholds.size), 1)
        reward = num_passed.astype(jnp.float32) / (active_count * thresholds_per_wall) * self.wall_pass_reward_weight
        return reward, state._replace(
            passed_thresholds_mask=passed_now,
            next_threshold_for_unit=jnp.sum(passed_now.astype(jnp.int32), axis=1),
            num_walls_passed=num_passed,
            walls_passed_reward=reward,
        )

    def _compute_wall_pass_thresholds_np(self, wall_y: np.ndarray) -> np.ndarray:
        if wall_y.size == 0:
            return np.asarray([], dtype=float)
        return np.sort((wall_y[:, None] + self.wall_pass_thresholds[None, :]).reshape(-1))


def _sample_pole_xy(pole: MjxPoleSpec, rng: np.random.Generator) -> tuple[float, float]:
    if isinstance(pole, MjxPoleParams):
        return mjx_eval_fodp(pole.x, rng), mjx_eval_fodp(pole.y, rng)
    mean = np.asarray(pole.mean, dtype=float)
    cov = np.asarray(pole.cov, dtype=float)
    if pole.low is None:
        xy = rng.multivariate_normal(mean=mean, cov=cov)
        return float(xy[0]), float(xy[1])
    low = np.asarray(pole.low, dtype=float)
    high = np.asarray(pole.high, dtype=float)
    if pole.sampling_mode == "clamp":
        xy = np.clip(rng.multivariate_normal(mean=mean, cov=cov), low, high)
        return float(xy[0]), float(xy[1])
    for _ in range(10_000):
        xy = rng.multivariate_normal(mean=mean, cov=cov)
        if np.all((xy >= low) & (xy <= high)):
            return float(xy[0]), float(xy[1])
    raise RuntimeError(f"Failed to sample pole position within bounds: {pole}")
