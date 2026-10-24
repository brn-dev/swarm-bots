from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from swarmbots.mjw_env.scenarios.scenario_spaces import make_action_space, make_observation_space
from swarmbots.mjw_env.parameter_distributions import (
    BoundedDistParams,
    FloatOrBoundedDistParams,
    FloatOrDistParams,
)
from swarmbots.mjw_env.scenarios.base_mjw_scenario import (
    BaseMJWScenario,
    MJWRecordingCameraConfig,
    MJWRuntimeBindings,
)
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm
from swarmbots.mjw_env.scenarios.scenario_validation import finite_positive


@dataclass(frozen=True, slots=True)
class MJWFindOpeningRuntimeMetadata:
    barrier_mocap_id: int


@dataclass
class MJWFindOpeningScenario(BaseMJWScenario):
    swarm: MJWHomogeneousSwarm
    timestep: float
    action_repeat: int
    actuator_strength: float
    progress_reward_weight: float
    potential_reward_discount_factor: float
    include_connectors_xpos_in_obs: bool
    include_connectors_xquat_in_obs: bool
    quat_rot6d_representation: bool
    connection_dist_threshold: float
    connection_angle_threshold: float
    disconnect_potential_threshold: float
    friction: float | Iterable[float] | None
    reset_settle_time: float
    reset_settle_timestep_scale: float
    swarm_start_x: FloatOrDistParams
    swarm_start_y: FloatOrDistParams
    street_width: float
    wall_y: float
    wall_height: float
    wall_thickness: float
    wall_segment_width: float
    opening_width: float
    opening_x: FloatOrBoundedDistParams
    opening_y_margin: float
    success_reward: float
    opening_distance_reward_weight: float
    opening_distance_reward_falloff_distance: float | None = None
    wall_exploration_cell_count: int = 0
    wall_exploration_cell_reward: float = 0.0
    wall_exploration_cell_depth: float = 1.0
    continuous_connector_actions: bool = True
    seed: int | None = None
    compile_reward_kernel: bool = False
    reward_kernel_compile_mode: str = "default"
    randomize_initial_swarm_z_rotation: bool = False

    def __post_init__(self) -> None:
        self.reset_settle_timestep_scale = finite_positive(
            self.reset_settle_timestep_scale, name="reset_settle_timestep_scale",
        )
        if self.include_connectors_xquat_in_obs:
            raise ValueError("MJWFindOpeningScenario currently does not support connector quats in observations")
        if self.timestep <= 0.0:
            raise ValueError(f"Expected timestep > 0, got {self.timestep}")
        if self.action_repeat <= 0:
            raise ValueError(f"Expected action_repeat > 0, got {self.action_repeat}")

        self.street_width = float(self.street_width)
        self.wall_y = float(self.wall_y)
        self.wall_height = float(self.wall_height)
        self.wall_thickness = float(self.wall_thickness)
        self.wall_segment_width = float(self.wall_segment_width)
        self.opening_width = float(self.opening_width)
        self.opening_y_margin = float(self.opening_y_margin)
        self.success_reward = float(self.success_reward)
        self.opening_distance_reward_weight = float(self.opening_distance_reward_weight)
        self.opening_distance_reward_falloff_distance = (
            None
            if self.opening_distance_reward_falloff_distance is None
            else float(self.opening_distance_reward_falloff_distance)
        )
        self.wall_exploration_cell_count = int(self.wall_exploration_cell_count)
        self.wall_exploration_cell_reward = float(self.wall_exploration_cell_reward)
        self.wall_exploration_cell_depth = float(self.wall_exploration_cell_depth)
        self.potential_reward_discount_factor = float(self.potential_reward_discount_factor)

        if self.street_width <= 0.0:
            raise ValueError(f"Expected street_width > 0, got {self.street_width}")
        if self.wall_height <= 0.0:
            raise ValueError(f"Expected wall_height > 0, got {self.wall_height}")
        if self.wall_thickness <= 0.0:
            raise ValueError(f"Expected wall_thickness > 0, got {self.wall_thickness}")
        if self.wall_segment_width < self.street_width:
            raise ValueError(
                f"Expected wall_segment_width >= street_width, got "
                f"{self.wall_segment_width} and {self.street_width}"
            )
        if not 0.0 < self.opening_width < self.street_width:
            raise ValueError(
                f"Expected 0 < opening_width < street_width, got {self.opening_width} and {self.street_width}"
            )
        if self.opening_y_margin <= 0.0:
            raise ValueError(f"Expected opening_y_margin > 0, got {self.opening_y_margin}")
        if (
            self.opening_distance_reward_falloff_distance is not None
            and self.opening_distance_reward_falloff_distance <= 0.0
        ):
            raise ValueError(
                "Expected opening_distance_reward_falloff_distance > 0 or None, "
                f"got {self.opening_distance_reward_falloff_distance}"
            )
        if self.wall_exploration_cell_count < 0:
            raise ValueError(f"Expected wall_exploration_cell_count >= 0, got {self.wall_exploration_cell_count}")
        if self.wall_exploration_cell_depth <= 0.0:
            raise ValueError(f"Expected wall_exploration_cell_depth > 0, got {self.wall_exploration_cell_depth}")

        self.side_wall_x = self.street_width / 2.0
        self.inactive_area_location = (self.street_width * 2.0, 0.0, 0.1)
        self._validate_opening_x()

    def _validate_opening_x(self) -> None:
        min_x = -self.side_wall_x + (self.opening_width / 2.0)
        max_x = self.side_wall_x - (self.opening_width / 2.0)
        if isinstance(self.opening_x, BoundedDistParams):
            if self.opening_x.low < min_x or self.opening_x.high > max_x:
                raise ValueError(
                    f"Expected opening_x bounds within [{min_x}, {max_x}], got {self.opening_x}"
                )
            return
        if not isinstance(self.opening_x, (int, float)):
            raise TypeError(f"Expected opening_x to be float or BoundedDistParams, got {self.opening_x!r}")
        opening_x = float(self.opening_x)
        if opening_x < min_x or opening_x > max_x:
            raise ValueError(f"Expected opening_x within [{min_x}, {max_x}], got {opening_x}")

    def get_settings(self) -> dict[str, Any]:
        return {
            "scenario_type": "find_opening",
            "swarm": self.swarm.get_settings(),
            "timestep": self.timestep,
            "action_repeat": self.action_repeat,
            "actuator_strength": self.actuator_strength,
            "reward_weights": {
                "progress_reward_weight": self.progress_reward_weight,
            },
            "potential_reward_discount_factor": self.potential_reward_discount_factor,
            "include_connectors_xpos_in_obs": self.include_connectors_xpos_in_obs,
            "include_connectors_xquat_in_obs": self.include_connectors_xquat_in_obs,
            "quat_rot6d_representation": self.quat_rot6d_representation,
            "connection_dist_threshold": self.connection_dist_threshold,
            "connection_angle_threshold": self.connection_angle_threshold,
            "disconnect_potential_threshold": self.disconnect_potential_threshold,
            "continuous_connector_actions": self.continuous_connector_actions,
            "swarm_start_x": self.swarm_start_x,
            "swarm_start_y": self.swarm_start_y,
            "randomize_initial_swarm_z_rotation": self.randomize_initial_swarm_z_rotation,
            "friction": self.friction,
            "seed": self.seed,
            "reset_settle_time": self.reset_settle_time,
            "reset_settle_timestep_scale": self.reset_settle_timestep_scale,
            "street_width": self.street_width,
            "wall_y": self.wall_y,
            "wall_height": self.wall_height,
            "wall_thickness": self.wall_thickness,
            "wall_segment_width": self.wall_segment_width,
            "opening_width": self.opening_width,
            "opening_x": self.opening_x,
            "opening_y_margin": self.opening_y_margin,
            "success_reward": self.success_reward,
            "opening_distance_reward_weight": self.opening_distance_reward_weight,
            "opening_distance_reward_falloff_distance": self.opening_distance_reward_falloff_distance,
            "wall_exploration_cell_count": self.wall_exploration_cell_count,
            "wall_exploration_cell_reward": self.wall_exploration_cell_reward,
            "wall_exploration_cell_depth": self.wall_exploration_cell_depth,
            "compile_reward_kernel": self.compile_reward_kernel,
            "reward_kernel_compile_mode": self.reward_kernel_compile_mode,
        }

    def get_default_recording_camera_config(self) -> MJWRecordingCameraConfig | None:
        return MJWRecordingCameraConfig(
            lookat=(0.0, self.wall_y, max(0.4, self.wall_height * 0.4)),
            distance=max(4.0, min(10.0, self.street_width * 0.75)),
            azimuth=135.0,
            elevation=-35.0,
        )

    def build_model(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody = spec.worldbody

        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[100, 100, 0.1],
            rgba=[0.2, 0.3, 0.4, 1],
            pos=[0, 0, 0],
        )
        worldbody.add_light(pos=[0, 0, 100], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 100, 100], dir=[-1, -1, -1])

        side_wall_height = max(5.0, self.wall_height)
        for x in (-self.side_wall_x, self.side_wall_x):
            worldbody.add_geom(
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[0.1, 100, side_wall_height / 2.0],
                rgba=[0.3, 0.4, 0.5, 0.1],
                pos=[x, 0, side_wall_height / 2.0],
            )

        swarm_site = worldbody.add_site(pos=[0, 0, 0], name="swarm_site")
        spec.attach(self.swarm.create_swarm_spec(seed=self.seed), "", site=swarm_site)

        barrier_body = worldbody.add_body(name="FindOpeningBarrier", mocap=True, pos=[0, 0, 0])
        segment_half_width = self.wall_segment_width / 2.0
        segment_center_offset = (self.opening_width / 2.0) + segment_half_width
        for name, x in (
            ("FindOpeningBarrier-left", -segment_center_offset),
            ("FindOpeningBarrier-right", segment_center_offset),
        ):
            barrier_body.add_geom(
                name=name,
                type=mujoco.mjtGeom.mjGEOM_BOX,
                size=[segment_half_width, self.wall_thickness / 2.0, self.wall_height / 2.0],
                rgba=[0.5, 0.5, 0.6, 1],
                pos=[x, 0, self.wall_height / 2.0],
            )

        model = spec.compile()
        model.opt.timestep = float(self.timestep)
        if self.friction is not None:
            if isinstance(self.friction, (int, float)):
                friction = np.asarray([float(self.friction), 0.005, 0.0001], dtype=float)
            else:
                friction = np.asarray(tuple(float(v) for v in self.friction), dtype=float)
            model.geom_friction[:] = friction
        return model

    def get_single_observation_space(self) -> spaces.Dict:
        return make_observation_space(
            self,
            hidden_global_vars_dim=1 + self.wall_exploration_cell_count,
        )

    def get_single_action_space(self) -> spaces.Dict:
        return make_action_space(self)

    def get_batched_observation_space(self, num_envs: int) -> spaces.Dict:
        return batch_space(self.get_single_observation_space(), n=num_envs)

    def get_batched_action_space(self, num_envs: int) -> spaces.Dict:
        return batch_space(self.get_single_action_space(), n=num_envs)

    def build_runtime_metadata(self, *, host_model: mujoco.MjModel) -> MJWFindOpeningRuntimeMetadata:
        barrier_body_id = mujoco.mj_name2id(
            host_model,
            mujoco.mjtObj.mjOBJ_BODY,
            "FindOpeningBarrier",
        )
        if barrier_body_id < 0:
            raise ValueError("Find-opening barrier body not found in model")
        return MJWFindOpeningRuntimeMetadata(
            barrier_mocap_id=int(host_model.body_mocapid[barrier_body_id]),
        )

    def create_runtime(self, *, bindings: MJWRuntimeBindings, runtime_metadata: Any) -> Any:
        from swarmbots.mjw_env.scenarios.mjw_find_opening_runtime import FindOpeningMJWScenarioRuntime

        return FindOpeningMJWScenarioRuntime(
            scenario=self,
            bindings=bindings,
            runtime_metadata=runtime_metadata,
        )
