from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

from swarmbots.mjw_env.scenarios.scenario_spaces import make_action_space, make_observation_space
from swarmbots.mjw_env.scenarios.scenario_validation import finite_positive
from swarmbots.mjw_env.parameter_distributions import (
    BoundedDistParams,
    FloatOrBoundedDistParams,
    FloatOrDistParams,
)
from swarmbots.mjw_env.scenarios.base_mjw_scenario import BaseMJWScenario, MJWRecordingCameraConfig, MJWRuntimeBindings
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm


@dataclass(frozen=True, slots=True)
class MJWBridgeRuntimeMetadata:
    bridge_mocap_id: int


@dataclass
class MJWBridgeScenario(BaseMJWScenario):
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
    bridge_width: float
    bridge_length: float
    bridge_x: FloatOrBoundedDistParams
    platform_length: float
    platform_height: float
    fall_z_threshold: float
    success_margin: float
    success_reward: float
    fell_off_bridge_reward: float
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
            raise ValueError("MJWBridgeScenario currently does not support connector quats in observations")
        if self.timestep <= 0.0:
            raise ValueError(f"Expected timestep > 0, got {self.timestep}")
        if self.action_repeat <= 0:
            raise ValueError(f"Expected action_repeat > 0, got {self.action_repeat}")
        self.street_width = float(self.street_width)
        self.bridge_width = float(self.bridge_width)
        self.bridge_length = float(self.bridge_length)
        self.platform_length = float(self.platform_length)
        self.platform_height = float(self.platform_height)
        self.fall_z_threshold = float(self.fall_z_threshold)
        self.success_margin = float(self.success_margin)
        self.success_reward = float(self.success_reward)
        self.fell_off_bridge_reward = float(self.fell_off_bridge_reward)
        self.potential_reward_discount_factor = float(self.potential_reward_discount_factor)
        if self.street_width <= 0.0:
            raise ValueError(f"Expected street_width > 0, got {self.street_width}")
        if self.bridge_width <= 0.0:
            raise ValueError(f"Expected bridge_width > 0, got {self.bridge_width}")
        if self.bridge_width >= self.street_width:
            raise ValueError(f"Expected bridge_width < street_width, got {self.bridge_width} >= {self.street_width}")
        if self.bridge_length <= 0.0:
            raise ValueError(f"Expected bridge_length > 0, got {self.bridge_length}")
        if self.platform_length <= 0.0:
            raise ValueError(f"Expected platform_length > 0, got {self.platform_length}")
        if self.platform_height <= 0.0:
            raise ValueError(f"Expected platform_height > 0, got {self.platform_height}")
        if self.success_margin < 0.0:
            raise ValueError(f"Expected success_margin >= 0, got {self.success_margin}")

        self.side_wall_x = self.street_width / 2.0
        self.platform1_min_y = -50.0
        self.platform1_max_y = self.platform_length / 2.0
        self.platform1_center_y = (self.platform1_min_y + self.platform1_max_y) / 2.0
        self.platform1_half_length = (self.platform1_max_y - self.platform1_min_y) / 2.0
        self.platform2_center_y = self.platform_length + self.bridge_length
        self.bridge_center_y = (self.platform_length / 2.0) + (self.bridge_length / 2.0)
        self.bridge_y_max = (self.platform_length / 2.0) + self.bridge_length
        self.success_y = self.bridge_y_max + self.success_margin
        self.platform_top_z = 0.0
        self.inactive_area_location = (-self.street_width * 1.5, 0.0, 0.1)
        self._validate_bridge_x()

    def _validate_bridge_x(self) -> None:
        min_x = -self.side_wall_x + (self.bridge_width / 2.0)
        max_x = self.side_wall_x - (self.bridge_width / 2.0)
        if isinstance(self.bridge_x, BoundedDistParams):
            if self.bridge_x.low < min_x or self.bridge_x.high > max_x:
                raise ValueError(f"Expected bridge_x bounds within [{min_x}, {max_x}], got {self.bridge_x}")
            return
        if not isinstance(self.bridge_x, (int, float)):
            raise TypeError(f"Expected bridge_x to be float or BoundedDistParams, got {self.bridge_x!r}")
        bridge_x = float(self.bridge_x)
        if bridge_x < min_x or bridge_x > max_x:
            raise ValueError(f"Expected bridge_x within [{min_x}, {max_x}], got {bridge_x}")

    def get_settings(self) -> dict[str, Any]:
        return {
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
            "bridge_width": self.bridge_width,
            "bridge_length": self.bridge_length,
            "bridge_x": self.bridge_x,
            "platform_length": self.platform_length,
            "platform_height": self.platform_height,
            "fall_z_threshold": self.fall_z_threshold,
            "success_margin": self.success_margin,
            "success_y": self.success_y,
            "success_min_z": self.platform_top_z,
            "success_reward": self.success_reward,
            "fell_off_bridge_reward": self.fell_off_bridge_reward,
            "compile_reward_kernel": self.compile_reward_kernel,
            "reward_kernel_compile_mode": self.reward_kernel_compile_mode,
        }

    def get_default_recording_camera_config(self) -> MJWRecordingCameraConfig | None:
        return MJWRecordingCameraConfig(
            lookat=(0.0, self.bridge_center_y, max(0.35, self.platform_height * 2.5)),
            distance=max(5.0, min(12.0, self.bridge_length + self.swarm.max_unit_extent * 4.0)),
            azimuth=180.0,
            elevation=-35.0,
        )

    def build_model(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody = spec.worldbody

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

        platform_center_z = self.platform_top_z - (self.platform_height / 2.0)
        platform_rgba = [0.45, 0.45, 0.5, 1.0]
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.street_width / 2.0, self.platform1_half_length, self.platform_height / 2.0],
            rgba=platform_rgba,
            pos=[0, self.platform1_center_y, platform_center_z],
        )
        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.street_width / 2.0, self.platform_length / 2.0, self.platform_height / 2.0],
            rgba=platform_rgba,
            pos=[0, self.platform2_center_y, platform_center_z],
        )

        swarm_site = worldbody.add_site(pos=[0, 0, 0], name="swarm_site")
        spec.attach(self.swarm.create_swarm_spec(seed=self.seed), "", site=swarm_site)

        bridge_body = worldbody.add_body(name="Bridge", mocap=True, pos=[0, 0, 0])
        bridge_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=[self.bridge_width / 2.0, self.bridge_length / 2.0, self.platform_height / 2.0],
            rgba=[0.7, 0.6, 0.3, 1.0],
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
            hidden_global_vars_dim=1,
        )

    def get_single_action_space(self) -> spaces.Dict:
        return make_action_space(self)

    def get_batched_observation_space(self, num_envs: int) -> spaces.Dict:
        return batch_space(self.get_single_observation_space(), n=num_envs)

    def get_batched_action_space(self, num_envs: int) -> spaces.Dict:
        return batch_space(self.get_single_action_space(), n=num_envs)

    def build_runtime_metadata(self, *, host_model: mujoco.MjModel) -> MJWBridgeRuntimeMetadata:
        bridge_body_id = mujoco.mj_name2id(host_model, mujoco.mjtObj.mjOBJ_BODY, "Bridge")
        if bridge_body_id < 0:
            raise ValueError("Bridge body not found in bridge model.")
        return MJWBridgeRuntimeMetadata(bridge_mocap_id=int(host_model.body_mocapid[bridge_body_id]))

    def create_runtime(self, *, bindings: MJWRuntimeBindings, runtime_metadata: Any) -> Any:
        from swarmbots.mjw_env.scenarios.mjw_bridge_runtime import BridgeMJWScenarioRuntime

        return BridgeMJWScenarioRuntime(
            scenario=self,
            bindings=bindings,
            runtime_metadata=runtime_metadata,
        )
