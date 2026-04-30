from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable

import mujoco
import numpy as np
from gymnasium import spaces
from gymnasium.vector.utils import batch_space

import swarmbots.mj_env.mujoco_utils as mj_utils
from swarmbots.mj_env.float_or_dist_params import FloatOrDistParams
from swarmbots.mjw_env.scenarios.base_mjw_scenario import BaseMJWScenario, MJWRecordingCameraConfig, MJWRuntimeBindings
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWHomogeneousSwarm


@dataclass(frozen=True, slots=True)
class MJWPayloadPlaneRuntimeMetadata:
    payload_qpos_indices: np.ndarray


def _hinges_per_limb(unit_config: tuple[object, ...]) -> int:
    total = 0
    for limb in unit_config:
        limb_type = str(limb.type).removeprefix("LimbType.")
        if limb_type in ("xy", "zx"):
            total += 2
        elif limb_type == "xyz":
            total += 3
        else:
            raise ValueError(f"Unsupported limb type: {limb_type}")
    return total


@dataclass
class MJWPayloadPlaneScenario(BaseMJWScenario):
    swarm: MJWHomogeneousSwarm
    timestep: float
    action_repeat: int
    actuator_strength: float
    progress_reward_weight: float
    forward_reward_weight: float
    guidance_reward_weight: float
    units_without_connections_reward_weight: float
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
    plane_size: float
    payload_radius: float
    payload_mass: float
    payload_offset_x: FloatOrDistParams
    payload_offset_y: FloatOrDistParams
    payload_centering_penalty_weight: float
    payload_centering_penalty_power: float
    forward_reward_max_y: float | None = None
    seed: int | None = None
    compile_reward_kernel: bool = False
    reward_kernel_compile_mode: str = "default"
    randomize_initial_swarm_z_rotation: bool = False

    def __post_init__(self) -> None:
        if self.include_connectors_xquat_in_obs:
            raise ValueError("MJWPayloadPlaneScenario currently does not support connector quats in observations")
        if self.timestep <= 0.0:
            raise ValueError(f"Expected timestep > 0, got {self.timestep}")
        if self.action_repeat <= 0:
            raise ValueError(f"Expected action_repeat > 0, got {self.action_repeat}")
        self.plane_size = float(self.plane_size)
        if self.plane_size <= 0.0:
            raise ValueError(f"Expected plane_size > 0, got {self.plane_size}")
        self.payload_radius = float(self.payload_radius)
        if self.payload_radius <= 0.0:
            raise ValueError(f"Expected payload_radius > 0, got {self.payload_radius}")
        self.payload_mass = float(self.payload_mass)
        if self.payload_mass <= 0.0:
            raise ValueError(f"Expected payload_mass > 0, got {self.payload_mass}")
        self.payload_centering_penalty_weight = float(self.payload_centering_penalty_weight)
        if self.payload_centering_penalty_weight < 0.0:
            raise ValueError(
                "Expected payload_centering_penalty_weight >= 0, "
                f"got {self.payload_centering_penalty_weight}"
            )
        self.payload_centering_penalty_power = float(self.payload_centering_penalty_power)
        if self.payload_centering_penalty_power <= 0.0:
            raise ValueError(
                f"Expected payload_centering_penalty_power > 0, got {self.payload_centering_penalty_power}"
            )
        if self.forward_reward_max_y is not None:
            self.forward_reward_max_y = float(self.forward_reward_max_y)
        self.inactive_area_location = (50.0, 0.0, 0.1)

    def get_settings(self) -> dict[str, Any]:
        return {
            "swarm": self.swarm.get_settings(),
            "timestep": self.timestep,
            "action_repeat": self.action_repeat,
            "actuator_strength": self.actuator_strength,
            "reward_weights": {
                "progress_reward_weight": self.progress_reward_weight,
                "guidance_reward_weight": self.guidance_reward_weight,
                "units_without_connections_reward_weight": self.units_without_connections_reward_weight,
            },
            "include_connectors_xpos_in_obs": self.include_connectors_xpos_in_obs,
            "include_connectors_xquat_in_obs": self.include_connectors_xquat_in_obs,
            "quat_rot6d_representation": self.quat_rot6d_representation,
            "connection_dist_threshold": self.connection_dist_threshold,
            "connection_angle_threshold": self.connection_angle_threshold,
            "disconnect_potential_threshold": self.disconnect_potential_threshold,
            "swarm_start_x": self.swarm_start_x,
            "swarm_start_y": self.swarm_start_y,
            "randomize_initial_swarm_z_rotation": self.randomize_initial_swarm_z_rotation,
            "friction": self.friction,
            "seed": self.seed,
            "reset_settle_time": self.reset_settle_time,
            "reset_settle_timestep_scale": self.reset_settle_timestep_scale,
            "plane_size": self.plane_size,
            "payload_radius": self.payload_radius,
            "payload_mass": self.payload_mass,
            "payload_offset_x": self.payload_offset_x,
            "payload_offset_y": self.payload_offset_y,
            "payload_centering_penalty_weight": self.payload_centering_penalty_weight,
            "payload_centering_penalty_power": self.payload_centering_penalty_power,
            "forward_reward_weight": self.forward_reward_weight,
            "forward_reward_max_y": self.forward_reward_max_y,
            "compile_reward_kernel": self.compile_reward_kernel,
            "reward_kernel_compile_mode": self.reward_kernel_compile_mode,
        }

    def get_default_recording_camera_config(self) -> MJWRecordingCameraConfig | None:
        payload_offset_y = self.payload_offset_y if isinstance(self.payload_offset_y, (int, float)) else 0.75
        distance = max(3.0, min(8.0, self.swarm.max_unit_extent * 6.0))
        return MJWRecordingCameraConfig(
            lookat=(0.0, float(payload_offset_y) * 0.75, max(0.35, self.payload_radius * 3.0)),
            distance=distance,
            azimuth=180.0,
            elevation=-35.0,
        )

    def build_model(self) -> mujoco.MjModel:
        spec = mujoco.MjSpec()
        spec.compiler.degree = 0
        worldbody = spec.worldbody

        worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_PLANE,
            size=[self.plane_size, self.plane_size, 0.1],
            rgba=[0.2, 0.3, 0.4, 1.0],
            pos=[0, 0, 0],
        )
        worldbody.add_light(pos=[0, 0, 100], dir=[0, 0, -1])
        worldbody.add_light(pos=[0, 100, 100], dir=[-1, -1, -1])

        swarm_site = worldbody.add_site(pos=[0, 0, 0], name="swarm_site")
        spec.attach(self.swarm.create_swarm_spec(seed=self.seed), "", site=swarm_site)

        payload_body = worldbody.add_body(name="Payload", pos=[0, 0, self.payload_radius])
        payload_body.add_freejoint(name="Payload_freejoint")
        payload_body.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[self.payload_radius, 0, 0],
            mass=self.payload_mass,
            rgba=[0.92, 0.72, 0.18, 1.0],
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
        limbs_per_unit = self.swarm.config.limbs_per_unit
        num_hinges = _hinges_per_limb(self.swarm.config.unit_config)
        free_joint_rot_dim = 6 if self.quat_rot6d_representation else 4
        qpos_obs_dim = 3 + free_joint_rot_dim + (2 * num_hinges)
        qvel_dim = 6 + num_hinges
        connector_obs_dim = limbs_per_unit * 5
        connectors_xpos_dim = limbs_per_unit * 3 if self.include_connectors_xpos_in_obs else 0
        local_obs_dim = qpos_obs_dim + qvel_dim + connector_obs_dim + connectors_xpos_dim
        return spaces.Dict(
            {
                "local_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(self.swarm.num_units, local_obs_dim), dtype=np.float32),
                "global_obs": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "hidden_local_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(self.swarm.num_units, 0), dtype=np.float32),
                "hidden_global_vars": spaces.Box(low=-np.inf, high=np.inf, shape=(0,), dtype=np.float32),
                "agent_mask": spaces.MultiBinary((self.swarm.num_units,)),
            }
        )

    def get_single_action_space(self) -> spaces.Dict:
        actuators_per_unit = _hinges_per_limb(self.swarm.config.unit_config)
        return spaces.Dict(
            {
                "actuators": spaces.Box(
                    low=-1.0,
                    high=1.0,
                    shape=(self.swarm.num_units, actuators_per_unit),
                    dtype=np.float32,
                ),
                "connectors": spaces.MultiBinary((self.swarm.num_units, self.swarm.config.limbs_per_unit)),
            }
        )

    def get_batched_observation_space(self, num_envs: int) -> spaces.Dict:
        return batch_space(self.get_single_observation_space(), n=num_envs)

    def get_batched_action_space(self, num_envs: int) -> spaces.Dict:
        return batch_space(self.get_single_action_space(), n=num_envs)

    def build_runtime_metadata(self, *, host_model: mujoco.MjModel) -> MJWPayloadPlaneRuntimeMetadata:
        payload_qpos_indices = np.asarray(mj_utils.qpos_indices_for_body(host_model, "Payload"), dtype=np.int64)
        if payload_qpos_indices.shape[0] < 7:
            raise ValueError("Payload body must expose a free joint with 7 qpos values.")
        return MJWPayloadPlaneRuntimeMetadata(payload_qpos_indices=payload_qpos_indices)

    def create_runtime(self, *, bindings: MJWRuntimeBindings, runtime_metadata: Any) -> Any:
        from swarmbots.mjw_env.scenarios.mjw_payload_plane_runtime import PayloadPlaneMJWScenarioRuntime

        return PayloadPlaneMJWScenarioRuntime(
            scenario=self,
            bindings=bindings,
            runtime_metadata=runtime_metadata,
        )
