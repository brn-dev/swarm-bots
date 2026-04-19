from __future__ import annotations

import math
from typing import Any

import mujoco_warp as mjw
import torch
import warp as wp
from gymnasium.vector import AutoresetMode, VectorEnv

from swarmbots.mjw_env.mjw_kernels import (
    apply_reset_unit_pose,
    compute_best_connection_candidates,
    gather_connector_frames,
)
from swarmbots.mjw_env.mjw_model_metadata import MJWModelMetadata, build_model_metadata
from swarmbots.mjw_env.mjw_torch_quat import quat_to_rot6d_torch
from swarmbots.mjw_env.mjw_torch_utils import masked_mean, sample_float_or_bounded_dist, sample_float_or_dist, to_device_bool_tensor
from swarmbots.mjw_env.scenarios.mjw_obstacle_street_scenario import MJWObstacleStreetScenario
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWSwarmPool


def _build_inactive_unit_positions(
    *,
    num_units: int,
    max_unit_extent: float,
    inactive_area_location: tuple[float, float, float],
    device: torch.device,
) -> torch.Tensor:
    num_units_sqrt = int(math.ceil(math.sqrt(num_units)))
    unit_spacing = max_unit_extent * 3.0
    positions = torch.zeros((num_units, 3), device=device, dtype=torch.float32)
    positions[:, 2] = max_unit_extent
    for unit_idx in range(num_units):
        positions[unit_idx, 0] = (unit_idx // num_units_sqrt) * unit_spacing
        positions[unit_idx, 1] = (unit_idx % num_units_sqrt) * unit_spacing
    positions[:, :2] -= positions[:, :2].mean(dim=0, keepdim=True)
    positions += torch.tensor(inactive_area_location, device=device, dtype=torch.float32)
    return positions


def _capture_step_graph(model: Any, data: Any, nstep: int) -> Any | None:
    if nstep <= 0:
        return None
    with wp.ScopedCapture() as capture:
        for _ in range(nstep):
            mjw.step(model, data)
    return capture.graph


class MJWSwarmBotsVectorEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP, "render_modes": []}

    def __init__(
        self,
        scenario: MJWObstacleStreetScenario,
        *,
        num_envs: int,
        episode_length: int = 500,
        action_repeat: int = 15,
        first_episode_length: int | None = None,
        simulation_unstable_reward: float = -1.0,
        device: str | torch.device = "cuda",
        nconmax: int | None = None,
        njmax: int | None = None,
    ) -> None:
        super().__init__()
        wp.init()

        if first_episode_length is not None and first_episode_length > episode_length:
            raise ValueError(
                f"first_episode_length can not be longer than episode_length ({episode_length}), got {first_episode_length}"
            )
        if num_envs <= 0:
            raise ValueError(f"Expected num_envs > 0, got {num_envs}")

        self.device = torch.device(device)
        if self.device.type == "cuda":
            cuda_index = 0 if self.device.index is None else int(self.device.index)
            self._wp_device = wp.device_from_torch(torch.device(f"cuda:{cuda_index}"))
        else:
            self._wp_device = wp.device_from_torch(self.device)
        self.scenario = scenario
        self.num_envs = int(num_envs)
        self.episode_length = int(episode_length)
        self.first_episode_length = first_episode_length
        self.action_repeat = int(action_repeat)
        self.simulation_unstable_reward = float(simulation_unstable_reward)
        self.render_mode = None
        self.action_backend = "torch"

        self.single_observation_space = scenario.get_single_observation_space()
        self.single_action_space = scenario.get_single_action_space()
        self.observation_space = scenario.get_batched_observation_space(self.num_envs)
        self.action_space = scenario.get_batched_action_space(self.num_envs)

        self._n_agents = scenario.swarm.num_units
        self._n_connectors = scenario.swarm.config.limbs_per_unit
        self._n_total_connectors = self._n_agents * self._n_connectors
        self._n_actuators = int(self.single_action_space["actuators"].shape[1])
        self._n_twists = len(self.scenario.swarm.config.connection_twist_values)
        self._wall_thresholds_per_wall = len(self.scenario.wall_pass_thresholds)

        self._host_model = scenario.build_model()
        self._metadata: MJWModelMetadata = build_model_metadata(self._host_model, scenario)

        default_nconmax = max(128, self._n_total_connectors * 8)
        default_njmax = max(512, int(self._host_model.nv * 8 + default_nconmax * 6))
        self._model = mjw.put_model(self._host_model)
        self._data = mjw.make_data(
            self._host_model,
            nworld=self.num_envs,
            nconmax=default_nconmax if nconmax is None else nconmax,
            njmax=default_njmax if njmax is None else njmax,
        )

        self._qpos = wp.to_torch(self._data.qpos)
        self._qvel = wp.to_torch(self._data.qvel)
        self._ctrl = wp.to_torch(self._data.ctrl)
        self._eq_active = wp.to_torch(self._data.eq_active)
        self._mocap_pos = wp.to_torch(self._data.mocap_pos)
        self._mocap_quat = wp.to_torch(self._data.mocap_quat)
        self._xpos = wp.to_torch(self._data.xpos)
        self._xquat = wp.to_torch(self._data.xquat)
        self._xmat = wp.to_torch(self._data.xmat)
        self._time = wp.to_torch(self._data.time)
        self._qacc_warmstart = wp.to_torch(self._data.qacc_warmstart)
        self._act = wp.to_torch(self._data.act)

        self._base_qpos = self._qpos[0].clone()
        self._base_mocap_pos = self._mocap_pos[0].clone()
        self._base_mocap_quat = self._mocap_quat[0].clone()

        self._qpos_flat_indices = torch.as_tensor(self._metadata.qpos_indices.reshape(-1), device=self.device, dtype=torch.long)
        self._qvel_flat_indices = torch.as_tensor(self._metadata.qvel_indices.reshape(-1), device=self.device, dtype=torch.long)
        self._ctrl_flat_indices = torch.as_tensor(self._metadata.ctrl_indices.reshape(-1), device=self.device, dtype=torch.long)
        self._qpos_indices = torch.as_tensor(self._metadata.qpos_indices, device=self.device, dtype=torch.long)
        self._qvel_indices = torch.as_tensor(self._metadata.qvel_indices, device=self.device, dtype=torch.long)
        self._connector_body_indices = torch.as_tensor(
            self._metadata.connector_body_indices.reshape(-1),
            device=self.device,
            dtype=torch.long,
        )
        self._connector_body_indices_i32 = self._connector_body_indices.to(dtype=torch.int32)
        self._eq_indices = torch.as_tensor(self._metadata.eq_indices, device=self.device, dtype=torch.long)
        self._unit_qpos_adr = torch.as_tensor(self._metadata.unit_qpos_adr, device=self.device, dtype=torch.long)
        self._unit_qpos_adr_i32 = self._unit_qpos_adr.to(dtype=torch.int32)
        self._unit_dof_adr = torch.as_tensor(self._metadata.unit_dof_adr, device=self.device, dtype=torch.long)
        self._wall_left_mocap_ids = torch.as_tensor(self._metadata.wall_left_mocap_ids, device=self.device, dtype=torch.long)
        self._wall_right_mocap_ids = torch.as_tensor(self._metadata.wall_right_mocap_ids, device=self.device, dtype=torch.long)
        self._ramp_mocap_ids = torch.as_tensor(self._metadata.ramp_mocap_ids, device=self.device, dtype=torch.long)
        self._connector_unit_idx = torch.arange(self._n_agents, device=self.device, dtype=torch.long).repeat_interleave(
            self._n_connectors
        )
        self._connector_unit_idx_i32 = self._connector_unit_idx.to(dtype=torch.int32)
        self._connector_connector_idx = torch.arange(self._n_connectors, device=self.device, dtype=torch.long).repeat(
            self._n_agents
        )
        self._flat_connector_idx = torch.arange(self._n_total_connectors, device=self.device, dtype=torch.long).view(1, -1)
        self._flat_connector_idx_i32 = self._flat_connector_idx.to(dtype=torch.int32)
        self._unit_indices = torch.arange(self._n_agents, device=self.device, dtype=torch.long).view(1, -1, 1)
        self._connector_indices = torch.arange(self._n_connectors, device=self.device, dtype=torch.long).view(1, 1, -1)
        self._threshold_values = torch.as_tensor(scenario.wall_pass_thresholds, device=self.device, dtype=torch.float32)
        self._threshold_index = torch.arange(scenario.total_thresholds, device=self.device, dtype=torch.long)
        self._twist_values = torch.as_tensor(scenario.swarm.config.connection_twist_values, device=self.device, dtype=torch.float32)
        self._distance_threshold_sq = float(self.scenario.connection_dist_threshold) ** 2
        self._twist_step = 2.0 * math.pi / self._n_twists

        self._pool: MJWSwarmPool = scenario.swarm.build_pool(device=self.device)
        self._pool_eq_active = self._build_pool_eq_active()

        self._inactive_unit_positions = _build_inactive_unit_positions(
            num_units=self._n_agents,
            max_unit_extent=scenario.swarm.max_unit_extent,
            inactive_area_location=(scenario.street_width * 2.0, 0.0, 0.1),
            device=self.device,
        )

        self.units_active_mask = torch.zeros((self.num_envs, self._n_agents), device=self.device, dtype=torch.bool)
        self.partner_unit = torch.full((self.num_envs, self._n_agents, self._n_connectors), -1, device=self.device, dtype=torch.long)
        self.partner_connector = torch.full_like(self.partner_unit, -1)
        self.connection_twist_idx = torch.full_like(self.partner_unit, -1)
        self.disconnect_potentials = torch.zeros((self.num_envs, self._n_agents, self._n_connectors), device=self.device, dtype=torch.float32)

        self.current_step = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int64)
        self.is_first_episode = torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
        self.progress = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self.hidden_global_vars = torch.zeros(
            (self.num_envs, int(self.single_observation_space["hidden_global_vars"].shape[0])),
            device=self.device,
            dtype=torch.float32,
        )
        self.passed_thresholds_mask = torch.zeros(
            (self.num_envs, self._n_agents, scenario.total_thresholds),
            device=self.device,
            dtype=torch.bool,
        )
        self.next_threshold_for_unit = torch.zeros((self.num_envs, self._n_agents), device=self.device, dtype=torch.long)
        self.wall_pass_absolute_thresholds = torch.zeros(
            (self.num_envs, scenario.total_thresholds),
            device=self.device,
            dtype=torch.float32,
        )
        self._best_partner_idx = torch.full((self.num_envs, self._n_total_connectors), -1, device=self.device, dtype=torch.int32)
        self._best_twist_idx = torch.full_like(self._best_partner_idx, -1)
        self._connector_positions = torch.empty(
            (self.num_envs, self._n_total_connectors, 3),
            device=self.device,
            dtype=torch.float32,
        )
        self._connector_x_axis = torch.empty_like(self._connector_positions)
        self._connector_y_axis = torch.empty_like(self._connector_positions)
        self._connector_z_axis = torch.empty_like(self._connector_positions)
        self._global_obs = torch.empty((self.num_envs, 0), device=self.device, dtype=torch.float32)
        self._hidden_local_obs = torch.zeros_like(self.passed_thresholds_mask, dtype=torch.float32)
        self._disconnect_update = torch.empty_like(self.disconnect_potentials)
        self._local_obs = torch.empty(
            (self.num_envs, *self.single_observation_space["local_obs"].shape),
            device=self.device,
            dtype=torch.float32,
        )
        self._episode_length_limit = torch.full((self.num_envs,), self.episode_length, device=self.device, dtype=torch.int64)
        self._first_episode_length_limit = None
        if self.first_episode_length is not None:
            self._first_episode_length_limit = torch.full(
                (self.num_envs,),
                int(self.first_episode_length),
                device=self.device,
                dtype=torch.int64,
            )

        self._physics_graph = _capture_step_graph(self._model, self._data, self.action_repeat) if self.device.type == "cuda" else None

        free_joint_rot_dim = 6 if self.scenario.quat_rot6d_representation else 4
        num_hinges = self._qvel_indices.shape[1] - 6
        self._free_joint_pos_slice = slice(0, 3)
        self._free_joint_rot_slice = slice(3, 3 + free_joint_rot_dim)
        self._hinge_obs_slice = slice(self._free_joint_rot_slice.stop, self._free_joint_rot_slice.stop + (2 * num_hinges))
        self._qvel_obs_slice = slice(self._hinge_obs_slice.stop, self._hinge_obs_slice.stop + self._qvel_indices.shape[1])
        self._connector_obs_slice = slice(self._qvel_obs_slice.stop, self._qvel_obs_slice.stop + (self._n_connectors * 5))
        self._connector_xpos_obs_slice = slice(
            self._connector_obs_slice.stop,
            self._connector_obs_slice.stop + (self._n_connectors * 3),
        )

        self._qpos_wp = wp.from_torch(self._qpos)
        self._xpos_wp = wp.from_torch(self._xpos, dtype=wp.vec3)
        self._xmat_wp = wp.from_torch(self._xmat)
        self._pool_active_mask_wp = wp.from_torch(self._pool.active_mask)
        self._pool_positions_wp = wp.from_torch(self._pool.positions, dtype=wp.vec3)
        self._pool_quats_wp = wp.from_torch(self._pool.quats)
        self._inactive_unit_positions_wp = wp.from_torch(self._inactive_unit_positions, dtype=wp.vec3)
        self._unit_qpos_adr_wp = wp.from_torch(self._unit_qpos_adr_i32)
        self._connector_body_indices_wp = wp.from_torch(self._connector_body_indices_i32)
        self._connector_unit_idx_wp = wp.from_torch(self._connector_unit_idx_i32)
        self._best_partner_idx_wp = wp.from_torch(self._best_partner_idx)
        self._best_twist_idx_wp = wp.from_torch(self._best_twist_idx)
        self._connector_positions_wp = wp.from_torch(self._connector_positions, dtype=wp.vec3)
        self._connector_x_axis_wp = wp.from_torch(self._connector_x_axis, dtype=wp.vec3)
        self._connector_y_axis_wp = wp.from_torch(self._connector_y_axis, dtype=wp.vec3)
        self._connector_z_axis_wp = wp.from_torch(self._connector_z_axis, dtype=wp.vec3)

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(42 if scenario.seed is None else int(scenario.seed))

    def get_settings(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario.get_settings(),
            "episode_length": self.episode_length,
            "action_repeat": self.action_repeat,
            "simulation_unstable_reward": self.simulation_unstable_reward,
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        if seed is not None:
            self._rng.manual_seed(int(seed))
        reset_mask = (
            torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
            if options is None or "reset_mask" not in options
            else to_device_bool_tensor(options["reset_mask"], device=self.device, expected_shape=(self.num_envs,))
        )
        self._reset_worlds(reset_mask)
        return self._build_obs(), {}

    def close(self) -> None:
        return None

    def step(
        self,
        actions: dict[str, torch.Tensor],
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        actuators = torch.as_tensor(actions["actuators"], device=self.device, dtype=torch.float32)
        connectors = torch.as_tensor(actions["connectors"], device=self.device, dtype=torch.bool)
        if actuators.shape != (self.num_envs, self._n_agents, self._n_actuators):
            raise ValueError(f"Unexpected actuators shape {tuple(actuators.shape)}")
        if connectors.shape != (self.num_envs, self._n_agents, self._n_connectors):
            raise ValueError(f"Unexpected connectors shape {tuple(connectors.shape)}")

        self._apply_actions(actuators=actuators, connectors=connectors)
        self._run_physics()

        unstable_mask = torch.isnan(self._qpos).any(dim=1) | torch.isnan(self._qvel).any(dim=1)
        stable_mask = ~unstable_mask

        progress_reward = self._update_progress_rewards(stable_mask)
        guidance_reward = self._compute_guidance_reward()
        rewards = progress_reward + guidance_reward

        terminations = unstable_mask.clone()
        rewards[unstable_mask] = self.simulation_unstable_reward
        progress_reward[unstable_mask] = 0.0
        guidance_reward[unstable_mask] = 0.0

        self.current_step[stable_mask] += 1
        if self._first_episode_length_limit is None:
            trunc_limit = self._episode_length_limit
        else:
            trunc_limit = torch.where(self.is_first_episode, self._first_episode_length_limit, self._episode_length_limit)
        truncations = stable_mask & (self.current_step >= trunc_limit)
        dones = terminations | truncations

        obs = self._build_obs()
        obs = self._apply_error_obs(obs, unstable_mask)

        infos: dict[str, Any] = {
            "progress_reward": progress_reward,
            "guidance_reward": guidance_reward,
        }

        if torch.any(dones):
            infos["final_obs"] = {key: value.clone() for key, value in obs.items()}
            infos["_final_obs"] = dones.clone()
            self.is_first_episode[dones] = False
            self._reset_worlds(dones)
            obs = self._build_obs()

        return obs, rewards, terminations, truncations, infos

    def _build_pool_eq_active(self) -> torch.Tensor:
        pool_eq_active = torch.zeros(
            (self._pool.size, int(self._host_model.neq)),
            device=self.device,
            dtype=self._eq_active.dtype if self._eq_active.numel() > 0 else torch.int32,
        )
        if pool_eq_active.numel() == 0:
            return pool_eq_active

        canonical_mask = self._pool.partner_unit >= 0
        canonical_mask &= self._connector_unit_idx.view(1, self._n_agents, self._n_connectors) < self._pool.partner_unit
        pool_idx, unit_idx, connector_idx = torch.nonzero(canonical_mask, as_tuple=True)
        if pool_idx.numel() == 0:
            return pool_eq_active

        other_unit = self._pool.partner_unit[pool_idx, unit_idx, connector_idx]
        other_connector = self._pool.partner_connector[pool_idx, unit_idx, connector_idx]
        twist_idx = self._pool.twist_idx[pool_idx, unit_idx, connector_idx]
        eq_idx = self._eq_indices[unit_idx, connector_idx, other_unit, other_connector, twist_idx]
        pool_eq_active[pool_idx, eq_idx] = 1
        return pool_eq_active

    def _run_physics(self) -> None:
        if self._physics_graph is not None:
            wp.capture_launch(self._physics_graph)
            return
        for _ in range(self.action_repeat):
            mjw.step(self._model, self._data)

    def _reset_worlds(self, reset_mask: torch.Tensor) -> None:
        world_idx = torch.nonzero(reset_mask, as_tuple=False).flatten()
        if world_idx.numel() == 0:
            return

        pool_idx = torch.randint(self._pool.size, (world_idx.numel(),), device=self.device, generator=self._rng)
        world_idx_i32 = world_idx.to(dtype=torch.int32)
        pool_idx_i32 = pool_idx.to(dtype=torch.int32)
        swarm_start = self._sample_swarm_start(world_idx.numel())

        self._qpos[world_idx] = self._base_qpos
        self._qvel[world_idx] = 0.0
        if self._ctrl.numel() > 0:
            self._ctrl[world_idx] = 0.0
        if self._qacc_warmstart.numel() > 0:
            self._qacc_warmstart[world_idx] = 0.0
        if self._act.numel() > 0:
            self._act[world_idx] = 0.0
        if self._eq_active.numel() > 0:
            self._eq_active[world_idx] = self._pool_eq_active[pool_idx]
        if self._mocap_pos.numel() > 0:
            self._mocap_pos[world_idx] = self._base_mocap_pos
        if self._mocap_quat.numel() > 0:
            self._mocap_quat[world_idx] = self._base_mocap_quat
        self._time[world_idx] = 0.0

        wp.launch(
            kernel=apply_reset_unit_pose,
            dim=(int(world_idx.numel()), self._n_agents),
            inputs=[
                wp.from_torch(world_idx_i32),
                wp.from_torch(pool_idx_i32),
                self._pool_active_mask_wp,
                self._pool_positions_wp,
                self._pool_quats_wp,
                self._inactive_unit_positions_wp,
                wp.from_torch(swarm_start, dtype=wp.vec3),
                self._unit_qpos_adr_wp,
            ],
            outputs=[self._qpos_wp],
            device=self._wp_device,
        )

        hidden_global_vars, wall_pass_thresholds = self._sample_wall_configuration(world_idx)
        mjw.forward(self._model, self._data)

        self.units_active_mask[world_idx] = self._pool.active_mask[pool_idx]
        self.partner_unit[world_idx] = self._pool.partner_unit[pool_idx].clamp(min=-1)
        self.partner_connector[world_idx] = self._pool.partner_connector[pool_idx].clamp(min=-1)
        self.connection_twist_idx[world_idx] = self._pool.twist_idx[pool_idx]
        self.disconnect_potentials[world_idx] = 0.0
        self.hidden_global_vars[world_idx] = hidden_global_vars
        self.wall_pass_absolute_thresholds[world_idx] = wall_pass_thresholds
        self.current_step[world_idx] = 0

        unit_y = self._get_unit_y()[world_idx]
        total_passed = (unit_y.unsqueeze(-1) > wall_pass_thresholds.unsqueeze(1)).sum(dim=-1)
        self.progress[world_idx] = masked_mean(unit_y, self.units_active_mask[world_idx], dim=1)
        self.next_threshold_for_unit[world_idx] = total_passed
        self.passed_thresholds_mask[world_idx] = self._threshold_index.view(1, 1, -1) < total_passed.unsqueeze(-1)
        self._hidden_local_obs[world_idx] = self.passed_thresholds_mask[world_idx].to(dtype=torch.float32)

    def _sample_swarm_start(self, n_reset: int) -> torch.Tensor:
        swarm_start_x = sample_float_or_dist(
            self.scenario.swarm_start_x,
            shape=(n_reset,),
            device=self.device,
            generator=self._rng,
        )
        swarm_start_y = sample_float_or_dist(
            self.scenario.swarm_start_y,
            shape=(n_reset,),
            device=self.device,
            generator=self._rng,
        )
        return torch.stack(
            (
                swarm_start_x,
                swarm_start_y,
                torch.full(
                    (n_reset,),
                    self.scenario.swarm.max_unit_extent * 1.1,
                    device=self.device,
                    dtype=torch.float32,
                ),
            ),
            dim=-1,
        )

    def _sample_wall_configuration(self, world_idx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        n_reset = int(world_idx.numel())
        hidden_global = torch.zeros((n_reset, self.hidden_global_vars.shape[1]), device=self.device, dtype=torch.float32)
        wall_pass_thresholds = torch.zeros((n_reset, self.scenario.total_thresholds), device=self.device, dtype=torch.float32)
        unusable_opening_offset = sample_float_or_dist(
            self.scenario.unusable_opening_offset,
            shape=(n_reset,),
            device=self.device,
            generator=self._rng,
        )

        current_wall_y = sample_float_or_dist(
            self.scenario.first_wall_distance,
            shape=(n_reset,),
            device=self.device,
            generator=self._rng,
        )
        hidden_col = 0
        ramp_idx = 0
        for wall_idx in range(self.scenario.num_walls):
            if wall_idx != 0:
                current_wall_y = current_wall_y + sample_float_or_bounded_dist(
                    self.scenario.inter_wall_distance,
                    shape=(n_reset,),
                    device=self.device,
                    generator=self._rng,
                )
            opening_width = sample_float_or_dist(
                self.scenario.opening_widths[wall_idx],
                shape=(n_reset,),
                device=self.device,
                generator=self._rng,
            )
            opening_range = self.scenario.side_wall_x - (opening_width / 2.0) + unusable_opening_offset
            opening_x = (torch.rand((n_reset,), device=self.device, generator=self._rng) * 2.0 - 1.0) * opening_range

            hidden_global[:, hidden_col] = current_wall_y
            hidden_global[:, hidden_col + 1] = opening_width
            hidden_global[:, hidden_col + 2] = opening_x
            hidden_col += 3

            wall_left_pos_x = opening_x - (opening_width / 2.0) - 12.5
            wall_right_pos_x = opening_x + (opening_width / 2.0) + 12.5
            left_mocap_id = int(self._metadata.wall_left_mocap_ids[wall_idx])
            right_mocap_id = int(self._metadata.wall_right_mocap_ids[wall_idx])
            self._mocap_pos[world_idx, left_mocap_id, 0] = wall_left_pos_x
            self._mocap_pos[world_idx, left_mocap_id, 1] = current_wall_y
            self._mocap_pos[world_idx, left_mocap_id, 2] = 0.0
            self._mocap_pos[world_idx, right_mocap_id, 0] = wall_right_pos_x
            self._mocap_pos[world_idx, right_mocap_id, 1] = current_wall_y
            self._mocap_pos[world_idx, right_mocap_id, 2] = 0.0

            start = wall_idx * self._wall_thresholds_per_wall
            stop = start + self._wall_thresholds_per_wall
            wall_pass_thresholds[:, start:stop] = current_wall_y.unsqueeze(-1) + self._threshold_values.unsqueeze(0)

            if wall_idx > 0 or not self.scenario.no_initial_ramp:
                ramp_x = (torch.rand((n_reset,), device=self.device, generator=self._rng) * 2.0 - 1.0) * self.scenario.ramp_range_x
                hidden_global[:, hidden_col] = ramp_x
                hidden_col += 1
                mocap_id = int(self._metadata.ramp_mocap_ids[ramp_idx])
                self._mocap_pos[world_idx, mocap_id, 0] = ramp_x
                self._mocap_pos[world_idx, mocap_id, 1] = current_wall_y - (self.scenario.ramp_distances_to_wall[wall_idx] / 2.0)
                self._mocap_pos[world_idx, mocap_id, 2] = (self.scenario.wall_heights[wall_idx] / 2.0) - 0.05
                ramp_idx += 1

        wall_pass_thresholds, _ = torch.sort(wall_pass_thresholds, dim=-1)
        return hidden_global, wall_pass_thresholds

    def _apply_actions(self, *, actuators: torch.Tensor, connectors: torch.Tensor) -> None:
        actuators = actuators.masked_fill(~self.units_active_mask.unsqueeze(-1), 0.0)
        if self._ctrl.numel() > 0:
            self._ctrl[:, self._ctrl_flat_indices] = actuators.reshape(self.num_envs, -1) * float(self.scenario.actuator_strength)

        connector_action = connectors & self.units_active_mask.unsqueeze(-1)
        self._try_connect(connector_action)
        self._disconnect(connector_action)

    def _try_connect(self, connector_action: torch.Tensor) -> None:
        newly_activated = (connector_action & (self.partner_unit < 0)).reshape(self.num_envs, self._n_total_connectors)

        wp.launch(
            kernel=gather_connector_frames,
            dim=(self.num_envs, self._n_total_connectors),
            inputs=[
                self._xpos_wp,
                self._xmat_wp,
                self._connector_body_indices_wp,
            ],
            outputs=[
                self._connector_positions_wp,
                self._connector_x_axis_wp,
                self._connector_y_axis_wp,
                self._connector_z_axis_wp,
            ],
            device=self._wp_device,
        )

        wp.launch(
            kernel=compute_best_connection_candidates,
            dim=(self.num_envs, self._n_total_connectors),
            inputs=[
                wp.from_torch(newly_activated),
                self._connector_positions_wp,
                self._connector_x_axis_wp,
                self._connector_y_axis_wp,
                self._connector_z_axis_wp,
                self._connector_unit_idx_wp,
                self._n_total_connectors,
                self._distance_threshold_sq,
                float(self.scenario.connection_angle_threshold),
                self._twist_step,
                self._n_twists,
            ],
            outputs=[self._best_partner_idx_wp, self._best_twist_idx_wp],
            device=self._wp_device,
        )

        partner_idx = self._best_partner_idx
        partner_idx_clamped = partner_idx.clamp(min=0).to(dtype=torch.long)
        mutual_partner = torch.gather(partner_idx, 1, partner_idx_clamped)
        canonical_match = partner_idx >= 0
        canonical_match &= mutual_partner == self._flat_connector_idx_i32
        canonical_match &= self._flat_connector_idx_i32 < partner_idx

        world_idx, connector_idx = torch.nonzero(canonical_match, as_tuple=True)
        partner_flat_idx = partner_idx[world_idx, connector_idx].to(dtype=torch.long)
        twist_idx = self._best_twist_idx[world_idx, connector_idx].to(dtype=torch.long)
        unit1 = self._connector_unit_idx[connector_idx]
        connector1 = self._connector_connector_idx[connector_idx]
        unit2 = self._connector_unit_idx[partner_flat_idx]
        connector2 = self._connector_connector_idx[partner_flat_idx]

        self.partner_unit[world_idx, unit1, connector1] = unit2
        self.partner_connector[world_idx, unit1, connector1] = connector2
        self.connection_twist_idx[world_idx, unit1, connector1] = twist_idx
        self.disconnect_potentials[world_idx, unit1, connector1] = 0.0

        self.partner_unit[world_idx, unit2, connector2] = unit1
        self.partner_connector[world_idx, unit2, connector2] = connector1
        self.connection_twist_idx[world_idx, unit2, connector2] = twist_idx
        self.disconnect_potentials[world_idx, unit2, connector2] = 0.0

        eq_idx = self._eq_indices[unit1, connector1, unit2, connector2, twist_idx]
        self._eq_active[world_idx, eq_idx] = 1

    def _disconnect(self, connector_action: torch.Tensor) -> None:
        currently_active = self.partner_unit >= 0
        newly_deactivated = ~connector_action & currently_active
        disconnect_update = self._disconnect_update
        disconnect_update.zero_()
        disconnect_update[newly_deactivated] = 1.0

        deactivated_world, deactivated_unit, deactivated_connector = torch.nonzero(newly_deactivated, as_tuple=True)
        partner_units = self.partner_unit[deactivated_world, deactivated_unit, deactivated_connector]
        partner_connectors = self.partner_connector[deactivated_world, deactivated_unit, deactivated_connector]
        disconnect_update.index_put_(
            (deactivated_world, partner_units, partner_connectors),
            torch.ones_like(partner_units, dtype=disconnect_update.dtype),
            accumulate=True,
        )

        stayed_active = currently_active & (disconnect_update == 0)
        disconnect_update[stayed_active] = -2.0
        self.disconnect_potentials += disconnect_update
        self.disconnect_potentials.clamp_min_(0.0)

        canonical_disconnect = self.disconnect_potentials >= float(self.scenario.disconnect_potential_threshold)
        canonical_disconnect &= currently_active
        canonical_disconnect &= (
            (self._unit_indices < self.partner_unit)
            | ((self._unit_indices == self.partner_unit) & (self._connector_indices < self.partner_connector))
        )

        world_idx, unit1, connector1 = torch.nonzero(canonical_disconnect, as_tuple=True)
        unit2 = self.partner_unit[world_idx, unit1, connector1]
        connector2 = self.partner_connector[world_idx, unit1, connector1]
        eq_variants = self._eq_indices[unit1, connector1, unit2, connector2]
        self._eq_active[world_idx.unsqueeze(1), eq_variants] = 0

        self.partner_unit[world_idx, unit1, connector1] = -1
        self.partner_connector[world_idx, unit1, connector1] = -1
        self.connection_twist_idx[world_idx, unit1, connector1] = -1
        self.disconnect_potentials[world_idx, unit1, connector1] = 0.0

        self.partner_unit[world_idx, unit2, connector2] = -1
        self.partner_connector[world_idx, unit2, connector2] = -1
        self.connection_twist_idx[world_idx, unit2, connector2] = -1
        self.disconnect_potentials[world_idx, unit2, connector2] = 0.0

    def _get_unit_y(self) -> torch.Tensor:
        return self._qpos[:, self._unit_qpos_adr + 1]

    def _compute_progress(self) -> torch.Tensor:
        return masked_mean(self._get_unit_y(), self.units_active_mask, dim=1)

    def _update_progress_rewards(self, stable_mask: torch.Tensor) -> torch.Tensor:
        unit_y = self._get_unit_y()
        safe_unit_y = torch.where(stable_mask.unsqueeze(1), unit_y, torch.zeros_like(unit_y))
        new_progress = masked_mean(safe_unit_y, self.units_active_mask, dim=1)
        progress_delta = new_progress - self.progress
        self.progress[stable_mask] = new_progress[stable_mask]

        wall_pass_reward = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        if self.scenario.total_thresholds > 0:
            total_passed = (safe_unit_y.unsqueeze(-1) > self.wall_pass_absolute_thresholds.unsqueeze(1)).sum(dim=-1)
            delta_passed = torch.clamp(total_passed - self.next_threshold_for_unit, min=0)
            delta_passed = delta_passed * self.units_active_mask.to(dtype=delta_passed.dtype)
            active_units_count = self.units_active_mask.sum(dim=-1)
            denom = active_units_count * max(self._wall_thresholds_per_wall, 1)
            valid = stable_mask & (denom > 0)
            wall_pass_reward[valid] = (
                delta_passed.sum(dim=-1)[valid].to(dtype=torch.float32)
                / denom[valid].to(dtype=torch.float32)
            ) * float(self.scenario.wall_pass_reward_weight)
            self.next_threshold_for_unit[stable_mask] = total_passed[stable_mask]
            self.passed_thresholds_mask[stable_mask] = self._threshold_index.view(1, 1, -1) < total_passed[stable_mask].unsqueeze(-1)
            self._hidden_local_obs[stable_mask] = self.passed_thresholds_mask[stable_mask].to(dtype=torch.float32)

        return progress_delta * float(self.scenario.progress_reward_weight) + wall_pass_reward

    def _compute_guidance_reward(self) -> torch.Tensor:
        connection_mask = self.partner_unit >= 0
        units_without_connections = (~connection_mask).all(dim=-1) & self.units_active_mask
        active_units_count = self.units_active_mask.sum(dim=-1)
        reward = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        valid = active_units_count > 0
        reward[valid] = (
            units_without_connections.sum(dim=-1)[valid].to(dtype=torch.float32)
            / active_units_count[valid].to(dtype=torch.float32)
        ) * float(self.scenario.units_without_connections_reward_weight)
        return reward * float(self.scenario.guidance_reward_weight)

    def _build_obs(self) -> dict[str, torch.Tensor]:
        local_obs = self._local_obs
        qpos = self._qpos[:, self._qpos_flat_indices].reshape(self.num_envs, self._n_agents, -1)
        qvel = self._qvel[:, self._qvel_flat_indices].reshape(self.num_envs, self._n_agents, -1)
        local_obs[:, :, self._free_joint_pos_slice] = qpos[:, :, :3]
        free_joint_quat = qpos[:, :, 3:7]
        if self.scenario.quat_rot6d_representation:
            local_obs[:, :, self._free_joint_rot_slice] = quat_to_rot6d_torch(free_joint_quat)
        else:
            local_obs[:, :, self._free_joint_rot_slice] = free_joint_quat

        hinge_qpos = qpos[:, :, 7:]
        hinge_obs = local_obs[:, :, self._hinge_obs_slice].view(self.num_envs, self._n_agents, -1, 2)
        hinge_obs[..., 0] = torch.sin(hinge_qpos)
        hinge_obs[..., 1] = torch.cos(hinge_qpos)
        local_obs[:, :, self._qvel_obs_slice] = qvel
        connector_active = self.partner_unit >= 0
        valid_twist = connector_active & (self.connection_twist_idx >= 0)
        twist_values = self._twist_values[self.connection_twist_idx.clamp(min=0)]
        connector_obs = local_obs[:, :, self._connector_obs_slice].view(self.num_envs, self._n_agents, self._n_connectors, 5)
        valid_twist_f = valid_twist.to(dtype=torch.float32)
        connector_obs[..., 0] = (~connector_active).to(dtype=torch.float32)
        connector_obs[..., 1] = connector_active.to(dtype=torch.float32)
        connector_obs[..., 2] = torch.sin(twist_values) * valid_twist_f
        connector_obs[..., 3] = torch.cos(twist_values) * valid_twist_f
        connector_obs[..., 4] = self.disconnect_potentials
        if self.scenario.include_connectors_xpos_in_obs:
            local_obs[:, :, self._connector_xpos_obs_slice] = self._xpos[:, self._connector_body_indices].reshape(
                self.num_envs,
                self._n_agents,
                -1,
            )
        return {
            "local_obs": local_obs,
            "global_obs": self._global_obs,
            "hidden_local_vars": self._hidden_local_obs,
            "hidden_global_vars": self.hidden_global_vars,
            "agent_mask": self.units_active_mask,
        }

    def _apply_error_obs(self, obs: dict[str, torch.Tensor], unstable_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        unstable_world_idx = torch.nonzero(unstable_mask, as_tuple=True)[0]
        obs["local_obs"][unstable_world_idx] = 0.0
        obs["global_obs"][unstable_world_idx] = 0.0
        obs["hidden_local_vars"][unstable_world_idx] = 0.0
        obs["hidden_global_vars"][unstable_world_idx] = 0.0
        return obs
