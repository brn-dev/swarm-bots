from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
import math
from typing import Any

import mujoco_warp as mjw
import torch
import warp as wp
from gymnasium.vector import AutoresetMode, VectorEnv
from loguru import logger

from swarmbots.mjw_env.mjw_kernels import (
    compute_best_connection_candidates,
    gather_connector_frames,
)
from swarmbots.mjw_env.mjw_model_metadata import MJWModelMetadata, build_model_metadata
from swarmbots.mjw_env.mjw_torch_quat import quat_to_rot6d_torch
from swarmbots.mjw_env.mjw_torch_utils import to_device_bool_tensor
from swarmbots.mjw_env.scenarios.base_mjw_scenario import BaseMJWScenario, MJWRuntimeBindings
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

@dataclass(slots=True)
class _PendingSettledReset:
    world_idx: torch.Tensor
    sampled_batch: Any
    future: Future[list[Any]]
    used: bool = False


class MJWSwarmBotsVectorEnv(VectorEnv):
    metadata = {"autoreset_mode": AutoresetMode.SAME_STEP, "render_modes": []}

    def __init__(
        self,
        scenario: BaseMJWScenario,
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
        if self._physics_graph is None and self.device.type == "cuda":
            logger.warning("Physics CUDA graph could not be captured")

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

        self._runtime_bindings = MJWRuntimeBindings(
            device=self.device,
            wp_device=self._wp_device,
            num_envs=self.num_envs,
            model=self._model,
            data=self._data,
            metadata=self._metadata,
            pool=self._pool,
            pool_eq_active=self._pool_eq_active,
            inactive_unit_positions=self._inactive_unit_positions,
            qpos=self._qpos,
            qvel=self._qvel,
            ctrl=self._ctrl,
            eq_active=self._eq_active,
            mocap_pos=self._mocap_pos,
            mocap_quat=self._mocap_quat,
            xpos=self._xpos,
            xquat=self._xquat,
            xmat=self._xmat,
            time=self._time,
            qacc_warmstart=self._qacc_warmstart,
            act=self._act,
            units_active_mask=self.units_active_mask,
            partner_unit=self.partner_unit,
            partner_connector=self.partner_connector,
            connection_twist_idx=self.connection_twist_idx,
            disconnect_potentials=self.disconnect_potentials,
            current_step=self.current_step,
            is_first_episode=self.is_first_episode,
            base_qpos=self._base_qpos,
            base_mocap_pos=self._base_mocap_pos,
            base_mocap_quat=self._base_mocap_quat,
            unit_qpos_adr=self._unit_qpos_adr,
            qpos_wp=self._qpos_wp,
            pool_active_mask_wp=self._pool_active_mask_wp,
            pool_positions_wp=self._pool_positions_wp,
            pool_quats_wp=self._pool_quats_wp,
            inactive_unit_positions_wp=self._inactive_unit_positions_wp,
            unit_qpos_adr_wp=self._unit_qpos_adr_wp,
        )
        self._scenario_runtime = self.scenario.create_runtime(bindings=self._runtime_bindings)

        self._rng = torch.Generator(device=self.device)
        self._rng.manual_seed(42 if scenario.seed is None else int(scenario.seed))
        self._use_settled_resets = float(self.scenario.reset_settle_time) > 0.0
        self._settle_executor: ThreadPoolExecutor | None = None
        self._pending_settled_reset: _PendingSettledReset | None = None
        if self._use_settled_resets:
            self._settle_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mjw-reset-settle")

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
        self._discard_pending_settled_reset()
        reset_mask = (
            torch.ones((self.num_envs,), device=self.device, dtype=torch.bool)
            if options is None or "reset_mask" not in options
            else to_device_bool_tensor(options["reset_mask"], device=self.device, expected_shape=(self.num_envs,))
        )
        self._reset_worlds(reset_mask)
        return self._build_obs(), {}

    def close(self) -> None:
        if self._settle_executor is not None:
            self._settle_executor.shutdown(wait=True, cancel_futures=True)
            self._settle_executor = None
        self._pending_settled_reset = None
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

        trunc_limit = self._get_current_truncation_limit()
        self._cleanup_pending_settled_reset()
        if self._use_settled_resets:
            self._maybe_start_settled_reset_prefetch(self.current_step + 1 >= trunc_limit)

        self._apply_actions(actuators=actuators, connectors=connectors)
        self._run_physics()

        unstable_mask = torch.isnan(self._qpos).any(dim=1) | torch.isnan(self._qvel).any(dim=1)
        stable_mask = ~unstable_mask

        terminations = unstable_mask.clone()

        self.current_step[stable_mask] += 1
        truncations = stable_mask & (self.current_step >= trunc_limit)
        dones = terminations | truncations

        step_result = self._scenario_runtime.compute_step_rewards(stable_mask=stable_mask)
        rewards = step_result.reward
        infos: dict[str, Any] = dict(step_result.info)

        rewards[unstable_mask] = self.simulation_unstable_reward
        for value in infos.values():
            value[unstable_mask] = 0.0

        obs = self._build_obs()
        obs = self._apply_error_obs(obs, unstable_mask)

        if torch.any(dones):
            infos["final_obs"] = {key: value.clone() for key, value in obs.items()}
            infos["_final_obs"] = dones.clone()
            self.is_first_episode[dones] = False
            self._reset_done_worlds(dones)
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

    def _get_current_truncation_limit(self) -> torch.Tensor:
        if self._first_episode_length_limit is None:
            return self._episode_length_limit
        return torch.where(self.is_first_episode, self._first_episode_length_limit, self._episode_length_limit)

    def _cleanup_pending_settled_reset(self) -> None:
        pending = self._pending_settled_reset
        if pending is None or not pending.used or not pending.future.done():
            return
        pending.future.result()
        self._pending_settled_reset = None

    def _discard_pending_settled_reset(self) -> None:
        pending = self._pending_settled_reset
        if pending is None:
            return
        if not pending.future.done():
            pending.future.cancel()
        self._pending_settled_reset = None

    def _maybe_start_settled_reset_prefetch(self, truncation_mask: torch.Tensor) -> None:
        if not self._use_settled_resets or self._settle_executor is None:
            return
        pending = self._pending_settled_reset
        if pending is not None:
            return

        world_idx = torch.nonzero(truncation_mask, as_tuple=False).flatten()
        if world_idx.numel() == 0:
            return

        sampled_batch = self._scenario_runtime.sample_reset_batch(n_reset=int(world_idx.numel()), rng=self._rng)
        specs = self._scenario_runtime.build_cpu_reset_specs(reset_batch=sampled_batch)
        future = self._settle_executor.submit(self._scenario_runtime.settle_cpu_reset_specs, specs=specs)
        self._pending_settled_reset = _PendingSettledReset(
            world_idx=world_idx,
            sampled_batch=sampled_batch,
            future=future,
        )

    def _reset_done_worlds(self, done_mask: torch.Tensor) -> None:
        remaining_done = done_mask.clone()
        pending = self._pending_settled_reset
        if pending is not None and not pending.used:
            pending_done_mask = done_mask[pending.world_idx]
            if torch.any(pending_done_mask):
                pending_world_idx = pending.world_idx[pending_done_mask]
                if pending.future.done():
                    snapshots = pending.future.result()
                    selected_indices = torch.nonzero(pending_done_mask, as_tuple=False).flatten().tolist()
                    self._scenario_runtime.apply_settled_reset_batch(
                        world_idx=pending_world_idx,
                        snapshots=[snapshots[idx] for idx in selected_indices],
                    )
                    self._pending_settled_reset = None
                else:
                    self._scenario_runtime.apply_reset_batch(
                        world_idx=pending_world_idx,
                        reset_batch=self._scenario_runtime.select_reset_batch(
                            reset_batch=pending.sampled_batch,
                            mask=pending_done_mask,
                        ),
                    )
                    pending.used = True
                remaining_done[pending_world_idx] = False

        if torch.any(remaining_done):
            self._reset_worlds(remaining_done)

    def _reset_worlds(self, reset_mask: torch.Tensor) -> None:
        world_idx = torch.nonzero(reset_mask, as_tuple=False).flatten()
        if world_idx.numel() == 0:
            return
        self._scenario_runtime.apply_reset_batch(
            world_idx=world_idx,
            reset_batch=self._scenario_runtime.sample_reset_batch(n_reset=int(world_idx.numel()), rng=self._rng),
        )

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
            "global_obs": self._scenario_runtime.global_obs,
            "hidden_local_vars": self._scenario_runtime.hidden_local_obs,
            "hidden_global_vars": self._scenario_runtime.hidden_global_obs,
            "agent_mask": self.units_active_mask,
        }

    def _apply_error_obs(self, obs: dict[str, torch.Tensor], unstable_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        unstable_world_idx = torch.nonzero(unstable_mask, as_tuple=True)[0]
        obs["local_obs"][unstable_world_idx] = 0.0
        obs["global_obs"][unstable_world_idx] = 0.0
        obs["hidden_local_vars"][unstable_world_idx] = 0.0
        obs["hidden_global_vars"][unstable_world_idx] = 0.0
        return obs
