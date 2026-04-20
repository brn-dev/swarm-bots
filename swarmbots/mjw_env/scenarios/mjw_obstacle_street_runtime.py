from __future__ import annotations

from dataclasses import dataclass
import shutil
import sys
from typing import Callable

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch

from swarmbots.mjw_env.scenarios.base_mjw_scenario import (
    BaseMJWCPUResetSettler,
    BaseMJWScenarioRuntime,
    MJWCommonResetBatch,
    MJWCommonResetSpec,
    MJWCommonSettledSnapshot,
    MJWRuntimeBindings,
    MJWStepResult,
)
from swarmbots.mjw_env.mjw_torch_utils import masked_mean, sample_float_or_bounded_dist, sample_float_or_dist


@dataclass(slots=True)
class ObstacleStreetResetBatch:
    common: MJWCommonResetBatch
    hidden_global_vars: torch.Tensor
    wall_pass_thresholds: torch.Tensor


@dataclass(slots=True)
class ObstacleStreetResetSpec:
    common: MJWCommonResetSpec
    hidden_global_vars: np.ndarray
    wall_pass_thresholds: np.ndarray


@dataclass(slots=True)
class ObstacleStreetSettledSnapshot:
    common: MJWCommonSettledSnapshot
    hidden_global_vars: np.ndarray
    wall_pass_thresholds: np.ndarray
    progress: float
    next_threshold_for_unit: np.ndarray
    passed_thresholds_mask: np.ndarray


def _compute_obstacle_street_reward_kernel(
    unit_y: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    partner_unit: torch.Tensor,
    wall_pass_absolute_thresholds: torch.Tensor,
    next_threshold_for_unit: torch.Tensor,
    progress: torch.Tensor,
    threshold_index_torch: torch.Tensor,
    progress_reward_weight: float,
    wall_pass_reward_weight: float,
    wall_thresholds_per_wall: int,
    units_without_connections_reward_weight: float,
    guidance_reward_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_unit_y = torch.where(stable_mask.unsqueeze(1), unit_y, torch.zeros_like(unit_y))
    new_progress = masked_mean(safe_unit_y, units_active_mask, dim=1)
    progress_delta = new_progress - progress

    wall_pass_reward = torch.zeros_like(progress, dtype=torch.float32)
    latched_thresholds = next_threshold_for_unit
    if wall_pass_absolute_thresholds.shape[1] > 0:
        total_passed = (safe_unit_y.unsqueeze(-1) > wall_pass_absolute_thresholds.unsqueeze(1)).sum(dim=-1)
        delta_passed = torch.clamp(total_passed - next_threshold_for_unit, min=0)
        delta_passed = delta_passed * units_active_mask.to(dtype=delta_passed.dtype)
        latched_thresholds = next_threshold_for_unit + delta_passed

        active_units_count = units_active_mask.sum(dim=-1)
        denom = active_units_count * max(wall_thresholds_per_wall, 1)
        valid = stable_mask & (denom > 0)
        wall_pass_reward = torch.where(
            valid,
            (
                delta_passed.sum(dim=-1).to(dtype=torch.float32)
                / denom.to(dtype=torch.float32)
            ) * float(wall_pass_reward_weight),
            torch.zeros_like(progress, dtype=torch.float32),
        )

    passed_thresholds_mask = threshold_index_torch.view(1, 1, -1) < latched_thresholds.unsqueeze(-1)
    forward_progress_reward = progress_delta * float(progress_reward_weight)

    connection_mask = partner_unit >= 0
    units_without_connections = (~connection_mask).all(dim=-1) & units_active_mask
    active_units_count = units_active_mask.sum(dim=-1)
    guidance_reward = torch.where(
        active_units_count > 0,
        (
            units_without_connections.sum(dim=-1).to(dtype=torch.float32)
            / active_units_count.to(dtype=torch.float32)
        ) * float(units_without_connections_reward_weight),
        torch.zeros_like(progress, dtype=torch.float32),
    )
    guidance_reward *= float(guidance_reward_weight)

    return (
        new_progress,
        forward_progress_reward,
        wall_pass_reward,
        guidance_reward,
        latched_thresholds,
        passed_thresholds_mask,
    )


class _ObstacleStreetCPUResetSettler(BaseMJWCPUResetSettler):
    def __init__(
        self,
        *,
        scenario: "MJWObstacleStreetScenario",
        bindings: MJWRuntimeBindings,
        threshold_index: np.ndarray,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings)
        self.scenario = scenario
        self.threshold_index = threshold_index
        self.wall_left_mocap_ids = self._build_wall_mocap_ids(kind="left")
        self.wall_right_mocap_ids = self._build_wall_mocap_ids(kind="right")
        self.ramp_mocap_ids = self._build_ramp_mocap_ids()

    def settle_batch(self, *, specs: list[ObstacleStreetResetSpec]) -> list[ObstacleStreetSettledSnapshot]:
        return [self._settle_one(spec=spec) for spec in specs]

    def _settle_one(self, *, spec: ObstacleStreetResetSpec) -> ObstacleStreetSettledSnapshot:
        self._apply_common_reset(common_reset_spec=spec.common)
        self._apply_wall_configuration(hidden_global_vars=spec.hidden_global_vars)
        mujoco.mj_forward(self.model, self.data)
        self._settle_physics()

        unit_y = self.data.qpos[self.bindings.metadata.unit_qpos_adr + 1]
        active_mask = self._pool_active_mask[spec.common.pool_idx]
        if spec.wall_pass_thresholds.size > 0:
            total_passed = (unit_y[:, np.newaxis] > spec.wall_pass_thresholds[np.newaxis, :]).sum(axis=-1)
            passed_thresholds_mask = self.threshold_index[np.newaxis, :] < total_passed[:, np.newaxis]
        else:
            total_passed = np.zeros((active_mask.shape[0],), dtype=np.int64)
            passed_thresholds_mask = np.zeros((active_mask.shape[0], 0), dtype=bool)

        progress = float(unit_y[active_mask].mean()) if active_mask.any() else 0.0
        return ObstacleStreetSettledSnapshot(
            common=self._build_common_snapshot(common_reset_spec=spec.common),
            hidden_global_vars=spec.hidden_global_vars.copy(),
            wall_pass_thresholds=spec.wall_pass_thresholds.copy(),
            progress=progress,
            next_threshold_for_unit=total_passed.astype(np.int64, copy=False),
            passed_thresholds_mask=passed_thresholds_mask.copy(),
        )

    def _apply_wall_configuration(self, *, hidden_global_vars: np.ndarray) -> None:
        if self.data.mocap_pos.size == 0:
            return
        hidden_col = 0
        ramp_idx = 0
        for wall_idx in range(self.scenario.num_walls):
            wall_y = float(hidden_global_vars[hidden_col])
            opening_width = float(hidden_global_vars[hidden_col + 1])
            opening_x = float(hidden_global_vars[hidden_col + 2])
            hidden_col += 3

            wall_left_pos_x = opening_x - (opening_width / 2.0) - 12.5
            wall_right_pos_x = opening_x + (opening_width / 2.0) + 12.5
            self.data.mocap_pos[int(self.wall_left_mocap_ids[wall_idx])] = [wall_left_pos_x, wall_y, 0.0]
            self.data.mocap_pos[int(self.wall_right_mocap_ids[wall_idx])] = [wall_right_pos_x, wall_y, 0.0]

            if wall_idx > 0 or not self.scenario.no_initial_ramp:
                ramp_x = float(hidden_global_vars[hidden_col])
                hidden_col += 1
                self.data.mocap_pos[int(self.ramp_mocap_ids[ramp_idx])] = [
                    ramp_x,
                    wall_y - (self.scenario.ramp_distances_to_wall[wall_idx] / 2.0),
                    (self.scenario.wall_heights[wall_idx] / 2.0) - 0.05,
                ]
                ramp_idx += 1

    def _build_wall_mocap_ids(self, *, kind: str) -> np.ndarray:
        ids = np.zeros((self.scenario.num_walls,), dtype=np.int64)
        for wall_idx in range(self.scenario.num_walls):
            body_name = f"Wall_{wall_idx}_{'Left' if kind == 'left' else 'Right'}"
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            ids[wall_idx] = self.model.body_mocapid[body_id]
        return ids

    def _build_ramp_mocap_ids(self) -> np.ndarray:
        ramp_ids: list[int] = []
        for wall_idx in range(self.scenario.num_walls):
            if wall_idx > 0 or not self.scenario.no_initial_ramp:
                body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, f"Ramp_{wall_idx}")
                ramp_ids.append(int(self.model.body_mocapid[body_id]))
        return np.asarray(ramp_ids, dtype=np.int64)


class ObstacleStreetMJWScenarioRuntime(BaseMJWScenarioRuntime):
    def __init__(self, *, scenario: "MJWObstacleStreetScenario", bindings: MJWRuntimeBindings) -> None:
        super().__init__(scenario=scenario, bindings=bindings)
        hidden_global_dim = int(scenario.get_single_observation_space()["hidden_global_vars"].shape[0])
        total_thresholds = scenario.total_thresholds
        self._global_obs = torch.empty((bindings.num_envs, 0), device=bindings.device, dtype=torch.float32)
        self.hidden_global_vars = torch.zeros((bindings.num_envs, hidden_global_dim), device=bindings.device, dtype=torch.float32)
        self.wall_pass_absolute_thresholds = torch.zeros((bindings.num_envs, total_thresholds), device=bindings.device, dtype=torch.float32)
        self.next_threshold_for_unit = torch.zeros(
            (bindings.num_envs, scenario.swarm.num_units),
            device=bindings.device,
            dtype=torch.long,
        )
        self.passed_thresholds_mask = torch.zeros(
            (bindings.num_envs, scenario.swarm.num_units, total_thresholds),
            device=bindings.device,
            dtype=torch.bool,
        )
        self._hidden_local_obs = torch.zeros_like(self.passed_thresholds_mask, dtype=torch.float32)
        self.progress = torch.zeros((bindings.num_envs,), device=bindings.device, dtype=torch.float32)
        self._threshold_values = torch.as_tensor(scenario.wall_pass_thresholds, device=bindings.device, dtype=torch.float32)
        self._threshold_index_torch = torch.arange(total_thresholds, device=bindings.device, dtype=torch.long)
        self._threshold_index_np = np.arange(total_thresholds, dtype=np.int64)
        self._wall_thresholds_per_wall = len(scenario.wall_pass_thresholds)
        self._cpu_settler = _ObstacleStreetCPUResetSettler(
            scenario=scenario,
            bindings=bindings,
            threshold_index=self._threshold_index_np,
        )
        self._reward_kernel: Callable[..., tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]
        self._reward_kernel = _compute_obstacle_street_reward_kernel
        if scenario.compile_reward_kernel:
            if not hasattr(torch, "compile"):
                raise RuntimeError("compile_reward_kernel=True requires torch.compile support.")
            if sys.platform == "win32" and shutil.which("cl") is None:
                raise RuntimeError(
                    "compile_reward_kernel=True on this Windows setup requires cl.exe on PATH for torch.compile."
                )
            self._reward_kernel = torch.compile(
                self._reward_kernel,
                mode=scenario.reward_kernel_compile_mode,
                fullgraph=False,
                dynamic=False,
            )
        self._wall_left_mocap_ids = self._cpu_settler.wall_left_mocap_ids
        self._wall_right_mocap_ids = self._cpu_settler.wall_right_mocap_ids
        self._ramp_mocap_ids = self._cpu_settler.ramp_mocap_ids

    @property
    def global_obs(self) -> torch.Tensor:
        return self._global_obs

    @property
    def hidden_local_obs(self) -> torch.Tensor:
        return self._hidden_local_obs

    @property
    def hidden_global_obs(self) -> torch.Tensor:
        return self.hidden_global_vars

    def sample_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> ObstacleStreetResetBatch:
        hidden_global_vars, wall_pass_thresholds = self._sample_wall_configuration_values(n_reset=n_reset, rng=rng)
        return ObstacleStreetResetBatch(
            common=self._sample_common_reset_batch(n_reset=n_reset, rng=rng),
            hidden_global_vars=hidden_global_vars,
            wall_pass_thresholds=wall_pass_thresholds,
        )

    def select_reset_batch(self, *, reset_batch: ObstacleStreetResetBatch, mask: torch.Tensor) -> ObstacleStreetResetBatch:
        return ObstacleStreetResetBatch(
            common=self._select_common_reset_batch(common_reset_batch=reset_batch.common, mask=mask),
            hidden_global_vars=reset_batch.hidden_global_vars[mask],
            wall_pass_thresholds=reset_batch.wall_pass_thresholds[mask],
        )

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: ObstacleStreetResetBatch) -> None:
        self._apply_common_reset_batch(world_idx=world_idx, common_reset_batch=reset_batch.common)
        self._apply_wall_configuration(world_idx=world_idx, hidden_global_vars=reset_batch.hidden_global_vars)
        mjw.forward(self.bindings.model, self.bindings.data)

        self.hidden_global_vars[world_idx] = reset_batch.hidden_global_vars
        self.wall_pass_absolute_thresholds[world_idx] = reset_batch.wall_pass_thresholds

        unit_y = self._get_unit_y()[world_idx]
        total_passed = (unit_y.unsqueeze(-1) > reset_batch.wall_pass_thresholds.unsqueeze(1)).sum(dim=-1)
        self.progress[world_idx] = masked_mean(unit_y, self.bindings.units_active_mask[world_idx], dim=1)
        self.next_threshold_for_unit[world_idx] = total_passed
        self.passed_thresholds_mask[world_idx] = self._threshold_index_torch.view(1, 1, -1) < total_passed.unsqueeze(-1)
        self._hidden_local_obs[world_idx] = self.passed_thresholds_mask[world_idx].to(dtype=torch.float32)

    def build_cpu_reset_specs(self, *, reset_batch: ObstacleStreetResetBatch) -> list[ObstacleStreetResetSpec]:
        common_specs = self._build_common_reset_specs(common_reset_batch=reset_batch.common)
        hidden_global_vars = reset_batch.hidden_global_vars.detach().cpu().numpy()
        wall_pass_thresholds = reset_batch.wall_pass_thresholds.detach().cpu().numpy()
        return [
            ObstacleStreetResetSpec(
                common=common_specs[i],
                hidden_global_vars=hidden_global_vars[i].copy(),
                wall_pass_thresholds=wall_pass_thresholds[i].copy(),
            )
            for i in range(len(common_specs))
        ]

    def settle_cpu_reset_specs(self, *, specs: list[ObstacleStreetResetSpec]) -> list[ObstacleStreetSettledSnapshot]:
        return self._cpu_settler.settle_batch(specs=specs)

    def apply_settled_reset_batch(self, *, world_idx: torch.Tensor, snapshots: list[ObstacleStreetSettledSnapshot]) -> None:
        self._apply_common_settled_snapshot_batch(world_idx=world_idx, snapshots=[snapshot.common for snapshot in snapshots])
        self.hidden_global_vars[world_idx] = torch.as_tensor(
            np.stack([snapshot.hidden_global_vars for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.hidden_global_vars.dtype,
        )
        self.wall_pass_absolute_thresholds[world_idx] = torch.as_tensor(
            np.stack([snapshot.wall_pass_thresholds for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.wall_pass_absolute_thresholds.dtype,
        )
        self.progress[world_idx] = torch.as_tensor(
            [snapshot.progress for snapshot in snapshots],
            device=self.bindings.device,
            dtype=self.progress.dtype,
        )
        self.next_threshold_for_unit[world_idx] = torch.as_tensor(
            np.stack([snapshot.next_threshold_for_unit for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.next_threshold_for_unit.dtype,
        )
        self.passed_thresholds_mask[world_idx] = torch.as_tensor(
            np.stack([snapshot.passed_thresholds_mask for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.passed_thresholds_mask.dtype,
        )
        self._hidden_local_obs[world_idx] = self.passed_thresholds_mask[world_idx].to(dtype=torch.float32)
        mjw.forward(self.bindings.model, self.bindings.data)

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        unit_y = self._get_unit_y()
        new_progress, forward_progress_reward, wall_pass_reward, guidance_reward, latched_thresholds, passed_thresholds_mask = self._reward_kernel(
            unit_y,
            stable_mask,
            self.bindings.units_active_mask,
            self.bindings.partner_unit,
            self.wall_pass_absolute_thresholds,
            self.next_threshold_for_unit,
            self.progress,
            self._threshold_index_torch,
            float(self.scenario.progress_reward_weight),
            float(self.scenario.wall_pass_reward_weight),
            int(self._wall_thresholds_per_wall),
            float(self.scenario.units_without_connections_reward_weight),
            float(self.scenario.guidance_reward_weight),
        )
        self.progress[stable_mask] = new_progress[stable_mask]
        self.next_threshold_for_unit[stable_mask] = latched_thresholds[stable_mask]
        self.passed_thresholds_mask[stable_mask] = passed_thresholds_mask[stable_mask]
        self._hidden_local_obs[stable_mask] = self.passed_thresholds_mask[stable_mask].to(dtype=torch.float32)
        progress_reward = forward_progress_reward + wall_pass_reward

        return MJWStepResult(
            reward=progress_reward + guidance_reward,
            info={
                "progress_reward": progress_reward,
                "forward_progress_reward": forward_progress_reward,
                "wall_pass_reward": wall_pass_reward,
                "guidance_reward": guidance_reward,
                "reward_terms": {
                    "progress": forward_progress_reward,
                    "wall": wall_pass_reward,
                },
            },
        )

    def _get_unit_y(self) -> torch.Tensor:
        return self.bindings.qpos[:, self.bindings.unit_qpos_adr + 1]

    def _sample_wall_configuration_values(
        self,
        *,
        n_reset: int,
        rng: torch.Generator,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        hidden_global = torch.zeros((n_reset, self.hidden_global_vars.shape[1]), device=self.bindings.device, dtype=torch.float32)
        wall_pass_thresholds = torch.zeros(
            (n_reset, self.wall_pass_absolute_thresholds.shape[1]),
            device=self.bindings.device,
            dtype=torch.float32,
        )
        unusable_opening_offset = sample_float_or_dist(
            self.scenario.unusable_opening_offset,
            shape=(n_reset,),
            device=self.bindings.device,
            generator=rng,
        )
        current_wall_y = sample_float_or_dist(
            self.scenario.first_wall_distance,
            shape=(n_reset,),
            device=self.bindings.device,
            generator=rng,
        )
        hidden_col = 0
        for wall_idx in range(self.scenario.num_walls):
            if wall_idx != 0:
                current_wall_y = current_wall_y + sample_float_or_bounded_dist(
                    self.scenario.inter_wall_distance,
                    shape=(n_reset,),
                    device=self.bindings.device,
                    generator=rng,
                )
            opening_width = sample_float_or_dist(
                self.scenario.opening_widths[wall_idx],
                shape=(n_reset,),
                device=self.bindings.device,
                generator=rng,
            )
            opening_range = self.scenario.side_wall_x - (opening_width / 2.0) + unusable_opening_offset
            opening_x = (torch.rand((n_reset,), device=self.bindings.device, generator=rng) * 2.0 - 1.0) * opening_range
            hidden_global[:, hidden_col] = current_wall_y
            hidden_global[:, hidden_col + 1] = opening_width
            hidden_global[:, hidden_col + 2] = opening_x
            hidden_col += 3

            start = wall_idx * self._wall_thresholds_per_wall
            stop = start + self._wall_thresholds_per_wall
            wall_pass_thresholds[:, start:stop] = current_wall_y.unsqueeze(-1) + self._threshold_values.unsqueeze(0)

            if wall_idx > 0 or not self.scenario.no_initial_ramp:
                hidden_global[:, hidden_col] = (
                    (torch.rand((n_reset,), device=self.bindings.device, generator=rng) * 2.0 - 1.0)
                    * self.scenario.ramp_range_x
                )
                hidden_col += 1

        wall_pass_thresholds, _ = torch.sort(wall_pass_thresholds, dim=-1)
        return hidden_global, wall_pass_thresholds

    def _apply_wall_configuration(self, *, world_idx: torch.Tensor, hidden_global_vars: torch.Tensor) -> None:
        if self.bindings.mocap_pos.numel() == 0:
            return
        hidden_col = 0
        ramp_idx = 0
        for wall_idx in range(self.scenario.num_walls):
            wall_y = hidden_global_vars[:, hidden_col]
            opening_width = hidden_global_vars[:, hidden_col + 1]
            opening_x = hidden_global_vars[:, hidden_col + 2]
            hidden_col += 3

            wall_left_pos_x = opening_x - (opening_width / 2.0) - 12.5
            wall_right_pos_x = opening_x + (opening_width / 2.0) + 12.5
            left_mocap_id = int(self._wall_left_mocap_ids[wall_idx])
            right_mocap_id = int(self._wall_right_mocap_ids[wall_idx])
            self.bindings.mocap_pos[world_idx, left_mocap_id, 0] = wall_left_pos_x
            self.bindings.mocap_pos[world_idx, left_mocap_id, 1] = wall_y
            self.bindings.mocap_pos[world_idx, left_mocap_id, 2] = 0.0
            self.bindings.mocap_pos[world_idx, right_mocap_id, 0] = wall_right_pos_x
            self.bindings.mocap_pos[world_idx, right_mocap_id, 1] = wall_y
            self.bindings.mocap_pos[world_idx, right_mocap_id, 2] = 0.0

            if wall_idx > 0 or not self.scenario.no_initial_ramp:
                ramp_x = hidden_global_vars[:, hidden_col]
                hidden_col += 1
                mocap_id = int(self._ramp_mocap_ids[ramp_idx])
                self.bindings.mocap_pos[world_idx, mocap_id, 0] = ramp_x
                self.bindings.mocap_pos[world_idx, mocap_id, 1] = wall_y - (self.scenario.ramp_distances_to_wall[wall_idx] / 2.0)
                self.bindings.mocap_pos[world_idx, mocap_id, 2] = (self.scenario.wall_heights[wall_idx] / 2.0) - 0.05
                ramp_idx += 1
