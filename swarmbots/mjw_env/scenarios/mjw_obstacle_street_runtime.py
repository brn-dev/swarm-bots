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
    forward_progress_unit_y: np.ndarray
    wall_climb_potential: np.ndarray
    wall_climb_done_mask: np.ndarray
    next_threshold_for_unit: np.ndarray
    passed_thresholds_mask: np.ndarray


def _build_wall_pass_rank_weights(*, max_units: int, skew: float, device: torch.device) -> torch.Tensor:
    rank_weights = torch.zeros((max_units + 1, max_units), device=device, dtype=torch.float32)
    ranks = torch.arange(1, max_units + 1, device=device, dtype=torch.float32)
    for active_units_count in range(1, max_units + 1):
        active_ranks = ranks[:active_units_count]
        unnormalized = torch.pow(active_ranks / float(active_units_count), float(skew))
        rank_weights[active_units_count, :active_units_count] = (
            unnormalized * (float(active_units_count) / unnormalized.sum())
        )
    return rank_weights


def _compute_wall_climb_potential_torch(
    *,
    unit_y: torch.Tensor,
    unit_z: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    wall_y_by_wall: torch.Tensor,
    wall_heights: torch.Tensor,
    wall_climb_reward_distance: float,
    unit_ground_z: float,
) -> torch.Tensor:
    if wall_y_by_wall.shape[1] == 0:
        return torch.zeros(
            (*unit_y.shape, 0),
            device=unit_y.device,
            dtype=torch.float32,
        )

    safe_unit_y = torch.where(stable_mask.unsqueeze(1), unit_y, torch.zeros_like(unit_y))
    safe_unit_z = torch.where(stable_mask.unsqueeze(1), unit_z, torch.zeros_like(unit_z))
    distance_to_wall = wall_y_by_wall.unsqueeze(1) - safe_unit_y.unsqueeze(-1)
    approach = torch.clamp(1.0 - (distance_to_wall / float(wall_climb_reward_distance)), min=0.0, max=1.0)
    approach = torch.where(
        (distance_to_wall >= 0.0) & (distance_to_wall <= float(wall_climb_reward_distance)),
        approach,
        torch.zeros_like(approach),
    )
    approach = torch.sqrt(approach)
    target_lift = torch.clamp(wall_heights - float(unit_ground_z), min=1e-6)
    height = torch.clamp(
        (safe_unit_z.unsqueeze(-1) - float(unit_ground_z)) / target_lift.view(1, 1, -1),
        min=0.0,
        max=1.0,
    )
    active_mask = units_active_mask.unsqueeze(-1).to(dtype=torch.float32)
    return approach * height * active_mask


def _compute_forward_reward_wall_boost_mask_torch(
    *,
    unit_y: torch.Tensor,
    unit_z: torch.Tensor,
    stable_mask: torch.Tensor,
    wall_y_by_wall: torch.Tensor,
    wall_heights: torch.Tensor,
    boost_distance: float,
    height_margin: float,
    wall_half_thickness: float,
) -> torch.Tensor:
    if wall_y_by_wall.shape[1] == 0:
        return torch.zeros_like(unit_y, dtype=torch.bool)

    safe_unit_y = torch.where(stable_mask.unsqueeze(1), unit_y, torch.zeros_like(unit_y))
    safe_unit_z = torch.where(stable_mask.unsqueeze(1), unit_z, torch.zeros_like(unit_z))
    distance_to_wall = wall_y_by_wall.unsqueeze(1) - safe_unit_y.unsqueeze(-1)
    near_wall = (distance_to_wall >= -float(wall_half_thickness)) & (distance_to_wall <= float(boost_distance))
    high_enough = safe_unit_z.unsqueeze(-1) >= (wall_heights.view(1, 1, -1) + float(height_margin))
    return (near_wall & high_enough).any(dim=-1)


def _compute_forward_reward_wall_boost_mask_np(
    *,
    unit_y: np.ndarray,
    unit_z: np.ndarray,
    wall_y_by_wall: np.ndarray,
    wall_heights: np.ndarray,
    boost_distance: float,
    height_margin: float,
    wall_half_thickness: float,
) -> np.ndarray:
    if wall_y_by_wall.size == 0:
        return np.zeros_like(unit_y, dtype=bool)

    distance_to_wall = wall_y_by_wall[np.newaxis, :] - unit_y[:, np.newaxis]
    near_wall = (distance_to_wall >= -float(wall_half_thickness)) & (distance_to_wall <= float(boost_distance))
    high_enough = unit_z[:, np.newaxis] >= (wall_heights[np.newaxis, :] + float(height_margin))
    return (near_wall & high_enough).any(axis=-1)


def _compute_wall_climb_done_mask_torch(
    *,
    unit_y: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    wall_y_by_wall: torch.Tensor,
) -> torch.Tensor:
    if wall_y_by_wall.shape[1] == 0:
        return torch.zeros((*unit_y.shape, 0), device=unit_y.device, dtype=torch.bool)

    safe_unit_y = torch.where(stable_mask.unsqueeze(1), unit_y, torch.zeros_like(unit_y))
    crossed_wall_y = safe_unit_y.unsqueeze(-1) > wall_y_by_wall.unsqueeze(1)
    return crossed_wall_y & units_active_mask.unsqueeze(-1)


def _compute_obstacle_street_reward_kernel(
    unit_y: torch.Tensor,
    unit_z: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    partner_unit: torch.Tensor,
    wall_pass_absolute_thresholds: torch.Tensor,
    wall_y_by_wall: torch.Tensor,
    wall_heights: torch.Tensor,
    next_threshold_for_unit: torch.Tensor,
    wall_climb_potential: torch.Tensor,
    wall_climb_done_mask: torch.Tensor,
    progress: torch.Tensor,
    forward_progress_unit_y: torch.Tensor,
    threshold_index_torch: torch.Tensor,
    unit_rank_torch: torch.Tensor,
    wall_pass_rank_weights: torch.Tensor,
    progress_reward_weight: float,
    forward_reward_weight: float,
    forward_reward_max_y: float,
    wall_pass_reward_weight: float,
    wall_pass_reward_skew: float,
    wall_thresholds_per_wall: int,
    wall_climb_reward_weight: float,
    wall_climb_reward_distance: float,
    potential_reward_discount_factor: float,
    forward_reward_wall_boost_factor: float,
    forward_reward_wall_boost_distance: float,
    forward_reward_wall_boost_height_margin: float,
    unit_ground_z: float,
    units_without_connections_reward_weight: float,
    guidance_reward_weight: float,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    safe_unit_y = torch.where(stable_mask.unsqueeze(1), unit_y, torch.zeros_like(unit_y))
    capped_unit_y = torch.clamp(safe_unit_y, max=float(forward_reward_max_y))
    if forward_reward_wall_boost_factor == 1.0:
        forward_progress_unit_potential = capped_unit_y
        new_progress = masked_mean(forward_progress_unit_potential, units_active_mask, dim=1)
        progress_delta = (new_progress * float(potential_reward_discount_factor)) - progress
    else:
        boost_mask = _compute_forward_reward_wall_boost_mask_torch(
            unit_y=unit_y,
            unit_z=unit_z,
            stable_mask=stable_mask,
            wall_y_by_wall=wall_y_by_wall,
            wall_heights=wall_heights,
            boost_distance=forward_reward_wall_boost_distance,
            height_margin=forward_reward_wall_boost_height_margin,
            wall_half_thickness=0.1,
        )
        forward_progress_unit_potential = torch.where(
            boost_mask,
            capped_unit_y * float(forward_reward_wall_boost_factor),
            capped_unit_y,
        )
        unit_forward_delta = (
            forward_progress_unit_potential * float(potential_reward_discount_factor)
        ) - forward_progress_unit_y
        new_progress = masked_mean(forward_progress_unit_potential, units_active_mask, dim=1)
        progress_delta = masked_mean(unit_forward_delta, units_active_mask, dim=1)

    wall_pass_reward = torch.zeros_like(progress, dtype=torch.float32)
    latched_thresholds = next_threshold_for_unit
    if wall_pass_absolute_thresholds.shape[1] > 0:
        total_passed = (safe_unit_y.unsqueeze(-1) > wall_pass_absolute_thresholds.unsqueeze(1)).sum(dim=-1)
        delta_passed = torch.clamp(total_passed - next_threshold_for_unit, min=0)
        delta_passed = delta_passed * units_active_mask.to(dtype=delta_passed.dtype)
        latched_thresholds = next_threshold_for_unit + delta_passed

        active_units_count = units_active_mask.sum(dim=-1)
        if wall_pass_reward_skew == 0.0:
            crossed_rank_weight_sum = delta_passed.sum(dim=-1).to(dtype=torch.float32)
        else:
            rank_weights = wall_pass_rank_weights[active_units_count]
            threshold_indices = threshold_index_torch.view(1, 1, -1)
            previous_passed_count = (
                (next_threshold_for_unit.unsqueeze(-1) > threshold_indices) & units_active_mask.unsqueeze(-1)
            ).sum(dim=1)
            new_passed_count = (
                (latched_thresholds.unsqueeze(-1) > threshold_indices) & units_active_mask.unsqueeze(-1)
            ).sum(dim=1)
            crossed_rank_mask = (
                (unit_rank_torch.view(1, 1, -1) > previous_passed_count.unsqueeze(-1))
                & (unit_rank_torch.view(1, 1, -1) <= new_passed_count.unsqueeze(-1))
            )
            crossed_rank_weight_sum = (rank_weights.unsqueeze(1) * crossed_rank_mask.to(dtype=torch.float32)).sum(dim=(1, 2))
        denom = active_units_count * max(wall_thresholds_per_wall, 1)
        valid = stable_mask & (denom > 0)
        wall_pass_reward = torch.where(
            valid,
            (
                crossed_rank_weight_sum
                / denom.to(dtype=torch.float32)
            ) * float(wall_pass_reward_weight),
            torch.zeros_like(progress, dtype=torch.float32),
        )

    passed_thresholds_mask = threshold_index_torch.view(1, 1, -1) < latched_thresholds.unsqueeze(-1)
    forward_component_reward = progress_delta * float(forward_reward_weight)
    if wall_climb_reward_weight == 0.0:
        new_wall_climb_potential = wall_climb_potential
        new_wall_climb_done_mask = wall_climb_done_mask
        wall_climb_reward = torch.zeros_like(progress, dtype=torch.float32)
    else:
        current_wall_climb_potential = _compute_wall_climb_potential_torch(
            unit_y=unit_y,
            unit_z=unit_z,
            stable_mask=stable_mask,
            units_active_mask=units_active_mask,
            wall_y_by_wall=wall_y_by_wall,
            wall_heights=wall_heights,
            wall_climb_reward_distance=wall_climb_reward_distance,
            unit_ground_z=unit_ground_z,
        )
        crossed_wall_y_mask = _compute_wall_climb_done_mask_torch(
            unit_y=unit_y,
            stable_mask=stable_mask,
            units_active_mask=units_active_mask,
            wall_y_by_wall=wall_y_by_wall,
        )
        newly_done_mask = crossed_wall_y_mask & ~wall_climb_done_mask
        new_wall_climb_done_mask = torch.where(
            stable_mask.view(-1, 1, 1),
            wall_climb_done_mask | newly_done_mask,
            wall_climb_done_mask,
        )
        new_wall_climb_potential = torch.where(
            new_wall_climb_done_mask,
            torch.zeros_like(current_wall_climb_potential),
            current_wall_climb_potential,
        )
        wall_climb_delta = torch.where(
            stable_mask.view(-1, 1, 1),
            (new_wall_climb_potential * float(potential_reward_discount_factor)) - wall_climb_potential,
            torch.zeros_like(wall_climb_potential),
        )
        wall_climb_delta = torch.where(
            newly_done_mask & (wall_climb_delta < 0.0),
            torch.zeros_like(wall_climb_delta),
            wall_climb_delta,
        )
        wall_climb_reward = (
            masked_mean(wall_climb_delta.sum(dim=-1), units_active_mask, dim=1)
            * float(wall_climb_reward_weight)
        )
    progress_reward = (forward_component_reward + wall_pass_reward + wall_climb_reward) * float(progress_reward_weight)
    forward_reward = forward_component_reward * float(progress_reward_weight)
    wall_pass_reward = wall_pass_reward * float(progress_reward_weight)
    wall_climb_reward = wall_climb_reward * float(progress_reward_weight)

    connection_mask = partner_unit >= 0
    units_without_connections = (~connection_mask).all(dim=-1) & units_active_mask
    active_units_count = units_active_mask.sum(dim=-1)
    units_without_connections_reward = torch.where(
        active_units_count > 0,
        (
            units_without_connections.sum(dim=-1).to(dtype=torch.float32)
            / active_units_count.to(dtype=torch.float32)
        ) * float(units_without_connections_reward_weight),
        torch.zeros_like(progress, dtype=torch.float32),
    )
    units_without_connections_reward *= float(guidance_reward_weight)

    return (
        new_progress,
        progress_reward,
        forward_reward,
        wall_pass_reward,
        wall_climb_reward,
        units_without_connections_reward,
        latched_thresholds,
        new_wall_climb_potential,
        new_wall_climb_done_mask,
        passed_thresholds_mask,
        forward_progress_unit_potential,
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
        unit_z = self.data.qpos[self.bindings.metadata.unit_qpos_adr + 2]
        active_mask = self._pool_active_mask[spec.common.pool_idx]
        if spec.wall_pass_thresholds.size > 0:
            total_passed = (unit_y[:, np.newaxis] > spec.wall_pass_thresholds[np.newaxis, :]).sum(axis=-1)
            passed_thresholds_mask = self.threshold_index[np.newaxis, :] < total_passed[:, np.newaxis]
        else:
            total_passed = np.zeros((active_mask.shape[0],), dtype=np.int64)
            passed_thresholds_mask = np.zeros((active_mask.shape[0], 0), dtype=bool)

        wall_y_by_wall = _extract_wall_y_by_wall_np(
            hidden_global_vars=spec.hidden_global_vars[np.newaxis, :],
            num_walls=self.scenario.num_walls,
            no_initial_ramp=self.scenario.no_initial_ramp,
        )[0]
        forward_progress_unit_y = _compute_forward_progress_unit_y_np(
            unit_y=unit_y,
            forward_reward_max_y=self.scenario.forward_reward_max_y,
        )
        boost_distance = (
            float(self.scenario.forward_reward_wall_boost_distance)
            if self.scenario.forward_reward_wall_boost_distance is not None
            else float(self.scenario.wall_climb_reward_distance)
        )
        height_margin = (
            float(self.scenario.forward_reward_wall_boost_height_margin)
            if self.scenario.forward_reward_wall_boost_height_margin is not None
            else float(self.scenario.swarm.body_radius)
        )
        forward_progress_unit_y = _apply_forward_reward_wall_boost_to_potential_np(
            unit_y=unit_y,
            unit_z=unit_z,
            forward_progress_unit_y=forward_progress_unit_y,
            wall_y_by_wall=wall_y_by_wall,
            wall_heights=np.asarray(self.scenario.wall_heights, dtype=float),
            forward_reward_wall_boost_factor=float(self.scenario.forward_reward_wall_boost_factor),
            boost_distance=boost_distance,
            height_margin=height_margin,
        )
        progress = _mean_active_forward_progress_np(
            forward_progress_unit_y=forward_progress_unit_y,
            active_mask=active_mask,
        )
        if self.scenario.wall_climb_reward_weight == 0.0:
            wall_climb_potential = np.zeros((active_mask.shape[0], self.scenario.num_walls), dtype=np.float32)
            wall_climb_done_mask = np.zeros((active_mask.shape[0], self.scenario.num_walls), dtype=bool)
        else:
            wall_climb_done_mask = (unit_y[:, np.newaxis] > wall_y_by_wall[np.newaxis, :]) & active_mask[:, np.newaxis]
            wall_climb_potential = _compute_wall_climb_potential_np(
                unit_y=unit_y,
                unit_z=unit_z,
                active_mask=active_mask,
                wall_y_by_wall=wall_y_by_wall,
                wall_heights=np.asarray(self.scenario.wall_heights, dtype=float),
                wall_climb_reward_distance=self.scenario.wall_climb_reward_distance,
                unit_ground_z=float(self.scenario.swarm.body_radius),
            )
        return ObstacleStreetSettledSnapshot(
            common=self._build_common_snapshot(common_reset_spec=spec.common),
            hidden_global_vars=spec.hidden_global_vars.copy(),
            wall_pass_thresholds=spec.wall_pass_thresholds.copy(),
            progress=progress,
            forward_progress_unit_y=forward_progress_unit_y,
            wall_climb_potential=wall_climb_potential,
            wall_climb_done_mask=wall_climb_done_mask,
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
    def __init__(
        self,
        *,
        scenario: "MJWObstacleStreetScenario",
        bindings: MJWRuntimeBindings,
        runtime_metadata: None = None,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        hidden_global_dim = int(scenario.get_single_observation_space()["hidden_global_vars"].shape[0])
        total_thresholds = scenario.total_thresholds
        self._global_obs = torch.empty((bindings.num_envs, 0), device=bindings.device, dtype=torch.float32)
        self.hidden_global_vars = torch.zeros((bindings.num_envs, hidden_global_dim), device=bindings.device, dtype=torch.float32)
        self.wall_y_by_wall = torch.zeros((bindings.num_envs, scenario.num_walls), device=bindings.device, dtype=torch.float32)
        self.wall_pass_absolute_thresholds = torch.zeros((bindings.num_envs, total_thresholds), device=bindings.device, dtype=torch.float32)
        self.wall_climb_potential = torch.zeros(
            (bindings.num_envs, scenario.swarm.num_units, scenario.num_walls),
            device=bindings.device,
            dtype=torch.float32,
        )
        self.wall_climb_done_mask = torch.zeros(
            (bindings.num_envs, scenario.swarm.num_units, scenario.num_walls),
            device=bindings.device,
            dtype=torch.bool,
        )
        self._wall_heights = torch.as_tensor(scenario.wall_heights, device=bindings.device, dtype=torch.float32)
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
        self.forward_progress_unit_y = torch.zeros(
            (bindings.num_envs, scenario.swarm.num_units),
            device=bindings.device,
            dtype=torch.float32,
        )
        self._threshold_values = torch.as_tensor(scenario.wall_pass_thresholds, device=bindings.device, dtype=torch.float32)
        self._threshold_index_torch = torch.arange(total_thresholds, device=bindings.device, dtype=torch.long)
        self._unit_rank_torch = torch.arange(1, scenario.swarm.num_units + 1, device=bindings.device, dtype=torch.long)
        self._wall_pass_rank_weights = torch.zeros(
            (scenario.swarm.num_units + 1, scenario.swarm.num_units),
            device=bindings.device,
            dtype=torch.float32,
        )
        self._wall_pass_rank_weights_skew: float | None = None
        self._threshold_index_np = np.arange(total_thresholds, dtype=np.int64)
        self._wall_thresholds_per_wall = len(scenario.wall_pass_thresholds)
        self._cpu_settler = _ObstacleStreetCPUResetSettler(
            scenario=scenario,
            bindings=bindings,
            threshold_index=self._threshold_index_np,
        )
        self._reward_kernel: Callable[
            ...,
            tuple[
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
                torch.Tensor,
            ],
        ]
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
        self.wall_y_by_wall[world_idx] = _extract_wall_y_by_wall_torch(
            hidden_global_vars=reset_batch.hidden_global_vars,
            num_walls=self.scenario.num_walls,
            no_initial_ramp=self.scenario.no_initial_ramp,
        )
        self.wall_pass_absolute_thresholds[world_idx] = reset_batch.wall_pass_thresholds

        unit_y = self._get_unit_y()[world_idx]
        unit_z = self._get_unit_z()[world_idx]
        total_passed = (unit_y.unsqueeze(-1) > reset_batch.wall_pass_thresholds.unsqueeze(1)).sum(dim=-1)
        forward_progress_unit_y = _compute_forward_progress_unit_y_torch(
            unit_y=unit_y,
            forward_reward_max_y=self.scenario.forward_reward_max_y,
        )
        forward_progress_unit_y = _apply_forward_reward_wall_boost_to_potential_torch(
            unit_y=unit_y,
            unit_z=unit_z,
            stable_mask=torch.ones((unit_y.shape[0],), device=unit_y.device, dtype=torch.bool),
            forward_progress_unit_y=forward_progress_unit_y,
            wall_y_by_wall=self.wall_y_by_wall[world_idx],
            wall_heights=self._wall_heights,
            forward_reward_wall_boost_factor=float(self.scenario.forward_reward_wall_boost_factor),
            boost_distance=(
                float(self.scenario.forward_reward_wall_boost_distance)
                if self.scenario.forward_reward_wall_boost_distance is not None
                else float(self.scenario.wall_climb_reward_distance)
            ),
            height_margin=(
                float(self.scenario.forward_reward_wall_boost_height_margin)
                if self.scenario.forward_reward_wall_boost_height_margin is not None
                else float(self.scenario.swarm.body_radius)
            ),
        )
        self.progress[world_idx] = masked_mean(
            forward_progress_unit_y,
            self.bindings.units_active_mask[world_idx],
            dim=1,
        )
        self.forward_progress_unit_y[world_idx] = forward_progress_unit_y
        if self.scenario.wall_climb_reward_weight == 0.0:
            self.wall_climb_potential[world_idx] = 0.0
            self.wall_climb_done_mask[world_idx] = False
        else:
            self.wall_climb_done_mask[world_idx] = (
                (unit_y.unsqueeze(-1) > self.wall_y_by_wall[world_idx].unsqueeze(1))
                & self.bindings.units_active_mask[world_idx].unsqueeze(-1)
            )
            self.wall_climb_potential[world_idx] = _compute_wall_climb_potential_torch(
                unit_y=unit_y,
                unit_z=unit_z,
                stable_mask=torch.ones((unit_y.shape[0],), device=unit_y.device, dtype=torch.bool),
                units_active_mask=self.bindings.units_active_mask[world_idx],
                wall_y_by_wall=self.wall_y_by_wall[world_idx],
                wall_heights=self._wall_heights,
                wall_climb_reward_distance=float(self.scenario.wall_climb_reward_distance),
                unit_ground_z=float(self.scenario.swarm.body_radius),
            )
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
        self.wall_y_by_wall[world_idx] = _extract_wall_y_by_wall_torch(
            hidden_global_vars=self.hidden_global_vars[world_idx],
            num_walls=self.scenario.num_walls,
            no_initial_ramp=self.scenario.no_initial_ramp,
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
        self.forward_progress_unit_y[world_idx] = torch.as_tensor(
            np.stack([snapshot.forward_progress_unit_y for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.forward_progress_unit_y.dtype,
        )
        self.wall_climb_potential[world_idx] = torch.as_tensor(
            np.stack([snapshot.wall_climb_potential for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.wall_climb_potential.dtype,
        )
        self.wall_climb_done_mask[world_idx] = torch.as_tensor(
            np.stack([snapshot.wall_climb_done_mask for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.wall_climb_done_mask.dtype,
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
        wall_climb_reward_weight = float(getattr(self.scenario, "wall_climb_reward_weight", 0.0))
        forward_reward_wall_boost_factor = float(getattr(self.scenario, "forward_reward_wall_boost_factor", 1.0))
        needs_wall_height_context = wall_climb_reward_weight != 0.0 or forward_reward_wall_boost_factor != 1.0
        if not needs_wall_height_context:
            unit_z = torch.empty_like(unit_y)
            wall_y_by_wall = torch.empty((unit_y.shape[0], 0), device=unit_y.device, dtype=torch.float32)
            wall_heights = torch.empty((0,), device=unit_y.device, dtype=torch.float32)
            wall_climb_potential = torch.empty((*unit_y.shape, 0), device=unit_y.device, dtype=torch.float32)
            wall_climb_done_mask = torch.empty((*unit_y.shape, 0), device=unit_y.device, dtype=torch.bool)
            wall_climb_reward_distance = 1.0
            unit_ground_z = 0.1
        else:
            unit_z = self._get_unit_z()
            wall_y_by_wall = self.wall_y_by_wall
            wall_heights = self._wall_heights
            wall_climb_potential = self.wall_climb_potential
            wall_climb_done_mask = self.wall_climb_done_mask
            wall_climb_reward_distance = float(self.scenario.wall_climb_reward_distance)
            unit_ground_z = float(self.scenario.swarm.body_radius)
        forward_reward_wall_boost_distance = (
            float(self.scenario.forward_reward_wall_boost_distance)
            if getattr(self.scenario, "forward_reward_wall_boost_distance", None) is not None
            else wall_climb_reward_distance
        )
        forward_reward_wall_boost_height_margin = (
            float(self.scenario.forward_reward_wall_boost_height_margin)
            if getattr(self.scenario, "forward_reward_wall_boost_height_margin", None) is not None
            else unit_ground_z
        )
        wall_pass_reward_skew = float(self.scenario.wall_pass_reward_skew)
        wall_pass_rank_weights = (
            self._wall_pass_rank_weights
            if wall_pass_reward_skew == 0.0
            else self._get_wall_pass_rank_weights(wall_pass_reward_skew)
        )
        (
            new_progress,
            progress_reward,
            forward_reward,
            wall_pass_reward,
            wall_climb_reward,
            units_without_connections_reward,
            latched_thresholds,
            wall_climb_potential,
            wall_climb_done_mask,
            passed_thresholds_mask,
            forward_progress_unit_y,
        ) = self._reward_kernel(
            unit_y,
            unit_z,
            stable_mask,
            self.bindings.units_active_mask,
            self.bindings.partner_unit,
            self.wall_pass_absolute_thresholds,
            wall_y_by_wall,
            wall_heights,
            self.next_threshold_for_unit,
            wall_climb_potential,
            wall_climb_done_mask,
            self.progress,
            self.forward_progress_unit_y,
            self._threshold_index_torch,
            self._unit_rank_torch,
            wall_pass_rank_weights,
            float(self.scenario.progress_reward_weight),
            float(self.scenario.forward_reward_weight),
            float("inf") if self.scenario.forward_reward_max_y is None else float(self.scenario.forward_reward_max_y),
            float(self.scenario.wall_pass_reward_weight),
            wall_pass_reward_skew,
            int(self._wall_thresholds_per_wall),
            wall_climb_reward_weight,
            wall_climb_reward_distance,
            float(getattr(self.scenario, "potential_reward_discount_factor", 1.0)),
            forward_reward_wall_boost_factor,
            forward_reward_wall_boost_distance,
            forward_reward_wall_boost_height_margin,
            unit_ground_z,
            float(self.scenario.units_without_connections_reward_weight),
            float(self.scenario.guidance_reward_weight),
        )
        self.progress[stable_mask] = new_progress[stable_mask]
        self.forward_progress_unit_y[stable_mask] = forward_progress_unit_y[stable_mask]
        if wall_climb_reward_weight != 0.0:
            self.wall_climb_potential[stable_mask] = wall_climb_potential[stable_mask]
            self.wall_climb_done_mask[stable_mask] = wall_climb_done_mask[stable_mask]
        self.next_threshold_for_unit[stable_mask] = latched_thresholds[stable_mask]
        self.passed_thresholds_mask[stable_mask] = passed_thresholds_mask[stable_mask]
        self._hidden_local_obs[stable_mask] = self.passed_thresholds_mask[stable_mask].to(dtype=torch.float32)

        return MJWStepResult(
            reward=progress_reward + units_without_connections_reward,
            info={
                "progress_reward": progress_reward,
                "forward_reward": forward_reward,
                "forward_progress_reward": forward_reward,
                "wall_pass_reward": wall_pass_reward,
                "wall_climb_reward": wall_climb_reward,
                "units_without_connections_reward": units_without_connections_reward,
                "guidance_reward": units_without_connections_reward,
                "reward_terms": {
                    "forward": forward_reward,
                    "wall": wall_pass_reward,
                    "climb": wall_climb_reward,
                    "units_without_connections": units_without_connections_reward,
                },
            },
        )

    def _get_unit_y(self) -> torch.Tensor:
        return self.bindings.qpos[:, self.bindings.unit_qpos_adr + 1]

    def _get_unit_z(self) -> torch.Tensor:
        return self.bindings.qpos[:, self.bindings.unit_qpos_adr + 2]

    def _get_wall_pass_rank_weights(self, skew: float) -> torch.Tensor:
        if not hasattr(self, "_wall_pass_rank_weights"):
            max_units = int(self._unit_rank_torch.numel())
            self._wall_pass_rank_weights = torch.zeros(
                (max_units + 1, max_units),
                device=self.bindings.device,
                dtype=torch.float32,
            )
            self._wall_pass_rank_weights_skew = None
        if skew != 0.0 and self._wall_pass_rank_weights_skew != skew:
            self._wall_pass_rank_weights.copy_(
                _build_wall_pass_rank_weights(
                    max_units=int(self._unit_rank_torch.numel()),
                    skew=skew,
                    device=self.bindings.device,
                )
            )
            self._wall_pass_rank_weights_skew = skew
        return self._wall_pass_rank_weights

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


def _compute_forward_progress_baseline_np(
    *,
    unit_y: np.ndarray,
    active_mask: np.ndarray,
    forward_reward_max_y: float | None,
) -> float:
    capped_unit_y = np.asarray(unit_y, dtype=float)
    if forward_reward_max_y is not None:
        capped_unit_y = np.minimum(capped_unit_y, forward_reward_max_y)
    active_units_mask = np.asarray(active_mask, dtype=bool)
    if not active_units_mask.any():
        return 0.0
    return float(capped_unit_y[active_units_mask].mean())


def _mean_active_forward_progress_np(*, forward_progress_unit_y: np.ndarray, active_mask: np.ndarray) -> float:
    active_units_mask = np.asarray(active_mask, dtype=bool)
    if not active_units_mask.any():
        return 0.0
    return float(forward_progress_unit_y[active_units_mask].mean())


def _compute_forward_progress_unit_y_np(*, unit_y: np.ndarray, forward_reward_max_y: float | None) -> np.ndarray:
    capped_unit_y = np.asarray(unit_y, dtype=float)
    if forward_reward_max_y is not None:
        capped_unit_y = np.minimum(capped_unit_y, forward_reward_max_y)
    return capped_unit_y.astype(np.float32, copy=False)


def _apply_forward_reward_wall_boost_to_potential_np(
    *,
    unit_y: np.ndarray,
    unit_z: np.ndarray,
    forward_progress_unit_y: np.ndarray,
    wall_y_by_wall: np.ndarray,
    wall_heights: np.ndarray,
    forward_reward_wall_boost_factor: float,
    boost_distance: float,
    height_margin: float,
) -> np.ndarray:
    if forward_reward_wall_boost_factor == 1.0:
        return forward_progress_unit_y

    boost_mask = _compute_forward_reward_wall_boost_mask_np(
        unit_y=unit_y,
        unit_z=unit_z,
        wall_y_by_wall=wall_y_by_wall,
        wall_heights=wall_heights,
        boost_distance=boost_distance,
        height_margin=height_margin,
        wall_half_thickness=0.1,
    )
    return np.where(
        boost_mask,
        forward_progress_unit_y * float(forward_reward_wall_boost_factor),
        forward_progress_unit_y,
    ).astype(np.float32, copy=False)


def _compute_wall_climb_potential_np(
    *,
    unit_y: np.ndarray,
    unit_z: np.ndarray,
    active_mask: np.ndarray,
    wall_y_by_wall: np.ndarray,
    wall_heights: np.ndarray,
    wall_climb_reward_distance: float,
    unit_ground_z: float,
) -> np.ndarray:
    if wall_y_by_wall.size == 0:
        return np.zeros((unit_y.shape[0], 0), dtype=np.float32)
    distance_to_wall = wall_y_by_wall[np.newaxis, :] - unit_y[:, np.newaxis]
    approach = np.clip(1.0 - (distance_to_wall / float(wall_climb_reward_distance)), 0.0, 1.0)
    approach = np.where(
        (distance_to_wall >= 0.0) & (distance_to_wall <= float(wall_climb_reward_distance)),
        approach,
        0.0,
    )
    approach = np.sqrt(approach)
    target_lift = np.maximum(wall_heights - float(unit_ground_z), 1e-6)
    height = np.clip(
        (unit_z[:, np.newaxis] - float(unit_ground_z)) / target_lift[np.newaxis, :],
        0.0,
        1.0,
    )
    return (approach * height * active_mask[:, np.newaxis].astype(np.float32)).astype(np.float32, copy=False)


def _compute_forward_progress_baseline_torch(
    *,
    unit_y: torch.Tensor,
    active_mask: torch.Tensor,
    forward_reward_max_y: float | None,
) -> torch.Tensor:
    capped_unit_y = unit_y
    if forward_reward_max_y is not None:
        capped_unit_y = torch.clamp(capped_unit_y, max=float(forward_reward_max_y))
    return masked_mean(capped_unit_y, active_mask, dim=1)


def _compute_forward_progress_unit_y_torch(*, unit_y: torch.Tensor, forward_reward_max_y: float | None) -> torch.Tensor:
    if forward_reward_max_y is None:
        return unit_y
    return torch.clamp(unit_y, max=float(forward_reward_max_y))


def _apply_forward_reward_wall_boost_to_potential_torch(
    *,
    unit_y: torch.Tensor,
    unit_z: torch.Tensor,
    stable_mask: torch.Tensor,
    forward_progress_unit_y: torch.Tensor,
    wall_y_by_wall: torch.Tensor,
    wall_heights: torch.Tensor,
    forward_reward_wall_boost_factor: float,
    boost_distance: float,
    height_margin: float,
) -> torch.Tensor:
    if forward_reward_wall_boost_factor == 1.0:
        return forward_progress_unit_y

    boost_mask = _compute_forward_reward_wall_boost_mask_torch(
        unit_y=unit_y,
        unit_z=unit_z,
        stable_mask=stable_mask,
        wall_y_by_wall=wall_y_by_wall,
        wall_heights=wall_heights,
        boost_distance=boost_distance,
        height_margin=height_margin,
        wall_half_thickness=0.1,
    )
    return torch.where(
        boost_mask,
        forward_progress_unit_y * float(forward_reward_wall_boost_factor),
        forward_progress_unit_y,
    )


def _extract_wall_y_by_wall_np(
    *,
    hidden_global_vars: np.ndarray,
    num_walls: int,
    no_initial_ramp: bool,
) -> np.ndarray:
    wall_y_columns: list[int] = []
    hidden_col = 0
    for wall_idx in range(num_walls):
        wall_y_columns.append(hidden_col)
        hidden_col += 3
        if wall_idx > 0 or not no_initial_ramp:
            hidden_col += 1
    return hidden_global_vars[:, wall_y_columns].astype(np.float32, copy=False)


def _extract_wall_y_by_wall_torch(
    *,
    hidden_global_vars: torch.Tensor,
    num_walls: int,
    no_initial_ramp: bool,
) -> torch.Tensor:
    wall_y_columns: list[int] = []
    hidden_col = 0
    for wall_idx in range(num_walls):
        wall_y_columns.append(hidden_col)
        hidden_col += 3
        if wall_idx > 0 or not no_initial_ramp:
            hidden_col += 1
    return hidden_global_vars[:, wall_y_columns]
