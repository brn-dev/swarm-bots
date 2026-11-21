from __future__ import annotations

from dataclasses import dataclass
import math
from typing import TYPE_CHECKING

import mujoco
import numpy as np
import torch

from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWRuntimeBindings, MJWStepResult, mjw_forward
from swarmbots.mjw_env.mujoco_utils import dof_indices_for_body
from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_runtime import (
    MultiPayloadGoalResetBatch,
    MultiPayloadGoalResetSpec,
    MultiPayloadGoalSettledSnapshot,
    _MultiPayloadGoalCPUResetSettler,
    _compute_payload_progress_torch,
)
from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_scenario import MJWMultiPayloadGoalRuntimeMetadata
from swarmbots.mjw_env.scenarios.mjw_payload_goal_runtime import (
    PayloadGoalMJWScenarioRuntime,
    _compute_payload_goal_approach_progress,
)
from swarmbots.mjw_env.scenarios.reward_compilation import compile_reward_kernel

if TYPE_CHECKING:
    from swarmbots.mjw_env.scenarios.mjw_payload_goal_sequence_scenario import MJWPayloadGoalSequenceScenario


@dataclass(slots=True)
class PayloadGoalSequenceResetBatch:
    payload: MultiPayloadGoalResetBatch
    goals: torch.Tensor


@dataclass(slots=True)
class PayloadGoalSequenceResetSpec:
    payload: MultiPayloadGoalResetSpec
    goals: np.ndarray


@dataclass(slots=True)
class PayloadGoalSequenceSettledSnapshot:
    payload: MultiPayloadGoalSettledSnapshot
    goals: np.ndarray


def _rotate_xy(vector: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    x, y = vector.unbind(dim=-1)
    cosine, sine = yaw.cos(), yaw.sin()
    return torch.stack((cosine * x - sine * y, sine * x + cosine * y), dim=-1)


def _yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    half = yaw / 2.
    return torch.stack((half.cos(), torch.zeros_like(half), torch.zeros_like(half), half.sin()), dim=-1)


def _sample_goal_sequence(
    *, origin: torch.Tensor, yaw: torch.Tensor,
    distance_ranges: tuple[tuple[float, float], ...], turn_arc_degrees: tuple[float, ...],
    rng: torch.Generator,
) -> torch.Tensor:
    position, heading = origin, yaw
    goals = []
    for (low, high), arc in zip(distance_ranges, turn_arc_degrees, strict=True):
        distance = torch.empty_like(yaw).uniform_(low, high, generator=rng)
        turn = torch.empty_like(yaw).uniform_(-math.radians(arc) / 2., math.radians(arc) / 2., generator=rng)
        heading = heading + turn
        position = position + distance[:, None] * torch.stack((-heading.sin(), heading.cos()), dim=-1)
        goals.append(position)
    return torch.stack(goals, dim=1)


def _advance_goal_sequence(
    *, inside_goal: torch.Tensor, stable_mask: torch.Tensor, completed: torch.Tensor, inside_samples: torch.Tensor,
    goals: torch.Tensor, payload_position: torch.Tensor, unit_xy: torch.Tensor, units_active_mask: torch.Tensor,
    payload_progress: torch.Tensor, approach_progress: torch.Tensor, required_hold_steps: int,
    payload_radius: float, goal_radius: float, approach_radius: float,
    waypoint_reward: float, success_reward: float, progress_reward_weight: float,
    approach_potential: str = "zone", approach_softness: float = 0.5,
    hold_reward_fraction: float = 0.0,
) -> tuple[torch.Tensor, ...]:
    num_goals = goals.shape[1]
    unfinished = completed < num_goals
    next_hold = torch.where(inside_goal & unfinished, inside_samples + 1, 0)
    next_hold = torch.where(stable_mask & unfinished, next_hold, inside_samples)
    # First entry starts the clock; N confirmed intervals require N+1 inside samples.
    reached = stable_mask & unfinished & (next_hold >= required_hold_steps + 1)
    next_completed = completed + reached.to(dtype=completed.dtype)
    finished_now = reached & (next_completed == num_goals)
    hold_intervals = max(1, required_hold_steps)
    first_entry_offset = int(required_hold_steps > 0)
    old_hold_fraction = (inside_samples - first_entry_offset).clamp(min=0, max=hold_intervals) / hold_intervals
    new_hold_fraction = (next_hold - first_entry_offset).clamp(min=0, max=hold_intervals) / hold_intervals
    goal_bonus = waypoint_reward + (success_reward - waypoint_reward) * (completed == num_goals - 1)
    hold_reward = torch.where(
        stable_mask & unfinished,
        goal_bonus * hold_reward_fraction * progress_reward_weight * (new_hold_fraction - old_hold_fraction),
        0.,
    )
    # Score the completed hold before resetting its clock for the next goal.
    next_hold = torch.where(reached & ~finished_now, 0, next_hold)
    next_goal = goals.gather(1, next_completed.clamp_max(num_goals - 1)[:, None, None].expand(-1, 1, 2))

    forward_baseline = _compute_payload_progress_torch(
        payload_position=payload_position[:, None], goal_position=next_goal,
        active_payload_mask=units_active_mask.new_ones((goals.shape[0], 1)),
        payload_radius=payload_radius, goal_radius=goal_radius,
    )
    approach_baseline = _compute_payload_goal_approach_progress(
        unit_xy, payload_position, next_goal[:, 0], units_active_mask, payload_radius, approach_radius,
        approach_potential, approach_softness,
    )
    next_payload_progress = torch.where(reached[:, None], forward_baseline, payload_progress)
    next_approach_progress = torch.where(reached, approach_baseline, approach_progress)
    waypoint_bonus = ((reached & ~finished_now).to(payload_position.dtype) * waypoint_reward
                      * progress_reward_weight * (1. - hold_reward_fraction))
    final_bonus = (finished_now.to(payload_position.dtype) * success_reward
                   * progress_reward_weight * (1. - hold_reward_fraction))
    success = stable_mask & (next_completed == num_goals)
    return (
        next_completed, next_hold, next_goal, next_payload_progress, next_approach_progress,
        waypoint_bonus, final_bonus, success, hold_reward,
    )


class _PayloadGoalSequenceCPUResetSettler(_MultiPayloadGoalCPUResetSettler):
    def __init__(self, *, scenario: MJWPayloadGoalSequenceScenario, bindings: MJWRuntimeBindings) -> None:
        super().__init__(scenario=scenario, bindings=bindings)
        self._route_mocap_ids = np.array([
            self.model.body(f"PayloadGoal{i}").mocapid[0] for i in range(scenario.num_goals)
        ])

    def settle_batch(self, *, specs: list[PayloadGoalSequenceResetSpec]) -> list[PayloadGoalSequenceSettledSnapshot]:
        snapshots = []
        for spec in specs:
            self._prepare_reset_state(spec=spec.payload)
            half_yaw = spec.payload.common.initial_z_rotation / 2.
            self.data.qpos[self._payload_qpos_indices[0, 3:7]] = [math.cos(half_yaw), 0., 0., math.sin(half_yaw)]
            self.data.mocap_pos[self._route_mocap_ids, :2] = spec.goals
            self.data.mocap_pos[self._route_mocap_ids, 2] = 0.
            mujoco.mj_forward(self.model, self.data)
            self._settle_physics()
            snapshots.append(PayloadGoalSequenceSettledSnapshot(
                payload=self._build_settled_snapshot(spec=spec.payload), goals=spec.goals.copy(),
            ))
        return snapshots


class PayloadGoalSequenceMJWScenarioRuntime(PayloadGoalMJWScenarioRuntime):
    def __init__(
        self, *, scenario: MJWPayloadGoalSequenceScenario, bindings: MJWRuntimeBindings,
        runtime_metadata: MJWMultiPayloadGoalRuntimeMetadata,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        self.goal_sequence = torch.zeros((bindings.num_envs, scenario.num_goals, 2), device=bindings.device)
        self.completed_goals = torch.zeros(bindings.num_envs, dtype=torch.long, device=bindings.device)
        self.goal_inside_samples = torch.zeros_like(self.completed_goals)
        self._goal_indices = torch.arange(scenario.num_goals, device=bindings.device)
        self._sequence_obs = torch.zeros((bindings.num_envs, scenario.global_obs_dim), device=bindings.device)
        self._payload_xy_qvel_indices = torch.tensor(
            dof_indices_for_body(bindings.host_model, "Payload0")[:2], device=bindings.device, dtype=torch.long,
        )
        self._route_mocap_ids = torch.tensor([
            bindings.host_model.body(f"PayloadGoal{i}").mocapid[0] for i in range(scenario.num_goals)
        ], device=bindings.device, dtype=torch.long)
        self._sequence_kernel = compile_reward_kernel(
            _advance_goal_sequence, enabled=scenario.compile_reward_kernel, mode=scenario.reward_kernel_compile_mode,
        )

    @property
    def global_obs(self) -> torch.Tensor:
        return self._sequence_obs

    def _make_cpu_reset_settler(self, *, bindings: MJWRuntimeBindings) -> _PayloadGoalSequenceCPUResetSettler:
        return _PayloadGoalSequenceCPUResetSettler(scenario=self.scenario, bindings=bindings)

    def sample_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> PayloadGoalSequenceResetBatch:
        common = self._sample_common_reset_batch(n_reset=n_reset, rng=rng)
        offset = common.swarm_start.new_tensor((0., self.scenario.payload_spawn_y)).expand(n_reset, -1)
        payload_xy = common.swarm_start[:, :2] + _rotate_xy(offset, common.initial_z_rotation)
        payload_position = torch.cat((payload_xy, torch.full_like(payload_xy[:, :1], self.scenario.payload_radius)), dim=-1)
        goals = _sample_goal_sequence(
            origin=payload_xy, yaw=common.initial_z_rotation,
            distance_ranges=self.scenario.goal_distance_ranges,
            turn_arc_degrees=self.scenario.goal_turn_arc_degrees, rng=rng,
        )
        return PayloadGoalSequenceResetBatch(
            payload=MultiPayloadGoalResetBatch(
                common=common, payload_position=payload_position[:, None],
                active_payload_mask=torch.ones((n_reset, 1), dtype=torch.bool, device=self.bindings.device),
                goal_position=goals[:, :1],
            ),
            goals=goals,
        )

    def _restore_sequence(self, *, world_idx: torch.Tensor, goals: torch.Tensor) -> None:
        self.goal_sequence[world_idx] = goals
        self.completed_goals[world_idx] = 0
        self.goal_inside_samples[world_idx] = 0

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: PayloadGoalSequenceResetBatch) -> None:
        self._restore_sequence(world_idx=world_idx, goals=reset_batch.goals)
        super().apply_reset_batch(world_idx=world_idx, reset_batch=reset_batch.payload)
        self.bindings.qpos[world_idx[:, None], self._payload_quat_qpos_indices[0][None]] = _yaw_quaternion(
            reset_batch.payload.common.initial_z_rotation,
        )
        mjw_forward(self.bindings)
        self._update_payload_obs(world_idx=world_idx)

    def build_cpu_reset_specs(self, *, reset_batch: PayloadGoalSequenceResetBatch) -> list[PayloadGoalSequenceResetSpec]:
        payload_specs = super().build_cpu_reset_specs(reset_batch=reset_batch.payload)
        goals = reset_batch.goals.detach().cpu().numpy()
        return [PayloadGoalSequenceResetSpec(payload=spec, goals=goals[i].copy()) for i, spec in enumerate(payload_specs)]

    def settle_cpu_reset_specs(
        self, *, specs: list[PayloadGoalSequenceResetSpec],
    ) -> list[PayloadGoalSequenceSettledSnapshot]:
        return self._cpu_settler.settle_batch(specs=specs)

    def apply_settled_reset_batch(
        self, *, world_idx: torch.Tensor, snapshots: list[PayloadGoalSequenceSettledSnapshot],
    ) -> None:
        goals = torch.as_tensor(np.stack([s.goals for s in snapshots]), device=self.bindings.device)
        self._restore_sequence(world_idx=world_idx, goals=goals)
        super().apply_settled_reset_batch(world_idx=world_idx, snapshots=[s.payload for s in snapshots])

    def _apply_goal_markers(
        self, *, world_idx: torch.Tensor, goal_position: torch.Tensor, active_payload_mask: torch.Tensor,
    ) -> None:
        positions = torch.zeros((*self.goal_sequence[world_idx].shape[:2], 3), device=self.bindings.device)
        positions[..., :2] = self.goal_sequence[world_idx]
        self.bindings.mocap_pos[world_idx[:, None], self._route_mocap_ids[None]] = positions

    def _update_payload_obs(self, *, world_idx: torch.Tensor | None = None) -> None:
        super()._update_payload_obs(world_idx=world_idx)
        self._update_sequence_obs(world_idx=world_idx)

    def _update_sequence_obs(self, *, world_idx: torch.Tensor | None = None) -> None:
        rows = slice(None) if world_idx is None else world_idx
        num_goals = self.scenario.num_goals
        self._sequence_obs[rows, :12] = self._global_obs[rows]
        self._sequence_obs[rows, 12:12 + 2 * num_goals] = self.goal_sequence[rows].flatten(start_dim=1)
        hold_index = 12 + 3 * num_goals
        self._sequence_obs[rows, 12 + 2 * num_goals:hold_index] = (
            self.completed_goals[rows, None] == self._goal_indices
        ).to(dtype=self._sequence_obs.dtype)
        self._sequence_obs[rows, hold_index] = self.goal_inside_samples[rows] / (self.scenario.goal_hold_steps + 1)
        extra_start = hold_index + 1
        if self.scenario.include_goal_delta_obs:
            self._sequence_obs[rows, extra_start:extra_start + 2] = self._global_obs[rows, 10:12] - self._global_obs[rows, 1:3]
            extra_start += 2
        if self.scenario.include_payload_velocity_obs:
            self._sequence_obs[rows, extra_start:extra_start + 2] = self.bindings.qvel[rows][:, self._payload_xy_qvel_indices]
            extra_start += 2
        if self.scenario.include_goal_inside_obs:
            remaining = _compute_payload_progress_torch(
                payload_position=self.payload_position[rows], goal_position=self.goal_position[rows],
                active_payload_mask=self.active_payload_mask[rows], payload_radius=self.scenario.payload_radius,
                goal_radius=self.scenario.goal_radius,
            )
            self._sequence_obs[rows, extra_start] = ((remaining == 0.) & self.active_payload_mask[rows])[:, 0].to(
                dtype=self._sequence_obs.dtype,
            )

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        result = super().compute_step_rewards(stable_mask=stable_mask)
        completed, hold, goal, forward_potential, approach_potential, waypoint_bonus, final_bonus, success, hold_reward = self._sequence_kernel(
            inside_goal=result.info["success"], stable_mask=stable_mask,
            completed=self.completed_goals, inside_samples=self.goal_inside_samples,
            goals=self.goal_sequence, payload_position=self.payload_position[:, 0],
            unit_xy=self.bindings.qpos[:, self._unit_xy_qpos_indices],
            units_active_mask=self.bindings.units_active_mask,
            payload_progress=self.payload_progress, approach_progress=self.towards_payload_progress,
            required_hold_steps=self.scenario.goal_hold_steps, payload_radius=self.scenario.payload_radius,
            goal_radius=self.scenario.goal_radius, approach_radius=self.scenario.towards_payload_goal_radius,
            waypoint_reward=self.scenario.waypoint_reward, success_reward=self.scenario.success_reward,
            progress_reward_weight=self.scenario.progress_reward_weight,
            approach_potential=self.scenario.towards_payload_potential,
            approach_softness=self.scenario.towards_payload_softness,
            hold_reward_fraction=self.scenario.goal_hold_reward_fraction,
        )
        self.completed_goals.copy_(completed)
        self.goal_inside_samples.copy_(hold)
        self.goal_position.copy_(goal)
        self.payload_progress.copy_(forward_potential)
        self.towards_payload_progress.copy_(approach_potential)
        self._global_obs[:, 10:12] = goal[:, 0]
        self._update_sequence_obs()
        result.terminations = success
        result.reward = (result.info["forward_reward"] + result.info["towards_payload_reward"]
                         + waypoint_bonus + final_bonus + hold_reward)
        result.info.update(
            success=success, goal_success_reward=final_bonus, waypoint_reward=waypoint_bonus,
            hold_progress_reward=hold_reward,
            progress_reward=result.reward, goals_completed=completed.clone(), current_goal_index=completed.clone(),
            goal_hold_progress=self._sequence_obs[:, 12 + 3 * self.scenario.num_goals].clone(),
        )
        result.info["reward_terms"].update(success=final_bonus, waypoint=waypoint_bonus, hold_progress=hold_reward)
        for i in range(self.scenario.num_goals):
            result.info[f"goal_{i + 1}_reached"] = completed > i
        return result
