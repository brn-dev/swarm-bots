from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch

from swarmbots.mjw_env.mjw_torch_utils import copy_masked_rows_, masked_mean
from swarmbots.mjw_env.scenarios.base_mjw_scenario import MJWCommonResetBatch, MJWRuntimeBindings, MJWStepResult
from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_runtime import (
    MultiPayloadGoalMJWScenarioRuntime,
    MultiPayloadGoalResetBatch,
    MultiPayloadGoalSettledSnapshot,
)
from swarmbots.mjw_env.scenarios.mjw_multi_payload_goal_scenario import MJWMultiPayloadGoalRuntimeMetadata
from swarmbots.mjw_env.scenarios.mjw_payload_plane_runtime import _compute_towards_payload_progress_baseline_torch
from swarmbots.mjw_env.scenarios.reward_compilation import compile_reward_kernel

if TYPE_CHECKING:
    from swarmbots.mjw_env.scenarios.mjw_payload_goal_scenario import MJWPayloadGoalScenario


def _sample_arc_goal_positions(
    *, origin: torch.Tensor, radius: float, arc_degrees: float, noise_std: float, rng: torch.Generator,
) -> torch.Tensor:
    half_angle = math.radians(arc_degrees) / 2.0
    angle = torch.empty(origin.shape[0], device=origin.device, dtype=origin.dtype).uniform_(
        -half_angle, half_angle, generator=rng,
    )
    offset = float(radius) * torch.stack((angle.sin(), angle.cos()), dim=-1)
    if noise_std > 0.0:
        offset = offset + float(noise_std) * torch.randn(
            offset.shape, device=origin.device, dtype=origin.dtype, generator=rng,
        )
    return origin + offset


def _compute_payload_goal_approach_progress(
    unit_xy: torch.Tensor,
    payload_position: torch.Tensor,
    goal_position: torch.Tensor,
    units_active_mask: torch.Tensor,
    payload_radius: float,
    towards_payload_goal_radius: float,
    approach_potential: str = "zone",
    approach_softness: float = 0.5,
) -> torch.Tensor:
    payload_xy = payload_position[..., :2]
    goal_delta = goal_position - payload_xy
    goal_distance = torch.hypot(goal_delta[..., 0], goal_delta[..., 1]).unsqueeze(-1)
    # Coincident payload/goal centres have no preferred pushing direction.
    pushing_zone_center = payload_xy - float(payload_radius) * goal_delta / goal_distance.clamp_min(1e-6)
    if approach_potential == "pseudo_huber":
        squared_distance = (unit_xy - pushing_zone_center[:, None]).square().sum(dim=-1)
        # Rationalizing sqrt(d^2 + l^2) - l preserves small distances in float32.
        cost = squared_distance / (torch.sqrt(squared_distance + approach_softness**2) + approach_softness)
        return -masked_mean(cost, units_active_mask, dim=1)
    return _compute_towards_payload_progress_baseline_torch(
        unit_xy=unit_xy,
        virtual_goal_position=pushing_zone_center,
        units_active_mask=units_active_mask,
        goal_radius=towards_payload_goal_radius,
    )


def _compute_payload_goal_approach_reward_kernel(
    unit_xy: torch.Tensor,
    payload_position: torch.Tensor,
    goal_position: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    towards_payload_progress: torch.Tensor,
    payload_radius: float,
    towards_payload_goal_radius: float,
    towards_payload_reward_weight: float,
    progress_reward_weight: float,
    potential_reward_discount_factor: float,
    approach_potential: str = "zone",
    approach_softness: float = 0.5,
) -> tuple[torch.Tensor, torch.Tensor]:
    new_progress = _compute_payload_goal_approach_progress(
        torch.where(stable_mask[:, None, None], unit_xy, 0.0),
        torch.where(stable_mask[:, None], payload_position, 0.0),
        goal_position,
        units_active_mask,
        payload_radius,
        towards_payload_goal_radius,
        approach_potential,
        approach_softness,
    )
    reward = (
        (float(potential_reward_discount_factor) * new_progress - towards_payload_progress)
        * float(towards_payload_reward_weight)
        * float(progress_reward_weight)
    )
    return new_progress, torch.where(stable_mask, reward, 0.0)


class PayloadGoalMJWScenarioRuntime(MultiPayloadGoalMJWScenarioRuntime):
    def __init__(
        self,
        *,
        scenario: MJWPayloadGoalScenario,
        bindings: MJWRuntimeBindings,
        runtime_metadata: MJWMultiPayloadGoalRuntimeMetadata,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        self.towards_payload_progress = torch.zeros(bindings.num_envs, device=bindings.device)
        self._unit_xy_qpos_indices = bindings.unit_qpos_adr[:, None] + torch.arange(2, device=bindings.device)
        self._approach_reward_kernel = compile_reward_kernel(
            _compute_payload_goal_approach_reward_kernel,
            enabled=scenario.compile_reward_kernel,
            mode=scenario.reward_kernel_compile_mode,
        )

    def _sample_goal_positions(
        self, *, common_reset_batch: MJWCommonResetBatch, active_payload_mask: torch.Tensor, rng: torch.Generator,
    ) -> torch.Tensor:
        origin = common_reset_batch.swarm_start[:, :2] + common_reset_batch.swarm_start.new_tensor(
            (0.0, self.scenario.payload_spawn_y),
        )
        return _sample_arc_goal_positions(
            origin=origin,
            radius=self.scenario.goal_arc_radius,
            arc_degrees=self.scenario.goal_arc_degrees,
            noise_std=self.scenario.goal_noise_std,
            rng=rng,
        ).unsqueeze(1)

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: MultiPayloadGoalResetBatch) -> None:
        super().apply_reset_batch(world_idx=world_idx, reset_batch=reset_batch)
        self._reset_approach_progress(world_idx)

    def apply_settled_reset_batch(
        self, *, world_idx: torch.Tensor, snapshots: list[MultiPayloadGoalSettledSnapshot],
    ) -> None:
        super().apply_settled_reset_batch(world_idx=world_idx, snapshots=snapshots)
        self._reset_approach_progress(world_idx)

    def _reset_approach_progress(self, world_idx: torch.Tensor) -> None:
        self.towards_payload_progress[world_idx] = _compute_payload_goal_approach_progress(
            self.bindings.qpos[world_idx][:, self._unit_xy_qpos_indices],
            self.payload_position[world_idx, 0],
            self.goal_position[world_idx, 0],
            self.bindings.units_active_mask[world_idx],
            self.scenario.payload_radius,
            self.scenario.towards_payload_goal_radius,
            self.scenario.towards_payload_potential,
            self.scenario.towards_payload_softness,
        )

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        result = super().compute_step_rewards(stable_mask=stable_mask)
        new_progress, towards_payload_reward = self._approach_reward_kernel(
            self.bindings.qpos[:, self._unit_xy_qpos_indices],
            self.payload_position[:, 0],
            self.goal_position[:, 0],
            stable_mask,
            self.bindings.units_active_mask,
            self.towards_payload_progress,
            self.scenario.payload_radius,
            self.scenario.towards_payload_goal_radius,
            self.scenario.towards_payload_reward_weight,
            self.scenario.progress_reward_weight,
            self.scenario.potential_reward_discount_factor,
            self.scenario.towards_payload_potential,
            self.scenario.towards_payload_softness,
        )
        copy_masked_rows_(self.towards_payload_progress, new_progress, stable_mask)
        result.reward = result.reward + towards_payload_reward
        result.info["progress_reward"] = result.reward
        result.info["towards_payload_reward"] = towards_payload_reward
        result.info["reward_terms"]["towards_payload"] = towards_payload_reward
        return result
