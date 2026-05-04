from __future__ import annotations

from dataclasses import dataclass
import shutil
import sys
from typing import Callable

import mujoco
import mujoco_warp as mjw
import numpy as np
import torch

import swarmbots.mj_env.mujoco_utils as mj_utils
from swarmbots.mjw_env.mjw_torch_utils import sample_float_or_dist
from swarmbots.mjw_env.scenarios.base_mjw_scenario import (
    BaseMJWCPUResetSettler,
    BaseMJWScenarioRuntime,
    MJWCommonResetBatch,
    MJWCommonResetSpec,
    MJWCommonSettledSnapshot,
    MJWRuntimeBindings,
    MJWStepResult,
)
from swarmbots.mjw_env.scenarios.mjw_payload_plane_scenario import MJWPayloadPlaneRuntimeMetadata


@dataclass(slots=True)
class PayloadPlaneResetBatch:
    common: MJWCommonResetBatch
    payload_position: torch.Tensor


@dataclass(slots=True)
class PayloadPlaneResetSpec:
    common: MJWCommonResetSpec
    payload_position: np.ndarray


@dataclass(slots=True)
class PayloadPlaneSettledSnapshot:
    common: MJWCommonSettledSnapshot
    payload_position: np.ndarray
    progress: float


def _compute_payload_plane_reward_kernel(
    payload_position: torch.Tensor,
    stable_mask: torch.Tensor,
    units_active_mask: torch.Tensor,
    partner_unit: torch.Tensor,
    progress: torch.Tensor,
    progress_reward_weight: float,
    forward_reward_weight: float,
    forward_reward_max_y: float,
    payload_centering_penalty_weight: float,
    payload_centering_penalty_power: float,
    payload_centering_tolerance: float,
    units_without_connections_reward_weight: float,
    guidance_reward_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    safe_payload_x = torch.where(stable_mask, payload_position[:, 0], torch.zeros_like(progress))
    safe_payload_y = torch.where(stable_mask, payload_position[:, 1], torch.zeros_like(progress))
    new_progress = torch.clamp(safe_payload_y, max=float(forward_reward_max_y))
    progress_delta = new_progress - progress

    forward_component_reward = progress_delta * float(forward_reward_weight)
    centered_payload_x = torch.clamp(
        torch.abs(safe_payload_x) - float(payload_centering_tolerance),
        min=0.0,
    )
    if float(payload_centering_penalty_power) == 1.0:
        penalty_magnitude = centered_payload_x
    else:
        penalty_magnitude = centered_payload_x.pow(float(payload_centering_penalty_power))
    payload_x_penalty = -penalty_magnitude * float(payload_centering_penalty_weight)

    progress_reward = (forward_component_reward + payload_x_penalty) * float(progress_reward_weight)
    forward_reward = forward_component_reward * float(progress_reward_weight)
    payload_x_penalty = payload_x_penalty * float(progress_reward_weight)

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

    return new_progress, progress_reward, forward_reward, payload_x_penalty, guidance_reward


class _PayloadPlaneCPUResetSettler(BaseMJWCPUResetSettler):
    def __init__(self, *, scenario: "MJWPayloadPlaneScenario", bindings: MJWRuntimeBindings) -> None:
        super().__init__(scenario=scenario, bindings=bindings)
        self.scenario = scenario
        self._payload_qpos_indices = np.asarray(mj_utils.qpos_indices_for_body(self.model, "Payload"), dtype=np.int64)
        if self._payload_qpos_indices.shape[0] < 7:
            raise ValueError("Payload body must expose a free joint with 7 qpos values.")

    def settle_batch(self, *, specs: list[PayloadPlaneResetSpec]) -> list[PayloadPlaneSettledSnapshot]:
        return [self._settle_one(spec=spec) for spec in specs]

    def _settle_one(self, *, spec: PayloadPlaneResetSpec) -> PayloadPlaneSettledSnapshot:
        self._apply_common_reset(common_reset_spec=spec.common)
        self._apply_payload_position(payload_position=spec.payload_position)
        mujoco.mj_forward(self.model, self.data)
        self._settle_physics()

        payload_position = np.asarray(self.data.qpos[self._payload_qpos_indices[:3]], dtype=np.float64).copy()
        progress = _compute_payload_progress_baseline_np(
            payload_y=float(payload_position[1]),
            forward_reward_max_y=self.scenario.forward_reward_max_y,
        )
        return PayloadPlaneSettledSnapshot(
            common=self._build_common_snapshot(common_reset_spec=spec.common),
            payload_position=payload_position,
            progress=progress,
        )

    def _apply_payload_position(self, *, payload_position: np.ndarray) -> None:
        self.data.qpos[self._payload_qpos_indices[:3]] = payload_position
        self.data.qpos[self._payload_qpos_indices[3:7]] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


class PayloadPlaneMJWScenarioRuntime(BaseMJWScenarioRuntime):
    def __init__(
        self,
        *,
        scenario: "MJWPayloadPlaneScenario",
        bindings: MJWRuntimeBindings,
        runtime_metadata: MJWPayloadPlaneRuntimeMetadata,
    ) -> None:
        super().__init__(scenario=scenario, bindings=bindings, runtime_metadata=runtime_metadata)
        if runtime_metadata.payload_qpos_indices.shape[0] < 7:
            raise ValueError("PayloadPlaneMJWScenarioRuntime requires payload_qpos_indices in runtime metadata.")

        self.payload_position = torch.zeros((bindings.num_envs, 3), device=bindings.device, dtype=torch.float32)
        self._hidden_local_obs = torch.zeros((bindings.num_envs, scenario.swarm.num_units, 0), device=bindings.device, dtype=torch.float32)
        self._hidden_global_obs = torch.zeros((bindings.num_envs, 0), device=bindings.device, dtype=torch.float32)
        self.progress = torch.zeros((bindings.num_envs,), device=bindings.device, dtype=torch.float32)
        self._payload_qpos_indices = torch.as_tensor(
            runtime_metadata.payload_qpos_indices,
            device=bindings.device,
            dtype=torch.long,
        )
        self._cpu_settler = _PayloadPlaneCPUResetSettler(scenario=scenario, bindings=bindings)
        self._reward_kernel: Callable[
            ...,
            tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        ] = _compute_payload_plane_reward_kernel
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

    @property
    def global_obs(self) -> torch.Tensor:
        return self.payload_position

    @property
    def hidden_local_obs(self) -> torch.Tensor:
        return self._hidden_local_obs

    @property
    def hidden_global_obs(self) -> torch.Tensor:
        return self._hidden_global_obs

    def sample_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> PayloadPlaneResetBatch:
        common = self._sample_common_reset_batch(n_reset=n_reset, rng=rng)
        return PayloadPlaneResetBatch(
            common=common,
            payload_position=self._sample_payload_positions(common_reset_batch=common, rng=rng),
        )

    def select_reset_batch(self, *, reset_batch: PayloadPlaneResetBatch, mask: torch.Tensor) -> PayloadPlaneResetBatch:
        return PayloadPlaneResetBatch(
            common=self._select_common_reset_batch(common_reset_batch=reset_batch.common, mask=mask),
            payload_position=reset_batch.payload_position[mask],
        )

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: PayloadPlaneResetBatch) -> None:
        self._apply_common_reset_batch(world_idx=world_idx, common_reset_batch=reset_batch.common)
        self._apply_payload_position(world_idx=world_idx, payload_position=reset_batch.payload_position)
        mjw.forward(self.bindings.model, self.bindings.data)

        self.payload_position[world_idx] = reset_batch.payload_position
        self.progress[world_idx] = _compute_payload_progress_baseline_torch(
            payload_y=reset_batch.payload_position[:, 1],
            forward_reward_max_y=self.scenario.forward_reward_max_y,
        )

    def build_cpu_reset_specs(self, *, reset_batch: PayloadPlaneResetBatch) -> list[PayloadPlaneResetSpec]:
        common_specs = self._build_common_reset_specs(common_reset_batch=reset_batch.common)
        payload_position = reset_batch.payload_position.detach().cpu().numpy()
        return [
            PayloadPlaneResetSpec(
                common=common_specs[i],
                payload_position=payload_position[i].copy(),
            )
            for i in range(len(common_specs))
        ]

    def settle_cpu_reset_specs(self, *, specs: list[PayloadPlaneResetSpec]) -> list[PayloadPlaneSettledSnapshot]:
        return self._cpu_settler.settle_batch(specs=specs)

    def apply_settled_reset_batch(self, *, world_idx: torch.Tensor, snapshots: list[PayloadPlaneSettledSnapshot]) -> None:
        self._apply_common_settled_snapshot_batch(world_idx=world_idx, snapshots=[snapshot.common for snapshot in snapshots])
        self.payload_position[world_idx] = torch.as_tensor(
            np.stack([snapshot.payload_position for snapshot in snapshots]),
            device=self.bindings.device,
            dtype=self.payload_position.dtype,
        )
        self.progress[world_idx] = torch.as_tensor(
            [snapshot.progress for snapshot in snapshots],
            device=self.bindings.device,
            dtype=self.progress.dtype,
        )
        mjw.forward(self.bindings.model, self.bindings.data)

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> MJWStepResult:
        current_payload_position = self._get_payload_position()
        self.payload_position[:] = current_payload_position
        (
            new_progress,
            progress_reward,
            forward_reward,
            payload_x_penalty,
            guidance_reward,
        ) = self._reward_kernel(
            current_payload_position,
            stable_mask,
            self.bindings.units_active_mask,
            self.bindings.partner_unit,
            self.progress,
            float(self.scenario.progress_reward_weight),
            float(self.scenario.forward_reward_weight),
            float("inf") if self.scenario.forward_reward_max_y is None else float(self.scenario.forward_reward_max_y),
            float(self.scenario.payload_centering_penalty_weight),
            float(self.scenario.payload_centering_penalty_power),
            float(self.scenario.payload_centering_tolerance),
            float(self.scenario.units_without_connections_reward_weight),
            float(self.scenario.guidance_reward_weight),
        )
        self.progress[stable_mask] = new_progress[stable_mask]

        return MJWStepResult(
            reward=progress_reward + guidance_reward,
            info={
                "progress_reward": progress_reward,
                "forward_reward": forward_reward,
                "forward_progress_reward": forward_reward,
                "payload_x_penalty": payload_x_penalty,
                "guidance_reward": guidance_reward,
                "reward_terms": {
                    "forward": forward_reward,
                    "payload_x": payload_x_penalty,
                    "guidance": guidance_reward,
                },
            },
        )

    def _sample_payload_positions(
        self,
        *,
        common_reset_batch: MJWCommonResetBatch,
        rng: torch.Generator,
    ) -> torch.Tensor:
        n_reset = int(common_reset_batch.swarm_start.shape[0])
        payload_offset_x = sample_float_or_dist(
            self.scenario.payload_offset_x,
            shape=(n_reset,),
            device=self.bindings.device,
            generator=rng,
        )
        payload_offset_y = sample_float_or_dist(
            self.scenario.payload_offset_y,
            shape=(n_reset,),
            device=self.bindings.device,
            generator=rng,
        )
        return torch.stack(
            (
                common_reset_batch.swarm_start[:, 0] + payload_offset_x,
                common_reset_batch.swarm_start[:, 1] + payload_offset_y,
                torch.full(
                    (n_reset,),
                    float(self.scenario.payload_radius),
                    device=self.bindings.device,
                    dtype=torch.float32,
                ),
            ),
            dim=-1,
        )

    def _apply_payload_position(self, *, world_idx: torch.Tensor, payload_position: torch.Tensor) -> None:
        self.bindings.qpos[world_idx.unsqueeze(1), self._payload_qpos_indices[:3].unsqueeze(0)] = payload_position
        self.bindings.qpos[world_idx.unsqueeze(1), self._payload_qpos_indices[3:7].unsqueeze(0)] = torch.tensor(
            [1.0, 0.0, 0.0, 0.0],
            device=self.bindings.device,
            dtype=self.bindings.qpos.dtype,
        )

    def _get_payload_position(self) -> torch.Tensor:
        return self.bindings.qpos[:, self._payload_qpos_indices[:3]]


def _compute_payload_progress_baseline_np(*, payload_y: float, forward_reward_max_y: float | None) -> float:
    if forward_reward_max_y is not None:
        return min(float(payload_y), float(forward_reward_max_y))
    return float(payload_y)


def _compute_payload_progress_baseline_torch(
    *,
    payload_y: torch.Tensor,
    forward_reward_max_y: float | None,
) -> torch.Tensor:
    if forward_reward_max_y is not None:
        return torch.clamp(payload_y, max=float(forward_reward_max_y))
    return payload_y
