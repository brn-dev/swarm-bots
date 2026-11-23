from __future__ import annotations

from _thread import LockType, RLock
from collections.abc import Callable
from copy import deepcopy
from inspect import signature
from threading import Lock
from typing import Any, Self

import numpy as np
import torch

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper, TorchObs
from swarmbots.learn.env_wrappers.learn_wrappers.torch_env_wrapper import TorchEnvWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_tensor_ops import (
    NormalizeObservations,
    build_obs_norm_tensor_operation,
    should_compile_obs_norm_tensor_operations_by_default,
)
from swarmbots.learn.tensor_conversion import to_torch_tensor
from swarmbots.learn.torch_running_mean_std import TorchRunningMeanStd


def _accepts_sample_lanes(
    observe: Callable[..., TorchObs], observations: TorchObs, sample_lanes: torch.Tensor,
) -> bool:
    try:
        observe_signature = signature(observe)
    except (TypeError, ValueError):
        return False
    try:
        observe_signature.bind(observations, sample_lanes=sample_lanes)
    except TypeError:
        return False
    return True


class TorchFeatureWiseObsNormWrapper(TorchEnvWrapper):
    def __init__(
        self,
        env: BaseLearnEnvWrapper,
        obs_key: str,
        scalar_feature_indices: list[int] | np.ndarray,
        quaternion_indices: list[int] | np.ndarray,
        eps: float = 1e-6,
        compile_tensor_operations: bool | None = None,
        tensor_operations_compile_mode: str = "reduce-overhead",
    ) -> None:
        super().__init__(env)
        if obs_key not in self.observation_space.spaces:
            raise ValueError(f'Expected "{obs_key}" key in observation space')

        self.obs_key = obs_key
        obs_dim = int(self.observation_space[obs_key].shape[-1])
        self._scalar_indices = self._as_index_array(
            scalar_feature_indices,
            max_index=obs_dim,
            name=f"{obs_key}_scalar_feature_indices",
        ).to(self.device)
        self._quaternion_slices = self._as_quaternion_slices(
            quaternion_indices,
            max_index=obs_dim,
            name=f"{obs_key}_quaternion_indices",
        ).to(self.device)
        self._ensure_no_overlap(self._scalar_indices, self._quaternion_slices, name=obs_key)

        self.obs_rms: TorchRunningMeanStd | None = None
        if self._scalar_indices.numel() > 0:
            self.obs_rms = TorchRunningMeanStd(shape=(self._scalar_indices.numel(),), device=self.device)

        self._eps = float(eps)
        self._update_running_mean = True
        self._reset_sample_lanes: torch.Tensor | None = None
        self.normalize_output = True
        self._compile_tensor_operations = compile_tensor_operations
        self._tensor_operations_compile_mode = tensor_operations_compile_mode
        self._rebuild_tensor_operations()

    @property
    def scalar_indices(self) -> torch.Tensor:
        return self._scalar_indices

    @property
    def epsilon(self) -> float:
        return self._eps

    @property
    def update_running_mean(self) -> bool:
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool) -> None:
        self._update_running_mean = setting

    def reset(self, **kwargs: Any) -> tuple[TorchObs, dict[str, Any]]:
        reset_mask = (kwargs.get("options") or {}).get("reset_mask")
        if reset_mask is None or not self._update_running_mean or self.obs_rms is None:
            return super().reset(**kwargs)

        obs, info = self.env.reset(**kwargs)
        sample_lanes = to_torch_tensor(reset_mask, device=self.device, dtype=torch.bool).reshape(self._n_envs)
        observe = self.observations
        if getattr(observe, "__func__", None) is TorchFeatureWiseObsNormWrapper.observations:
            return observe(obs, sample_lanes=sample_lanes), info

        # A custom hook may call the base transform without forwarding the reset mask.
        previous_sample_lanes = self._reset_sample_lanes
        self._reset_sample_lanes = sample_lanes
        try:
            if _accepts_sample_lanes(observe, obs, sample_lanes):
                return observe(obs, sample_lanes=sample_lanes), info
            return observe(obs), info
        finally:
            self._reset_sample_lanes = previous_sample_lanes

    def frozen_observation_copy(self, *, replacements: dict[int, Any] | None = None) -> Self:
        """Snapshot a custom observation transform while sharing its simulator.

        Direct tensor views are detached and cloned, and direct locks are replaced
        with fresh unlocked locks. Override this method for hooks that close over
        training state or need a custom snapshot of external resources. Pass
        ``{id(resource): snapshot}`` as replacements to reuse the wrapper copy.
        """
        frozen = object.__new__(type(self))
        # Copying the wrapper directly can resolve missing attributes through its live environment.
        state = object.__getstate__(self)
        attributes, slots = state if isinstance(state, tuple) else (state, {})
        memo = {**(replacements or {}), id(self): frozen, id(self.env): self.env}
        for value in (*attributes.values(), *slots.values()):
            if id(value) in memo:
                continue
            if isinstance(value, torch.Tensor) and not value.is_leaf:
                memo[id(value)] = value.detach().clone()
            elif isinstance(value, LockType):
                memo[id(value)] = Lock()
            elif isinstance(value, RLock):
                memo[id(value)] = RLock()
        attributes, slots = deepcopy((attributes, slots), memo)
        frozen.__dict__.update(attributes)
        for name, value in slots.items():
            object.__setattr__(frozen, name, value)
        frozen.update_running_mean = False
        frozen._reset_sample_lanes = None
        return frozen

    def observations(self, observations: TorchObs, *, sample_lanes: torch.Tensor | None = None) -> TorchObs:
        obs = observations[self.obs_key]

        if self.obs_rms is None and self._quaternion_slices.numel() == 0:
            return observations

        statistics_mask = self._statistics_mask(
            obs=obs,
            agent_mask=observations.get("agent_mask", None),
        )
        if sample_lanes is None:
            sample_lanes = self._reset_sample_lanes
        if sample_lanes is not None:
            lane_shape = (self._n_envs,) + (1,) * (statistics_mask.ndim - 1)
            statistics_mask = statistics_mask & sample_lanes.reshape(lane_shape)
        if self.obs_rms is None:
            running_mean = torch.empty((0,), device=obs.device, dtype=torch.float64)
            running_var = torch.empty((0,), device=obs.device, dtype=torch.float64)
            running_count = torch.zeros((), device=obs.device, dtype=torch.float64)
        else:
            running_mean = self.obs_rms.mean
            running_var = self.obs_rms.var
            running_count = self.obs_rms.count

        update_statistics = self._update_running_mean and self.obs_rms is not None
        if not self.normalize_output and not update_statistics:
            return observations
        normalize = (
            self._update_observation_statistics if not self.normalize_output
            else self._normalize_observations if update_statistics else self._normalize_frozen_observations
        )
        normalized_obs, next_mean, next_var, next_count = normalize(
            obs,
            statistics_mask,
            self._scalar_indices,
            self._quaternion_slices,
            running_mean,
            running_var,
            running_count,
            self._eps,
            update_statistics,
        )
        if self._tensor_operations_are_compiled and self.normalize_output:
            # reduce-overhead reuses CUDA-graph output buffers, while returned observations
            # remain live until the rollout snapshots them before the next environment step.
            normalized_obs = normalized_obs.clone()
        if self.obs_rms is not None and self._update_running_mean:
            self.obs_rms.mean.copy_(next_mean)
            self.obs_rms.var.copy_(next_var)
            self.obs_rms.count.copy_(next_count)

        if not self.normalize_output:
            return observations

        normalized_observations = dict(observations)
        normalized_observations[self.obs_key] = normalized_obs
        return normalized_observations

    def _transform_infos(self, infos: dict[str, Any]) -> dict[str, Any]:
        if "final_obs" not in infos or "_final_obs" not in infos:
            return infos

        old_update_running_mean = self._update_running_mean
        self._update_running_mean = False
        try:
            return super()._transform_infos(infos)
        finally:
            self._update_running_mean = old_update_running_mean

    def set_device(self, device: torch.device | str) -> None:
        old_device_type = self.device.type
        if self.obs_rms is not None:
            self.obs_rms.to(device)
        super().set_device(device)
        self._scalar_indices = self._scalar_indices.to(self.device)
        self._quaternion_slices = self._quaternion_slices.to(self.device)
        if self.device.type != old_device_type and self._compile_tensor_operations is None:
            self._rebuild_tensor_operations()

    def _rebuild_tensor_operations(self) -> None:
        self._normalize_observations = self._build_normalize_observations()
        self._default_frozen_observations = self._build_normalize_observations(update_statistics=False)
        self._normalize_frozen_observations = self._default_frozen_observations
        self._update_observation_statistics = self._build_normalize_observations(normalize_output=False)

    def _statistics_mask(
        self,
        *,
        obs: torch.Tensor,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            agent_mask is not None
            and agent_mask.ndim == 2
            and obs.ndim >= 3
            and tuple(agent_mask.shape) == tuple(obs.shape[:2])
        ):
            return agent_mask
        return torch.ones(obs.shape[:-1], device=obs.device, dtype=torch.bool)

    def _build_normalize_observations(
        self, *, update_statistics: bool = True, normalize_output: bool = True,
    ) -> NormalizeObservations:
        compile_operation = (
            should_compile_obs_norm_tensor_operations_by_default(self.device)
            if self._compile_tensor_operations is None
            else self._compile_tensor_operations
        )
        self._tensor_operations_are_compiled = compile_operation
        return build_obs_norm_tensor_operation(
            compile_operation=compile_operation,
            compile_mode=self._tensor_operations_compile_mode,
            update_statistics=update_statistics,
            normalize_output=normalize_output,
        )

    @staticmethod
    def _as_index_array(
        indices: list[int] | np.ndarray,
        *,
        max_index: int,
        name: str,
    ) -> torch.Tensor:
        arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return torch.empty((0,), dtype=torch.long)
        if arr.min() < 0 or arr.max() >= max_index:
            raise ValueError(f"{name} out of bounds for dimension {max_index}: {arr}")
        return torch.as_tensor(arr, dtype=torch.long)

    @staticmethod
    def _as_quaternion_slices(
        indices: list[int] | np.ndarray,
        *,
        max_index: int,
        name: str,
    ) -> torch.Tensor:
        starts = np.asarray(indices, dtype=np.int64).reshape(-1)
        if starts.size == 0:
            return torch.empty((0, 4), dtype=torch.long)
        if starts.min() < 0 or np.any(starts + 3 >= max_index):
            raise ValueError(f"{name} out of bounds for dimension {max_index}: {starts}")
        return torch.as_tensor(np.stack([starts + i for i in range(4)], axis=1), dtype=torch.long)

    @staticmethod
    def _ensure_no_overlap(
        scalar_indices: torch.Tensor,
        quaternion_slices: torch.Tensor,
        *,
        name: str,
    ) -> None:
        if scalar_indices.numel() == 0 or quaternion_slices.numel() == 0:
            return
        quat_indices = torch.unique(quaternion_slices.reshape(-1))
        overlap = np.intersect1d(scalar_indices.cpu().numpy(), quat_indices.cpu().numpy())
        if overlap.size > 0:
            raise ValueError(f"{name} scalar indices overlap with quaternion indices: {overlap.tolist()}")
