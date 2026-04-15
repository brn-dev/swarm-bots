from __future__ import annotations

from typing import Any

import numpy as np
import torch

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper, TorchObs
from swarmbots.learn.env_wrappers.learn_wrappers.torch_env_wrapper import TorchEnvWrapper
from swarmbots.learn.torch_running_mean_std import TorchRunningMeanStd


class TorchFeatureWiseObsNormWrapper(TorchEnvWrapper):
    def __init__(
        self,
        env: BaseLearnEnvWrapper,
        obs_key: str,
        scalar_feature_indices: list[int] | np.ndarray,
        quaternion_indices: list[int] | np.ndarray,
        eps: float = 1e-6,
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
        )
        self._quaternion_slices = self._as_quaternion_slices(
            quaternion_indices,
            max_index=obs_dim,
            name=f"{obs_key}_quaternion_indices",
        )
        self._ensure_no_overlap(self._scalar_indices, self._quaternion_slices, name=obs_key)

        self.obs_rms: TorchRunningMeanStd | None = None
        if self._scalar_indices.numel() > 0:
            self.obs_rms = TorchRunningMeanStd(shape=(self._scalar_indices.numel(),), device=self.device)

        self._eps = float(eps)
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool) -> None:
        self._update_running_mean = setting

    def observations(self, observations: TorchObs) -> TorchObs:
        obs = observations[self.obs_key]

        if self._update_running_mean and self.obs_rms is not None:
            batch = self._prepare_batch(
                samples=obs[..., self._scalar_indices],
                agent_mask=observations.get("agent_mask", None),
            )
            if batch.numel() > 0:
                if batch.shape[-1] != self.obs_rms.mean.shape[-1]:
                    raise ValueError(
                        f'TorchFeatureWiseObsNormWrapper("{self.obs_key}") batch width mismatch: '
                        f"{batch.shape[-1]} != {self.obs_rms.mean.shape[-1]}"
                    )
                self.obs_rms.update(batch)

        if self.obs_rms is None and self._quaternion_slices.numel() == 0:
            return observations

        normalized_obs = obs.clone()
        if self.obs_rms is not None:
            scalar_mean = self.obs_rms.mean.to(device=obs.device, dtype=obs.dtype)
            scalar_var = self.obs_rms.var.to(device=obs.device, dtype=obs.dtype)
            scalars = normalized_obs[..., self._scalar_indices]
            normalized_obs[..., self._scalar_indices] = (scalars - scalar_mean) / torch.sqrt(scalar_var + self._eps)

        if self._quaternion_slices.numel() > 0:
            self._normalize_quaternion_signs(normalized_obs, self._quaternion_slices.to(obs.device))

        normalized_observations = dict(observations)
        normalized_observations[self.obs_key] = normalized_obs
        return normalized_observations

    def set_device(self, device: torch.device | str) -> None:
        if self.obs_rms is not None:
            self.obs_rms.to(device)
        super().set_device(device)

    def _prepare_batch(
        self,
        *,
        samples: torch.Tensor,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if (
            agent_mask is not None
            and agent_mask.ndim == 2
            and samples.ndim >= 3
            and tuple(agent_mask.shape) == tuple(samples.shape[:2])
        ):
            masked_samples = samples[agent_mask]
            if masked_samples.ndim == 1:
                return masked_samples.reshape(1, -1)
            return masked_samples.reshape(-1, masked_samples.shape[-1])
        if samples.ndim == 1:
            return samples.reshape(1, -1)
        return samples.reshape(-1, samples.shape[-1])

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
    def _normalize_quaternion_signs(obs: torch.Tensor, quat_slices: torch.Tensor) -> None:
        quats = obs[..., quat_slices]
        sign = torch.where(quats[..., 0] < 0, -1.0, 1.0).to(dtype=obs.dtype)
        obs[..., quat_slices] = quats * sign.unsqueeze(-1)

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
