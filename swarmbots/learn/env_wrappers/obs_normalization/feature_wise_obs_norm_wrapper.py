from typing import Any

import numpy as np
import gymnasium as gym
from gymnasium.core import ObsType
from gymnasium.logger import warn
from gymnasium.vector.vector_env import (
    AutoresetMode,
    VectorEnv,
    VectorObservationWrapper,
)
from gymnasium.wrappers.utils import RunningMeanStd


class FeatureWiseObsNormWrapper(VectorObservationWrapper, gym.utils.RecordConstructorArgs):
    """
    Normalizes selected observation features and canonicalizes quaternion signs.

    Metric indices are normalized independently using running mean/variance.
    Quaternion indices represent the first element of a quaternion (4 values total) and are
    sign-normalized so that the first component is non-negative.
    """

    def __init__(
        self,
        env: VectorEnv,
        local_scalar_feature_indices: list[int] | np.ndarray,
        local_quaternion_indices: list[int] | np.ndarray,
        global_scalar_feature_indices: list[int] | np.ndarray,
        global_quaternion_indices: list[int] | np.ndarray,
        eps: float = 1e-6,
    ):
        gym.utils.RecordConstructorArgs.__init__(
            self,
            local_scalar_feature_indices=local_scalar_feature_indices,
            local_quaternion_indices=local_quaternion_indices,
            global_scalar_feature_indices=global_scalar_feature_indices,
            global_quaternion_indices=global_quaternion_indices,
            eps=eps,
        )
        VectorObservationWrapper.__init__(self, env)

        if "autoreset_mode" not in self.env.metadata:
            warn(
                f"{self} is missing `autoreset_mode` data. Assuming that the vector environment it follows the "
                f"`NextStep` autoreset api or autoreset is disabled. Read https://farama.org/Vector-Autoreset-Mode "
                f"for more details."
            )
        else:
            assert self.env.metadata["autoreset_mode"] in {AutoresetMode.NEXT_STEP}

        if not isinstance(self.env.single_observation_space, gym.spaces.Dict):
            raise ValueError(f"Expected Dict observation space, got {type(self.env.single_observation_space)}")
        if "local_obs" not in self.env.single_observation_space.spaces:
            raise ValueError('Expected "local_obs" key in observation space')
        if "global_obs" not in self.env.single_observation_space.spaces:
            raise ValueError('Expected "global_obs" key in observation space')

        local_space: gym.spaces.Box = self.env.single_observation_space["local_obs"]  # type: ignore[assignment]
        global_space: gym.spaces.Box = self.env.single_observation_space["global_obs"]  # type: ignore[assignment]
        local_obs_dim = int(local_space.shape[-1])
        global_obs_dim = int(global_space.shape[-1])

        self._local_scalar_indices = self._as_index_array(
            local_scalar_feature_indices, max_index=local_obs_dim, name="local_scalar_feature_indices"
        )
        self._global_scalar_indices = self._as_index_array(
            global_scalar_feature_indices, max_index=global_obs_dim, name="global_scalar_feature_indices"
        )
        self._local_quaternion_slices = self._as_quaternion_slices(
            local_quaternion_indices, max_index=local_obs_dim, name="local_quaternion_indices"
        )
        self._global_quaternion_slices = self._as_quaternion_slices(
            global_quaternion_indices, max_index=global_obs_dim, name="global_quaternion_indices"
        )
        self._ensure_no_overlap(
            self._local_scalar_indices, self._local_quaternion_slices, name="local"
        )
        self._ensure_no_overlap(
            self._global_scalar_indices, self._global_quaternion_slices, name="global"
        )

        self.local_obs_rms: RunningMeanStd | None = None
        if self._local_scalar_indices.size > 0:
            self.local_obs_rms = RunningMeanStd(
                shape=(self._local_scalar_indices.size,),
                dtype=local_space.dtype,
            )

        self.global_obs_rms: RunningMeanStd | None = None
        if self._global_scalar_indices.size > 0:
            self.global_obs_rms = RunningMeanStd(
                shape=(self._global_scalar_indices.size,),
                dtype=global_space.dtype,
            )

        self._eps = float(eps)
        self._update_running_mean = True

    @property
    def update_running_mean(self) -> bool:
        return self._update_running_mean

    @update_running_mean.setter
    def update_running_mean(self, setting: bool):
        self._update_running_mean = setting

    def observations(self, observations: ObsType) -> ObsType:
        local_obs = observations["local_obs"]
        global_obs = observations["global_obs"]

        if self._update_running_mean:
            if self.local_obs_rms is not None:
                self.local_obs_rms.update(local_obs[..., self._local_scalar_indices])
            if self.global_obs_rms is not None:
                self.global_obs_rms.update(global_obs[..., self._global_scalar_indices])

        if self.local_obs_rms is not None:
            local_scalars = local_obs[..., self._local_scalar_indices]
            local_obs[..., self._local_scalar_indices] = (
                local_scalars - self.local_obs_rms.mean
            ) / np.sqrt(self.local_obs_rms.var + self._eps)

        if self.global_obs_rms is not None:
            global_scalars = global_obs[..., self._global_scalar_indices]
            global_obs[..., self._global_scalar_indices] = (
                global_scalars - self.global_obs_rms.mean
            ) / np.sqrt(self.global_obs_rms.var + self._eps)

        if self._local_quaternion_slices.size > 0:
            self._normalize_quaternion_signs(local_obs, self._local_quaternion_slices)
        if self._global_quaternion_slices.size > 0:
            self._normalize_quaternion_signs(global_obs, self._global_quaternion_slices)

        return observations

    @staticmethod
    def _as_index_array(
        indices: list[int] | np.ndarray, *, max_index: int, name: str
    ) -> np.ndarray:
        arr = np.asarray(indices, dtype=np.int64).reshape(-1)
        if arr.size == 0:
            return arr
        if arr.min() < 0 or arr.max() >= max_index:
            raise ValueError(f"{name} out of bounds for dimension {max_index}: {arr}")
        return arr

    @staticmethod
    def _as_quaternion_slices(
        indices: list[int] | np.ndarray, *, max_index: int, name: str
    ) -> np.ndarray:
        starts = np.asarray(indices, dtype=np.int64).reshape(-1)
        if starts.size == 0:
            return np.empty((0, 4), dtype=np.int64)
        if starts.min() < 0 or np.any(starts + 3 >= max_index):
            raise ValueError(f"{name} out of bounds for dimension {max_index}: {starts}")
        return np.stack([starts + i for i in range(4)], axis=1)

    @staticmethod
    def _normalize_quaternion_signs(obs: np.ndarray, quat_slices: np.ndarray) -> None:
        quats = obs[..., quat_slices]
        sign = np.where(quats[..., 0] < 0, -1.0, 1.0)
        quats = quats * sign[..., None]
        obs[..., quat_slices] = quats

    @staticmethod
    def _ensure_no_overlap(
        scalar_indices: np.ndarray, quaternion_slices: np.ndarray, *, name: str
    ) -> None:
        if scalar_indices.size == 0 or quaternion_slices.size == 0:
            return
        quat_indices = np.unique(quaternion_slices.reshape(-1))
        overlap = np.intersect1d(scalar_indices, quat_indices)
        if overlap.size > 0:
            raise ValueError(
                f"{name} scalar indices overlap with quaternion indices: {overlap.tolist()}"
            )
