"""Adaptive normalization at policy and learner boundaries; replay retains raw data."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import lru_cache
from typing import TYPE_CHECKING, Any

import torch

from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_tensor_ops import _apply_normalization
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_normalize_reward_wrapper import TorchNormalizeRewardWrapper
from swarmbots.learn.env_wrappers.torch_transition_obs_wrapper import TorchTransitionObsWrapper

if TYPE_CHECKING:
    from swarmbots.learn.algos.off_policy.replay_buffer import (
        OffPolicyReplayBatch, OffPolicyReplayEpisodeSegmentBatch,
    )


NormalizationParameters = tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, float]
NormalizeTensors = Callable[..., tuple[torch.Tensor, ...]]


def _normalize_tensors(
    tensors: tuple[torch.Tensor, ...], positions: tuple[int, ...],
    parameters: tuple[NormalizationParameters, ...],
) -> tuple[torch.Tensor, ...]:
    outputs = list(tensors)
    for position, (scalar_indices, quaternion_slices, mean, variance, epsilon) in zip(
        positions, parameters, strict=True,
    ):
        outputs[position] = _apply_normalization(
            outputs[position], scalar_indices, quaternion_slices, mean, variance, epsilon,
        )
    return tuple(outputs)


@lru_cache(maxsize=None)
def _normalization_operation(compile_operation: bool) -> NormalizeTensors:
    if not compile_operation:
        return _normalize_tensors
    # Learner inputs remain live through backward and policy CUDA-graph step boundaries.
    return torch.compile(
        _normalize_tensors, fullgraph=True, dynamic=True, options={"triton.cudagraphs": False},
    )


@dataclass
class _ObservationLayout:
    wrapper: TorchFeatureWiseObsNormWrapper
    scalar_indices: torch.Tensor
    quaternion_slices: torch.Tensor
    repetitions: int = 1


@dataclass(frozen=True)
class NormalizationSnapshot:
    observations: tuple[tuple[str, NormalizationParameters], ...]
    reward_scale: torch.Tensor | None
    operation: NormalizeTensors

    def _normalize_fields(self, fields: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        if not self.observations:
            return fields
        names = tuple(fields)
        positions: list[int] = []
        parameters: list[NormalizationParameters] = []
        for key, normalization in self.observations:
            for name in (key, f"next_{key}"):
                if name in fields:
                    positions.append(names.index(name))
                    parameters.append(normalization)
        if not positions:
            return fields
        tensors = self.operation(tuple(fields.values()), tuple(positions), tuple(parameters))
        return dict(zip(names, tensors, strict=True))

    @torch.no_grad()
    def normalize_observations(self, observations: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        fields = {key: observations[key] for key, _ in self.observations if key in observations}
        return {**observations, **self._normalize_fields(fields)}

    @torch.no_grad()
    def normalize_batch(
        self, batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch:
        fields = {
            name: getattr(batch, name)
            for key, _ in self.observations for name in (key, f"next_{key}")
        }
        fields = self._normalize_fields(fields)
        if self.reward_scale is not None:
            fields["rewards"] = batch.rewards * self.reward_scale
        return replace(batch, **fields) if fields else batch


class ReplayNormalization:
    @classmethod
    def for_policy(cls, env: Any, policy: Any) -> ReplayNormalization | None:
        from swarmbots.learn.algos.sac.base_sac_policy import BaseSACPolicy
        from swarmbots.learn.algos.td3.td3_policy import TD3Policy

        while hasattr(policy, "_orig_mod"):
            policy = policy._orig_mod
        return cls(env) if isinstance(policy, (BaseSACPolicy, TD3Policy)) else None

    def __init__(self, env: Any) -> None:
        self.observation_layouts: list[_ObservationLayout] = []
        self.reward_wrappers: list[TorchNormalizeRewardWrapper] = []
        self._indices: dict[torch.device, tuple[tuple[torch.Tensor, torch.Tensor], ...]] = {}
        wrappers = []
        observation_keys: set[str] = set()
        while hasattr(env, "env"):
            if isinstance(env, TorchFeatureWiseObsNormWrapper):
                if env.obs_key in observation_keys:
                    raise ValueError(
                        "Off-policy normalization does not support duplicate observation normalization wrappers "
                        f"for key {env.obs_key!r}. Combine feature indices in one wrapper."
                    )
                observation_keys.add(env.obs_key)
            wrappers.append(env)
            env = env.env
        for wrapper in reversed(wrappers):
            if isinstance(wrapper, TorchFeatureWiseObsNormWrapper):
                wrapper.normalize_output = False
                if wrapper.obs_rms is not None or wrapper._quaternion_slices.numel():
                    self.observation_layouts.append(_ObservationLayout(
                        wrapper, wrapper._scalar_indices.cpu(), wrapper._quaternion_slices.cpu(),
                    ))
            elif isinstance(wrapper, TorchNormalizeRewardWrapper):
                self.reward_wrappers.append(wrapper)
                wrapper.normalize_rewards = False
            elif isinstance(wrapper, TorchTransitionObsWrapper):
                for layout in self.observation_layouts:
                    key = layout.wrapper.obs_key
                    if key not in {"local_obs", "global_obs"}:
                        continue
                    width = int(wrapper.env.observation_space[key].shape[-1])
                    offset = width + (wrapper._n_action_features if key == "local_obs" else 0)
                    layout.scalar_indices = torch.cat((layout.scalar_indices, layout.scalar_indices + offset))
                    layout.quaternion_slices = torch.cat((
                        layout.quaternion_slices, layout.quaternion_slices + offset,
                    ))
                    layout.repetitions *= 2

    @torch.no_grad()
    def snapshot(
        self, *, device: torch.device, copy_statistics: bool = True, public_only: bool = False,
    ) -> NormalizationSnapshot:
        if device not in self._indices:
            self._indices[device] = tuple(
                (layout.scalar_indices.to(device), layout.quaternion_slices.to(device))
                for layout in self.observation_layouts
            )
        observations = []
        compile_operation = device.type == "cuda"
        for layout, (scalar_indices, quaternion_slices) in zip(
            self.observation_layouts, self._indices[device], strict=True,
        ):
            wrapper = layout.wrapper
            if public_only and wrapper.obs_key not in {"local_obs", "global_obs"}:
                continue
            # A fused pass must respect every participating wrapper's explicit opt-out.
            compile_operation = compile_operation and wrapper._compile_tensor_operations is not False
            if wrapper.obs_rms is None:
                mean = variance = torch.empty(0, device=device, dtype=torch.float64)
            else:
                mean = wrapper.obs_rms.mean.to(device=device, copy=copy_statistics)
                variance = wrapper.obs_rms.var.to(device=device, copy=copy_statistics)
                if layout.repetitions != 1:
                    mean = mean.repeat(layout.repetitions)
                    variance = variance.repeat(layout.repetitions)
            observations.append((wrapper.obs_key, (
                scalar_indices, quaternion_slices, mean, variance, wrapper._eps,
            )))
        reward_scale = None
        if not public_only:
            for wrapper in self.reward_wrappers:
                scale = wrapper.reward_scale(device=device, dtype=torch.float32)
                reward_scale = scale if reward_scale is None else reward_scale * scale
        operation = _normalization_operation(compile_operation and bool(observations))
        return NormalizationSnapshot(tuple(observations), reward_scale, operation)
