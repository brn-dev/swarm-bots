"""Scalar normalization frames used by multi-step PPO observation targets."""

from collections import deque
from dataclasses import dataclass

import torch

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_transition_obs_wrapper import TorchTransitionObsWrapper


@dataclass(slots=True)
class _HistoryPadding:
    indices: torch.Tensor
    # A literal zero needs no time- or lane-dependent storage.
    values: torch.Tensor | None = None


@dataclass(slots=True)
class ScalarFrames:
    indices: torch.Tensor
    mean: torch.Tensor
    scale: torch.Tensor
    padding: tuple[_HistoryPadding, ...] = ()
    episode_ages: torch.Tensor | None = None

    def segment(self, start: int, length: int, lane: int) -> "ScalarFrames":
        # Include the final successor, including SAME_STEP terminal observations.
        steps = slice(start, start + length + 1)
        if self.episode_ages is None:
            return ScalarFrames(self.indices, self.mean[steps], self.scale[steps])
        return ScalarFrames(
            self.indices, self.mean[steps], self.scale[steps],
            tuple(p if p.values is None else _HistoryPadding(p.indices, p.values[steps]) for p in self.padding),
            self.episode_ages[lane:lane + 1, steps],
        )

    def to(self, device: torch.device | str) -> "ScalarFrames":
        return ScalarFrames(
            self.indices.to(device), self.mean.to(device), self.scale.to(device),
            tuple(_HistoryPadding(p.indices.to(device), None if p.values is None else p.values.to(device))
                  for p in self.padding),
            None if self.episode_ages is None else self.episode_ages.to(device),
        )

    @staticmethod
    def compose(layers: list["ScalarFrames"]) -> "ScalarFrames":
        if len(layers) == 1:
            return layers[0]
        indices, positions = torch.unique(
            torch.cat([layer.indices for layer in layers]), return_inverse=True,
        )
        shape = (layers[0].mean.shape[0], indices.numel())
        mean = layers[0].mean.new_zeros(shape)
        scale = torch.ones_like(mean)
        for layer, columns in zip(layers, positions.split([layer.indices.numel() for layer in layers]), strict=True):
            # Invert outer normalization before inner: raw = (z * s_outer + m_outer) * s_inner + m_inner.
            inner_scale = scale[:, columns]
            mean[:, columns] += layer.mean * inner_scale
            scale[:, columns] = layer.scale * inner_scale
        return ScalarFrames(indices, mean, scale)


@dataclass(slots=True)
class FrameHistory:
    # Keyed by wrapper identity within the live rollout so the state holds no environment references.
    moments: dict[int, torch.Tensor]
    episode_age: torch.Tensor | None


@dataclass(slots=True)
class _ObservationCopy:
    offset: int
    lag: int


@dataclass(slots=True)
class _RecordedLayer:
    wrapper: TorchFeatureWiseObsNormWrapper
    copies: list[_ObservationCopy]
    lookback: int
    moments: torch.Tensor


def _history_padding(layers: list[tuple[int, ScalarFrames]], age: int, *, single_layer: bool) -> _HistoryPadding:
    # A stack inserted zero after every layer whose source predates the reset.
    # Only the enclosing normalizers operated on that zero, in inner-to-outer order.
    missing_parts = [frame.indices for lag, frame in layers if lag > age]
    if single_layer:
        # Copies of one normalizer are disjoint, so no enclosing normalizer touched these zeros.
        return _HistoryPadding(missing_parts[0] if len(missing_parts) == 1 else torch.cat(missing_parts))
    missing = torch.unique(torch.cat(missing_parts))
    enclosing = [frame for lag, frame in layers if lag <= age]
    widths = [missing.numel(), *(frame.indices.numel() for frame in enclosing)]
    indices, positions = torch.unique(
        torch.cat([missing, *(frame.indices for frame in enclosing)]), return_inverse=True,
    )
    missing_columns, *layer_columns = positions.split(widths)
    template = layers[0][1].mean
    values = template.new_zeros((template.shape[0], indices.numel()))
    for frame, columns in zip(enclosing, layer_columns, strict=True):
        values[:, columns] = (values[:, columns] - frame.mean) / frame.scale
    return _HistoryPadding(missing, values[:, missing_columns])


class RolloutScalarFrameRecorder:
    def __init__(
        self, env: BaseLearnEnvWrapper, *, n_steps: int, history: FrameHistory | None = None,
        history_only: bool = False,
    ) -> None:
        wrappers: dict[str, list[tuple[TorchFeatureWiseObsNormWrapper, list[_ObservationCopy], int]]] = {}
        copies = {key: [_ObservationCopy(0, 0)] for key in ("local_obs", "global_obs")}
        stack_depth = lookback = 0
        while hasattr(env, "env"):
            if isinstance(env, TorchTransitionObsWrapper):
                stack_depth += 1
                for key in copies:
                    width = env.env.observation_space[key].shape[-1]
                    stride = width + (env.action_space.total_agent_action_dim if key == "local_obs" else 0)
                    current, previous = (0, stride) if env.new_obs_first else (stride, 0)
                    copies[key] = [
                        _ObservationCopy(outer.offset + offset, outer.lag + lag)
                        for outer in copies[key] for offset, lag in ((current, 0), (previous, 1))
                    ]
            if (
                isinstance(env, TorchFeatureWiseObsNormWrapper) and env.obs_key in copies
                and env.obs_rms is not None and env.normalize_output
            ):
                wrappers.setdefault(env.obs_key, []).append((env, copies[env.obs_key], stack_depth))
                lookback = stack_depth
            env = env.env

        previous_moments = {} if history is None else history.moments
        active_layers = [
            layer for layers in wrappers.values()
            if any(wrapper.update_running_mean or id(wrapper) in previous_moments for wrapper, _, _ in layers)
            for layer in reversed(layers)
        ]
        self._first_step = max(0, n_steps - lookback) if history_only else 0
        self.layers: list[_RecordedLayer] = []
        self._lookback = lookback
        self._initial_age = None if history is None else history.episode_age
        # Every scored successor is at least one step old, including terminal observations.
        retain_tail = history_only or not active_layers
        self._reset_masks = deque(maxlen=lookback if retain_tail else None) if lookback > 1 else None
        self._ages: torch.Tensor | None = None
        self._frozen_history: dict[int, torch.Tensor] = {}
        for wrapper, locations, lag in active_layers:
            if history_only and lag == 0:
                continue
            # Each wrapper's moments are shared by all lanes, agents and copies of the observation.
            shape = (2, lag + n_steps - self._first_step + 1, wrapper.scalar_indices.numel())
            moments = torch.empty(shape, device=wrapper.device, dtype=wrapper.obs_dtype)
            if lag and self._first_step == 0:
                prefix = previous_moments.get(id(wrapper))
                count = 0 if prefix is None else prefix.shape[1]
                if prefix is not None:
                    moments[:, :count].copy_(prefix)
                    if not wrapper.update_running_mean and n_steps < count:
                        # A short rollout can end before all pre-freeze frames leave the stack.
                        self._frozen_history[id(wrapper)] = prefix[:, n_steps:]
                if count < lag:
                    # Frames absent from the history were frozen or precede the initial reset.
                    moments[0, count:lag].copy_(wrapper.obs_rms.mean)
                    moments[1, count:lag].copy_(wrapper.obs_rms.var)
            self.layers.append(_RecordedLayer(wrapper, locations, lag, moments))
        self.record(0)

    def record(self, step: int, dones: torch.Tensor | None = None) -> None:
        if self._reset_masks is not None and dones is not None:
            self._reset_masks.append(dones)
        if step < self._first_step:
            return
        for layer in self.layers:
            index = step - self._first_step + layer.lookback
            layer.moments[0, index].copy_(layer.wrapper.obs_rms.mean)
            layer.moments[1, index].copy_(layer.wrapper.obs_rms.var)

    def _episode_ages(self) -> torch.Tensor | None:
        if self._reset_masks and self._ages is None:
            resets = torch.stack(tuple(self._reset_masks), dim=1)
            if self._initial_age is None:
                start = torch.zeros(resets.shape[0], device=resets.device, dtype=torch.long)
            else:
                start = self._initial_age.to(resets.device)
            steps = torch.arange(1, resets.shape[1] + 1, device=resets.device)
            last_reset = torch.where(resets, steps, -start[:, None]).cummax(dim=1).values
            # Warmup only retains the last lookback masks, which suffice for this capped age.
            ages = (steps - last_reset).clamp_max(self._lookback)
            self._ages = torch.cat((start[:, None], ages), dim=1)
        return self._ages

    def history(self) -> FrameHistory:
        # The next rollout records the boundary observation itself at step zero.
        moments = self._frozen_history.copy()
        for layer in self.layers:
            if layer.lookback and layer.wrapper.update_running_mean:
                moments[id(layer.wrapper)] = layer.moments[:, -layer.lookback - 1:-1].clone()
        ages = self._episode_ages()
        return FrameHistory(moments, None if ages is None else ages[:, -1].clone())

    def finish(self, device: torch.device | str) -> dict[str, ScalarFrames]:
        layers: dict[str, list[ScalarFrames]] = {}
        lagged_layers: dict[str, list[tuple[int, ScalarFrames]]] = {}
        for layer in self.layers:
            wrapper = layer.wrapper
            mean, variance = layer.moments.unbind(0)
            scale = (variance + wrapper.epsilon).sqrt()
            if mean.dtype in (torch.float16, torch.bfloat16):
                # Preserve the observation transform's rounded moments but compose and divide in FP32.
                mean, scale = mean.float(), scale.float()
            copies = []
            for copy in layer.copies:
                # History observed at t was normalized when the original observation was produced at t - lag.
                steps = slice(layer.lookback - copy.lag, mean.shape[0] - copy.lag)
                indices = wrapper.scalar_indices + copy.offset if copy.offset else wrapper.scalar_indices
                copies.append(ScalarFrames(indices, mean[steps], scale[steps]))
            if self._reset_masks is not None:
                lagged_layers.setdefault(wrapper.obs_key, []).extend(
                    (copy.lag, frame) for copy, frame in zip(layer.copies, copies, strict=True)
                )
            # Separate copies of one wrapper occupy disjoint columns; only different layers can overlap.
            combined = copies[0] if len(copies) == 1 else ScalarFrames(
                torch.cat([frame.indices for frame in copies]),
                torch.cat([frame.mean for frame in copies], dim=-1),
                torch.cat([frame.scale for frame in copies], dim=-1),
            )
            layers.setdefault(wrapper.obs_key, []).append(combined)
        result = {}
        for key, frames in layers.items():
            composed = ScalarFrames.compose(frames)
            recorded = lagged_layers.get(key, [])
            max_lag = max((lag for lag, _ in recorded), default=0)
            if max_lag > 1:
                composed.padding = tuple(
                    _history_padding(recorded, age, single_layer=len(frames) == 1) for age in range(1, max_lag)
                )
                composed.episode_ages = self._episode_ages()
            result[key] = composed.to(device)
        return result


def rebase_scalar_targets_(
    targets: torch.Tensor, valid: torch.Tensor, frames: list[ScalarFrames],
) -> None:
    """Express every valid successor in its prediction origin's affine frame.

    Inputs have (lane, time, horizon, ..., feature) layout. A dense rollout
    shares one history across all lanes. Segments can start at different times.
    """
    if len(frames) == 1:
        means, scales = frames[0].mean.unsqueeze(0), frames[0].scale.unsqueeze(0)
    else:
        means = torch.stack([frame.mean for frame in frames])
        scales = torch.stack([frame.scale for frame in frames])
    indices = frames[0].indices
    length, horizon = targets.shape[1:3]
    time = torch.arange(length, device=targets.device)[:, None]
    ahead = torch.arange(1, horizon + 1, device=targets.device)
    successor = (time + ahead).clamp_max(length)
    feature_axes = (1,) * (targets.ndim - 4)
    shape = (means.shape[0], length, horizon, *feature_axes, indices.numel())
    origin_scale = scales[:, :-1, None]
    ratio = (scales[:, successor] / origin_scale).reshape(shape)
    shift = ((means[:, successor] - means[:, :-1, None]) / origin_scale).reshape(shape)
    mask_shape = (*valid.shape, *([1] * (targets.ndim - valid.ndim)))
    targets[..., indices] = torch.where(
        valid.reshape(mask_shape), torch.addcmul(shift, targets[..., indices], ratio), 0,
    ).to(targets.dtype)
    if frames[0].padding:
        ages = frames[0].episode_ages if len(frames) == 1 else torch.cat([f.episode_ages for f in frames])
        # A valid terminal successor belongs to the origin episode even when its lane autoresets.
        successor_ages = ages[:, :-1, None] + ahead[:len(frames[0].padding)]
        for age, padding in enumerate(frames[0].padding, start=1):
            # A horizon longer than this episode age cannot contain this padding.
            prefix = targets[:, :, :age]
            missing = valid[:, :, :age] & (successor_ages[:, :, :age] == age)
            missing = missing.reshape(*missing.shape, *feature_axes, 1)
            sentinel = 0
            if padding.values is not None:
                values = (padding.values.unsqueeze(0) if len(frames) == 1
                          else torch.stack([frame.padding[age - 1].values for frame in frames]))
                sentinel = values[:, :-1].reshape(-1, length, 1, *feature_axes, padding.indices.numel())
            prefix[..., padding.indices] = torch.where(
                missing, sentinel, prefix[..., padding.indices],
            ).to(targets.dtype)
