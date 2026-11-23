import abc
import math
from typing import Optional, Self, Any, NamedTuple

import torch
import torch.distributions as torchdist
from torch import nn
from torch.nn.modules import module as nn_module

from swarmbots.learn.action_dists.action_dist import (
    BOUNDED_ACTION_HISTOGRAM,
    ActionNetInitialization,
    ActionDist,
    ActionMetricsSplitterInput,
    _accepts_keyword,
    compute_action_metrics,
    resolve_action_metrics_splitter,
)
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.action_dists.entropy_utils import compute_entropy_metrics
from swarmbots.learn.action_dists.tanh_bijector import TanhBijector


STD_HISTOGRAM_BINS = 11
_COMPILES_TENSOR_VERSION = tuple(map(int, torch.__version__.split(".", 2)[:2])) >= (2, 9)


class _CapturedGaussianDraw(NamedTuple):
    reference: int | torch.Tensor
    noise: torch.Tensor | None
    standard_noise: torch.Tensor | None
    use_rsample: bool


def gaussian_scale(log_stds: torch.Tensor) -> torch.Tensor:
    # d(log p)/d(scale) can overflow FP16 before exp's backward yields a finite log-std gradient.
    if log_stds.dtype in (torch.float16, torch.bfloat16):
        log_stds = log_stds.float()
    return log_stds.exp()


class ContinuousActionDist(ActionDist, abc.ABC):

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        scorer = cls.log_prob
        cls._gaussian_scorer_contract = scorer, _accepts_keyword(scorer, "gaussian_actions")

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            action_net_initialization: ActionNetInitialization,
            squash_output: bool = False,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            action_net_initialization=action_net_initialization,
        )
        self.squash_output = squash_output

        self.distribution: Optional[torchdist.Normal] = None
        self._fp16_mean_binding: torchdist.Normal | None = None
        self._last_gaussian_actions: torch.Tensor | None = None
        self._last_squashed_actions: torch.Tensor | None = None
        self._capturing_draw = False
        self._captured_draw: _CapturedGaussianDraw | None = None
        self._custom_squash_check: tuple[torch.Tensor, torch.Tensor | bool] | None = None
        self._distribution_log_scale: torch.Tensor | None = None
        self._unbounded_log_scale = False

    def update_latent_features(self, latent_pi: torch.Tensor) -> Self:
        return self._update_gaussian_from_latent(latent_pi, self.log_stds)

    def _update_gaussian_from_latent(self, latent_pi: torch.Tensor, log_stds: torch.Tensor) -> Self:
        if self._fp16_mean_binding is not None:
            self._fp16_mean_binding = None
        device_type = latent_pi.device.type
        # Custom forwards and hooks must see the caller's autocast state and choose their own output dtype.
        if (
            torch.is_autocast_enabled(device_type) and torch.get_autocast_dtype(device_type) == torch.float16
            and type(self.action_net) is nn.Linear and self.action_net.weight.dtype == torch.float32
            and "forward" not in vars(self.action_net)
            and not (self.action_net._forward_pre_hooks or self.action_net._forward_hooks
                     or nn_module._global_forward_pre_hooks or nn_module._global_forward_hooks)
            and latent_pi.dtype in (torch.float16, torch.float32)
        ):
            # A finite mean score of order 1/std can overflow the half-precision head's backward
            # before reaching its FP32 weights. Keep the score path wide and round the emitted action separately.
            with torch.autocast(device_type, enabled=False):
                self.update_distribution_params(self.action_net(latent_pi.float()), log_stds)
            self._fp16_mean_binding = self.distribution
            return self
        return self.update_distribution_params(self.action_net(latent_pi), log_stds)

    def _bind_gaussian(self, means: torch.Tensor, scale: torch.Tensor) -> Self:
        self.distribution = torchdist.Normal(loc=means, scale=scale, validate_args=False)
        self._fp16_mean_binding = None
        return self

    @abc.abstractmethod
    def update_distribution_params(self, means: torch.Tensor, log_stds: torch.Tensor) -> Self:
        raise NotImplementedError

    @torch.no_grad()
    def sample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._draw_gaussian(agent=agent)[0]

    def rsample(
            self, agent: int | None = None, previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self._draw_gaussian(agent=agent, use_rsample=True)[0]

    def mode(self, previous_actions: torch.Tensor | None = None) -> torch.Tensor:
        return self._draw_gaussian(deterministic=True)[0]

    def _draw_gaussian(
            self, *, agent: int | None = None, deterministic: bool = False, use_rsample: bool = False,
            action_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        normal = self.distribution
        # A narrow draw can change the rounded action even when mean + noise would round back to the mean.
        gaussian = normal.loc.double() if self.squash_output else normal.loc
        noise = standard_noise = None
        if not deterministic:
            scale = normal.scale
            if self.squash_output:
                scale = scale.to(torch.promote_types(scale.dtype, torch.float32))
            noise, standard_noise = self._sample_gaussian_noise(agent, scale)
            gaussian = gaussian + noise
            if not use_rsample:
                gaussian = gaussian.detach()
        actions = gaussian
        mean_dtype = torch.float16 if normal is self._fp16_mean_binding else normal.loc.dtype
        if self.squash_output:
            self._last_gaussian_actions = gaussian
            actions = TanhBijector.forward(gaussian, dtype=mean_dtype if action_dtype is None else action_dtype)
            if self._capturing_draw:
                # A custom hook can write even when rounding made the new value look unchanged.
                if torch.compiler.is_compiling() and not _COMPILES_TENSOR_VERSION:
                    reference = actions.detach().clone()
                else:
                    if torch.compiler.is_compiling() or torch.is_inference_mode_enabled():
                        with torch.inference_mode(False):
                            actions = actions.clone()
                    reference = actions._version
                self._captured_draw = _CapturedGaussianDraw(reference, noise, standard_noise, use_rsample)
            elif self._custom_squash_check is not None:
                self._custom_squash_check = None
            self._last_squashed_actions = actions
        elif action_dtype is not None:
            actions = actions.to(action_dtype)
        elif mean_dtype in (torch.float16, torch.bfloat16):
            # Sampling used the native mean and log-scale dtypes before promoting the scale for backward.
            if not deterministic:
                log_stds = self._distribution_log_scale
                if log_stds is None:
                    log_stds = getattr(self, "log_stds", normal.scale)
                mean_dtype = torch.promote_types(mean_dtype, log_stds.dtype)
            actions = actions.to(mean_dtype)
        return actions, gaussian, noise, standard_noise

    def _uses_native_gaussian_draw(self, deterministic: bool, use_rsample: bool) -> bool:
        draw = "mode" if deterministic else "rsample" if use_rsample else "sample"
        cls = type(self)
        return (
            cls.get_actions is ActionDist.get_actions
            and getattr(cls, draw) is getattr(ContinuousActionDist, draw)
            and "get_actions" not in vars(self) and draw not in vars(self)
        )

    def _draw_with_public_hooks(
            self, deterministic: bool, agent: int | None, previous_actions: torch.Tensor | None, use_rsample: bool,
    ) -> tuple[torch.Tensor, _CapturedGaussianDraw | None, bool | torch.Tensor]:
        self._last_gaussian_actions = self._last_squashed_actions = None
        self._custom_squash_check = None
        self._capturing_draw = True
        try:
            actions = self.get_actions(
                deterministic=deterministic, agent=agent, previous_actions=previous_actions, use_rsample=use_rsample,
            )
            draw = self._captured_draw
        finally:
            self._capturing_draw = False
            self._captured_draw = None
        if draw is not None and self._last_squashed_actions is actions:
            if torch.compiler.is_compiling() and not _COMPILES_TENSOR_VERSION:
                # Older compilers cannot track writes invisible in the sampled dtype. Keep the emitted values.
                self._custom_squash_check = actions, False
                return actions, draw, (actions.detach() == draw.reference).all()
            unchanged = actions._version == draw.reference
            if torch.compiler.is_compiling():
                unchanged = torch.scalar_tensor(unchanged, device=actions.device, dtype=torch.bool)
            if unchanged is not True:
                self._custom_squash_check = actions, unchanged
            return actions, draw, unchanged
        return actions, None, False

    def _score_public_gaussian_draw(
            self, actions: torch.Tensor, previous_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        scorer = self.log_prob
        if self._last_gaussian_actions is not None and (
            self._last_squashed_actions is None or self._last_squashed_actions is actions
        ):
            implementation, accepts_gaussian = self._gaussian_scorer_contract
            if type(self).log_prob is not implementation or "log_prob" in vars(self):
                accepts_gaussian = _accepts_keyword(scorer, "gaussian_actions")
            if accepts_gaussian:
                return scorer(
                    actions, previous_actions=previous_actions, gaussian_actions=self._last_gaussian_actions,
                )
        return scorer(actions, previous_actions=previous_actions)

    def _sample_gaussian_noise(
            self, agent: int | None, scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        standard_noise = torch.randn_like(scale)
        return standard_noise * scale, standard_noise

    def log_prob(
            self,
            actions: torch.Tensor,
            previous_actions: torch.Tensor | None = None,
            gaussian_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.squash_output:
            return self.sum_action_dim(TanhBijector.normal_log_prob(
                self.distribution, actions, gaussian_actions, self._distribution_log_scale, self._unbounded_log_scale,
            ))
        # Normal squares (action - loc) before dividing by the variance. Both operands may
        # be half even when the scale is FP32, which overflows the residual's backward multiplier.
        if actions.dtype in (torch.float16, torch.bfloat16):
            actions = actions.float()
        return self.sum_action_dim(self.distribution.log_prob(actions))

    def get_on_policy_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            *,
            storage_dtype: torch.dtype | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.update_latent_features(latent_pi)
        if self._uses_native_gaussian_draw(deterministic, False):
            # Round once to the precision used by the buffer, which determines the action cell on replay.
            actions = self._draw_gaussian(agent=agent, deterministic=deterministic, action_dtype=storage_dtype)[0]
        else:
            actions, _, _ = self._draw_with_public_hooks(deterministic, agent, previous_actions, False)
            if storage_dtype is not None and actions.dtype != storage_dtype:
                actions = self.sampled_action_in_dtype(actions, storage_dtype)
        return actions, self.log_prob(actions, previous_actions=previous_actions)

    def sampled_action_in_dtype(self, actions: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        if actions is self._last_squashed_actions:
            check = self._custom_squash_check
            unchanged = check[1] if check is not None and actions is check[0] else None
            if unchanged is False:
                return actions.to(dtype)
            projected = TanhBijector.forward(self._last_gaussian_actions, dtype=dtype)
            if unchanged is not None:
                projected = torch.where(unchanged, projected, actions.to(dtype))
                self._custom_squash_check = projected, unchanged
            self._last_squashed_actions = projected
            return projected
        return actions.to(dtype)

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            use_rsample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.update_latent_features(latent_pi)
        if not self._uses_native_gaussian_draw(deterministic, use_rsample):
            actions, draw, unchanged = self._draw_with_public_hooks(deterministic, agent, previous_actions, use_rsample)
            if draw is None or self._has_custom_scorer():
                return actions, self._score_public_gaussian_draw(actions, previous_actions)
            return actions, self._score_captured_gaussian_draw(actions, draw, unchanged, previous_actions)
        actions, gaussian, noise, standard_noise = self._draw_gaussian(
            agent=agent, deterministic=deterministic, use_rsample=use_rsample,
        )
        if self._has_custom_scorer():
            return actions, self._score_public_gaussian_draw(actions, previous_actions)
        if not self.squash_output:
            return actions, self.log_prob(actions, previous_actions=previous_actions).to(actions.dtype)
        return actions, self._score_gaussian_draw(actions.dtype, gaussian, noise, standard_noise, use_rsample)

    def _has_custom_scorer(self) -> bool:
        return type(self).log_prob is not ContinuousActionDist.log_prob or "log_prob" in vars(self)

    def _score_captured_gaussian_draw(
            self, actions: torch.Tensor, draw: _CapturedGaussianDraw, unchanged: bool | torch.Tensor,
            previous_actions: torch.Tensor | None,
    ) -> torch.Tensor:
        def score_native() -> torch.Tensor:
            return self._score_gaussian_draw(
                actions.dtype, self._last_gaussian_actions, draw.noise, draw.standard_noise, draw.use_rsample,
            )

        if unchanged is True:
            return score_native()
        if unchanged is False:
            return self.log_prob(actions, previous_actions=previous_actions)
        score_dtype = torch.promote_types(torch.promote_types(actions.dtype, self.distribution.scale.dtype), torch.float32)
        # Preserve native rounding and the fallback likelihood's range. Backward must use only the chosen score.
        return torch.cond(
            unchanged,
            lambda: score_native().to(score_dtype),
            lambda: self.log_prob(actions, previous_actions=previous_actions).to(score_dtype),
        )

    def _score_gaussian_draw(
            self, score_dtype: torch.dtype, gaussian: torch.Tensor, noise: torch.Tensor | None,
            standard_noise: torch.Tensor | None, use_rsample: bool,
    ) -> torch.Tensor:
        normal = self.distribution
        scale = normal.scale.to(torch.promote_types(score_dtype, torch.float32))
        score = -scale.log() - 0.5 * math.log(2.0 * math.pi)
        if noise is not None:
            if not use_rsample and torch.is_grad_enabled():
                # The sampled value is fixed for a score-function gradient. Keep its residual before mean rounding.
                standard_noise = (noise.detach() + (normal.loc.detach() - normal.loc)) / scale
            elif standard_noise is None:
                # gSDE's normalized noise depends on the learned projection, even for a pathwise draw.
                standard_noise = noise / scale
            score = score - 0.5 * standard_noise.square()
        score = score - TanhBijector.log_abs_det_jacobian(gaussian.to(scale.dtype))
        return self.sum_action_dim(score).to(score_dtype)

    @torch.no_grad()
    def get_metrics(
            self,
            actions: torch.Tensor,
            action_splitter: ActionMetricsSplitterInput = None,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> dict[str, Any]:
        splitter = resolve_action_metrics_splitter(action_splitter)
        metrics = compute_action_metrics(actions, splitter, histogram=BOUNDED_ACTION_HISTOGRAM)
        distribution = getattr(self, "distribution", None)
        if distribution is not None:
            metrics.update(compute_entropy_metrics(
                distribution.entropy(),
                agent_mask=agent_mask,
                action_splitter=splitter,
                name="ent_normal" if self.squash_output else "ent",
            ))

        if not hasattr(self, "log_stds"):
            return metrics
        log_stds = getattr(self, "log_stds")
        if not isinstance(log_stds, torch.Tensor):
            return metrics

        std_values = self._std_values_for_metrics(agent_mask)

        can_split_stds = not (std_values.ndim >= 1 and std_values.shape[-1] == 1 and self.action_dim > 1)
        if splitter is not None and can_split_stds:
            for key, split_stds in splitter(std_values).items():
                metrics[f"std_{key}"] = compute_summary_statistics(
                    split_stds, find_min=True, find_max=True, make_histogram=STD_HISTOGRAM_BINS
                )
        else:
            metrics["std"] = compute_summary_statistics(
                std_values, find_min=True, find_max=True, make_histogram=STD_HISTOGRAM_BINS
            )

        return metrics

    def _std_values_for_metrics(self, agent_mask: torch.Tensor | None) -> torch.Tensor:
        return gaussian_scale(self.log_stds)

    @staticmethod
    def sum_action_dim(tensor: torch.Tensor) -> torch.Tensor:
        return tensor.sum(dim=-1)
