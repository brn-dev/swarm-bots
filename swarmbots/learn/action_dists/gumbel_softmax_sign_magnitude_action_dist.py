from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from swarmbots.learn.action_dists.action_dist import AGENT_ACTIONS_DIM
from swarmbots.learn.action_dists.sign_magnitude_beta_action_dist import (
    SignMagnitudeBetaActionDist,
    SignMagnitudeBetaConfig,
    beta_log_density,
)


@dataclass(frozen=True)
class GumbelSoftmaxSignMagnitudeBetaConfig(SignMagnitudeBetaConfig):
    gumbel_temperature: float = 1.0


def _straight_through_gumbel_softmax(
        logits: torch.Tensor,
        temperature: torch.Tensor,
) -> torch.Tensor:
    working_logits = logits.to(torch.promote_types(logits.dtype, torch.float32))
    gumbel_noise = -torch.empty_like(working_logits).exponential_().log()
    perturbed = (working_logits - working_logits.amax(dim=-1, keepdim=True)) + gumbel_noise
    peak, indices = perturbed.max(dim=-1, keepdim=True)
    # Temperature controls only the surrogate gradient. Softmax probabilities can round to a tie.
    relaxed_selection = F.softmax((perturbed - peak) / temperature, dim=-1)
    hard_selection = F.one_hot(
        indices.squeeze(-1),
        num_classes=logits.shape[-1],
    )
    return (hard_selection + (relaxed_selection - relaxed_selection.detach())).to(logits.dtype)


def _temperature_tensor(value: float, dtype: torch.dtype) -> torch.Tensor:
    temperature = torch.tensor(value, dtype=dtype, device="cpu")
    if not 0.0 < temperature.item() < float("inf"):
        raise ValueError(f"gumbel_temperature must be finite and positive in {dtype}, got {value}")
    return temperature


class GumbelSoftmaxSignMagnitudeBetaActionDist(SignMagnitudeBetaActionDist):
    """Uses an exact hard component in forward and a biased straight-through gradient."""

    def __init__(self, *args: Any, gumbel_temperature: float = 1.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.register_buffer(
            "_gumbel_temperature",
            _temperature_tensor(gumbel_temperature, torch.float32).to(self.output_net.weight.device),
        )

    @property
    def gumbel_temperature(self) -> float:
        return float(self._gumbel_temperature.item())

    def rsample(
            self,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = (agent, previous_actions)
        actions, _selection, _negative_magnitudes, _positive_magnitudes = self._rsample_with_selection()
        return actions

    def get_actions_with_log_probs(
            self,
            latent_pi: torch.Tensor,
            deterministic: bool = False,
            agent: int | None = None,
            previous_actions: torch.Tensor | None = None,
            use_rsample: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if deterministic or not use_rsample:
            return super().get_actions_with_log_probs(
                latent_pi=latent_pi,
                deterministic=deterministic,
                agent=agent,
                previous_actions=previous_actions,
                use_rsample=False,
            )

        _ = (agent, previous_actions)
        self.update_latent_features(latent_pi)
        actions, selection, negative_magnitudes, positive_magnitudes = self._rsample_with_selection()
        negative, positive = self.negative_magnitude_dist, self.positive_beta_dist
        assert isinstance(negative, torch.distributions.Beta) and positive is not None
        negative_score = beta_log_density(negative_magnitudes, negative.concentration1, negative.concentration0)
        positive_score = beta_log_density(positive_magnitudes, positive.concentration1, positive.concentration0)
        component_log_probs = self._component_log_probs(previous_actions) + torch.stack((negative_score, positive_score), dim=-1)
        log_probs = (selection * component_log_probs).sum(dim=-1).sum(dim=AGENT_ACTIONS_DIM)
        return actions, log_probs

    def _rsample_with_selection(
            self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._assert_ready()
        assert self.weight_logits is not None
        assert self.negative_magnitude_dist is not None and self.positive_magnitude_dist is not None
        selection = _straight_through_gumbel_softmax(self.weight_logits, self._gumbel_temperature)
        negative_magnitudes = self.negative_magnitude_dist.rsample()
        positive_magnitudes = self.positive_magnitude_dist.rsample()
        actions = (
            selection[..., self._POSITIVE_INDEX] * positive_magnitudes
            - selection[..., self._NEGATIVE_INDEX] * negative_magnitudes
        )
        return actions, selection, negative_magnitudes, positive_magnitudes

    def set_gumbel_temperature(self, value: float) -> None:
        self._gumbel_temperature.copy_(_temperature_tensor(value, self._gumbel_temperature.dtype))

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "gumbel_temperature": self.gumbel_temperature,
        }
