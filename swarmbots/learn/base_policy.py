import abc
from collections.abc import Iterable
from typing import Any

import torch
from torch import nn


class BasePolicy(nn.Module, abc.ABC):

    @property
    @abc.abstractmethod
    def gsde_enabled(self) -> bool:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_hyper_parameters(self) -> dict[str, Any]:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_grad_norms(self) -> dict[str, float]:
        raise NotImplementedError()

    @abc.abstractmethod
    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError()

    @abc.abstractmethod
    def update_loss_weights(self, **weights: float) -> None:
        """
        Updates weights for extra losses. If weights contains an unknown key, a ValueError is thrown
        :param weights:
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def requires_previous_actions(self) -> bool:
        raise NotImplementedError()

    @staticmethod
    def _grad_norm_from_parameters(parameters: Iterable[nn.Parameter]) -> float:
        gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
        if not gradients:
            return 0.0
        return float(nn.utils.get_total_norm(gradients, norm_type=2.0).item())

    @classmethod
    def _module_grad_norm(cls, module: nn.Module | None) -> float:
        if module is None:
            return 0.0
        return cls._grad_norm_from_parameters(module.parameters())

    @classmethod
    def _parameter_grad_norm(cls, parameter: nn.Parameter | None) -> float:
        if parameter is None:
            return 0.0
        return cls._grad_norm_from_parameters((parameter,))
