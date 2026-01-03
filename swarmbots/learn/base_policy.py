import abc
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
    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            deterministic: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError()