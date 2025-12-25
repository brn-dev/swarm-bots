import abc

import torch
from torch import nn


class BasePolicy(nn.Module, abc.ABC):

    @abc.abstractmethod
    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            deterministic: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError()