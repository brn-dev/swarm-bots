import abc

import torch


class TemporallyCorrelatedActionDist(abc.ABC):

    @abc.abstractmethod
    def reset_on_ep_start(self, mask: torch.Tensor) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_on_step(
            self,
            mask: torch.Tensor | None = None,
            batch_shape: tuple[int, ...] | None = None,
    ) -> None:
        raise NotImplementedError()
