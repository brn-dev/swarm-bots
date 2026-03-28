import abc


class StickyActionDist(abc.ABC):

    @abc.abstractmethod
    def set_stickiness(self, stickiness: float) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_stickiness(self) -> float:
        raise NotImplementedError()
