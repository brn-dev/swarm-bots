import abc

import mujoco


class BaseSwarm(abc.ABC):

    @abc.abstractmethod
    def create_swarm_spec(self) -> mujoco.MjSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_swarm(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_unit_prefixes(self) -> set[str]:
        raise NotImplementedError()
