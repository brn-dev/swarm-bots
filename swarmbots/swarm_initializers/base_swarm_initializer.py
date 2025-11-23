import abc

import mujoco


class BaseSwarmInitializer(abc.ABC):

    @abc.abstractmethod
    def build_swarm_spec(self) -> mujoco.MjsBody:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_swarm(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        raise NotImplementedError()