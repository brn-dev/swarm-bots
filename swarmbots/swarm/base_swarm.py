import abc
from dataclasses import dataclass

import mujoco
import numpy as np


class BaseSwarm(abc.ABC):

    def __init__(self, seed: int):
        self.rng = np.random.default_rng(seed)

    @abc.abstractmethod
    def create_swarm_spec(self) -> mujoco.MjSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_swarm(self, model: mujoco.MjModel, data: mujoco.MjData) -> dict:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_unit_prefixes(self) -> list[str]:
        raise NotImplementedError()
