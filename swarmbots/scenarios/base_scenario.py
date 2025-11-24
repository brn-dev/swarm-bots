import abc
from typing import Generic, TypeVar

import mujoco
import numpy as np

TState = TypeVar('TState')

class BaseScenario(abc.ABC, Generic[TState]):

    def get_start_location(self):
        return np.array([0.0, 0.0, 1.0])

    @abc.abstractmethod
    def build_scenario_spec(self) -> mujoco.MjSpec:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_scenario(self, model: mujoco.MjModel, data: mujoco.MjData) -> TState:
        raise NotImplementedError()

    @abc.abstractmethod
    def modify_obs(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            state: TState | None,
            obs: np.ndarray
    ) -> np.ndarray:
        raise NotImplementedError()

    @abc.abstractmethod
    def scenario_step(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            old_state: TState,
    ) -> tuple[TState, float, bool]:
        """
        :return: (new_state, reward for step, done)
        """
        raise NotImplementedError()
