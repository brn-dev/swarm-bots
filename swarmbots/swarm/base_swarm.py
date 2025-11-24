import abc

import mujoco
import numpy as np

import swarmbots.mujoco_utils as muju


class BaseSwarm(abc.ABC):

    def __init__(self):
        # caching indices, shape (n_units, n_x_per_unit)
        self._qpos_indices: np.ndarray | None = None
        self._qvel_indices: np.ndarray | None = None
        self._ctrl_indices: np.ndarray | None = None

    @abc.abstractmethod
    def build_swarm_spec(self) -> mujoco.MjsBody:
        raise NotImplementedError()

    @abc.abstractmethod
    def reset_swarm(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
        raise NotImplementedError()

    @abc.abstractmethod
    def get_unit_prefixes(self) -> set[str]:
        raise NotImplementedError()

    def get_obs(self, model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
        if self._qpos_indices is None or self._qvel_indices is None:
            self._compute_indices(model)

        qpos = data.qpos[self._qpos_indices]
        qvel = data.qvel[self._qvel_indices]
        return np.concatenate([qpos, qvel], axis=1)

    def get_n_obs_features_per_agent(self, model: mujoco.MjModel) -> int:
        if self._qpos_indices is None or self._qvel_indices is None:
            self._compute_indices(model)
        return self._qpos_indices.shape[1] + self._qvel_indices.shape[1]

    def get_n_actions_per_agent(self, model: mujoco.MjModel) -> int:
        if self._ctrl_indices is None:
            self._compute_indices(model)
        return self._ctrl_indices.shape[1]

    def apply_action(
            self,
            model: mujoco.MjModel,
            data: mujoco.MjData,
            action: np.ndarray,
            action_scale: float = 1.0
    ) -> None:
        if self._ctrl_indices is None:
            self._compute_indices(model)
        data.ctrl[self._ctrl_indices] = action * action_scale

    def _compute_indices(self, model: mujoco.MjModel):
        prefixes = self.get_unit_prefixes()
        self._qpos_indices = np.array(
            [muju.qpos_indices_for_prefix(model, prefix) for prefix in prefixes],
            dtype=int,
        )
        self._qvel_indices = np.array(
            [muju.dof_indices_for_prefix(model, prefix) for prefix in prefixes],
            dtype=int,
        )
        self._ctrl_indices = np.array(
            [muju.ctrl_indices_for_prefix(model, prefix) for prefix in prefixes],
            dtype=int,
        )
