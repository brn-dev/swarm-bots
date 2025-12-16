from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Literal, Sequence

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import VectorEnv
from gymnasium.vector.vector_env import AutoresetMode

from swarmbots.learn.hybrid_action_space import HybridActionSpace
from swarmbots.scenarios.base_scenario import SwarmActDict, SwarmObsDict
from swarmbots.swarm_bots_env import SwarmBotsEnv


def _to_torch(x: Any, *, device: torch.device | str, dtype: torch.dtype | None = None) -> torch.Tensor:
    """Best-effort conversion to torch without copies when possible."""
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype)
    t = torch.as_tensor(x, device=device, dtype=dtype)
    return t


class TorchHybridActionSpace(HybridActionSpace):
    """A HybridActionSpace that samples batched (n_envs, ...) torch tensors on a device."""

    def __init__(
        self,
        *,
        single_action_space: spaces.Dict,
        n_envs: int,
        device: torch.device | str,
        dtype: torch.dtype = torch.float32,
    ):
        # Preserve key order explicitly (HybridActionSpace sorts plain mappings).
        ordered = OrderedDict(
            [
                ("actuators", single_action_space["actuators"]),
                ("connectors", single_action_space["connectors"]),
            ]
        )
        super().__init__(spaces=ordered)
        self.n_envs = int(n_envs)
        self.device = device
        self.dtype = dtype

    def sample(self, mask: dict[str, Any] | None = None, probability: dict[str, Any] | None = None) -> dict[str, Any]:
        if mask is not None or probability is not None:
            raise NotImplementedError("mask/probability sampling is not implemented")

        out: dict[str, torch.Tensor] = {}
        for k, space in self.space_map.items():
            # space.sample() returns shape (n_agents, n_actions_per_agent)
            samples = [space.sample() for _ in range(self.n_envs)]
            arr = np.stack(samples, axis=0)
            out[k] = _to_torch(arr, device=self.device, dtype=self.dtype)
        return out

    def concat_actions(self, actions_dict: dict[str, Any]) -> torch.Tensor:
        # Accept either numpy or torch; always return torch on self.device.
        parts = [_to_torch(actions_dict[k], device=self.device, dtype=self.dtype) for k in self.key_order]
        return torch.cat(parts, dim=-1)


@dataclass(frozen=True)
class ActionSplitSpec:
    n_agents: int
    actuators_dim: int
    connectors_dim: int

    @property
    def total_dim(self) -> int:
        return self.actuators_dim + self.connectors_dim


class SwarmBotsLearnWrapper(VectorEnv):
    """
    A minimal VectorEnv wrapper for SwarmBotsEnv that is compatible with swarmbots/learn:

    - Returns torch tensors (on `device`) for observations / rewards / terminations / truncations.
    - Accepts actions either as:
      - a dict with keys {'actuators', 'connectors'} of shape (n_envs, n_agents, dim)
      - or a flat tensor/array of shape (n_envs, n_agents, total_dim) which is split into the dict form.
    - Enforces autoreset mode NEXT_STEP only.
    """

    metadata: dict[str, Any] = {"autoreset_mode": AutoresetMode.NEXT_STEP.value}

    def __init__(
        self,
        env_fns: Sequence[Callable[[], SwarmBotsEnv]] | None = None,
        envs: Sequence[SwarmBotsEnv] | None = None,
        *,
        device: torch.device | str = "cpu",
        torch_dtype: torch.dtype = torch.float32,
        autoreset_mode: AutoresetMode | Literal["NextStep"] = AutoresetMode.NEXT_STEP,
    ):
        if (env_fns is None) == (envs is None):
            raise ValueError("Provide exactly one of env_fns or envs")

        if autoreset_mode == "NextStep":
            autoreset_mode = AutoresetMode.NEXT_STEP
        if autoreset_mode is not AutoresetMode.NEXT_STEP:
            raise ValueError("Only AutoresetMode.NEXT_STEP is supported (buffer accumulator assumes next-step autoreset).")

        self.device = device
        self.torch_dtype = torch_dtype

        self.envs: list[SwarmBotsEnv] = [fn() for fn in env_fns] if envs is None else list(envs)
        if len(self.envs) == 0:
            raise ValueError("Need at least one sub-environment")

        self.num_envs = len(self.envs)

        # Single spaces (assumed identical across envs)
        self.single_observation_space: spaces.Dict = self.envs[0].observation_space
        self.single_action_space: spaces.Dict = self.envs[0].action_space

        # NOTE: `swarmbots/learn` code (buffer) expects *single* shapes for obs/action spaces.
        # We still expose the canonical VectorEnv attributes, but keep them equal to the single spaces.
        self.observation_space = self.single_observation_space

        self._action_split = self._infer_action_split_spec(self.single_action_space)
        self.action_space: TorchHybridActionSpace = TorchHybridActionSpace(
            single_action_space=self.single_action_space,
            n_envs=self.num_envs,
            device=self.device,
            dtype=self.torch_dtype,
        )

        # Next-step autoreset bookkeeping
        self._needs_reset = np.zeros((self.num_envs,), dtype=bool)

    @staticmethod
    def _infer_action_split_spec(single_action_space: spaces.Dict) -> ActionSplitSpec:
        act_space = single_action_space["actuators"]
        conn_space = single_action_space["connectors"]

        if not (hasattr(act_space, "shape") and hasattr(conn_space, "shape")):
            raise ValueError("Expected shaped subspaces for actuators/connectors")

        act_shape = act_space.shape
        conn_shape = conn_space.shape
        if len(act_shape) != 2 or len(conn_shape) != 2:
            raise ValueError(f"Expected action shapes (n_agents, dim), got {act_shape=} and {conn_shape=}")
        if act_shape[0] != conn_shape[0]:
            raise ValueError(f"Mismatched n_agents between actuators/connectors: {act_shape[0]} vs {conn_shape[0]}")

        return ActionSplitSpec(
            n_agents=int(act_shape[0]),
            actuators_dim=int(act_shape[1]),
            connectors_dim=int(conn_shape[1]),
        )

    def _split_flat_actions(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        if actions.ndim != 3:
            raise ValueError(f"Expected flat actions with shape (n_envs, n_agents, total_dim), got {tuple(actions.shape)}")
        if actions.shape[0] != self.num_envs:
            raise ValueError(f"Expected n_envs={self.num_envs}, got {actions.shape[0]}")
        if actions.shape[1] != self._action_split.n_agents:
            raise ValueError(f"Expected n_agents={self._action_split.n_agents}, got {actions.shape[1]}")
        if actions.shape[2] != self._action_split.total_dim:
            raise ValueError(f"Expected total_dim={self._action_split.total_dim}, got {actions.shape[2]}")

        a = actions[..., : self._action_split.actuators_dim]
        c = actions[..., self._action_split.actuators_dim :]
        return {"actuators": a, "connectors": c}

    def _coerce_actions(self, actions: Any) -> dict[str, torch.Tensor]:
        if isinstance(actions, dict):
            if "actuators" not in actions or "connectors" not in actions:
                raise ValueError("Dict actions must contain keys {'actuators', 'connectors'}")
            return {
                "actuators": _to_torch(actions["actuators"], device=self.device, dtype=self.torch_dtype),
                "connectors": _to_torch(actions["connectors"], device=self.device, dtype=self.torch_dtype),
            }

        flat = _to_torch(actions, device=self.device, dtype=self.torch_dtype)
        return self._split_flat_actions(flat)

    def _torch_obs_from_np(self, obs_list: list[SwarmObsDict]) -> dict[str, torch.Tensor]:
        local = np.stack([o["local_obs"] for o in obs_list], axis=0)
        glob = np.stack([o["global_obs"] for o in obs_list], axis=0)
        return {
            "local_obs": _to_torch(local, device=self.device, dtype=self.torch_dtype),
            "global_obs": _to_torch(glob, device=self.device, dtype=self.torch_dtype),
        }

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        obs_list: list[SwarmObsDict] = []
        infos_list: list[dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            s = None if seed is None else int(seed) + i
            obs, info = env.reset(seed=s, options=options)
            obs_list.append(obs)
            infos_list.append(info)

        self._needs_reset[:] = False
        obs_t = self._torch_obs_from_np(obs_list)
        infos = self._vectorize_infos(infos_list)
        return obs_t, infos

    def step(
        self, actions: Any
    ) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        actions_t = self._coerce_actions(actions)

        # Connectors: treat >0 as True when feeding mujoco env
        actuators = actions_t["actuators"]
        connectors = actions_t["connectors"]

        if actuators.shape[:2] != (self.num_envs, self._action_split.n_agents):
            raise ValueError(f"Bad actuators shape {tuple(actuators.shape)}, expected (n_envs, n_agents, dim)")
        if connectors.shape[:2] != (self.num_envs, self._action_split.n_agents):
            raise ValueError(f"Bad connectors shape {tuple(connectors.shape)}, expected (n_envs, n_agents, dim)")

        # Run each env sequentially (sync vector env)
        obs_list: list[SwarmObsDict] = []
        rewards = np.zeros((self.num_envs,), dtype=np.float32)
        terms = np.zeros((self.num_envs,), dtype=bool)
        truncs = np.zeros((self.num_envs,), dtype=bool)
        infos_list: list[dict[str, Any]] = []

        for i, env in enumerate(self.envs):
            if self._needs_reset[i]:
                env.reset()
                self._needs_reset[i] = False

            act_i = np.asarray(actuators[i].detach().cpu().numpy(), dtype=np.float32)
            conn_i = connectors[i].detach().cpu().numpy()
            conn_i = np.asarray(conn_i > 0, dtype=bool)

            action_dict: SwarmActDict = {"actuators": act_i, "connectors": conn_i}

            obs, r, terminated, truncated, info = env.step(action_dict)
            obs_list.append(obs)
            rewards[i] = r
            terms[i] = bool(terminated)
            truncs[i] = bool(truncated)
            infos_list.append(info if isinstance(info, dict) else {"info": info})

            if terms[i] or truncs[i]:
                self._needs_reset[i] = True

        obs_t = self._torch_obs_from_np(obs_list)
        rewards_t = _to_torch(rewards, device=self.device, dtype=self.torch_dtype)
        terms_t = _to_torch(terms, device=self.device, dtype=torch.bool)
        truncs_t = _to_torch(truncs, device=self.device, dtype=torch.bool)
        infos = self._vectorize_infos(infos_list)
        return obs_t, rewards_t, terms_t, truncs_t, infos

    @staticmethod
    def _vectorize_infos(infos_list: list[dict[str, Any]]) -> dict[str, Any]:
        # Gymnasium v0.25+ expects dict-of-batched, but swarmbots/learn doesn't rely on info.
        # We'll store per-env info values in lists under each key.
        if len(infos_list) == 0:
            return {}
        keys = set()
        for info in infos_list:
            keys.update(info.keys())
        return {k: [info.get(k, None) for info in infos_list] for k in sorted(keys)}

    def close(self):
        for env in self.envs:
            env.close()

