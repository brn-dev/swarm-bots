from typing import Any

import numpy as np
import torch

from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.tensor_conversion import to_torch_tensor


def append_episode_infos(
        *,
        episode_infos: list[dict[str, Any]],
        infos: dict[str, Any],
        dones: torch.Tensor,
) -> None:
    info_source = infos
    if "episode" not in info_source:
        final_info = infos.get("final_info", None)
        if isinstance(final_info, dict) and "episode" in final_info:
            info_source = final_info
        else:
            return

    episode_stats = info_source["episode"]
    if not isinstance(episode_stats, dict):
        raise ValueError(f"Expected infos['episode'] to be a dict, got {type(episode_stats)}")

    episode_mask = to_torch_tensor(
        info_source.get("_episode", dones),
        device=dones.device,
        dtype=torch.bool,
    ).reshape(-1)
    if tuple(episode_mask.shape) != tuple(dones.shape):
        raise ValueError(f"Expected infos['_episode'] shape {tuple(dones.shape)}, got {tuple(episode_mask.shape)}")
    if not torch.equal(episode_mask, dones):
        raise ValueError("Expected infos['_episode'] to match computed dones.")

    for env_idx in torch.nonzero(episode_mask, as_tuple=False).flatten().tolist():
        episode_infos.append(
            {
                key: to_python_episode_stat(values[env_idx])
                for key, values in episode_stats.items()
                if not key.startswith("_")
            }
        )


def extract_bootstrap_obs(
        *,
        env: BaseLearnEnvWrapper,
        next_obs: dict[str, torch.Tensor],
        infos: dict[str, Any],
        dones: torch.Tensor,
) -> dict[str, torch.Tensor]:
    if not torch.any(dones):
        return next_obs

    if "final_obs" not in infos or "_final_obs" not in infos:
        raise ValueError("SAME_STEP rollouts require infos['final_obs'] and infos['_final_obs'] for done environments.")

    final_obs_mask = to_torch_tensor(infos["_final_obs"], device=dones.device, dtype=torch.bool).reshape(-1)
    if tuple(final_obs_mask.shape) != tuple(dones.shape):
        raise ValueError(f"Expected infos['_final_obs'] shape {tuple(dones.shape)}, got {tuple(final_obs_mask.shape)}")
    if not torch.equal(final_obs_mask, dones):
        raise ValueError("Expected infos['_final_obs'] to match computed dones.")

    bootstrap_obs = {key: value.clone() for key, value in next_obs.items()}
    final_obs_value = infos["final_obs"]
    if isinstance(final_obs_value, dict):
        final_obs = env._obs_to_torch(final_obs_value)
        for key, value in final_obs.items():
            bootstrap_obs[key][final_obs_mask] = value[final_obs_mask]
        if "agent_mask" in bootstrap_obs and "agent_mask" not in final_obs:
            raise ValueError("Expected final_obs to contain 'agent_mask' when the observation space includes it.")
        return bootstrap_obs

    final_obs_entries = np.asarray(final_obs_value, dtype=object).reshape(-1)

    for env_idx in torch.nonzero(final_obs_mask, as_tuple=False).flatten().tolist():
        final_obs = env._obs_to_torch(final_obs_entries[env_idx])
        for key, value in final_obs.items():
            bootstrap_obs[key][env_idx] = value
        if "agent_mask" in bootstrap_obs and "agent_mask" not in final_obs:
            raise ValueError("Expected final_obs to contain 'agent_mask' when the observation space includes it.")

    return bootstrap_obs


def to_python_episode_stat(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value


def snapshot_obs(obs: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in obs.items()}


def initial_previous_actions(
        *,
        policy: BasePolicy,
        obs: dict[str, torch.Tensor],
        n_agent_actions: int,
) -> torch.Tensor | None:
    if not policy.requires_previous_actions():
        return None

    local_obs = obs["local_obs"]
    n_envs, n_agents = local_obs.shape[:2]
    return torch.zeros(
        (n_envs, n_agents, n_agent_actions),
        dtype=local_obs.dtype,
        device=local_obs.device,
    )


def sample_random_actions(
        action_space: VectorHybridActionSpace,
        *,
        device: torch.device,
        dtype: torch.dtype,
) -> torch.Tensor:
    action_dict = action_space.sample()
    actions = action_space.concat_actions(action_dict)
    return to_torch_tensor(actions, device=device, dtype=dtype)
