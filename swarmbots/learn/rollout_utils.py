from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.hybrid_action_space import VectorHybridActionSpace
from swarmbots.learn.tensor_conversion import to_torch_tensor


_FINAL_OBS_REQUIRED_MESSAGE = (
    "SAME_STEP rollouts require infos['final_obs'] and infos['_final_obs'] for done environments."
)


@dataclass(frozen=True, slots=True)
class RolloutStepMetadata:
    completed_indices: list[int]
    has_truncations: bool
    has_ongoing: bool
    final_obs_mask: torch.Tensor | None


def _episode_info_source(infos: dict[str, Any]) -> dict[str, Any] | None:
    if "episode" in infos:
        source = infos
    else:
        source = infos.get("final_info")
        if not isinstance(source, dict) or "episode" not in source:
            return None
    if not isinstance(source["episode"], dict):
        raise ValueError(f"Expected infos['episode'] to be a dict, got {type(source['episode'])}")
    return source


def _info_mask(infos: dict[str, Any], key: str, dones: torch.Tensor) -> torch.Tensor:
    mask = to_torch_tensor(infos.get(key, dones), device=dones.device, dtype=torch.bool).reshape(-1)
    if tuple(mask.shape) != tuple(dones.shape):
        raise ValueError(f"Expected infos['{key}'] shape {tuple(dones.shape)}, got {tuple(mask.shape)}")
    return mask


def collect_rollout_step_metadata(
    *, infos: dict[str, Any], dones: torch.Tensor, terminations: torch.Tensor, truncations: torch.Tensor,
) -> RolloutStepMetadata:
    source = _episode_info_source(infos)
    episode_mask = dones if source is None else _info_mask(source, "_episode", dones)
    final_obs_mask = (
        _info_mask(infos, "_final_obs", dones)
        if "final_obs" in infos and "_final_obs" in infos else None
    )
    if source is None and final_obs_mask is None:
        has_dones, has_truncations = torch.stack((dones, truncations & ~terminations)).any(dim=1).tolist()
        if has_dones:
            raise ValueError(_FINAL_OBS_REQUIRED_MESSAGE)
        return RolloutStepMetadata([], has_truncations, dones.numel() > 0, None)
    # Bootstrap decisions, episode indices and mask validation share one device transfer.
    dones_cpu, truncations_cpu, episode_mask_cpu, final_obs_mask_cpu = torch.stack((
        dones, truncations & ~terminations, episode_mask, dones if final_obs_mask is None else final_obs_mask,
    )).detach().cpu()
    if not torch.equal(episode_mask_cpu, dones_cpu):
        raise ValueError("Expected infos['_episode'] to match computed dones.")
    completed_indices = torch.nonzero(dones_cpu, as_tuple=False).flatten().tolist()
    if completed_indices:
        if final_obs_mask is None:
            raise ValueError(_FINAL_OBS_REQUIRED_MESSAGE)
        if not torch.equal(final_obs_mask_cpu, dones_cpu):
            raise ValueError("Expected infos['_final_obs'] to match computed dones.")
    return RolloutStepMetadata(
        completed_indices=completed_indices,
        has_truncations=bool(truncations_cpu.any()),
        has_ongoing=len(completed_indices) < dones.numel(),
        final_obs_mask=final_obs_mask,
    )


def append_episode_infos(
        *,
        episode_infos: list[dict[str, Any]],
        infos: dict[str, Any],
        dones: torch.Tensor,
        step_metadata: RolloutStepMetadata | None = None,
) -> None:
    info_source = _episode_info_source(infos)
    if info_source is None:
        return
    episode_stats = info_source["episode"]
    if step_metadata is None:
        episode_mask = _info_mask(info_source, "_episode", dones)
        episode_mask_cpu, dones_cpu = torch.stack((episode_mask, dones)).detach().cpu()
        if not torch.equal(episode_mask_cpu, dones_cpu):
            raise ValueError("Expected infos['_episode'] to match computed dones.")
        completed_indices = torch.nonzero(episode_mask_cpu, as_tuple=False).flatten().tolist()
    else:
        completed_indices = step_metadata.completed_indices
    if not completed_indices:
        return
    statistic_names = tuple(key for key in episode_stats if not key.startswith("_"))
    columns = {key: episode_stats[key] for key in statistic_names}
    for key in statistic_names:
        mask_key = f"_{key}"
        if episode_stats.get(mask_key) is not None:
            columns[mask_key] = episode_stats[mask_key]
    scenario_columns = {
        key: info_source.get(key, infos.get(key))
        for key in ("scenario_id", "scenario_name")
        if info_source.get(key, infos.get(key)) is not None
    }
    columns.update(scenario_columns)
    completed_columns = _episode_columns_to_cpu(columns, completed_indices)

    for row_idx in range(len(completed_indices)):
        episode_info = {}
        for key in statistic_names:
            value_mask = completed_columns.get(f"_{key}")
            if value_mask is not None and not bool(to_python_episode_stat(value_mask[row_idx])):
                continue
            episode_info[key] = to_python_episode_stat(completed_columns[key][row_idx])
        for key in scenario_columns:
            episode_info[key] = to_python_episode_stat(completed_columns[key][row_idx])
        episode_infos.append(episode_info)


def _episode_columns_to_cpu(columns: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    completed_columns: dict[str, Any] = {}
    tensor_groups: dict[tuple[torch.device, torch.dtype], list[tuple[str, torch.Tensor]]] = {}
    device_indices: dict[torch.device, torch.Tensor] = {}
    for key, values in columns.items():
        if isinstance(values, torch.Tensor) and values.device.type != "cpu":
            if values.device not in device_indices:
                device_indices[values.device] = torch.tensor(indices, device=values.device)
            selected = values.detach().index_select(0, device_indices[values.device])
            tensor_groups.setdefault((values.device, values.dtype), []).append((key, selected))
        else:
            completed_columns[key] = [values[index] for index in indices]

    # Group by dtype so integer counts and Boolean presence masks retain their
    # types without one device transfer per completed episode and field.
    for fields in tensor_groups.values():
        packed = torch.cat([values.reshape(-1) for _, values in fields]).cpu()
        offset = 0
        for key, values in fields:
            completed = packed[offset:offset + values.numel()].reshape(values.shape)
            completed_columns[key] = (
                completed.flatten().tolist() if values.numel() == len(indices) else completed.numpy()
            )
            offset += values.numel()
    return completed_columns


def extract_bootstrap_obs(
        *,
        env: BaseLearnEnvWrapper,
        next_obs: dict[str, torch.Tensor],
        infos: dict[str, Any],
        dones: torch.Tensor,
        has_dones: bool | None = None,
        step_metadata: RolloutStepMetadata | None = None,
) -> dict[str, torch.Tensor]:
    if step_metadata is not None:
        has_dones = bool(step_metadata.completed_indices)
    if not (bool(dones.any()) if has_dones is None else has_dones):
        return next_obs

    final_obs_mask = (
        _validated_final_obs_mask(infos=infos, dones=dones)
        if step_metadata is None else step_metadata.final_obs_mask
    )
    assert final_obs_mask is not None
    expected_obs_keys = set(next_obs)
    final_obs_value = infos["final_obs"]
    if isinstance(final_obs_value, dict):
        final_obs = _final_obs_to_torch(env=env, final_obs=final_obs_value)
        _validate_final_obs(final_obs=final_obs, expected_obs=next_obs, batched=True)
        return {
            key: torch.where(
                final_obs_mask.reshape((-1,) + (1,) * (value.ndim - 1)), final_obs[key], value,
            )
            for key, value in next_obs.items()
        }

    bootstrap_obs = {key: value.clone() for key, value in next_obs.items()}
    final_obs_entries = np.asarray(final_obs_value, dtype=object).reshape(-1)
    completed_indices = (
        torch.nonzero(final_obs_mask, as_tuple=False).flatten().tolist()
        if step_metadata is None else step_metadata.completed_indices
    )
    for env_idx in completed_indices:
        final_obs = _final_obs_to_torch(env=env, final_obs=final_obs_entries[env_idx])
        _validate_final_obs(final_obs=final_obs, expected_obs=next_obs)
        for key in expected_obs_keys:
            bootstrap_obs[key][env_idx] = final_obs[key]

    return bootstrap_obs


def extract_terminal_obs(
        *,
        env: BaseLearnEnvWrapper,
        next_obs: dict[str, torch.Tensor],
        infos: dict[str, Any],
        dones: torch.Tensor,
) -> dict[str, torch.Tensor] | None:
    if not torch.any(dones):
        return None

    final_obs_mask = _validated_final_obs_mask(infos=infos, dones=dones)
    expected_obs_keys = set(next_obs)
    final_obs_value = infos["final_obs"]
    if isinstance(final_obs_value, dict):
        final_obs = _final_obs_to_torch(env=env, final_obs=final_obs_value)
        _validate_final_obs(final_obs=final_obs, expected_obs=next_obs, batched=True)
        return {key: final_obs[key][final_obs_mask] for key in expected_obs_keys}

    final_obs_entries = np.asarray(final_obs_value, dtype=object).reshape(-1)
    terminal_obs: dict[str, list[torch.Tensor]] = {key: [] for key in expected_obs_keys}

    for env_idx in torch.nonzero(final_obs_mask, as_tuple=False).flatten().tolist():
        final_obs = _final_obs_to_torch(env=env, final_obs=final_obs_entries[env_idx])
        _validate_final_obs(final_obs=final_obs, expected_obs=next_obs)
        for key in expected_obs_keys:
            terminal_obs[key].append(final_obs[key])

    return {key: torch.stack(values, dim=0) for key, values in terminal_obs.items()}


def _validated_final_obs_mask(
        *,
        infos: dict[str, Any],
        dones: torch.Tensor,
) -> torch.Tensor:
    if "final_obs" not in infos or "_final_obs" not in infos:
        raise ValueError(_FINAL_OBS_REQUIRED_MESSAGE)

    final_obs_mask = _info_mask(infos, "_final_obs", dones)
    if not torch.equal(final_obs_mask, dones):
        raise ValueError("Expected infos['_final_obs'] to match computed dones.")
    return final_obs_mask


def _final_obs_to_torch(
        *,
        env: BaseLearnEnvWrapper,
        final_obs: Any,
) -> dict[str, torch.Tensor]:
    try:
        return env._obs_to_torch(final_obs)
    except KeyError as exc:
        missing_key = exc.args[0]
        raise ValueError(f"Expected final_obs to contain observation key {missing_key!r}.") from exc


def _validate_final_obs(
        *,
        final_obs: dict[str, torch.Tensor],
        expected_obs: dict[str, torch.Tensor],
        batched: bool = False,
) -> None:
    missing_keys = expected_obs.keys() - final_obs.keys()
    if missing_keys:
        raise ValueError(f"Expected final_obs to contain observation keys {sorted(missing_keys)}.")
    if batched:
        for key, value in expected_obs.items():
            if final_obs[key].shape != value.shape:
                raise ValueError(
                    f"Expected final_obs[{key!r}] shape {tuple(value.shape)}, got {tuple(final_obs[key].shape)}."
                )


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
        obs: dict[str, torch.Tensor],
        n_agent_actions: int,
        policy: BasePolicy | None = None,
) -> torch.Tensor | None:
    if policy is not None and not policy.requires_previous_actions():
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
