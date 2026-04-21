from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment, PPORolloutBuffer
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEResetMode, GSDEIntervalResetMode, GSDEProbabilityResetMode
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.tensor_conversion import to_torch_tensor


@dataclass(slots=True)
class PPORolloutState:
    obs: dict[str, torch.Tensor]
    episode_start_mask: torch.Tensor
    previous_actions: torch.Tensor | None
    rollout_step_idx: int


@dataclass(slots=True)
class _RolloutTimers:
    reset_noise_timings: list[float]
    policy_forward_timings: list[float]
    env_step_timings: list[float]
    buffer_add_timings: list[float]
    reset_noise_timer: PerformanceTimer
    policy_forward_timer: PerformanceTimer
    env_step_timer: PerformanceTimer
    buffer_add_timer: PerformanceTimer


def _parse_gsde_reset_mode(
        policy: BasePPOPolicy[Any, Any],
        gsde_reset_mode: GSDEResetMode | None,
) -> tuple[bool, bool, int, float]:
    assert not policy.gsde_enabled or gsde_reset_mode is not None
    gsde_enabled = policy.gsde_enabled
    is_gsde_interval_reset_mode = False
    gsde_reset_interval = -1
    gsde_reset_prob = -1.0
    if gsde_enabled:
        if isinstance(gsde_reset_mode, GSDEIntervalResetMode):
            is_gsde_interval_reset_mode = True
            gsde_reset_interval = gsde_reset_mode.interval
            if gsde_reset_interval <= 0:
                raise ValueError(f"GSDEIntervalResetMode.interval must be > 0, got {gsde_reset_interval}")
        elif isinstance(gsde_reset_mode, GSDEProbabilityResetMode):
            gsde_reset_prob = gsde_reset_mode.probability
            if not (0.0 < gsde_reset_prob < 1.0):
                raise ValueError(f"GSDEProbabilityResetMode.probability must be in (0, 1), got {gsde_reset_prob}")
        else:
            raise TypeError(f"Unknown gsde_reset_mode type: {type(gsde_reset_mode)}")
    return gsde_enabled, is_gsde_interval_reset_mode, gsde_reset_interval, gsde_reset_prob


def _init_rollout_timers() -> _RolloutTimers:
    return _RolloutTimers(
        reset_noise_timings=[],
        policy_forward_timings=[],
        env_step_timings=[],
        buffer_add_timings=[],
        reset_noise_timer=PerformanceTimer(),
        policy_forward_timer=PerformanceTimer(),
        env_step_timer=PerformanceTimer(),
        buffer_add_timer=PerformanceTimer(),
    )


def _build_gsde_step_reset_mask(
        *,
        gsde_enabled: bool,
        is_gsde_interval_reset_mode: bool,
        gsde_reset_interval: int,
        gsde_reset_prob: float,
        rollout_step_idx: int,
        batch_shape: tuple[int, ...],
        rollout_device: torch.device,
) -> torch.Tensor | None:
    if not gsde_enabled:
        return None
    if is_gsde_interval_reset_mode:
        if (rollout_step_idx % gsde_reset_interval) != 0:
            return None
        return torch.ones(batch_shape, device=rollout_device, dtype=torch.bool)
    return torch.empty(
        batch_shape,
        device=rollout_device,
        dtype=torch.bool,
    ).bernoulli_(gsde_reset_prob)


def _reset_temporal_correlations(
        *,
        policy: BasePPOPolicy[Any, Any],
        episode_start_mask: torch.Tensor | None = None,
        step_reset_mask: torch.Tensor | None = None,
        batch_shape: tuple[int, ...] | None = None,
) -> None:
    if episode_start_mask is not None:
        policy.action_dist.reset_temporal_correlations_on_ep_start(episode_start_mask)
    if step_reset_mask is not None or batch_shape is not None:
        policy.action_dist.reset_temporal_correlations_on_step(mask=step_reset_mask, batch_shape=batch_shape)


def _append_episode_infos(
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
                key: _to_python_episode_stat(values[env_idx])
                for key, values in episode_stats.items()
                if not key.startswith("_")
            }
        )


def _extract_bootstrap_obs(
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


def _to_python_episode_stat(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().numpy()
    if isinstance(value, np.ndarray) and value.size == 1:
        return value.item()
    return value


def _evaluate_values(
        *,
        policy: BasePPOPolicy[Any, Any],
        obs: dict[str, torch.Tensor],
        previous_actions: torch.Tensor | None,
        terminated_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    temporal_state_snapshot = policy.get_temporal_state_snapshot()
    try:
        values = policy.predict_values(
            local_obs=obs["local_obs"],
            global_obs=obs["global_obs"],
            hidden_local_vars=obs["hidden_local_vars"],
            hidden_global_vars=obs["hidden_global_vars"],
            agent_mask=obs.get("agent_mask", None),
            previous_actions=previous_actions,
        )
    finally:
        policy.restore_temporal_state_snapshot(temporal_state_snapshot)

    if terminated_mask is None:
        return values
    return values.masked_fill(terminated_mask, 0.0)


def _collect_rollout_step(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        buffer: PPORolloutBuffer,
        obs: dict[str, torch.Tensor],
        episode_start_mask: torch.Tensor,
        previous_actions: torch.Tensor | None,
        rollout_step_idx: int,
        timers: _RolloutTimers,
        episode_infos: list[dict[str, Any]],
        gsde_enabled: bool,
        is_gsde_interval_reset_mode: bool,
        gsde_reset_interval: int,
        gsde_reset_prob: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None, int]:
    local_obs = obs["local_obs"]
    batch_shape = tuple(local_obs.shape[:-1])
    step_reset_mask = _build_gsde_step_reset_mask(
        gsde_enabled=gsde_enabled,
        is_gsde_interval_reset_mode=is_gsde_interval_reset_mode,
        gsde_reset_interval=gsde_reset_interval,
        gsde_reset_prob=gsde_reset_prob,
        rollout_step_idx=rollout_step_idx,
        batch_shape=batch_shape,
        rollout_device=buffer.rollout_device,
    )

    with timers.reset_noise_timer:
        policy.reset_temporal_state(episode_start_mask=episode_start_mask)
        _reset_temporal_correlations(
            policy=policy,
            episode_start_mask=episode_start_mask,
            step_reset_mask=step_reset_mask,
            batch_shape=batch_shape if (gsde_enabled and is_gsde_interval_reset_mode and step_reset_mask is None) else None,
        )
    timers.reset_noise_timings.append(timers.reset_noise_timer.get_duration())

    with timers.policy_forward_timer:
        actions, log_probs, values = policy(
            obs["local_obs"],
            obs["global_obs"],
            hidden_local_vars=obs["hidden_local_vars"],
            hidden_global_vars=obs["hidden_global_vars"],
            agent_mask=obs.get("agent_mask", None),
            previous_actions=previous_actions,
        )
    timers.policy_forward_timings.append(timers.policy_forward_timer.get_duration())

    with timers.env_step_timer:
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
    timers.env_step_timings.append(timers.env_step_timer.get_duration())

    dones = torch.logical_or(terminations, truncations)
    _append_episode_infos(episode_infos=episode_infos, infos=infos, dones=dones)

    bootstrap_obs = _extract_bootstrap_obs(
        env=env,
        next_obs=next_obs,
        infos=infos,
        dones=dones,
    )
    bootstrap_previous_actions = None if previous_actions is None else actions.detach()
    bootstrap_values = _evaluate_values(
        policy=policy,
        obs=bootstrap_obs,
        previous_actions=bootstrap_previous_actions,
        terminated_mask=terminations,
    )

    policy.reset_temporal_state(episode_start_mask=dones)
    _reset_temporal_correlations(policy=policy, episode_start_mask=dones)

    with timers.buffer_add_timer:
        buffer.add(
            local_obs=obs["local_obs"],
            global_obs=obs["global_obs"],
            hidden_local_vars=obs["hidden_local_vars"],
            hidden_global_vars=obs["hidden_global_vars"],
            agent_mask=obs.get("agent_mask", None),
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            previous_actions=previous_actions,
            episode_start_mask=episode_start_mask,
            bootstrap_obs=bootstrap_obs,
            bootstrap_values=bootstrap_values,
            dones=dones,
        )
    timers.buffer_add_timings.append(timers.buffer_add_timer.get_duration())

    next_previous_actions: torch.Tensor | None = None
    if previous_actions is not None:
        next_previous_actions = actions.detach().masked_fill(dones.unsqueeze(-1).unsqueeze(-1), 0.0)

    return next_obs, dones, next_previous_actions, rollout_step_idx + 1


def _build_rollout_metrics(
        *,
        env_reset_time: float,
        to_rollout_device_time: float,
        timers: _RolloutTimers,
        buffer_get_whole_episodes_time: float,
) -> dict[str, Any]:
    return {
        "env_reset_time": env_reset_time,
        "to_rollout_device_time": to_rollout_device_time,
        "reset_noise_time": compute_summary_statistics(timers.reset_noise_timings, find_min=True, find_max=True),
        "total_reset_noise_time": sum(timers.reset_noise_timings),
        "policy_forward_time": compute_summary_statistics(timers.policy_forward_timings, find_min=True, find_max=True),
        "total_policy_forward_time": sum(timers.policy_forward_timings),
        "env_step_time": compute_summary_statistics(timers.env_step_timings, find_min=True, find_max=True),
        "total_env_step_time": sum(timers.env_step_timings),
        "buffer_add_time": compute_summary_statistics(timers.buffer_add_timings),
        "total_buffer_add_time": sum(timers.buffer_add_timings),
        "buffer_get_whole_episodes_time": buffer_get_whole_episodes_time,
    }


@torch.no_grad()
def collect_whole_episodes(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        buffer: PPORolloutBuffer,
        n_episodes: int,
        gsde_reset_mode: GSDEResetMode | None = None,
) -> tuple[list[PPOEpisodeSegment], list[dict[str, Any]], dict[str, Any]]:
    gsde_enabled, is_gsde_interval_reset_mode, gsde_reset_interval, gsde_reset_prob = _parse_gsde_reset_mode(
        policy,
        gsde_reset_mode,
    )

    buffer.reset()
    with PerformanceTimer() as env_reset_timer:
        obs, _info = env.reset()
    episode_start_mask = torch.ones((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
    previous_actions: torch.Tensor | None = None
    if policy.requires_previous_actions():
        previous_actions = torch.zeros(
            (buffer.n_envs, buffer.n_agents, buffer.n_agent_actions),
            dtype=buffer.rollout_dtype,
            device=buffer.rollout_device,
        )

    with PerformanceTimer() as to_rollout_device_timer:
        policy.to(buffer.rollout_device)
        policy.eval()

    episode_infos: list[dict[str, Any]] = []
    rollout_step_idx = 0
    timers = _init_rollout_timers()

    while len(buffer.episodes) < n_episodes:
        obs, episode_start_mask, previous_actions, rollout_step_idx = _collect_rollout_step(
            env=env,
            policy=policy,
            buffer=buffer,
            obs=obs,
            episode_start_mask=episode_start_mask,
            previous_actions=previous_actions,
            rollout_step_idx=rollout_step_idx,
            timers=timers,
            episode_infos=episode_infos,
            gsde_enabled=gsde_enabled,
            is_gsde_interval_reset_mode=is_gsde_interval_reset_mode,
            gsde_reset_interval=gsde_reset_interval,
            gsde_reset_prob=gsde_reset_prob,
        )

    with PerformanceTimer() as buffer_get_whole_episodes_timer:
        episodes = buffer.get_whole_episodes()

    metrics = _build_rollout_metrics(
        env_reset_time=env_reset_timer.get_duration(),
        to_rollout_device_time=to_rollout_device_timer.get_duration(),
        timers=timers,
        buffer_get_whole_episodes_time=buffer_get_whole_episodes_timer.get_duration(),
    )
    return episodes, episode_infos, metrics


@torch.no_grad()
def collect_steps(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        buffer: PPORolloutBuffer,
        n_steps: int,
        rollout_state: PPORolloutState | None = None,
        gsde_reset_mode: GSDEResetMode | None = None,
) -> tuple[list[PPOEpisodeSegment], list[dict[str, Any]], dict[str, Any], PPORolloutState]:
    gsde_enabled, is_gsde_interval_reset_mode, gsde_reset_interval, gsde_reset_prob = _parse_gsde_reset_mode(
        policy,
        gsde_reset_mode,
    )

    buffer.reset()

    env_reset_time = 0.0
    if rollout_state is None:
        with PerformanceTimer() as env_reset_timer:
            obs, _info = env.reset()
        env_reset_time = env_reset_timer.get_duration()
        episode_start_mask = torch.ones((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
        previous_actions: torch.Tensor | None = None
        if policy.requires_previous_actions():
            previous_actions = torch.zeros(
                (buffer.n_envs, buffer.n_agents, buffer.n_agent_actions),
                dtype=buffer.rollout_dtype,
                device=buffer.rollout_device,
            )
        rollout_step_idx = 0
    else:
        obs = rollout_state.obs
        episode_start_mask = rollout_state.episode_start_mask
        previous_actions = rollout_state.previous_actions
        rollout_step_idx = rollout_state.rollout_step_idx

    with PerformanceTimer() as to_rollout_device_timer:
        policy.to(buffer.rollout_device)
        policy.eval()

    episode_infos: list[dict[str, Any]] = []
    timers = _init_rollout_timers()
    initial_episode_count = len(buffer.episodes)

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")

    transitions_collected = 0
    while transitions_collected < n_steps:
        transitions_collected += buffer.n_envs
        obs, episode_start_mask, previous_actions, rollout_step_idx = _collect_rollout_step(
            env=env,
            policy=policy,
            buffer=buffer,
            obs=obs,
            episode_start_mask=episode_start_mask,
            previous_actions=previous_actions,
            rollout_step_idx=rollout_step_idx,
            timers=timers,
            episode_infos=episode_infos,
            gsde_enabled=gsde_enabled,
            is_gsde_interval_reset_mode=is_gsde_interval_reset_mode,
            gsde_reset_interval=gsde_reset_interval,
            gsde_reset_prob=gsde_reset_prob,
        )

    final_values = _evaluate_values(
        policy=policy,
        obs=obs,
        previous_actions=previous_actions,
    )

    with PerformanceTimer() as buffer_get_whole_episodes_timer:
        completed_episodes = buffer.get_whole_episodes()[initial_episode_count:]
    buffer_get_whole_episodes_time = buffer_get_whole_episodes_timer.get_duration()
    with PerformanceTimer() as buffer_dump_partial_episodes_timer:
        partial_episodes = buffer.dump_partial_episodes(final_obs=obs, final_values=final_values)
    buffer_get_whole_episodes_time += buffer_dump_partial_episodes_timer.get_duration()

    episodes = completed_episodes + partial_episodes

    metrics = _build_rollout_metrics(
        env_reset_time=env_reset_time,
        to_rollout_device_time=to_rollout_device_timer.get_duration(),
        timers=timers,
        buffer_get_whole_episodes_time=buffer_get_whole_episodes_time,
    )
    new_state = PPORolloutState(
        obs=obs,
        episode_start_mask=episode_start_mask,
        previous_actions=previous_actions,
        rollout_step_idx=rollout_step_idx,
    )
    return episodes, episode_infos, metrics, new_state
