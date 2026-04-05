from dataclasses import dataclass
from typing import Any

import torch

from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisode, PPORolloutBuffer
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.gsde_reset import GSDEResetMode, GSDEIntervalResetMode, GSDEProbabilityResetMode
from swarmbots.learn.summary_statistics import compute_summary_statistics


@dataclass(slots=True)
class PPORolloutState:
    obs: dict[str, torch.Tensor]
    is_final: torch.Tensor
    was_terminated: torch.Tensor
    previous_actions: torch.Tensor | None
    rollout_step_idx: int
    pending_episode_infos: list[dict[str, Any] | None]


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


def _collect_rollout_step(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        buffer: PPORolloutBuffer,
        obs: dict[str, torch.Tensor],
        is_final: torch.Tensor,
        was_terminated: torch.Tensor,
        previous_actions: torch.Tensor | None,
        rollout_step_idx: int,
        timers: _RolloutTimers,
        episode_infos: list[dict[str, Any]],
        episode_info_buffers: list[dict[str, Any] | None],
        gsde_enabled: bool,
        is_gsde_interval_reset_mode: bool,
        gsde_reset_interval: int,
        gsde_reset_prob: float,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor, torch.Tensor | None, int]:
    local_obs = obs['local_obs']
    global_obs = obs['global_obs']
    hidden_local_vars = obs["hidden_local_vars"]
    hidden_global_vars = obs["hidden_global_vars"]
    agent_mask = obs.get("agent_mask", None)
    if previous_actions is not None:
        previous_actions = previous_actions.masked_fill(is_final.unsqueeze(-1).unsqueeze(-1), 0.0)

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
        _reset_temporal_correlations(
            policy=policy,
            episode_start_mask=is_final,
            step_reset_mask=step_reset_mask,
            batch_shape=batch_shape if (gsde_enabled and is_gsde_interval_reset_mode and step_reset_mask is None) else None,
        )
    timers.reset_noise_timings.append(timers.reset_noise_timer.get_duration())

    with timers.policy_forward_timer:
        actions, log_probs, values = policy(
            local_obs,
            global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
        )
    timers.policy_forward_timings.append(timers.policy_forward_timer.get_duration())
    policy.reset_temporal_state(episode_start_mask=is_final)
    _reset_temporal_correlations(policy=policy, episode_start_mask=is_final)

    values = values.masked_fill(was_terminated, 0.0)

    with timers.env_step_timer:
        new_obs, rewards, terminations, truncations, infos = env.step(actions)
    timers.env_step_timings.append(timers.env_step_timer.get_duration())
    dones = torch.logical_or(terminations, truncations)

    if "episode" in infos:
        episode_stats = infos["episode"]
        for i, has_ep_info in enumerate(infos["_episode"]):
            if has_ep_info:
                episode_info_buffers[i] = {
                    key: values[i]
                    for key, values in episode_stats.items()
                }

    with timers.buffer_add_timer:
        buffer.add(
            local_obs=local_obs,
            global_obs=global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            actions=actions,
            rewards=rewards,
            log_probs=log_probs,
            values=values,
            previous_actions=previous_actions,
            is_final=is_final,
        )
    timers.buffer_add_timings.append(timers.buffer_add_timer.get_duration())

    final_env_indices = torch.where(is_final)[0]
    for env_idx in final_env_indices.tolist():
        buffered_info = episode_info_buffers[env_idx]
        if buffered_info is not None:
            episode_infos.append(buffered_info)
            episode_info_buffers[env_idx] = None

    next_previous_actions: torch.Tensor | None = None
    if previous_actions is not None:
        next_previous_actions = actions.detach().masked_fill(is_final.unsqueeze(-1).unsqueeze(-1), 0.0)
    return new_obs, dones, terminations, next_previous_actions, rollout_step_idx + 1


def _build_rollout_metrics(
        *,
        env_reset_time: float,
        to_rollout_device_time: float,
        timers: _RolloutTimers,
        buffer_get_whole_episodes_time: float,
) -> dict[str, Any]:
    return {
        'env_reset_time': env_reset_time,
        'to_rollout_device_time': to_rollout_device_time,
        'reset_noise_time': compute_summary_statistics(timers.reset_noise_timings, find_min=True, find_max=True),
        'total_reset_noise_time': sum(timers.reset_noise_timings),
        'policy_forward_time': compute_summary_statistics(timers.policy_forward_timings, find_min=True, find_max=True),
        'total_policy_forward_time': sum(timers.policy_forward_timings),
        'env_step_time': compute_summary_statistics(timers.env_step_timings, find_min=True, find_max=True),
        'total_env_step_time': sum(timers.env_step_timings),
        'buffer_add_time': compute_summary_statistics(timers.buffer_add_timings),
        'total_buffer_add_time': sum(timers.buffer_add_timings),
        'buffer_get_whole_episodes_time': buffer_get_whole_episodes_time,
    }


@torch.no_grad()
def collect_whole_episodes(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        buffer: PPORolloutBuffer,
        n_episodes: int,
        gsde_reset_mode: GSDEResetMode | None = None,
) -> tuple[list[PPOEpisode], list[dict[str, Any]], dict[str, Any]]:
    gsde_enabled, is_gsde_interval_reset_mode, gsde_reset_interval, gsde_reset_prob = _parse_gsde_reset_mode(
        policy,
        gsde_reset_mode,
    )

    buffer.reset()
    with PerformanceTimer() as env_reset_timer:
        obs, info = env.reset()
    is_final = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
    was_terminated = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
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
        initial_mask = torch.ones((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
        policy.reset_temporal_state(episode_start_mask=initial_mask)
        _reset_temporal_correlations(policy=policy, episode_start_mask=initial_mask)

    episode_infos: list[dict[str, Any]] = []
    episode_info_buffers: list[dict[str, Any] | None] = [None for _ in range(buffer.n_envs)]
    rollout_step_idx = 0

    timers = _init_rollout_timers()

    while len(buffer.episodes) < n_episodes:
        obs, is_final, was_terminated, previous_actions, rollout_step_idx = _collect_rollout_step(
            env=env,
            policy=policy,
            buffer=buffer,
            obs=obs,
            is_final=is_final,
            was_terminated=was_terminated,
            previous_actions=previous_actions,
            rollout_step_idx=rollout_step_idx,
            timers=timers,
            episode_infos=episode_infos,
            episode_info_buffers=episode_info_buffers,
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
) -> tuple[list[PPOEpisode], list[dict[str, Any]], dict[str, Any], PPORolloutState]:
    gsde_enabled, is_gsde_interval_reset_mode, gsde_reset_interval, gsde_reset_prob = _parse_gsde_reset_mode(
        policy,
        gsde_reset_mode,
    )

    buffer.reset()

    env_reset_time = 0.0
    if rollout_state is None:
        with PerformanceTimer() as env_reset_timer:
            obs, info = env.reset()
        env_reset_time = env_reset_timer.get_duration()
        is_final = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
        was_terminated = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
        previous_actions: torch.Tensor | None = None
        if policy.requires_previous_actions():
            previous_actions = torch.zeros(
                (buffer.n_envs, buffer.n_agents, buffer.n_agent_actions),
                dtype=buffer.rollout_dtype,
                device=buffer.rollout_device,
            )
        rollout_step_idx = 0
        episode_info_buffers: list[dict[str, Any] | None] = [None for _ in range(buffer.n_envs)]
    else:
        obs = rollout_state.obs
        is_final = rollout_state.is_final
        was_terminated = rollout_state.was_terminated
        previous_actions = rollout_state.previous_actions
        rollout_step_idx = rollout_state.rollout_step_idx
        episode_info_buffers = rollout_state.pending_episode_infos

    with PerformanceTimer() as to_rollout_device_timer:
        policy.to(buffer.rollout_device)
        policy.eval()
        if rollout_state is None:
            initial_mask = torch.ones((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
            policy.reset_temporal_state(episode_start_mask=initial_mask)
            _reset_temporal_correlations(policy=policy, episode_start_mask=initial_mask)

    episode_infos: list[dict[str, Any]] = []
    timers = _init_rollout_timers()

    initial_episode_count = len(buffer.episodes)

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")

    transitions_collected = 0
    while transitions_collected < n_steps:
        transitions_collected += int(torch.count_nonzero(torch.logical_not(is_final)).item())
        obs, is_final, was_terminated, previous_actions, rollout_step_idx = _collect_rollout_step(
            env=env,
            policy=policy,
            buffer=buffer,
            obs=obs,
            is_final=is_final,
            was_terminated=was_terminated,
            previous_actions=previous_actions,
            rollout_step_idx=rollout_step_idx,
            timers=timers,
            episode_infos=episode_infos,
            episode_info_buffers=episode_info_buffers,
            gsde_enabled=gsde_enabled,
            is_gsde_interval_reset_mode=is_gsde_interval_reset_mode,
            gsde_reset_interval=gsde_reset_interval,
            gsde_reset_prob=gsde_reset_prob,
        )

    final_env_indices = torch.where(is_final)[0]
    for env_idx in final_env_indices.tolist():
        buffered_info = episode_info_buffers[env_idx]
        if buffered_info is not None:
            episode_infos.append(buffered_info)
            episode_info_buffers[env_idx] = None

    local_obs = obs["local_obs"]
    global_obs = obs["global_obs"]
    hidden_local_vars = obs["hidden_local_vars"]
    hidden_global_vars = obs["hidden_global_vars"]
    agent_mask = obs.get("agent_mask", None)
    previous_actions_for_value: torch.Tensor | None = None
    if previous_actions is not None:
        previous_actions_for_value = previous_actions.masked_fill(is_final.unsqueeze(-1).unsqueeze(-1), 0.0)
    temporal_state_snapshot = policy.get_temporal_state_snapshot()
    try:
        _, _, final_values = policy(
            local_obs,
            global_obs,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions_for_value,
            deterministic=True,
        )
    finally:
        policy.restore_temporal_state_snapshot(temporal_state_snapshot)
    final_values = final_values.masked_fill(was_terminated, 0.0)

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
        is_final=is_final,
        was_terminated=was_terminated,
        previous_actions=previous_actions,
        rollout_step_idx=rollout_step_idx,
        pending_episode_infos=episode_info_buffers,
    )
    return episodes, episode_infos, metrics, new_state
