from dataclasses import dataclass
import math
from typing import Any, TypedDict

import torch

from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_batch import PPORolloutBatch, PPORolloutBatchBuilder
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment, PPORolloutBuffer
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import (
    GSDEResetMode,
    build_gsde_step_reset_mask,
    resolve_gsde_reset_mode,
)
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.rollout_utils import append_episode_infos, extract_bootstrap_obs, initial_previous_actions, snapshot_obs
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.temporal_state import move_temporal_state
from swarmbots.learn.torch_device import as_device


@dataclass(slots=True)
class PPORolloutState:
    obs: dict[str, torch.Tensor]
    episode_start_mask: torch.Tensor
    previous_actions: torch.Tensor | None
    temporal_state: Any
    rollout_step_idx: int


class _TransitionData(TypedDict):
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: torch.Tensor | None
    actions: torch.Tensor
    rewards: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    previous_actions: torch.Tensor | None
    episode_start_mask: torch.Tensor
    bootstrap_obs: dict[str, torch.Tensor]
    bootstrap_values: torch.Tensor
    dones: torch.Tensor


@dataclass(slots=True)
class _RolloutTransition:
    data: _TransitionData
    terminations: torch.Tensor
    truncations: torch.Tensor
    next_state: PPORolloutState


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


def _reset_temporal_correlations(
        *,
        policy: BasePPOPolicy[Any, Any],
        episode_start_mask: torch.Tensor | None = None,
        step_reset_mask: torch.Tensor | None = None,
) -> None:
    if episode_start_mask is not None:
        policy.action_dist.reset_temporal_correlations_on_ep_start(episode_start_mask)
    if step_reset_mask is not None:
        policy.action_dist.reset_temporal_correlations_on_step(mask=step_reset_mask)


def _evaluate_values(
        *,
        policy: BasePPOPolicy[Any, Any],
        obs: dict[str, torch.Tensor],
        previous_actions: torch.Tensor | None,
        temporal_state: Any,
        episode_start_mask: torch.Tensor | None = None,
        terminated_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    values, _ = policy.predict_values_with_temporal_state(
        local_obs=obs["local_obs"],
        global_obs=obs["global_obs"],
        hidden_local_vars=obs["hidden_local_vars"],
        hidden_global_vars=obs["hidden_global_vars"],
        agent_mask=obs.get("agent_mask", None),
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        episode_start_mask=episode_start_mask,
    )

    if terminated_mask is None:
        return values
    return values.masked_fill(terminated_mask, 0.0)


def _collect_bootstrap_values(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        next_obs: dict[str, torch.Tensor],
        infos: dict[str, Any],
        dones: torch.Tensor,
        terminations: torch.Tensor,
        truncations: torch.Tensor,
        previous_actions: torch.Tensor | None,
        temporal_state: Any,
        values: torch.Tensor,
        timers: _RolloutTimers,
        final_step: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    # Share the autoreset host decision instead of synchronizing again for truncations.
    has_dones, has_truncations, has_ongoing = torch.stack(
        (dones, truncations & ~terminations, ~dones),
    ).any(dim=1).tolist()
    bootstrap_obs = extract_bootstrap_obs(
        env=env, next_obs=next_obs, infos=infos, dones=dones, has_dones=has_dones,
    )
    if not (has_truncations or (final_step and has_ongoing)):
        return bootstrap_obs, torch.zeros_like(values)

    # Full batches keep compiled critic shapes stable as the set of truncated lanes changes.
    with timers.policy_forward_timer:
        bootstrap_values = _evaluate_values(
            policy=policy, obs=bootstrap_obs, previous_actions=previous_actions,
            temporal_state=temporal_state, terminated_mask=terminations,
        )
    timers.policy_forward_timings.append(timers.policy_forward_timer.get_duration())
    return bootstrap_obs, bootstrap_values


def _prepare_rollout_device(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        rollout_state: PPORolloutState | None,
        device: torch.device,
) -> PPORolloutState | None:
    env.set_device(device)
    policy.to(device)
    policy.eval()
    if rollout_state is None or rollout_state.obs["local_obs"].device == device:
        return rollout_state
    return PPORolloutState(
        obs={key: value.to(device=device) for key, value in rollout_state.obs.items()},
        episode_start_mask=rollout_state.episode_start_mask.to(device=device),
        previous_actions=(
            None if rollout_state.previous_actions is None else rollout_state.previous_actions.to(device=device)
        ),
        temporal_state=move_temporal_state(rollout_state.temporal_state, device=device),
        rollout_step_idx=rollout_state.rollout_step_idx,
    )


def _reset_rollout_state(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        n_agent_actions: int,
) -> PPORolloutState:
    obs, _info = env.reset()
    local_obs = obs["local_obs"]
    episode_start_mask = torch.ones((local_obs.shape[0],), dtype=torch.bool, device=local_obs.device)
    previous_actions = initial_previous_actions(
        policy=policy,
        obs=obs,
        n_agent_actions=n_agent_actions,
    )
    temporal_state = policy.initial_temporal_state(
        batch_size=local_obs.shape[0],
        n_agents=local_obs.shape[1],
        device=local_obs.device,
        dtype=local_obs.dtype,
    )
    return PPORolloutState(
        obs=obs,
        episode_start_mask=episode_start_mask,
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        rollout_step_idx=0,
    )


def _collect_rollout_step(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        state: PPORolloutState,
        timers: _RolloutTimers,
        episode_infos: list[dict[str, Any]],
        gsde_reset_mode: GSDEResetMode | None,
        final_step: bool,
) -> _RolloutTransition:
    obs_snapshot = snapshot_obs(state.obs)
    local_obs = obs_snapshot["local_obs"]
    batch_shape = tuple(local_obs.shape[:-1])
    step_reset_mask = build_gsde_step_reset_mask(
        gsde_reset_mode=gsde_reset_mode,
        rollout_step_idx=state.rollout_step_idx,
        batch_shape=batch_shape,
        rollout_device=local_obs.device,
        initialize=state.rollout_step_idx == 0,
    )

    with timers.reset_noise_timer:
        _reset_temporal_correlations(
            policy=policy,
            episode_start_mask=state.episode_start_mask,
            step_reset_mask=step_reset_mask,
        )
    timers.reset_noise_timings.append(timers.reset_noise_timer.get_duration())

    with timers.policy_forward_timer:
        actions, log_probs, values, next_temporal_state = policy.forward_with_temporal_state(
            local_obs=obs_snapshot["local_obs"],
            global_obs=obs_snapshot["global_obs"],
            hidden_local_vars=obs_snapshot["hidden_local_vars"],
            hidden_global_vars=obs_snapshot["hidden_global_vars"],
            agent_mask=obs_snapshot.get("agent_mask", None),
            previous_actions=state.previous_actions,
            temporal_state=state.temporal_state,
            episode_start_mask=state.episode_start_mask,
        )
    timers.policy_forward_timings.append(timers.policy_forward_timer.get_duration())

    with timers.env_step_timer:
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
    timers.env_step_timings.append(timers.env_step_timer.get_duration())

    dones = torch.logical_or(terminations, truncations)
    append_episode_infos(episode_infos=episode_infos, infos=infos, dones=dones)

    bootstrap_previous_actions = None if state.previous_actions is None else actions.detach()
    bootstrap_obs, bootstrap_values = _collect_bootstrap_values(
        env=env, policy=policy, next_obs=next_obs, infos=infos, dones=dones,
        terminations=terminations, truncations=truncations,
        previous_actions=bootstrap_previous_actions, temporal_state=next_temporal_state,
        values=values, timers=timers, final_step=final_step,
    )

    _reset_temporal_correlations(policy=policy, episode_start_mask=dones)

    next_previous_actions: torch.Tensor | None = None
    if state.previous_actions is not None:
        next_previous_actions = actions.detach().masked_fill(dones.unsqueeze(-1).unsqueeze(-1), 0.0)

    return _RolloutTransition(
        data=_TransitionData(
            local_obs=obs_snapshot["local_obs"],
            global_obs=obs_snapshot["global_obs"],
            hidden_local_vars=obs_snapshot["hidden_local_vars"],
            hidden_global_vars=obs_snapshot["hidden_global_vars"],
            agent_mask=obs_snapshot.get("agent_mask"),
            actions=actions, rewards=rewards, log_probs=log_probs, values=values,
            previous_actions=state.previous_actions, episode_start_mask=state.episode_start_mask,
            bootstrap_obs=bootstrap_obs, bootstrap_values=bootstrap_values, dones=dones,
        ),
        terminations=terminations,
        truncations=truncations,
        next_state=PPORolloutState(
            obs=next_obs, episode_start_mask=dones, previous_actions=next_previous_actions,
            temporal_state=next_temporal_state, rollout_step_idx=state.rollout_step_idx + 1,
        ),
    )


def _warmup_rollout_step(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        obs: dict[str, torch.Tensor],
        episode_start_mask: torch.Tensor,
        previous_actions: torch.Tensor | None,
        temporal_state: Any,
        rollout_step_idx: int,
        gsde_reset_mode: GSDEResetMode | None,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None, Any, int]:
    local_obs = obs["local_obs"]
    batch_shape = tuple(local_obs.shape[:-1])
    step_reset_mask = build_gsde_step_reset_mask(
        gsde_reset_mode=gsde_reset_mode,
        rollout_step_idx=rollout_step_idx,
        batch_shape=batch_shape,
        rollout_device=local_obs.device,
        initialize=rollout_step_idx == 0,
    )

    _reset_temporal_correlations(
        policy=policy,
        episode_start_mask=episode_start_mask,
        step_reset_mask=step_reset_mask,
    )

    actions, _log_probs, _values, next_temporal_state = policy.forward_with_temporal_state(
        local_obs=local_obs,
        global_obs=obs["global_obs"],
        hidden_local_vars=obs["hidden_local_vars"],
        hidden_global_vars=obs["hidden_global_vars"],
        agent_mask=obs.get("agent_mask", None),
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        episode_start_mask=episode_start_mask,
    )

    next_obs, _rewards, terminations, truncations, _infos = env.step(actions)
    dones = torch.logical_or(terminations, truncations)

    _reset_temporal_correlations(policy=policy, episode_start_mask=dones)

    next_previous_actions: torch.Tensor | None = None
    if previous_actions is not None:
        next_previous_actions = actions.detach().masked_fill(dones.unsqueeze(-1).unsqueeze(-1), 0.0)

    return next_obs, dones, next_previous_actions, next_temporal_state, rollout_step_idx + 1


def _build_rollout_metrics(
        *,
        env_reset_time: float,
        to_rollout_device_time: float,
        timers: _RolloutTimers,
        rollout_finalize_time: float,
        episode_infos: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        "env_reset_time": env_reset_time,
        "to_rollout_device_time": to_rollout_device_time,
        "reset_noise_time": compute_summary_statistics(timers.reset_noise_timings),
        "total_reset_noise_time": sum(timers.reset_noise_timings),
        "policy_forward_time": compute_summary_statistics(timers.policy_forward_timings),
        "total_policy_forward_time": sum(timers.policy_forward_timings),
        "env_step_time": compute_summary_statistics(timers.env_step_timings),
        "total_env_step_time": sum(timers.env_step_timings),
        "buffer_add_time": compute_summary_statistics(timers.buffer_add_timings),
        "total_buffer_add_time": sum(timers.buffer_add_timings),
        "rollout_finalize_time": rollout_finalize_time,
    }
    success_values = [float(ep_info["success"]) for ep_info in episode_infos if "success" in ep_info]
    if success_values:
        metrics["ep_success_rate"] = 100.0 * (sum(success_values) / len(success_values))
    return metrics


@torch.no_grad()
def warmup_rollout_steps(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        n_steps: int,
        rollout_state: PPORolloutState | None = None,
        gsde_reset_mode: GSDEResetMode | None = None,
        rollout_device: torch.device | str | None = None,
) -> PPORolloutState:
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")

    gsde_reset_mode = resolve_gsde_reset_mode(
        gsde_enabled=policy.gsde_enabled,
        reset_mode=gsde_reset_mode,
    )

    rollout_state = _prepare_rollout_device(
        env=env,
        policy=policy,
        rollout_state=rollout_state,
        device=as_device(env.device if rollout_device is None else rollout_device),
    )
    if rollout_state is None:
        rollout_state = _reset_rollout_state(
            env=env,
            policy=policy,
            n_agent_actions=env.action_space.total_agent_action_dim,
        )

    obs = rollout_state.obs
    episode_start_mask = rollout_state.episode_start_mask
    previous_actions = rollout_state.previous_actions
    temporal_state = rollout_state.temporal_state
    rollout_step_idx = rollout_state.rollout_step_idx

    transitions_collected = 0
    while transitions_collected < n_steps:
        transitions_collected += obs["local_obs"].shape[0]
        obs, episode_start_mask, previous_actions, temporal_state, rollout_step_idx = _warmup_rollout_step(
            env=env,
            policy=policy,
            obs=obs,
            episode_start_mask=episode_start_mask,
            previous_actions=previous_actions,
            temporal_state=temporal_state,
            rollout_step_idx=rollout_step_idx,
            gsde_reset_mode=gsde_reset_mode,
        )

    return PPORolloutState(
        obs=obs,
        episode_start_mask=episode_start_mask,
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        rollout_step_idx=rollout_step_idx,
    )


@torch.no_grad()
def collect_step_rollout_batch(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        buffer: PPORolloutBuffer,
        n_steps: int,
        rollout_state: PPORolloutState | None = None,
        gsde_reset_mode: GSDEResetMode | None = None,
) -> tuple[PPORolloutBatch, list[dict[str, Any]], dict[str, Any], PPORolloutState]:
    gsde_reset_mode = resolve_gsde_reset_mode(
        gsde_enabled=policy.gsde_enabled,
        reset_mode=gsde_reset_mode,
    )

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")

    buffer.reset()

    with PerformanceTimer() as to_rollout_device_timer:
        rollout_state = _prepare_rollout_device(
            env=env, policy=policy, rollout_state=rollout_state, device=buffer.rollout_device,
        )
    env_reset_time = 0.0
    if rollout_state is None:
        with PerformanceTimer() as env_reset_timer:
            rollout_state = _reset_rollout_state(
                env=env, policy=policy, n_agent_actions=buffer.n_agent_actions,
            )
        env_reset_time = env_reset_timer.get_duration()
    vector_steps = math.ceil(n_steps / buffer.n_envs)
    with PerformanceTimer() as batch_builder_init_timer:
        batch_builder = PPORolloutBatchBuilder(
            n_envs=buffer.n_envs,
            n_steps=vector_steps,
            n_agents=buffer.n_agents,
            agent_obs_shape=buffer.agent_obs_shape,
            global_obs_shape=buffer.global_obs_shape,
            hidden_local_vars_shape=buffer.hidden_local_vars_shape,
            hidden_global_vars_shape=buffer.hidden_global_vars_shape,
            n_agent_actions=buffer.n_agent_actions,
            has_agent_mask=buffer.has_agent_mask,
            has_previous_actions=policy.requires_previous_actions(),
            gamma=buffer.gamma,
            gae_lambda=buffer.gae_lambda,
            storage_device=buffer.rollout_device,
            storage_dtype=buffer.rollout_dtype,
        )

    episode_infos: list[dict[str, Any]] = []
    timers = _init_rollout_timers()

    for batch_step_idx in range(vector_steps):
        transition = _collect_rollout_step(
            env=env, policy=policy, state=rollout_state, timers=timers,
            episode_infos=episode_infos, gsde_reset_mode=gsde_reset_mode,
            final_step=batch_step_idx == vector_steps - 1,
        )
        with timers.buffer_add_timer:
            batch_builder.add(
                step_idx=batch_step_idx,
                terminations=transition.terminations,
                truncations=transition.truncations,
                **transition.data,
            )
        timers.buffer_add_timings.append(timers.buffer_add_timer.get_duration())
        rollout_state = transition.next_state

    with PerformanceTimer() as batch_finalize_timer:
        rollout_batch = batch_builder.build().to(device=buffer.train_device, dtype=buffer.train_dtype)

    metrics = _build_rollout_metrics(
        env_reset_time=env_reset_time,
        to_rollout_device_time=to_rollout_device_timer.get_duration(),
        timers=timers,
        rollout_finalize_time=batch_finalize_timer.get_duration(),
        episode_infos=episode_infos,
    )
    metrics["step_rollout_batch_path"] = True
    metrics["rollout_batch_builder_init_time"] = batch_builder_init_timer.get_duration()
    metrics["rollout_batch_finalize_time"] = batch_finalize_timer.get_duration()
    return rollout_batch, episode_infos, metrics, rollout_state


@torch.no_grad()
def collect_steps(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy[Any, Any],
        buffer: PPORolloutBuffer,
        n_steps: int,
        rollout_state: PPORolloutState | None = None,
        gsde_reset_mode: GSDEResetMode | None = None,
) -> tuple[list[PPOEpisodeSegment], list[dict[str, Any]], dict[str, Any], PPORolloutState]:
    gsde_reset_mode = resolve_gsde_reset_mode(
        gsde_enabled=policy.gsde_enabled,
        reset_mode=gsde_reset_mode,
    )

    buffer.reset()

    with PerformanceTimer() as to_rollout_device_timer:
        rollout_state = _prepare_rollout_device(
            env=env, policy=policy, rollout_state=rollout_state, device=buffer.rollout_device,
        )
    env_reset_time = 0.0
    if rollout_state is None:
        with PerformanceTimer() as env_reset_timer:
            rollout_state = _reset_rollout_state(
                env=env, policy=policy, n_agent_actions=buffer.n_agent_actions,
            )
        env_reset_time = env_reset_timer.get_duration()
    episode_infos: list[dict[str, Any]] = []
    timers = _init_rollout_timers()

    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")

    transitions_collected = 0
    while transitions_collected < n_steps:
        transitions_collected += buffer.n_envs
        transition = _collect_rollout_step(
            env=env, policy=policy, state=rollout_state, timers=timers,
            episode_infos=episode_infos, gsde_reset_mode=gsde_reset_mode,
            final_step=transitions_collected >= n_steps,
        )
        with timers.buffer_add_timer:
            buffer.add(
                temporal_state=rollout_state.temporal_state,
                rollout_step_idx=rollout_state.rollout_step_idx,
                **transition.data,
            )
        timers.buffer_add_timings.append(timers.buffer_add_timer.get_duration())
        rollout_state = transition.next_state
        final_values = transition.data["bootstrap_values"]

    with PerformanceTimer() as rollout_finalize_timer:
        completed_episodes = buffer.get_completed_episodes()
    rollout_finalize_time = rollout_finalize_timer.get_duration()
    with PerformanceTimer() as buffer_dump_partial_episodes_timer:
        partial_episodes = buffer.dump_partial_episodes(
            final_obs=rollout_state.obs,
            final_values=final_values,
            clone_tensors=False,
        )
    rollout_finalize_time += buffer_dump_partial_episodes_timer.get_duration()

    episodes = completed_episodes + partial_episodes

    metrics = _build_rollout_metrics(
        env_reset_time=env_reset_time,
        to_rollout_device_time=to_rollout_device_timer.get_duration(),
        timers=timers,
        rollout_finalize_time=rollout_finalize_time,
        episode_infos=episode_infos,
    )
    return episodes, episode_infos, metrics, rollout_state
