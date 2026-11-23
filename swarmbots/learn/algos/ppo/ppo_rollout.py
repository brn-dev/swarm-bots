from dataclasses import dataclass
import math
from typing import Any, TypedDict

import torch

from swarmbots.learn.algos.ppo.base_ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_batch import PPORolloutBatch, PPORolloutBatchBuilder
from swarmbots.learn.algos.ppo.ppo_scalar_frames import FrameHistory, RolloutScalarFrameRecorder
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment, PPORolloutBuffer
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import (
    GSDEResetMode,
    reset_gsde_noise,
    resolve_gsde_reset_mode,
)
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.rollout_utils import (
    RolloutStepMetadata,
    append_episode_infos,
    collect_rollout_step_metadata,
    extract_bootstrap_obs,
    initial_previous_actions,
    snapshot_obs,
)
from swarmbots.learn.summary_statistics import compute_summary_statistics
from swarmbots.learn.task_episode_metrics import episode_success_metrics
from swarmbots.learn.temporal_state import move_temporal_state
from swarmbots.learn.torch_device import as_device


@dataclass(slots=True)
class PPORolloutState:
    obs: dict[str, torch.Tensor]
    episode_start_mask: torch.Tensor
    previous_actions: torch.Tensor | None
    temporal_state: Any
    rollout_step_idx: int
    scalar_frame_history: FrameHistory | None = None


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
        step_metadata: RolloutStepMetadata,
        previous_actions: torch.Tensor | None,
        temporal_state: Any,
        values: torch.Tensor,
        timers: _RolloutTimers,
        final_step: bool = False,
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    bootstrap_obs = extract_bootstrap_obs(
        env=env, next_obs=next_obs, infos=infos, dones=dones, step_metadata=step_metadata,
    )
    if not (step_metadata.has_truncations or (final_step and step_metadata.has_ongoing)):
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
        scalar_frame_history=rollout_state.scalar_frame_history,
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
    with timers.reset_noise_timer:
        reset_gsde_noise(
            action_dist=policy.action_dist, local_obs=local_obs, episode_start_mask=state.episode_start_mask,
            rollout_step_idx=state.rollout_step_idx, gsde_reset_mode=gsde_reset_mode,
            initialize=state.rollout_step_idx == 0,
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
    step_metadata = collect_rollout_step_metadata(
        infos=infos, dones=dones, terminations=terminations, truncations=truncations,
    )
    append_episode_infos(episode_infos=episode_infos, infos=infos, dones=dones, step_metadata=step_metadata)

    bootstrap_previous_actions = None if state.previous_actions is None else actions.detach()
    bootstrap_obs, bootstrap_values = _collect_bootstrap_values(
        env=env, policy=policy, next_obs=next_obs, infos=infos, dones=dones,
        terminations=terminations, step_metadata=step_metadata,
        previous_actions=bootstrap_previous_actions, temporal_state=next_temporal_state,
        values=values, timers=timers, final_step=final_step,
    )

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
    reset_gsde_noise(
        action_dist=policy.action_dist, local_obs=local_obs, episode_start_mask=episode_start_mask,
        rollout_step_idx=rollout_step_idx, gsde_reset_mode=gsde_reset_mode, initialize=rollout_step_idx == 0,
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
    return {
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
        **episode_success_metrics(episode_infos),
    }


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

    vector_steps = math.ceil(n_steps / obs["local_obs"].shape[0])
    frame_recorder = (
        RolloutScalarFrameRecorder(
            env, n_steps=vector_steps, history=rollout_state.scalar_frame_history, history_only=True,
        ) if getattr(policy, "uses_world_model_targets", False) else None
    )
    for step in range(vector_steps):
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
        if frame_recorder is not None:
            frame_recorder.record(step + 1, episode_start_mask)

    return PPORolloutState(
        obs=obs,
        episode_start_mask=episode_start_mask,
        previous_actions=previous_actions,
        temporal_state=temporal_state,
        rollout_step_idx=rollout_step_idx,
        scalar_frame_history=frame_recorder.history() if frame_recorder is not None else None,
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
    frame_recorder = (
        RolloutScalarFrameRecorder(env, n_steps=vector_steps, history=rollout_state.scalar_frame_history)
        if getattr(policy, "uses_world_model_targets", False) else None
    )
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
            if frame_recorder is not None:
                frame_recorder.record(batch_step_idx + 1, transition.data["dones"])
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
        if frame_recorder is not None:
            rollout_batch.scalar_frames = frame_recorder.finish(buffer.train_device)
            rollout_state.scalar_frame_history = frame_recorder.history()

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

    vector_steps = math.ceil(n_steps / buffer.n_envs)
    frame_recorder = (
        RolloutScalarFrameRecorder(env, n_steps=vector_steps, history=rollout_state.scalar_frame_history)
        if getattr(policy, "uses_world_model_targets", False) else None
    )
    start_step = rollout_state.rollout_step_idx
    for step in range(vector_steps):
        transition = _collect_rollout_step(
            env=env, policy=policy, state=rollout_state, timers=timers,
            episode_infos=episode_infos, gsde_reset_mode=gsde_reset_mode,
            final_step=step == vector_steps - 1,
        )
        with timers.buffer_add_timer:
            if frame_recorder is not None:
                frame_recorder.record(step + 1, transition.data["dones"])
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
    if frame_recorder is not None:
        frames = frame_recorder.finish(buffer.train_device)
        rollout_state.scalar_frame_history = frame_recorder.history()
        for episode in episodes:
            offset = episode.rollout_start_step - start_step
            episode.scalar_frames = {
                key: history.segment(offset, episode.local_obs.shape[0], episode.rollout_env_idx)
                for key, history in frames.items()
            }

    metrics = _build_rollout_metrics(
        env_reset_time=env_reset_time,
        to_rollout_device_time=to_rollout_device_timer.get_duration(),
        timers=timers,
        rollout_finalize_time=rollout_finalize_time,
        episode_infos=episode_infos,
    )
    return episodes, episode_infos, metrics, rollout_state
