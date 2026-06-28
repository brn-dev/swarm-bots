from dataclasses import dataclass
from typing import Any

import torch

from swarmbots.learn.algos.off_policy.off_policy_replay_buffer import OffPolicyEpisodeSegment, OffPolicyReplayBuffer
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.rollout_utils import (
    append_episode_infos,
    extract_bootstrap_obs,
    sample_random_actions,
    snapshot_obs,
)
from swarmbots.learn.summary_statistics import compute_summary_statistics


@dataclass(slots=True)
class OffPolicyRolloutState:
    obs: dict[str, torch.Tensor]
    episode_start_mask: torch.Tensor
    previous_actions: torch.Tensor | None
    rollout_step_idx: int


@dataclass(slots=True)
class _RolloutTimers:
    policy_forward_timings: list[float]
    env_step_timings: list[float]
    buffer_add_timings: list[float]
    policy_forward_timer: PerformanceTimer
    env_step_timer: PerformanceTimer
    buffer_add_timer: PerformanceTimer


def _init_rollout_timers() -> _RolloutTimers:
    return _RolloutTimers(
        policy_forward_timings=[],
        env_step_timings=[],
        buffer_add_timings=[],
        policy_forward_timer=PerformanceTimer(),
        env_step_timer=PerformanceTimer(),
        buffer_add_timer=PerformanceTimer(),
    )


def _reset_rollout_state(
        *,
        env: BaseLearnEnvWrapper,
        n_agent_actions: int,
) -> OffPolicyRolloutState:
    obs, _info = env.reset()
    local_obs = obs["local_obs"]
    return OffPolicyRolloutState(
        obs=obs,
        episode_start_mask=torch.ones((local_obs.shape[0],), dtype=torch.bool, device=local_obs.device),
        previous_actions=_zero_previous_actions(obs, n_agent_actions),
        rollout_step_idx=0,
    )


def _zero_previous_actions(obs: dict[str, torch.Tensor], n_agent_actions: int) -> torch.Tensor:
    local_obs = obs["local_obs"]
    n_envs, n_agents = local_obs.shape[:2]
    return torch.zeros(
        (n_envs, n_agents, n_agent_actions),
        dtype=local_obs.dtype,
        device=local_obs.device,
    )


def _act(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePolicy,
        obs: dict[str, torch.Tensor],
        previous_actions: torch.Tensor | None,
        deterministic: bool,
        random_actions: bool,
) -> torch.Tensor:
    if random_actions:
        return sample_random_actions(
            env.action_space,
            device=obs["local_obs"].device,
            dtype=obs["local_obs"].dtype,
        )

    policy_previous_actions = previous_actions if policy.requires_previous_actions() else None
    return policy.act(
        local_obs=obs["local_obs"],
        global_obs=obs["global_obs"],
        hidden_local_vars=obs["hidden_local_vars"],
        hidden_global_vars=obs["hidden_global_vars"],
        agent_mask=obs.get("agent_mask", None),
        previous_actions=policy_previous_actions,
        deterministic=deterministic,
    )


def _collect_rollout_step(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePolicy,
        replay_buffer: OffPolicyReplayBuffer,
        obs: dict[str, torch.Tensor],
        episode_start_mask: torch.Tensor,
        previous_actions: torch.Tensor | None,
        rollout_step_idx: int,
        deterministic: bool,
        random_actions: bool,
        timers: _RolloutTimers,
        episode_infos: list[dict[str, Any]],
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor | None, int, list[OffPolicyEpisodeSegment]]:
    obs_snapshot = snapshot_obs(obs)

    with timers.policy_forward_timer:
        actions = _act(
            env=env,
            policy=policy,
            obs=obs_snapshot,
            previous_actions=previous_actions,
            deterministic=deterministic,
            random_actions=random_actions,
        )
    timers.policy_forward_timings.append(timers.policy_forward_timer.get_duration())

    with timers.env_step_timer:
        next_obs, rewards, terminations, truncations, infos = env.step(actions)
    timers.env_step_timings.append(timers.env_step_timer.get_duration())

    dones = torch.logical_or(terminations, truncations)
    append_episode_infos(episode_infos=episode_infos, infos=infos, dones=dones)
    transition_next_obs = extract_bootstrap_obs(
        env=env,
        next_obs=next_obs,
        infos=infos,
        dones=dones,
    )

    with timers.buffer_add_timer:
        completed_segments = replay_buffer.add(
            local_obs=obs_snapshot["local_obs"],
            global_obs=obs_snapshot["global_obs"],
            hidden_local_vars=obs_snapshot["hidden_local_vars"],
            hidden_global_vars=obs_snapshot["hidden_global_vars"],
            agent_mask=obs_snapshot.get("agent_mask", None),
            previous_actions=previous_actions,
            actions=actions,
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            next_obs=transition_next_obs,
            episode_start_mask=episode_start_mask,
            rollout_step_idx=rollout_step_idx,
        )
    timers.buffer_add_timings.append(timers.buffer_add_timer.get_duration())

    next_previous_actions: torch.Tensor | None = None
    if previous_actions is not None:
        next_previous_actions = actions.detach().masked_fill(dones.unsqueeze(-1).unsqueeze(-1), 0.0)

    return next_obs, dones, next_previous_actions, rollout_step_idx + 1, completed_segments


def _build_rollout_metrics(
        *,
        env_reset_time: float,
        to_rollout_device_time: float,
        timers: _RolloutTimers,
        flush_partial_segments_time: float,
        episode_infos: list[dict[str, Any]],
) -> dict[str, Any]:
    metrics = {
        "env_reset_time": env_reset_time,
        "to_rollout_device_time": to_rollout_device_time,
        "policy_forward_time": compute_summary_statistics(timers.policy_forward_timings),
        "total_policy_forward_time": sum(timers.policy_forward_timings),
        "env_step_time": compute_summary_statistics(timers.env_step_timings),
        "total_env_step_time": sum(timers.env_step_timings),
        "buffer_add_time": compute_summary_statistics(timers.buffer_add_timings),
        "total_buffer_add_time": sum(timers.buffer_add_timings),
        "flush_partial_segments_time": flush_partial_segments_time,
    }
    success_values = [float(ep_info["success"]) for ep_info in episode_infos if "success" in ep_info]
    if success_values:
        metrics["ep_success_rate"] = 100.0 * (sum(success_values) / len(success_values))
    return metrics


@torch.no_grad()
def collect_steps(
        env: BaseLearnEnvWrapper,
        policy: BasePolicy,
        replay_buffer: OffPolicyReplayBuffer,
        n_steps: int,
        rollout_state: OffPolicyRolloutState | None = None,
        *,
        deterministic: bool = False,
) -> tuple[list[OffPolicyEpisodeSegment], list[dict[str, Any]], dict[str, Any], OffPolicyRolloutState]:
    return _collect_steps(
        env=env,
        policy=policy,
        replay_buffer=replay_buffer,
        n_steps=n_steps,
        rollout_state=rollout_state,
        deterministic=deterministic,
        random_actions=False,
    )


@torch.no_grad()
def warmup_random_steps(
        env: BaseLearnEnvWrapper,
        replay_buffer: OffPolicyReplayBuffer,
        n_steps: int,
        rollout_state: OffPolicyRolloutState | None = None,
) -> tuple[list[OffPolicyEpisodeSegment], list[dict[str, Any]], dict[str, Any], OffPolicyRolloutState]:
    class _RandomPolicy(BasePolicy):
        @property
        def gsde_enabled(self) -> bool:
            return False

        def get_hyper_parameters(self) -> dict[str, Any]:
            return {}

        def get_grad_norms(self) -> dict[str, float]:
            return {}

        def act(
                self,
                local_obs: torch.Tensor,
                global_obs: torch.Tensor,
                hidden_local_vars: torch.Tensor | None = None,
                hidden_global_vars: torch.Tensor | None = None,
                agent_mask: torch.Tensor | None = None,
                previous_actions: torch.Tensor | None = None,
                deterministic: bool = False,
        ) -> torch.Tensor:
            raise NotImplementedError

        def update_loss_weights(self, **weights: float) -> None:
            if weights:
                raise ValueError(f"Unknown loss weights: {sorted(weights)}")

        def requires_previous_actions(self) -> bool:
            return False

    return _collect_steps(
        env=env,
        policy=_RandomPolicy(),
        replay_buffer=replay_buffer,
        n_steps=n_steps,
        rollout_state=rollout_state,
        deterministic=False,
        random_actions=True,
    )


def _collect_steps(
        *,
        env: BaseLearnEnvWrapper,
        policy: BasePolicy,
        replay_buffer: OffPolicyReplayBuffer,
        n_steps: int,
        rollout_state: OffPolicyRolloutState | None,
        deterministic: bool,
        random_actions: bool,
) -> tuple[list[OffPolicyEpisodeSegment], list[dict[str, Any]], dict[str, Any], OffPolicyRolloutState]:
    if n_steps <= 0:
        raise ValueError(f"n_steps must be > 0, got {n_steps}")

    env_reset_time = 0.0
    if rollout_state is None:
        with PerformanceTimer() as env_reset_timer:
            rollout_state = _reset_rollout_state(
                env=env,
                n_agent_actions=replay_buffer.n_agent_actions,
            )
        env_reset_time = env_reset_timer.get_duration()

    obs = rollout_state.obs
    episode_start_mask = rollout_state.episode_start_mask
    previous_actions = rollout_state.previous_actions
    if previous_actions is None:
        previous_actions = _zero_previous_actions(obs, replay_buffer.n_agent_actions)
    rollout_step_idx = rollout_state.rollout_step_idx

    with PerformanceTimer() as to_rollout_device_timer:
        if not random_actions:
            policy.to(replay_buffer.rollout_device)
            policy.eval()

    episode_infos: list[dict[str, Any]] = []
    timers = _init_rollout_timers()
    added_segments: list[OffPolicyEpisodeSegment] = []

    transitions_collected = 0
    while transitions_collected < n_steps:
        transitions_collected += replay_buffer.n_envs
        obs, episode_start_mask, previous_actions, rollout_step_idx, completed_segments = _collect_rollout_step(
            env=env,
            policy=policy,
            replay_buffer=replay_buffer,
            obs=obs,
            episode_start_mask=episode_start_mask,
            previous_actions=previous_actions,
            rollout_step_idx=rollout_step_idx,
            deterministic=deterministic,
            random_actions=random_actions,
            timers=timers,
            episode_infos=episode_infos,
        )
        added_segments.extend(completed_segments)

    with PerformanceTimer() as flush_partial_segments_timer:
        partial_segments = replay_buffer.flush_partial_segments()
    added_segments.extend(partial_segments)

    metrics = _build_rollout_metrics(
        env_reset_time=env_reset_time,
        to_rollout_device_time=to_rollout_device_timer.get_duration(),
        timers=timers,
        flush_partial_segments_time=flush_partial_segments_timer.get_duration(),
        episode_infos=episode_infos,
    )
    new_state = OffPolicyRolloutState(
        obs=obs,
        episode_start_mask=episode_start_mask,
        previous_actions=previous_actions,
        rollout_step_idx=rollout_step_idx,
    )
    return added_segments, episode_infos, metrics, new_state
