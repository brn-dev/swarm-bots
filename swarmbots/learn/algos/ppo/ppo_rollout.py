from typing import Any

import torch

from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisode, PPORolloutBuffer
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.performance_timer import PerformanceTimer
from swarmbots.learn.gsde_reset import GSDEResetMode, GSDEIntervalResetMode, GSDEProbabilityResetMode
from swarmbots.learn.summary_statistics import compute_summary_statistics


@torch.no_grad()
def collect_whole_episodes(
        env: BaseLearnEnvWrapper,
        policy: BasePPOPolicy,
        buffer: PPORolloutBuffer,
        gsde_reset_mode: GSDEResetMode | None = None,
) -> tuple[list[PPOEpisode], list[dict], dict[str, Any]]:
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

    buffer.reset()
    with PerformanceTimer() as env_reset_timer:
        obs, info = env.reset()
    is_final = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)
    was_terminated = torch.zeros((buffer.n_envs,), dtype=torch.bool, device=buffer.rollout_device)

    with PerformanceTimer() as to_rollout_device_timer:
        policy.to(buffer.rollout_device)
        policy.eval()

    episode_infos = []
    rollout_step_idx = 0

    reset_noise_timings: list[float] = []
    policy_forward_timings: list[float] = []
    env_step_timings: list[float] = []
    buffer_add_timings: list[float] = []

    reset_noise_timer = PerformanceTimer()
    policy_forward_timer = PerformanceTimer()
    env_step_timer = PerformanceTimer()
    buffer_add_timer = PerformanceTimer()

    while not buffer.is_ready():
        local_obs = obs['local_obs']
        global_obs = obs['global_obs']
        hidden_vars = obs["hidden_vars"]
        agent_mask = obs.get("agent_mask", None)

        if gsde_enabled:
            batch_shape = tuple(local_obs.shape[:-1])
            with reset_noise_timer:
                if is_gsde_interval_reset_mode:
                    if (rollout_step_idx % gsde_reset_interval) == 0:
                        policy.action_dist.reset_noise(batch_shape=batch_shape)
                else:  # probability reset mode
                    mask = torch.empty(
                        batch_shape,
                        device=buffer.rollout_device,
                        dtype=torch.bool,
                    ).bernoulli_(gsde_reset_prob)
                    policy.action_dist.reset_noise_masked(mask)

            reset_noise_timings.append(reset_noise_timer.get_duration())

        with policy_forward_timer:
            actions, log_probs, values = policy(
                local_obs,
                global_obs,
                hidden_vars=hidden_vars,
                agent_mask=agent_mask,
            )
        policy_forward_timings.append(policy_forward_timer.get_duration())

        values = values.masked_fill(was_terminated, 0.0)

        with env_step_timer:
            new_obs, rewards, terminations, truncations, infos = env.step(actions)
        env_step_timings.append(env_step_timer.get_duration())
        dones = torch.logical_or(terminations, truncations)

        if "episode" in infos:
            for i, has_ep_info in enumerate(infos["_episode"]):
                if has_ep_info:
                    episode_infos.append({
                        'r': infos['episode']['r'][i],
                        'l': infos['episode']['l'][i],
                        't': infos['episode']['t'][i],
                    })

        with buffer_add_timer:
            buffer.add(
                local_obs=local_obs,
                global_obs=global_obs,
                hidden_vars=hidden_vars,
                agent_mask=agent_mask,
                actions=actions,
                rewards=rewards,
                log_probs=log_probs,
                values=values,
                is_final=is_final,
            )
        buffer_add_timings.append(buffer_add_timer.get_duration())

        obs = new_obs
        is_final = dones
        was_terminated = terminations
        rollout_step_idx += 1

    with PerformanceTimer() as buffer_get_whole_episodes_timer:
        episodes = buffer.get_whole_episodes()

    metrics = {
        'env_reset_time': env_reset_timer.get_duration(),
        'to_rollout_device_time': to_rollout_device_timer.get_duration(),
        'reset_noise_time': compute_summary_statistics(reset_noise_timings),
        'total_reset_noise_time': sum(reset_noise_timings),
        'policy_forward_time': compute_summary_statistics(policy_forward_timings),
        'total_policy_forward_time': sum(policy_forward_timings),
        'env_step_time': compute_summary_statistics(env_step_timings),
        'total_env_step_time': sum(env_step_timings),
        'buffer_add_time': compute_summary_statistics(buffer_add_timings),
        'total_buffer_add_time': sum(buffer_add_timings),
        'buffer_get_whole_episodes_time': buffer_get_whole_episodes_timer.get_duration(),
    }
    return episodes, episode_infos, metrics
