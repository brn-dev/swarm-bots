from dataclasses import dataclass
from typing import TypeVar

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import MaybeTensor, PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSampler, PPOSamplerConfig
from swarmbots.learn.algos.world_modeling.base_wm_sampler import BaseWMSampler, BaseWMSamples
from swarmbots.learn.algos.world_modeling.wm_sampler_helper import (
    build_wm_episode_windows_batch,
    ensure_wm_window_helper_compile_available,
)


@dataclass
class PPOWMSamples(PPOSamples, BaseWMSamples):
    wm_actions: torch.Tensor  # shape (batch, n_next_steps, n_agents, n_actions)
    next_local_obs: torch.Tensor  # shape (batch, n_next_steps, n_agents, n_obs_features)
    wm_target_time_mask: torch.Tensor  # shape (batch, n_next_steps)
    next_global_obs: torch.Tensor  # shape (batch, n_next_steps, n_global_obs_features)
    wm_agent_mask: MaybeTensor  # shape (batch, n_next_steps, n_agents)
    wm_loss_agent_mask: MaybeTensor  # shape (batch, n_next_steps, n_agents)

@dataclass(frozen=True)
class PPOWMSamplerConfig(PPOSamplerConfig):
    num_next_steps: int
    compile_wm_window_helper: bool = False
    wm_window_helper_compile_mode: str = "default"


PPOWMSamplesType = TypeVar('PPOSamplesType', bound=PPOWMSamples, covariant=True)
PPOWMSamplerConfigType = TypeVar('PPOSamplerConfigType', bound=PPOWMSamplerConfig, covariant=True)

class PPOWMSampler(
    PPOSampler[PPOWMSamples, PPOWMSamplerConfigType],
    BaseWMSampler[PPOWMSamples, PPOWMSamplerConfigType],
):

    def __init__(
            self,
            episodes: list[PPOEpisodeSegment],
            config: PPOWMSamplerConfigType,
            requires_previous_actions: bool = False,
    ):
        num_next_steps = config.num_next_steps
        if num_next_steps < 1:
            raise ValueError(f'num_next_steps must be >= 1, got {num_next_steps}')
        if config.compile_wm_window_helper:
            ensure_wm_window_helper_compile_available(
                compile_mode=config.wm_window_helper_compile_mode,
            )
        super().__init__(
            episodes=episodes,
            config=config,
            requires_previous_actions=requires_previous_actions,
        )

        has_agent_mask = self.agent_mask is not None
        self.multi_step_actions = torch.empty(
            (self.actions.shape[0], num_next_steps, *self.actions.shape[1:]),
            dtype=self.actions.dtype,
            device=self.actions.device,
        )
        self.next_local_obs = torch.empty(
            (self.local_obs.shape[0], num_next_steps, *self.local_obs.shape[1:]),
            dtype=self.local_obs.dtype,
            device=self.local_obs.device,
        )
        self.wm_target_time_mask = torch.empty(
            (self.local_obs.shape[0], num_next_steps),
            dtype=torch.bool,
            device=self.local_obs.device,
        )
        self.next_global_obs = torch.empty(
            (self.global_obs.shape[0], num_next_steps, *self.global_obs.shape[1:]),
            dtype=self.global_obs.dtype,
            device=self.global_obs.device,
        )
        self.wm_agent_mask = (
            torch.empty(
                (self.agent_mask.shape[0], num_next_steps, *self.agent_mask.shape[1:]),
                dtype=self.agent_mask.dtype,
                device=self.agent_mask.device,
            )
            if has_agent_mask
            else None
        )
        self.wm_loss_agent_mask = (
            torch.empty(
                (self.agent_mask.shape[0], num_next_steps, *self.agent_mask.shape[1:]),
                dtype=self.agent_mask.dtype,
                device=self.agent_mask.device,
            )
            if has_agent_mask
            else None
        )

        episode_sample_ranges: list[tuple[int, int]] = []
        episode_groups_by_length: dict[int, list[int]] = {}
        sample_start_idx = 0
        for episode_idx, episode in enumerate(episodes):
            sample_end_idx = sample_start_idx + episode.local_obs.shape[0]
            episode_sample_ranges.append((sample_start_idx, sample_end_idx))
            episode_groups_by_length.setdefault(int(episode.local_obs.shape[0]), []).append(episode_idx)
            sample_start_idx = sample_end_idx

        for episode_indices in episode_groups_by_length.values():
            episode_windows = build_wm_episode_windows_batch(
                [episodes[idx] for idx in episode_indices],
                num_next_steps=num_next_steps,
                compile_modules=config.compile_wm_window_helper,
                compile_mode=config.wm_window_helper_compile_mode,
            )
            flat_batch_indices = torch.cat(
                tuple(
                    torch.arange(start_idx, end_idx, device=self.local_obs.device)
                    for start_idx, end_idx in (episode_sample_ranges[idx] for idx in episode_indices)
                ),
                dim=0,
            )

            self.multi_step_actions[flat_batch_indices] = episode_windows.multi_step_actions.flatten(0, 1)
            self.next_local_obs[flat_batch_indices] = episode_windows.next_local_obs.flatten(0, 1)
            self.wm_target_time_mask[flat_batch_indices] = episode_windows.wm_target_time_mask.flatten(0, 1)
            self.next_global_obs[flat_batch_indices] = episode_windows.next_global_obs.flatten(0, 1)

            if has_agent_mask:
                if episode_windows.wm_agent_mask is None:
                    raise ValueError("wm_agent_mask must be set when agent_mask is enabled")
                if episode_windows.wm_loss_agent_mask is None:
                    raise ValueError("wm_loss_agent_mask must be set when agent_mask is enabled")
                assert self.wm_agent_mask is not None
                assert self.wm_loss_agent_mask is not None
                self.wm_agent_mask[flat_batch_indices] = episode_windows.wm_agent_mask.flatten(0, 1)
                self.wm_loss_agent_mask[flat_batch_indices] = episode_windows.wm_loss_agent_mask.flatten(0, 1)

    def _fetch_samples(self, batch_indices: torch.Tensor) -> PPOWMSamples:
        return PPOWMSamples(
            local_obs=self.local_obs[batch_indices],
            global_obs=self.global_obs[batch_indices],
            hidden_local_vars=self.hidden_local_vars[batch_indices],
            hidden_global_vars=self.hidden_global_vars[batch_indices],
            agent_mask=None if self.agent_mask is None else self.agent_mask[batch_indices],
            previous_actions=None if self.previous_actions is None else self.previous_actions[batch_indices],
            actions=self.actions[batch_indices],
            wm_actions=self.multi_step_actions[batch_indices],
            log_probs=self.log_probs[batch_indices],
            values=self.values[batch_indices],
            returns=self.returns[batch_indices],
            advantages=self.advantages[batch_indices],
            next_local_obs=self.next_local_obs[batch_indices],
            wm_target_time_mask=self.wm_target_time_mask[batch_indices],
            next_global_obs=self.next_global_obs[batch_indices],
            wm_agent_mask=None if self.wm_agent_mask is None else self.wm_agent_mask[batch_indices],
            wm_loss_agent_mask=None if self.wm_loss_agent_mask is None else self.wm_loss_agent_mask[batch_indices],
        )
