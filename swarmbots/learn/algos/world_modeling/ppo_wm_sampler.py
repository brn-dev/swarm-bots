from dataclasses import dataclass
from typing import TypeVar

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import MaybeTensor, PPOEpisodeSegment
from swarmbots.learn.algos.ppo.ppo_sampler import PPOSamples, PPOSampler, PPOSamplerConfig
from swarmbots.learn.algos.world_modeling.base_wm_sampler import BaseWMSampler, BaseWMSamples
from swarmbots.learn.algos.world_modeling.wm_sampler_helper import build_wm_episode_windows


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
        super().__init__(
            episodes=episodes,
            config=config,
            requires_previous_actions=requires_previous_actions,
        )
        
        multi_step_actions_list = []
        next_local_obs_list = []
        wm_target_time_mask_list = []
        next_global_obs_list = []
        wm_agent_mask_list: list[torch.Tensor] = []
        wm_loss_agent_mask_list: list[torch.Tensor] = []
        has_agent_mask = self.agent_mask is not None

        for ep in episodes:
            episode_windows = build_wm_episode_windows(
                ep,
                num_next_steps=num_next_steps,
            )
            multi_step_actions_list.append(episode_windows.multi_step_actions)
            next_local_obs_list.append(episode_windows.next_local_obs)
            wm_target_time_mask_list.append(episode_windows.wm_target_time_mask)
            next_global_obs_list.append(episode_windows.next_global_obs)
            if has_agent_mask:
                if episode_windows.wm_agent_mask is None:
                    raise ValueError("wm_agent_mask must be set when agent_mask is enabled")
                if episode_windows.wm_loss_agent_mask is None:
                    raise ValueError("wm_loss_agent_mask must be set when agent_mask is enabled")
                wm_agent_mask_list.append(episode_windows.wm_agent_mask)
                wm_loss_agent_mask_list.append(episode_windows.wm_loss_agent_mask)

        self.multi_step_actions = torch.cat(multi_step_actions_list, dim=0).contiguous()
        self.next_local_obs = torch.cat(next_local_obs_list, dim=0).contiguous()
        self.wm_target_time_mask = torch.cat(wm_target_time_mask_list, dim=0).contiguous()
        self.next_global_obs = torch.cat(next_global_obs_list, dim=0).contiguous()
        self.wm_agent_mask = (
            torch.cat(wm_agent_mask_list, dim=0).contiguous() if has_agent_mask else None
        )
        self.wm_loss_agent_mask = (
            torch.cat(wm_loss_agent_mask_list, dim=0).contiguous() if has_agent_mask else None
        )

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
