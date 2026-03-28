from dataclasses import dataclass

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOSampler, PPOEpisode, PPOSamples


@dataclass
class PPOWMSamples(PPOSamples):
    actions: torch.Tensor  # (batch, n_next_steps, n_agents, n_actions)

    next_local_obs: torch.Tensor  # shape (batch, n_next_steps, n_agents, n_obs_features)
    next_validity_mask: torch.Tensor  # shape (batch, n_next_steps)
    next_global_obs: torch.Tensor  # shape (batch, n_next_steps, n_global_obs_features)
    wm_agent_mask: torch.Tensor | None  # shape (batch, n_next_steps, n_agents)
    wm_loss_agent_mask: torch.Tensor | None  # shape (batch, n_next_steps, n_agents)


class PPOWMSampler(PPOSampler[PPOWMSamples]):

    def __init__(
            self,
            episodes: list[PPOEpisode],
            num_next_steps: int,
            requires_previous_actions: bool = False,
    ):
        if num_next_steps < 1:
            raise ValueError(f'num_next_steps must be >= 1, got {num_next_steps}')
        super().__init__(
            episodes=episodes,
            requires_previous_actions=requires_previous_actions,
        )
        
        multi_step_actions_list = []
        next_local_obs_list = []
        next_validity_mask_list = []
        next_global_obs_list = []
        wm_agent_mask_list: list[torch.Tensor] = []
        wm_loss_agent_mask_list: list[torch.Tensor] = []
        has_agent_mask = self.agent_mask is not None

        padding_len = num_next_steps - 1

        for ep in episodes:
            num_steps = ep.local_obs.shape[0]
            assert ep.final_local_obs is not None
            assert ep.final_global_obs is not None
            assert ep.actions.shape[0] == num_steps
            if has_agent_mask and ep.final_agent_mask is None:
                raise ValueError("final_agent_mask must be provided when agent_mask is enabled")

            next_obs = torch.cat([ep.local_obs[1:], ep.final_local_obs.unsqueeze(0)], dim=0)
            next_global_obs = torch.cat([ep.global_obs[1:], ep.final_global_obs.unsqueeze(0)], dim=0)
            wm_agent_mask: torch.Tensor | None = None
            wm_loss_agent_mask: torch.Tensor | None = None
            if has_agent_mask:
                if ep.agent_mask is None:
                    raise ValueError("agent_mask must be provided when agent_mask is enabled")
                wm_agent_mask = ep.agent_mask
                wm_loss_agent_mask = torch.cat(
                    [ep.agent_mask[1:], ep.final_agent_mask.unsqueeze(0)],
                    dim=0,
                )
            
            if padding_len > 0:
                padding = torch.zeros(
                    (padding_len, *next_obs.shape[1:]),
                    dtype=next_obs.dtype,
                    device=next_obs.device
                )
                padded_obs = torch.cat([next_obs, padding], dim=0)
                global_padding = torch.zeros(
                    (padding_len, *next_global_obs.shape[1:]),
                    dtype=next_global_obs.dtype,
                    device=next_global_obs.device,
                )
                padded_global_obs = torch.cat([next_global_obs, global_padding], dim=0)
                padded_wm_agent_mask: torch.Tensor | None = None
                padded_wm_loss_agent_mask: torch.Tensor | None = None
                if has_agent_mask:
                    if wm_agent_mask is None:
                        raise ValueError("wm_agent_mask must be set when agent_mask is enabled")
                    if wm_loss_agent_mask is None:
                        raise ValueError("wm_loss_agent_mask must be set when agent_mask is enabled")
                    wm_agent_mask_padding = torch.ones(
                        (padding_len, *wm_agent_mask.shape[1:]),
                        dtype=wm_agent_mask.dtype,
                        device=wm_agent_mask.device,
                    )
                    padded_wm_agent_mask = torch.cat([wm_agent_mask, wm_agent_mask_padding], dim=0)
                    agent_mask_padding = torch.ones(
                        (padding_len, *wm_loss_agent_mask.shape[1:]),
                        dtype=wm_loss_agent_mask.dtype,
                        device=wm_loss_agent_mask.device,
                    )
                    padded_wm_loss_agent_mask = torch.cat([wm_loss_agent_mask, agent_mask_padding], dim=0)
            else:
                padded_obs = next_obs
                padded_global_obs = next_global_obs
                padded_wm_agent_mask = None
                padded_wm_loss_agent_mask = None
                if has_agent_mask:
                    if wm_agent_mask is None:
                        raise ValueError("wm_agent_mask must be set when agent_mask is enabled")
                    if wm_loss_agent_mask is None:
                        raise ValueError("wm_loss_agent_mask must be set when agent_mask is enabled")
                    padded_wm_agent_mask = wm_agent_mask
                    padded_wm_loss_agent_mask = wm_loss_agent_mask

            windows = padded_obs.unfold(0, num_next_steps, 1)[:num_steps]  # (T, N, F, k)
            
            ep_next_obs = windows.permute(0, 3, 1, 2)  # (T, k, N, F)
            next_local_obs_list.append(ep_next_obs)
            
            global_windows = padded_global_obs.unfold(0, num_next_steps, 1)[:num_steps]  # (T, G, k)
            ep_next_global_obs = global_windows.permute(0, 2, 1)  # (T, k, G)
            next_global_obs_list.append(ep_next_global_obs)
            if has_agent_mask:
                if padded_wm_agent_mask is None:
                    raise ValueError("padded_wm_agent_mask must be set when agent_mask is enabled")
                if padded_wm_loss_agent_mask is None:
                    raise ValueError("padded_wm_loss_agent_mask must be set when agent_mask is enabled")
                wm_agent_mask_windows = padded_wm_agent_mask.unfold(0, num_next_steps, 1)[:num_steps]  # (T, N, k)
                wm_agent_mask_list.append(wm_agent_mask_windows.permute(0, 2, 1))  # (T, k, N)
                wm_loss_agent_mask_windows = padded_wm_loss_agent_mask.unfold(0, num_next_steps, 1)[:num_steps]  # (T, N, k)
                wm_loss_agent_mask_list.append(wm_loss_agent_mask_windows.permute(0, 2, 1))  # (T, k, N)

            if padding_len > 0:
                action_padding = torch.zeros(
                    (padding_len, *ep.actions.shape[1:]),
                    dtype=ep.actions.dtype,
                    device=ep.actions.device,
                )
                padded_actions = torch.cat([ep.actions, action_padding], dim=0)
            else:
                padded_actions = ep.actions

            action_windows = padded_actions.unfold(0, num_next_steps, 1)  # (T_out, N, A, k)
            ep_multi_step_actions = action_windows[:num_steps].permute(0, 3, 1, 2)  # (T, k, N, A)
            multi_step_actions_list.append(ep_multi_step_actions)
            
            validity = torch.ones(next_obs.shape[0], dtype=torch.bool, device=next_obs.device)
            if padding_len > 0:
                validity_padding = torch.zeros(padding_len, dtype=torch.bool, device=next_obs.device)
                padded_validity = torch.cat([validity, validity_padding], dim=0)
            else:
                padded_validity = validity
                
            validity_windows = padded_validity.unfold(0, num_next_steps, 1)[:num_steps]  # (T, k)
            next_validity_mask_list.append(validity_windows)

        self.multi_step_actions = torch.cat(multi_step_actions_list, dim=0).contiguous()
        self.next_local_obs = torch.cat(next_local_obs_list, dim=0).contiguous()
        self.next_validity_mask = torch.cat(next_validity_mask_list, dim=0).contiguous()
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
            actions=self.multi_step_actions[batch_indices],
            log_probs=self.log_probs[batch_indices],
            values=self.values[batch_indices],
            returns=self.returns[batch_indices],
            advantages=self.advantages[batch_indices],
            next_local_obs=self.next_local_obs[batch_indices],
            next_validity_mask=self.next_validity_mask[batch_indices],
            next_global_obs=self.next_global_obs[batch_indices],
            wm_agent_mask=None if self.wm_agent_mask is None else self.wm_agent_mask[batch_indices],
            wm_loss_agent_mask=None if self.wm_loss_agent_mask is None else self.wm_loss_agent_mask[batch_indices],
        )
