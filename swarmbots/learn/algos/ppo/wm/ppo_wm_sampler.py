from dataclasses import dataclass

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOSampler, PPOEpisode, PPOSamples


@dataclass
class PPOWMSamples(PPOSamples):
    actions: torch.Tensor  # (batch, n_next_steps, n_agents, n_actions)

    next_local_obs: torch.Tensor  # shape (batch, n_next_steps, n_agents, n_obs_features)
    next_validity_mask: torch.Tensor  # shape (batch, n_next_steps)
    next_global_obs: torch.Tensor  # shape (batch, n_next_steps, n_global_obs_features)


class PPOWMSampler(PPOSampler[PPOWMSamples]):

    def __init__(
            self,
            episodes: list[PPOEpisode],
            num_next_steps: int,
    ):
        if num_next_steps < 1:
            raise ValueError(f'num_next_steps must be >= 1, got {num_next_steps}')
        super().__init__(episodes)
        
        multi_step_actions_list = []
        next_local_obs_list = []
        next_validity_mask_list = []
        next_global_obs_list = []

        padding_len = num_next_steps - 1

        for ep in episodes:
            num_steps = ep.local_obs.shape[0]
            assert ep.final_local_obs is not None
            assert ep.final_global_obs is not None
            assert ep.actions.shape[0] == num_steps

            next_obs = torch.cat([ep.local_obs[1:], ep.final_local_obs.unsqueeze(0)], dim=0)
            next_global_obs = torch.cat([ep.global_obs[1:], ep.final_global_obs.unsqueeze(0)], dim=0)
            
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
            else:
                padded_obs = next_obs
                padded_global_obs = next_global_obs

            windows = padded_obs.unfold(0, num_next_steps, 1)[:num_steps]  # (T, N, F, k)
            
            ep_next_obs = windows.permute(0, 3, 1, 2)  # (T, k, N, F)
            next_local_obs_list.append(ep_next_obs)
            
            global_windows = padded_global_obs.unfold(0, num_next_steps, 1)[:num_steps]  # (T, G, k)
            ep_next_global_obs = global_windows.permute(0, 2, 1)  # (T, k, G)
            next_global_obs_list.append(ep_next_global_obs)

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

    def _fetch_samples(self, batch_indices: torch.Tensor) -> PPOWMSamples:
        return PPOWMSamples(
            local_obs=self.local_obs[batch_indices],
            global_obs=self.global_obs[batch_indices],
            actions=self.multi_step_actions[batch_indices],
            log_probs=self.log_probs[batch_indices],
            values=self.values[batch_indices],
            returns=self.returns[batch_indices],
            advantages=self.advantages[batch_indices],
            next_local_obs=self.next_local_obs[batch_indices],
            next_validity_mask=self.next_validity_mask[batch_indices],
            next_global_obs=self.next_global_obs[batch_indices],
        )