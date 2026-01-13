from dataclasses import dataclass

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOSampler, PPOEpisode


@dataclass
class PPOWMSamples:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor

    next_local_obs: torch.Tensor  # shape (batch, n_next_steps, n_agents, n_obs_features)
    next_local_obs_mask: torch.Tensor  # shape (batch, n_next_steps)

class PPOWMSampler(PPOSampler[PPOWMSamples]):

    def __init__(
            self,
            episodes: list[PPOEpisode],
            num_next_steps: int,
    ):
        super().__init__(episodes)
        
        next_local_obs_list = []
        next_local_obs_mask_list = []

        for ep in episodes:
            assert ep.final_local_obs is not None
            # T steps in episode. local_obs is (T, N, F)
            # We want predictions for t=0..T-1
            # Targets are at t+1..t+k
            
            # Construct full observation sequence: [obs_0, ..., obs_{T-1}, obs_T]
            # obs_T is final_local_obs
            all_obs = torch.cat([ep.local_obs, ep.final_local_obs.unsqueeze(0)], dim=0)
            
            # Pad with zeros for steps beyond T
            # We need to be able to access indices up to (T-1) + num_next_steps
            # all_obs has indices 0..T
            # Max index needed: T - 1 + num_next_steps
            # Current max index: T
            # Padding needed: (T - 1 + num_next_steps) - T = num_next_steps - 1
            
            padding_len = num_next_steps - 1
            if padding_len > 0:
                padding = torch.zeros(
                    (padding_len, *all_obs.shape[1:]),
                    dtype=all_obs.dtype,
                    device=all_obs.device
                )
                padded_obs = torch.cat([all_obs, padding], dim=0)
            else:
                padded_obs = all_obs

            # We want windows starting at t+1 for t in 0..T-1
            # i.e. starting at index 1, length T
            # window size = num_next_steps
            
            # unfold(dim, size, step)
            # shape: (batch, N, F) -> (batch_windows, N, F, window_size)
            # We take slice [1:] to start from t=1
            windows = padded_obs[1:].unfold(0, num_next_steps, 1)
            # windows shape: (T_out, N, F, num_next_steps)
            # We only need T steps (corresponding to t=0..T-1)
            T = ep.local_obs.shape[0]
            windows = windows[:T]
            
            # Permute to (T, num_next_steps, N, F)
            ep_next_obs = windows.permute(0, 3, 1, 2).contiguous()
            next_local_obs_list.append(ep_next_obs)
            
            # Create mask
            # Valid indices are 0..T
            validity = torch.ones(all_obs.shape[0], dtype=torch.bool, device=all_obs.device)
            if padding_len > 0:
                validity_padding = torch.zeros(padding_len, dtype=torch.bool, device=all_obs.device)
                padded_validity = torch.cat([validity, validity_padding], dim=0)
            else:
                padded_validity = validity
                
            validity_windows = padded_validity[1:].unfold(0, num_next_steps, 1)
            ep_mask = validity_windows[:T].contiguous() # (T, num_next_steps)
            next_local_obs_mask_list.append(ep_mask)

        self.next_local_obs = torch.cat(next_local_obs_list, dim=0)
        self.next_local_obs_mask = torch.cat(next_local_obs_mask_list, dim=0)

    def _fetch_samples(self, batch_indices: torch.Tensor) -> PPOWMSamples:
        return PPOWMSamples(
            local_obs=self.local_obs[batch_indices],
            global_obs=self.global_obs[batch_indices],
            actions=self.actions[batch_indices],
            log_probs=self.log_probs[batch_indices],
            values=self.values[batch_indices],
            returns=self.returns[batch_indices],
            advantages=self.advantages[batch_indices],
            next_local_obs=self.next_local_obs[batch_indices],
            next_local_obs_mask=self.next_local_obs_mask[batch_indices],
        )