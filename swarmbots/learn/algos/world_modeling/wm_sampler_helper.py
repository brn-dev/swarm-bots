from dataclasses import dataclass

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import MaybeTensor, PPOEpisodeSegment


@dataclass(slots=True)
class WMEpisodeWindows:
    multi_step_actions: torch.Tensor
    next_local_obs: torch.Tensor
    wm_target_time_mask: torch.Tensor
    next_global_obs: torch.Tensor
    wm_agent_mask: MaybeTensor
    wm_loss_agent_mask: MaybeTensor


def pad_time_axis(
        tensor: torch.Tensor,
        *,
        padding_len: int,
        pad_value: bool | int | float = 0,
) -> torch.Tensor:
    if padding_len < 0:
        raise ValueError(f"padding_len must be >= 0, got {padding_len}")
    if padding_len == 0:
        return tensor

    padding = torch.full(
        (padding_len, *tensor.shape[1:]),
        fill_value=pad_value,
        dtype=tensor.dtype,
        device=tensor.device,
    )
    return torch.cat((tensor, padding), dim=0)


def build_wm_episode_windows(
        episode: PPOEpisodeSegment,
        *,
        num_next_steps: int,
) -> WMEpisodeWindows:
    if num_next_steps < 1:
        raise ValueError(f"num_next_steps must be >= 1, got {num_next_steps}")

    num_steps = int(episode.local_obs.shape[0])
    if episode.actions.shape[0] != num_steps:
        raise ValueError(
            f"Expected actions to have {num_steps} timesteps, got {episode.actions.shape[0]}"
        )
    if episode.global_obs.shape[0] != num_steps:
        raise ValueError(
            f"Expected global_obs to have {num_steps} timesteps, got {episode.global_obs.shape[0]}"
        )
    if episode.final_local_obs is None:
        raise ValueError("final_local_obs must be provided to build world-model windows")
    if episode.final_global_obs is None:
        raise ValueError("final_global_obs must be provided to build world-model windows")

    has_agent_mask = episode.agent_mask is not None
    if has_agent_mask != (episode.final_agent_mask is not None):
        raise ValueError("agent_mask and final_agent_mask must either both be set or both be None")

    next_local_obs = torch.cat((episode.local_obs[1:], episode.final_local_obs.unsqueeze(0)), dim=0)
    next_global_obs = torch.cat((episode.global_obs[1:], episode.final_global_obs.unsqueeze(0)), dim=0)

    wm_agent_mask: MaybeTensor = None
    wm_loss_agent_mask: MaybeTensor = None
    if has_agent_mask:
        assert episode.agent_mask is not None
        assert episode.final_agent_mask is not None
        wm_agent_mask = episode.agent_mask
        wm_loss_agent_mask = torch.cat(
            (episode.agent_mask[1:], episode.final_agent_mask.unsqueeze(0)),
            dim=0,
        )

    padding_len = num_next_steps - 1
    padded_actions = pad_time_axis(episode.actions, padding_len=padding_len, pad_value=0)
    padded_next_local_obs = pad_time_axis(next_local_obs, padding_len=padding_len, pad_value=0)
    padded_next_global_obs = pad_time_axis(next_global_obs, padding_len=padding_len, pad_value=0)
    padded_validity = pad_time_axis(
        torch.ones(num_steps, dtype=torch.bool, device=episode.local_obs.device),
        padding_len=padding_len,
        pad_value=False,
    )

    padded_wm_agent_mask: MaybeTensor = None
    padded_wm_loss_agent_mask: MaybeTensor = None
    if has_agent_mask:
        assert wm_agent_mask is not None
        assert wm_loss_agent_mask is not None
        padded_wm_agent_mask = pad_time_axis(wm_agent_mask, padding_len=padding_len, pad_value=True)
        padded_wm_loss_agent_mask = pad_time_axis(wm_loss_agent_mask, padding_len=padding_len, pad_value=True)

    return WMEpisodeWindows(
        multi_step_actions=_build_time_windows(padded_actions, window_size=num_next_steps, num_steps=num_steps),
        next_local_obs=_build_time_windows(padded_next_local_obs, window_size=num_next_steps, num_steps=num_steps),
        wm_target_time_mask=_build_time_windows(padded_validity, window_size=num_next_steps, num_steps=num_steps),
        next_global_obs=_build_time_windows(padded_next_global_obs, window_size=num_next_steps, num_steps=num_steps),
        wm_agent_mask=(
            None
            if padded_wm_agent_mask is None
            else _build_time_windows(padded_wm_agent_mask, window_size=num_next_steps, num_steps=num_steps)
        ),
        wm_loss_agent_mask=(
            None
            if padded_wm_loss_agent_mask is None
            else _build_time_windows(padded_wm_loss_agent_mask, window_size=num_next_steps, num_steps=num_steps)
        ),
    )


def _build_time_windows(
        tensor: torch.Tensor,
        *,
        window_size: int,
        num_steps: int,
) -> torch.Tensor:
    windows = tensor.unfold(0, window_size, 1)[:num_steps]
    if tensor.ndim == 1:
        return windows.contiguous()

    permute_dims = (0, tensor.ndim, *range(1, tensor.ndim))
    return windows.permute(*permute_dims).contiguous()
