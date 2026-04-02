from dataclasses import dataclass

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import MaybeTensor, PPOEpisode
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
from swarmbots.learn.algos.world_modeling.wm_sampler_helper import build_wm_episode_windows, pad_time_axis
from swarmbots.learn.base_sampler import BaseSampler


@dataclass
class RPPOWMSamples:
    local_obs: torch.Tensor
    global_obs: torch.Tensor
    hidden_local_vars: torch.Tensor
    hidden_global_vars: torch.Tensor
    agent_mask: MaybeTensor
    previous_actions: MaybeTensor
    actions: torch.Tensor
    wm_actions: torch.Tensor
    log_probs: torch.Tensor
    values: torch.Tensor
    returns: torch.Tensor
    advantages: torch.Tensor
    time_mask: torch.Tensor
    segment_starts: torch.Tensor

    next_local_obs: torch.Tensor
    next_validity_mask: torch.Tensor
    next_global_obs: torch.Tensor
    wm_agent_mask: torch.Tensor | None
    wm_loss_agent_mask: torch.Tensor | None


@dataclass(frozen=True)
class RPPOWMSamplerConfig(PPOWMSamplerConfig):
    sequence_length: int


class RPPOWMSampler(BaseSampler[RPPOWMSamples, RPPOWMSamplerConfig]):

    def __init__(
            self,
            episodes: list[PPOEpisode],
            config: RPPOWMSamplerConfig,
            requires_previous_actions: bool = False,
    ):
        if config.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {config.sequence_length}")
        if config.num_next_steps < 1:
            raise ValueError(f"num_next_steps must be >= 1, got {config.num_next_steps}")

        has_agent_mask = any(ep.agent_mask is not None for ep in episodes)
        has_missing_agent_mask = any(ep.agent_mask is None for ep in episodes)
        if has_agent_mask and has_missing_agent_mask:
            raise ValueError("agent_mask must be provided for all episodes or none")

        local_obs_chunks: list[torch.Tensor] = []
        global_obs_chunks: list[torch.Tensor] = []
        hidden_local_vars_chunks: list[torch.Tensor] = []
        hidden_global_vars_chunks: list[torch.Tensor] = []
        agent_mask_chunks: list[torch.Tensor] = []
        previous_actions_chunks: list[torch.Tensor] = []
        actions_chunks: list[torch.Tensor] = []
        wm_actions_chunks: list[torch.Tensor] = []
        log_probs_chunks: list[torch.Tensor] = []
        values_chunks: list[torch.Tensor] = []
        returns_chunks: list[torch.Tensor] = []
        advantages_chunks: list[torch.Tensor] = []
        time_mask_chunks: list[torch.Tensor] = []
        segment_start_chunks: list[torch.Tensor] = []
        next_local_obs_chunks: list[torch.Tensor] = []
        next_validity_mask_chunks: list[torch.Tensor] = []
        next_global_obs_chunks: list[torch.Tensor] = []
        wm_agent_mask_chunks: list[torch.Tensor] = []
        wm_loss_agent_mask_chunks: list[torch.Tensor] = []

        sequence_length = config.sequence_length

        for episode in episodes:
            num_steps = int(episode.local_obs.shape[0])
            if num_steps == 0:
                continue

            episode_windows = build_wm_episode_windows(
                episode,
                num_next_steps=config.num_next_steps,
            )
            previous_actions = _build_previous_actions(episode) if requires_previous_actions else None

            for start_idx in range(0, num_steps, sequence_length):
                chunk_length = min(sequence_length, num_steps - start_idx)
                time_mask_chunks.append(_build_time_mask(chunk_length, sequence_length, episode.local_obs.device))
                segment_start_chunks.append(
                    torch.tensor(start_idx == 0, dtype=torch.bool, device=episode.local_obs.device)
                )

                local_obs_chunks.append(_slice_time_chunk(episode.local_obs, start_idx, sequence_length, pad_value=0))
                global_obs_chunks.append(_slice_time_chunk(episode.global_obs, start_idx, sequence_length, pad_value=0))
                hidden_local_vars_chunks.append(
                    _slice_time_chunk(episode.hidden_local_vars, start_idx, sequence_length, pad_value=0)
                )
                hidden_global_vars_chunks.append(
                    _slice_time_chunk(episode.hidden_global_vars, start_idx, sequence_length, pad_value=0)
                )
                actions_chunks.append(_slice_time_chunk(episode.actions, start_idx, sequence_length, pad_value=0))
                wm_actions_chunks.append(
                    _slice_time_chunk(episode_windows.multi_step_actions, start_idx, sequence_length, pad_value=0)
                )
                log_probs_chunks.append(_slice_time_chunk(episode.log_probs, start_idx, sequence_length, pad_value=0))
                values_chunks.append(_slice_time_chunk(episode.values, start_idx, sequence_length, pad_value=0))
                returns_chunks.append(_slice_time_chunk(episode.returns, start_idx, sequence_length, pad_value=0))
                advantages_chunks.append(_slice_time_chunk(episode.advantages, start_idx, sequence_length, pad_value=0))
                next_local_obs_chunks.append(
                    _slice_time_chunk(episode_windows.next_local_obs, start_idx, sequence_length, pad_value=0)
                )
                next_validity_mask_chunks.append(
                    _slice_time_chunk(episode_windows.next_validity_mask, start_idx, sequence_length, pad_value=False)
                )
                next_global_obs_chunks.append(
                    _slice_time_chunk(episode_windows.next_global_obs, start_idx, sequence_length, pad_value=0)
                )

                if has_agent_mask:
                    if episode.agent_mask is None:
                        raise ValueError("agent_mask must be provided when agent masks are enabled")
                    if episode_windows.wm_agent_mask is None:
                        raise ValueError("wm_agent_mask must be provided when agent masks are enabled")
                    if episode_windows.wm_loss_agent_mask is None:
                        raise ValueError("wm_loss_agent_mask must be provided when agent masks are enabled")
                    agent_mask_chunks.append(
                        _slice_time_chunk(episode.agent_mask, start_idx, sequence_length, pad_value=True)
                    )
                    wm_agent_mask_chunks.append(
                        _slice_time_chunk(episode_windows.wm_agent_mask, start_idx, sequence_length, pad_value=True)
                    )
                    wm_loss_agent_mask_chunks.append(
                        _slice_time_chunk(
                            episode_windows.wm_loss_agent_mask,
                            start_idx,
                            sequence_length,
                            pad_value=True,
                        )
                    )

                if previous_actions is not None:
                    previous_actions_chunks.append(
                        _slice_time_chunk(previous_actions, start_idx, sequence_length, pad_value=0)
                    )

        if not local_obs_chunks:
            raise ValueError("RPPOWMSampler requires at least one non-empty episode segment")

        self.local_obs = torch.stack(local_obs_chunks, dim=0).contiguous()
        self.global_obs = torch.stack(global_obs_chunks, dim=0).contiguous()
        self.hidden_local_vars = torch.stack(hidden_local_vars_chunks, dim=0).contiguous()
        self.hidden_global_vars = torch.stack(hidden_global_vars_chunks, dim=0).contiguous()
        self.agent_mask = torch.stack(agent_mask_chunks, dim=0).contiguous() if has_agent_mask else None
        self.previous_actions = (
            torch.stack(previous_actions_chunks, dim=0).contiguous()
            if requires_previous_actions else None
        )
        self.actions = torch.stack(actions_chunks, dim=0).contiguous()
        self.wm_actions = torch.stack(wm_actions_chunks, dim=0).contiguous()
        self.log_probs = torch.stack(log_probs_chunks, dim=0).contiguous()
        self.values = torch.stack(values_chunks, dim=0).contiguous()
        self.returns = torch.stack(returns_chunks, dim=0).contiguous()
        self.advantages = torch.stack(advantages_chunks, dim=0).contiguous()
        self.time_mask = torch.stack(time_mask_chunks, dim=0).contiguous()
        self.segment_starts = torch.stack(segment_start_chunks, dim=0).contiguous()
        self.next_local_obs = torch.stack(next_local_obs_chunks, dim=0).contiguous()
        self.next_validity_mask = torch.stack(next_validity_mask_chunks, dim=0).contiguous()
        self.next_global_obs = torch.stack(next_global_obs_chunks, dim=0).contiguous()
        self.wm_agent_mask = torch.stack(wm_agent_mask_chunks, dim=0).contiguous() if has_agent_mask else None
        self.wm_loss_agent_mask = (
            torch.stack(wm_loss_agent_mask_chunks, dim=0).contiguous() if has_agent_mask else None
        )

        super().__init__(config=config, n_samples=self.local_obs.shape[0])

    def _fetch_samples(self, batch_indices: torch.Tensor) -> RPPOWMSamples:
        return RPPOWMSamples(
            local_obs=self.local_obs[batch_indices],
            global_obs=self.global_obs[batch_indices],
            hidden_local_vars=self.hidden_local_vars[batch_indices],
            hidden_global_vars=self.hidden_global_vars[batch_indices],
            agent_mask=None if self.agent_mask is None else self.agent_mask[batch_indices],
            previous_actions=None if self.previous_actions is None else self.previous_actions[batch_indices],
            actions=self.actions[batch_indices],
            wm_actions=self.wm_actions[batch_indices],
            log_probs=self.log_probs[batch_indices],
            values=self.values[batch_indices],
            returns=self.returns[batch_indices],
            advantages=self.advantages[batch_indices],
            time_mask=self.time_mask[batch_indices],
            segment_starts=self.segment_starts[batch_indices],
            next_local_obs=self.next_local_obs[batch_indices],
            next_validity_mask=self.next_validity_mask[batch_indices],
            next_global_obs=self.next_global_obs[batch_indices],
            wm_agent_mask=None if self.wm_agent_mask is None else self.wm_agent_mask[batch_indices],
            wm_loss_agent_mask=None if self.wm_loss_agent_mask is None else self.wm_loss_agent_mask[batch_indices],
        )


def _build_previous_actions(episode: PPOEpisode) -> torch.Tensor:
    if episode.initial_previous_actions is None:
        return torch.zeros_like(episode.actions)
    return torch.cat((episode.initial_previous_actions.unsqueeze(0), episode.actions[:-1]), dim=0)


def _build_time_mask(
        chunk_length: int,
        sequence_length: int,
        device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    mask[:chunk_length] = True
    return mask


def _slice_time_chunk(
        tensor: torch.Tensor,
        start_idx: int,
        sequence_length: int,
        *,
        pad_value: bool | int | float,
) -> torch.Tensor:
    chunk = tensor[start_idx:start_idx + sequence_length]
    return pad_time_axis(
        chunk,
        padding_len=sequence_length - chunk.shape[0],
        pad_value=pad_value,
    )
