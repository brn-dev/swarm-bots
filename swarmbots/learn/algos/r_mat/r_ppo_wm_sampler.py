from dataclasses import dataclass

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import MaybeTensor, PPOEpisodeSegment
from swarmbots.learn.algos.world_modeling.base_wm_sampler import BaseWMSampler, BaseWMSamples
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSamplerConfig
from swarmbots.learn.algos.world_modeling.wm_sampler_helper import build_wm_episode_windows, pad_time_axis
from swarmbots.learn.base_sampler import BaseSampler


@dataclass
class RPPOWMSamples(BaseWMSamples):
    local_obs: torch.Tensor  # (batch, sequence_length, n_agents, n_local_obs_features)
    global_obs: torch.Tensor  # (batch, sequence_length, n_global_obs_features)
    hidden_local_vars: torch.Tensor  # (batch, sequence_length, n_agents, n_hidden_local_vars)
    hidden_global_vars: torch.Tensor  # (batch, sequence_length, n_hidden_global_vars)
    agent_mask: MaybeTensor  # (batch, sequence_length, n_agents)
    previous_actions: MaybeTensor  # (batch, sequence_length, n_agents, n_actions)
    actions: torch.Tensor  # (batch, sequence_length, n_agents, n_actions)
    wm_actions: torch.Tensor  # (batch, sequence_length, n_next_steps, n_agents, n_actions)
    log_probs: torch.Tensor  # (batch, sequence_length, n_agents)
    values: torch.Tensor  # (batch, sequence_length)
    returns: torch.Tensor  # (batch, sequence_length)
    advantages: torch.Tensor  # (batch, sequence_length)
    time_mask: torch.Tensor  # (batch, sequence_length)
    time_loss_mask: torch.Tensor  # (batch, sequence_length)
    is_true_episode_start: torch.Tensor  # (batch,)

    next_local_obs: torch.Tensor  # (batch, sequence_length, n_next_steps, n_agents, n_local_obs_features)
    wm_target_time_mask: torch.Tensor  # (batch, sequence_length, n_next_steps)
    next_global_obs: torch.Tensor  # (batch, sequence_length, n_next_steps, n_global_obs_features)
    wm_agent_mask: MaybeTensor  # (batch, sequence_length, n_next_steps, n_agents)
    wm_loss_agent_mask: MaybeTensor  # (batch, sequence_length, n_next_steps, n_agents)


@dataclass(frozen=True)
class RPPOWMSamplerConfig(PPOWMSamplerConfig):
    sequence_length: int
    burn_in_length: int = 0


class RPPOWMSampler(
    BaseSampler[RPPOWMSamples, RPPOWMSamplerConfig],
    BaseWMSampler[RPPOWMSamples, RPPOWMSamplerConfig],
):

    def __init__(
            self,
            episodes: list[PPOEpisodeSegment],
            config: RPPOWMSamplerConfig,
            requires_previous_actions: bool = False,
    ):
        if config.sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {config.sequence_length}")
        if config.burn_in_length < 0:
            raise ValueError(f"burn_in_length must be >= 0, got {config.burn_in_length}")
        if config.burn_in_length >= config.sequence_length:
            raise ValueError(
                f"burn_in_length must be < sequence_length, got {config.burn_in_length} >= {config.sequence_length}"
            )
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
        time_loss_mask_chunks: list[torch.Tensor] = []
        is_true_episode_start_chunks: list[torch.Tensor] = []
        next_local_obs_chunks: list[torch.Tensor] = []
        wm_target_time_mask_chunks: list[torch.Tensor] = []
        next_global_obs_chunks: list[torch.Tensor] = []
        wm_agent_mask_chunks: list[torch.Tensor] = []
        wm_loss_agent_mask_chunks: list[torch.Tensor] = []

        sequence_length = config.sequence_length
        burn_in_length = config.burn_in_length
        train_length = sequence_length - burn_in_length

        for episode in episodes:
            num_steps = int(episode.local_obs.shape[0])
            if num_steps == 0:
                continue

            episode_windows = build_wm_episode_windows(
                episode,
                num_next_steps=config.num_next_steps,
            )
            previous_actions = _build_previous_actions(episode) if requires_previous_actions else None

            train_start_idx = 0
            while train_start_idx < num_steps:
                start_idx = 0 if train_start_idx == 0 else train_start_idx - burn_in_length
                chunk_length = min(sequence_length, num_steps - start_idx)
                if train_start_idx == 0 and not episode.is_true_episode_start:
                    loss_start_idx = min(chunk_length, burn_in_length)
                else:
                    loss_start_idx = train_start_idx - start_idx
                time_mask_chunks.append(
                    _build_time_mask(chunk_length, sequence_length, episode.local_obs.device)
                )
                time_loss_mask_chunks.append(
                    _build_time_loss_mask(
                        chunk_length=chunk_length,
                        sequence_length=sequence_length,
                        loss_start_idx=loss_start_idx,
                        device=episode.local_obs.device,
                    )
                )
                is_true_episode_start_chunks.append(
                    torch.tensor(
                        train_start_idx == 0 and episode.is_true_episode_start,
                        dtype=torch.bool,
                        device=episode.local_obs.device,
                    )
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
                wm_target_time_mask_chunks.append(
                    _slice_time_chunk(
                        episode_windows.wm_target_time_mask,
                        start_idx,
                        sequence_length,
                        pad_value=False,
                    )
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
                if train_start_idx == 0:
                    train_start_idx = chunk_length
                else:
                    train_start_idx += train_length

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
        self.time_loss_mask = torch.stack(time_loss_mask_chunks, dim=0).contiguous()
        self.is_true_episode_start = torch.stack(is_true_episode_start_chunks, dim=0).contiguous()
        self.next_local_obs = torch.stack(next_local_obs_chunks, dim=0).contiguous()
        self.wm_target_time_mask = torch.stack(wm_target_time_mask_chunks, dim=0).contiguous()
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
            time_loss_mask=self.time_loss_mask[batch_indices],
            is_true_episode_start=self.is_true_episode_start[batch_indices],
            next_local_obs=self.next_local_obs[batch_indices],
            wm_target_time_mask=self.wm_target_time_mask[batch_indices],
            next_global_obs=self.next_global_obs[batch_indices],
            wm_agent_mask=None if self.wm_agent_mask is None else self.wm_agent_mask[batch_indices],
            wm_loss_agent_mask=None if self.wm_loss_agent_mask is None else self.wm_loss_agent_mask[batch_indices],
        )


def _build_previous_actions(episode: PPOEpisodeSegment) -> torch.Tensor:
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


def _build_time_loss_mask(
        *,
        chunk_length: int,
        sequence_length: int,
        loss_start_idx: int,
        device: torch.device,
) -> torch.Tensor:
    mask = torch.zeros(sequence_length, dtype=torch.bool, device=device)
    mask[loss_start_idx:chunk_length] = True
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
