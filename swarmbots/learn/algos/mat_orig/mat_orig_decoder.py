from dataclasses import dataclass

import torch
from torch import nn

from swarmbots.learn.algos.mat_orig.mat_orig_encoder import MATOrigSelfAttention
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class MATOrigDecodeBlock(nn.Module):

    def __init__(
            self,
            d_model: int,
            nhead: int,
            *,
            max_agents: int,
            dim_feedforward: int | None = None,
            transformer_ff_init_gain: float | None = 1.0,
    ) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ln3 = nn.LayerNorm(d_model)
        self.attn1 = MATOrigSelfAttention(d_model, nhead, max_agents=max_agents, masked=True)
        self.attn2 = MATOrigSelfAttention(d_model, nhead, max_agents=max_agents, masked=True)
        feedforward_dim = d_model if dim_feedforward is None else dim_feedforward
        self.mlp = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Linear(feedforward_dim, d_model),
        )
        if transformer_ff_init_gain is not None:
            init_linear_orthogonal(self.mlp[0], gain=transformer_ff_init_gain)
            init_linear_orthogonal(self.mlp[2], gain=1.0)

    def forward(
            self,
            x: torch.Tensor,
            obs_rep: torch.Tensor,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self.ln1(
            x + self.attn1(
                x,
                x,
                x,
                key_padding_mask=agent_mask,
                query_padding_mask=agent_mask,
            )
        )
        x = self.ln2(
            obs_rep + self.attn2(
                key=x,
                value=x,
                query=obs_rep,
                key_padding_mask=agent_mask,
                query_padding_mask=agent_mask,
            )
        )
        x = self.ln3(x + self.mlp(x))
        if agent_mask is not None:
            x = x.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        return x


@dataclass(frozen=True)
class MATOrigDecoderConfig:
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    latent_pi_dim: int | None = None
    dim_feedforward: int | None = None
    token_encoder_init_gain: float = 1.0
    transformer_ff_init_gain: float | None = 1.0
    actor_head_init_gain: float = 1.0


class MATOrigDecoder(nn.Module):

    def __init__(
            self,
            config: MATOrigDecoderConfig,
            *,
            max_agents: int,
            action_input_dim: int,
    ) -> None:
        super().__init__()
        self.max_agents = max_agents
        self.d_model = config.d_model
        self.action_input_dim = action_input_dim
        self.latent_pi_dim = config.d_model if config.latent_pi_dim is None else config.latent_pi_dim

        self.action_encoder = nn.Sequential(
            init_linear_orthogonal(nn.Linear(action_input_dim, config.d_model), gain=config.token_encoder_init_gain),
            nn.GELU(),
        )
        self.ln = nn.LayerNorm(config.d_model)
        self.blocks = nn.ModuleList([
            MATOrigDecodeBlock(
                config.d_model, config.nhead, max_agents=max_agents, dim_feedforward=config.dim_feedforward,
                transformer_ff_init_gain=config.transformer_ff_init_gain,
            )
            for _ in range(config.num_layers)
        ])
        self.head = nn.Sequential(
            init_linear_orthogonal(nn.Linear(config.d_model, config.d_model), gain=config.actor_head_init_gain),
            nn.GELU(),
            nn.LayerNorm(config.d_model),
            init_linear_orthogonal(nn.Linear(config.d_model, self.latent_pi_dim), gain=config.actor_head_init_gain),
        )

    def forward(
            self,
            shifted_actions: torch.Tensor,
            obs_rep: torch.Tensor,
            *,
            agent_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if shifted_actions.shape[1] > self.max_agents:
            raise ValueError(
                f"Expected shifted_actions second dim <= {self.max_agents}, got {shifted_actions.shape[1]}"
            )
        if shifted_actions.shape[:2] != obs_rep.shape[:2]:
            raise ValueError(
                "Expected shifted_actions and obs_rep to match in batch/agent dims, "
                f"got {tuple(shifted_actions.shape)} and {tuple(obs_rep.shape)}"
            )

        padding_mask = None if agent_mask is None else ~agent_mask.unsqueeze(-1)
        if padding_mask is not None:
            shifted_actions = shifted_actions.masked_fill(padding_mask, 0.0)
            obs_rep = obs_rep.masked_fill(padding_mask, 0.0)
        x = self.action_encoder(shifted_actions)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask, 0.0)
        x = self.ln(x)
        if padding_mask is not None:
            x = x.masked_fill(padding_mask, 0.0)

        for block in self.blocks:
            x = block(x, obs_rep, agent_mask=agent_mask)

        latent_pi = self.head(x)
        if padding_mask is not None:
            latent_pi = latent_pi.masked_fill(padding_mask, 0.0)
        return latent_pi
