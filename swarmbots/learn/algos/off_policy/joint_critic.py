"""Centralized team Q functions for padded, variable-size swarms."""

from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal


@dataclass(frozen=True)
class JointCriticConfig:
    kind: Literal["mlp", "deepset"] = "mlp"
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    element_hidden_dims: tuple[int, ...] = (512, 512, 512)
    # Deep Set conditioning; flattened MLP critics already receive all global inputs.
    context_in_elements: bool = True

    def __post_init__(self) -> None:
        if self.kind not in {"mlp", "deepset"}:
            raise ValueError(f"Unknown joint critic kind: {self.kind!r}")
        for dims in (self.hidden_dims, self.element_hidden_dims):
            if not dims or any(dim <= 0 for dim in dims):
                raise ValueError("Critic hidden dimensions must be nonempty and positive")


class JointQNetwork(nn.Module):
    def __init__(
        self,
        *,
        n_agents: int,
        local_obs_dim: int,
        global_obs_dim: int,
        hidden_local_vars_dim: int,
        hidden_global_vars_dim: int,
        action_dim: int,
        config: JointCriticConfig,
        act_fn_cls: ActivationFactory = nn.GELU,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.config = config
        self.hidden_local_vars_dim = hidden_local_vars_dim
        self.hidden_global_vars_dim = hidden_global_vars_dim
        local_dim = local_obs_dim + hidden_local_vars_dim + action_dim
        global_dim = global_obs_dim + hidden_global_vars_dim
        if config.kind == "mlp":
            # Every slot (including its presence bit) has a fixed position.
            self.network = MLP(
                input_dim=n_agents * (local_dim + 1) + global_dim,
                hidden_dims=[*config.hidden_dims, 1],
                end_with_act_fn=False,
                act_fn_cls=act_fn_cls,
                final_linear_init=make_init_linear_orthogonal(0.01),
            )
        else:
            # Masked mean pooling plus cardinality retains swarm-size information.
            self.network = DeepSetCritic(
                num_local_features=local_dim,
                num_global_features=global_dim + 1,
                local_projection_hidden_dims=list(config.element_hidden_dims),
                value_regressor_hidden_dims=list(config.hidden_dims),
                pool_mode="mean",
                act_fn_cls=act_fn_cls,
                context_in_elements=config.context_in_elements,
                context_after_pool=True,
            )

    def forward(
        self,
        **observations,
    ) -> torch.Tensor:
        return self.forward_with_latents(**observations)[0]

    @property
    def nop_source_latent_dim(self) -> int:
        if self.config.kind == "mlp":
            raise ValueError("MLP critics do not support NOP")
        return self.config.element_hidden_dims[-1]

    def _observation_features(
        self,
        *,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        actions: torch.Tensor,
        hidden_local_vars: torch.Tensor | None = None,
        hidden_global_vars: torch.Tensor | None = None,
        agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if local_obs.shape[-2] != self.n_agents:
            raise ValueError(f"Expected {self.n_agents} padded agent slots, got {local_obs.shape[-2]}")
        if agent_mask is None:
            agent_mask = torch.ones(local_obs.shape[:-1], dtype=torch.bool, device=local_obs.device)
        local_parts = [local_obs, actions]
        global_parts = [global_obs]
        if self.hidden_local_vars_dim:
            if hidden_local_vars is None:
                raise ValueError("hidden_local_vars are required by the critic")
            local_parts.append(hidden_local_vars)
        if self.hidden_global_vars_dim:
            if hidden_global_vars is None:
                raise ValueError("hidden_global_vars are required by the critic")
            global_parts.append(hidden_global_vars)
        # Sanitize before encoding, so even NaNs in padding cannot enter active Q values.
        local = torch.cat(local_parts, dim=-1).masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        global_features = torch.cat(global_parts, dim=-1)
        return local, global_features, agent_mask

    def encode(self, **observations) -> torch.Tensor:
        if self.config.kind == "mlp":
            raise ValueError("MLP critics do not support NOP")
        local, global_features, agent_mask = self._observation_features(**observations)
        context = self._deepset_context(global_features, agent_mask)
        latents = self.network.encode_elements(local, global_features=context)
        return latents.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

    @staticmethod
    def _deepset_context(global_features: torch.Tensor, agent_mask: torch.Tensor) -> torch.Tensor:
        cardinality = agent_mask.to(global_features.dtype).sum(-1, keepdim=True)
        return torch.cat((global_features, cardinality), dim=-1)

    def forward_with_latents(self, **observations) -> tuple[torch.Tensor, torch.Tensor | None]:
        local, global_features, agent_mask = self._observation_features(**observations)
        mask_features = agent_mask.to(local.dtype)
        if self.config.kind == "mlp":
            features = torch.cat((local.flatten(-2), global_features, mask_features), dim=-1)
            return self.network(features).squeeze(-1), None
        context = self._deepset_context(global_features, agent_mask)
        values, latents = self.network.forward_with_latents(local, global_features=context, agent_mask=agent_mask)
        return values, latents.masked_fill(~agent_mask.unsqueeze(-1), 0.0)


class JointTwinCritic(nn.Module):
    """Independent team Q networks with optional per-agent Deep Set NOP latents."""

    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.q1 = JointQNetwork(**kwargs)
        self.q2 = JointQNetwork(**kwargs)

    @property
    def nop_source_latent_dim(self) -> int:
        return self.q1.nop_source_latent_dim + self.q2.nop_source_latent_dim

    def encode(self, *, local_inputs, global_inputs, **observations) -> torch.Tensor:
        observations.update(local_obs=local_inputs, global_obs=global_inputs)
        return torch.cat((self.q1.encode(**observations), self.q2.encode(**observations)), dim=-1)

    def forward(self, **observations) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        q1, latents1 = self.q1.forward_with_latents(**observations)
        q2, latents2 = self.q2.forward_with_latents(**observations)
        latents = None if latents1 is None else torch.cat((latents1, latents2), dim=-1)
        return q1, q2, latents
