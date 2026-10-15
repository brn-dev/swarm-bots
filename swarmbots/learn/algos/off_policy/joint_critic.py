"""Centralized team Q functions for padded, variable-size swarms."""

from dataclasses import dataclass, field
from typing import Any, Literal

import torch
from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSet, DeepSetCritic, DeepSetInteractionConfig
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal


@dataclass(frozen=True)
class JointCriticConfig:
    kind: Literal["mlp", "deepset"] = "mlp"
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    element_hidden_dims: tuple[int, ...] = (512, 512, 512)
    # Deep Set conditioning; flattened MLP critics already receive all global inputs.
    context_in_elements: bool = True
    interaction_config: DeepSetInteractionConfig = field(
        default_factory=lambda: DeepSetInteractionConfig(num_layers=4),
    )
    independent_encoders: bool = False

    def __post_init__(self) -> None:
        if self.kind not in {"mlp", "deepset"}:
            raise ValueError(f"Unknown joint critic kind: {self.kind!r}")
        for dims in (self.hidden_dims, self.element_hidden_dims):
            if not dims or any(dim <= 0 for dim in dims):
                raise ValueError("Critic hidden dimensions must be nonempty and positive")


class _JointObservationInputs(nn.Module):
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
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.config = config
        self.hidden_local_vars_dim = hidden_local_vars_dim
        self.hidden_global_vars_dim = hidden_global_vars_dim
        self.local_dim = local_obs_dim + hidden_local_vars_dim + action_dim
        self.global_dim = global_obs_dim + hidden_global_vars_dim

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


class JointQNetwork(_JointObservationInputs):
    def __init__(self, *, config: JointCriticConfig, act_fn_cls: ActivationFactory = nn.GELU, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        if config.kind == "mlp":
            # Every slot (including its presence bit) has a fixed position.
            self.network = MLP(
                input_dim=self.n_agents * (self.local_dim + 1) + self.global_dim,
                hidden_dims=[*config.hidden_dims, 1],
                end_with_act_fn=False,
                act_fn_cls=act_fn_cls,
                final_linear_init=make_init_linear_orthogonal(0.01),
            )
        else:
            self.network = DeepSetCritic(
                num_local_features=self.local_dim,
                num_global_features=self.global_dim,
                local_projection_hidden_dims=list(config.element_hidden_dims),
                value_regressor_hidden_dims=list(config.hidden_dims),
                pool_mode="mean",
                act_fn_cls=act_fn_cls,
                context_in_elements=config.context_in_elements and self.global_dim > 0,
                context_after_pool=True,
                interaction_config=config.interaction_config,
            )

    def forward(
        self,
        **observations: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.forward_with_latents(**observations)[0]

    @property
    def nop_source_latent_dim(self) -> int:
        if self.config.kind == "mlp":
            raise ValueError("MLP critics do not support NOP")
        return self.config.element_hidden_dims[-1]

    def encode(self, **observations: torch.Tensor | None) -> torch.Tensor:
        if self.config.kind == "mlp":
            raise ValueError("MLP critics do not support NOP")
        local, global_features, agent_mask = self._observation_features(**observations)
        latents = self.network.encode_elements(local, global_features=global_features, agent_mask=agent_mask)
        return latents.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

    def forward_with_latents(self, **observations: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        local, global_features, agent_mask = self._observation_features(**observations)
        if self.config.kind == "mlp":
            mask_features = agent_mask.to(local.dtype)
            features = torch.cat((local.flatten(-2), global_features, mask_features), dim=-1)
            return self.network(features).squeeze(-1), None
        values, latents = self.network.forward_with_latents(local, global_features=global_features, agent_mask=agent_mask)
        return values, latents.masked_fill(~agent_mask.unsqueeze(-1), 0.0)


class JointDeepSetEncoder(_JointObservationInputs):
    """Action-conditioned equivariant trunk and masked pooling for twin team critics."""

    def __init__(self, *, config: JointCriticConfig, act_fn_cls: ActivationFactory = nn.GELU, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        self.d_model = config.element_hidden_dims[-1]
        context_in_elements = config.context_in_elements and self.global_dim > 0
        self.network = DeepSet.from_dimensions(
            element_input_features=self.local_dim + (self.global_dim if context_in_elements else 0),
            local_projection_hidden_dims=list(config.element_hidden_dims),
            set_decoder=nn.Identity(),
            pool_mode="mean",
            context_features=self.global_dim,
            context_in_elements=context_in_elements,
            context_after_pool=True,
            interaction_config=config.interaction_config,
            act_fn_cls=act_fn_cls,
            set_dim=-2,
        )

    def encode(self, **observations: torch.Tensor | None) -> torch.Tensor:
        local, global_features, agent_mask = self._observation_features(**observations)
        latents = self.network.encode_elements(local, context=global_features, element_mask=agent_mask)
        return latents.masked_fill(~agent_mask.unsqueeze(-1), 0.0)

    def forward_with_latents(self, **observations: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        local, global_features, agent_mask = self._observation_features(**observations)
        pooled, latents = self.network.forward_with_latents(local, context=global_features, element_mask=agent_mask)
        return pooled, latents.masked_fill(~agent_mask.unsqueeze(-1), 0.0)


class JointTwinCritic(nn.Module):
    """Two independent Q readouts with a shared Deep Set encoder by default."""

    def __init__(self, *, config: JointCriticConfig, act_fn_cls: ActivationFactory = nn.GELU, **kwargs: Any) -> None:
        super().__init__()
        self.config = config
        if config.kind == "mlp":
            self.encoder = self.encoder2 = None
            self.q1 = JointQNetwork(config=config, act_fn_cls=act_fn_cls, **kwargs)
            self.q2 = JointQNetwork(config=config, act_fn_cls=act_fn_cls, **kwargs)
            return
        self.encoder = JointDeepSetEncoder(config=config, act_fn_cls=act_fn_cls, **kwargs)
        self.encoder2 = (
            JointDeepSetEncoder(config=config, act_fn_cls=act_fn_cls, **kwargs) if config.independent_encoders else None
        )
        self.d_model = self.encoder.d_model

        def make_head() -> MLP:
            return MLP(
                input_dim=self.d_model + self.encoder.global_dim,
                hidden_dims=[*config.hidden_dims, 1], end_with_act_fn=False, act_fn_cls=act_fn_cls,
                final_linear_init=make_init_linear_orthogonal(0.01),
            )

        self.q1 = make_head()
        self.q2 = make_head()

    @property
    def nop_source_latent_dim(self) -> int:
        if self.encoder is None:
            raise ValueError("MLP critics do not support NOP")
        return self.d_model * (2 if self.encoder2 is not None else 1)

    def encode(
        self, *, local_inputs: torch.Tensor, global_inputs: torch.Tensor, **observations: torch.Tensor | None,
    ) -> torch.Tensor:
        observations.update(local_obs=local_inputs, global_obs=global_inputs)
        if self.encoder is None:
            raise ValueError("MLP critics do not support NOP")
        primary_latents = self.encoder.encode(**observations)
        if self.encoder2 is None:
            return primary_latents
        return torch.cat((primary_latents, self.encoder2.encode(**observations)), dim=-1)

    def forward(self, **observations: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if self.encoder is None:
            return self.q1(**observations), self.q2(**observations), None
        pooled1, latents1 = self.encoder.forward_with_latents(**observations)
        pooled2, latents2 = (
            (pooled1, latents1) if self.encoder2 is None else self.encoder2.forward_with_latents(**observations)
        )
        latents = latents1 if self.encoder2 is None else torch.cat((latents1, latents2), dim=-1)
        return self.q1(pooled1).squeeze(-1), self.q2(pooled2).squeeze(-1), latents
