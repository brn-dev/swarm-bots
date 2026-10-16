"""Centralized team Q functions for padded, variable-size swarms."""

from dataclasses import dataclass, field, replace
from typing import Any, Literal

import torch
from torch import nn

from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig
from swarmbots.learn.algos.off_policy.transformer_critic import (
    ActionConditionedEncoder,
    TransformerCriticConfig,
    build_q_network,
)
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSetInteractionConfig
from swarmbots.learn.nn_components.feed_forward import MLP, MLPConfig
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal


@dataclass(frozen=True)
class JointCriticConfig:
    kind: Literal["mlp", "deepset"] = "mlp"
    hidden_dims: tuple[int, ...] = (512, 512, 512)
    element_hidden_dims: tuple[int, ...] = (512, 512, 512)
    context_in_elements: bool = True
    interaction_config: DeepSetInteractionConfig = field(default_factory=DeepSetInteractionConfig)
    independent_encoders: bool = False
    encoder_config: MATEncoderConfig | None = None
    head_config: TransformerCriticConfig | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"mlp", "deepset"}:
            raise ValueError(f"Unknown joint critic kind: {self.kind!r}")
        for dims in (self.hidden_dims, self.element_hidden_dims):
            if not dims or any(dim <= 0 for dim in dims):
                raise ValueError("Critic hidden dimensions must be nonempty and positive")

    def resolved_encoder_config(self, act_fn_cls: ActivationFactory) -> MATEncoderConfig:
        if self.encoder_config is not None:
            return replace(self.encoder_config, act_fn_cls=act_fn_cls)
        token_width = self.element_hidden_dims[-1]
        local_hidden_dims = self.interaction_config.hidden_dims
        return MATEncoderConfig(
            d_model=token_width,
            nhead=1,
            num_layers=self.interaction_config.num_layers,
            dim_feedforward=token_width,
            transformer_ff_config=(None if local_hidden_dims is None else MLPConfig(hidden_dims=list(local_hidden_dims))),
            act_fn_cls=act_fn_cls,
        )

    def resolved_head_config(self) -> TransformerCriticConfig:
        if self.head_config is not None:
            return self.head_config
        token_width = self.element_hidden_dims[-1] if self.encoder_config is None else self.encoder_config.d_model
        return TransformerCriticConfig(
            action_coembed_hidden_dims=list(self.element_hidden_dims[:-1]),
            local_projection_hidden_dims=[token_width],
            value_regressor_hidden_dims=list(self.hidden_dims),
            interaction_config=replace(self.interaction_config, num_layers=2),
        )


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
        self.local_obs_dim = local_obs_dim
        self.global_obs_dim = global_obs_dim
        self.action_dim = action_dim
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
        self.encoder: JointDeepSetEncoder | None = None
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
            self.encoder = JointDeepSetEncoder(config=config, act_fn_cls=act_fn_cls, **kwargs)
            self.network = build_q_network(
                q_local_dim=self.encoder.d_model,
                critic_config=config.resolved_head_config(),
                act_fn_cls=act_fn_cls,
                num_global_features=self.global_dim if not config.context_in_elements else 0,
            )

    def forward(
        self,
        **observations: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.forward_with_latents(**observations)[0]

    @property
    def nop_source_latent_dim(self) -> int:
        if self.encoder is None:
            raise ValueError("MLP critics do not support NOP")
        return self.encoder.d_model

    def encode(self, **observations: torch.Tensor | None) -> torch.Tensor:
        if self.encoder is None:
            raise ValueError("MLP critics do not support NOP")
        return self.encoder.encode(**observations)

    def forward_with_latents(self, **observations: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.encoder is None:
            local, global_features, agent_mask = self._observation_features(**observations)
            mask_features = agent_mask.to(local.dtype)
            features = torch.cat((local.flatten(-2), global_features, mask_features), dim=-1)
            return self.network(features).squeeze(-1), None
        latents = self.encoder.encode(**observations)
        global_features = (
            None if self.config.context_in_elements else self._observation_features(**observations)[1]
        )
        return self.network(latents, global_features=global_features, agent_mask=observations.get("agent_mask")), latents


class JointDeepSetEncoder(_JointObservationInputs):
    """Transformer-compatible action embeddings with expanding Deep Set communication."""

    def __init__(self, *, config: JointCriticConfig, act_fn_cls: ActivationFactory = nn.GELU, **kwargs: Any) -> None:
        super().__init__(config=config, **kwargs)
        encoder_config = config.resolved_encoder_config(act_fn_cls)
        self.d_model = encoder_config.d_model
        self.network = ActionConditionedEncoder(
            max_agents=self.n_agents,
            local_input_dim=self.local_obs_dim,
            global_input_dim=self.global_obs_dim if config.context_in_elements else 0,
            hidden_local_vars_dim=self.hidden_local_vars_dim,
            hidden_global_vars_dim=self.hidden_global_vars_dim if config.context_in_elements else 0,
            action_dim=self.action_dim,
            encoder_config=encoder_config,
            critic_config=config.resolved_head_config(),
            interaction_config=config.interaction_config,
            act_fn_cls=act_fn_cls,
        )

    def encode(self, **observations: torch.Tensor | None) -> torch.Tensor:
        return self.network(
            local_inputs=observations["local_obs"],
            global_inputs=observations["global_obs"],
            actions=observations["actions"],
            hidden_local_vars=observations.get("hidden_local_vars"),
            hidden_global_vars=observations.get("hidden_global_vars"),
            agent_mask=observations.get("agent_mask"),
        )

    def forward_with_latents(self, **observations: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        latents = self.encode(**observations)
        agent_mask = observations.get("agent_mask")
        if agent_mask is None:
            return latents.mean(dim=1), latents
        count = agent_mask.sum(dim=1, keepdim=True).clamp_min(1)
        return latents.sum(dim=1) / count, latents


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

        head_config = config.resolved_head_config()
        head_global_dim = self.encoder.global_dim if not config.context_in_elements else 0
        self.q1 = build_q_network(
            q_local_dim=self.d_model, critic_config=head_config, act_fn_cls=act_fn_cls,
            num_global_features=head_global_dim,
        )
        self.q2 = build_q_network(
            q_local_dim=self.d_model, critic_config=head_config, act_fn_cls=act_fn_cls,
            num_global_features=head_global_dim,
        )

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
        latents1 = self.encoder.encode(**observations)
        latents2 = latents1 if self.encoder2 is None else self.encoder2.encode(**observations)
        latents = latents1 if self.encoder2 is None else torch.cat((latents1, latents2), dim=-1)
        agent_mask = observations.get("agent_mask")
        global_features = (
            None if self.config.context_in_elements else self.encoder._observation_features(**observations)[1]
        )
        return (
            self.q1(latents1, global_features=global_features, agent_mask=agent_mask),
            self.q2(latents2, global_features=global_features, agent_mask=agent_mask),
            latents,
        )
