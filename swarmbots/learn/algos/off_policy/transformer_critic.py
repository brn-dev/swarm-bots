"""Transformer team critics usable with stochastic or deterministic actors."""

from dataclasses import dataclass, field, replace
import copy
from typing import Any
import torch
from torch import nn
from swarmbots.learn.algos.mat.mat_encoder import MATEncoderConfig, MATEncoderLayer
from swarmbots.learn.algos.ppo.ppo_policy import PopArtConfig
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.feed_forward import MLP, feedforward_linear_layers, make_feedforward
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal, reinitialize_multihead_attention


@dataclass(frozen=True)
class TransformerCriticConfig:
    n_local_projection_hidden_layers: int = 1
    n_value_regressor_hidden_layers: int = 2
    action_coembed_hidden_dims: list[int] | None = None
    separate_observation_action_encoders: bool = False
    action_encoder_dim: int | None = None
    independent_encoders: bool = False
    use_popart: bool = False
    popart_config: PopArtConfig = field(default_factory=PopArtConfig)
    action_coembed_init_gain: float = 1.0
    action_coembed_output_init_gain: float = 1.0
    local_projection_init_gain: float = 1.0
    value_regressor_init_gain: float = 1.0
    value_head_init_gain: float = 0.01


class ObservationActionEncoder(nn.Module):
    def __init__(
        self,
        *,
        observation_input_dim: int,
        action_dim: int,
        d_model: int,
        action_encoder_dim: int | None,
        linear_init_gain: float,
        act_fn_cls: ActivationFactory,
    ) -> None:
        super().__init__()
        self.action_encoder_dim = max(1, d_model // 2) if action_encoder_dim is None else int(action_encoder_dim)
        linear_init = make_init_linear_orthogonal(linear_init_gain)
        self.observation_encoder = MLP(
            input_dim=observation_input_dim,
            hidden_dims=[d_model],
            end_with_act_fn=True,
            linear_init=linear_init,
            act_fn_cls=act_fn_cls,
        )
        self.action_encoder = MLP(
            input_dim=action_dim,
            hidden_dims=[self.action_encoder_dim],
            end_with_act_fn=True,
            linear_init=linear_init,
            act_fn_cls=act_fn_cls,
        )
        self.output_dim = d_model + self.action_encoder_dim

    def forward(
        self,
        observation_inputs: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                self.observation_encoder(observation_inputs),
                self.action_encoder(actions),
            ),
            dim=-1,
        )


class ActionConditionedEncoder(nn.Module):
    def __init__(
        self,
        *,
        max_agents: int,
        local_input_dim: int,
        global_input_dim: int,
        hidden_local_vars_dim: int,
        hidden_global_vars_dim: int,
        action_dim: int,
        encoder_config: MATEncoderConfig,
        critic_config: TransformerCriticConfig,
        act_fn_cls: ActivationFactory,
    ) -> None:
        super().__init__()
        self.max_agents = int(max_agents)
        self.local_input_dim = int(local_input_dim)
        self.global_input_dim = int(global_input_dim)
        self.hidden_local_vars_dim = int(hidden_local_vars_dim)
        self.hidden_global_vars_dim = int(hidden_global_vars_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(encoder_config.d_model)
        self.global_encoder_input_dim = self.global_input_dim + self.hidden_global_vars_dim
        self.has_global_input = self.global_encoder_input_dim > 0
        self.joint_obs_embedding = encoder_config.joint_obs_embedding and self.has_global_input

        local_observation_input_dim = self.local_input_dim + self.hidden_local_vars_dim
        self.observation_action_encoder = (
            ObservationActionEncoder(
                observation_input_dim=local_observation_input_dim,
                action_dim=self.action_dim,
                d_model=self.d_model,
                action_encoder_dim=critic_config.action_encoder_dim,
                linear_init_gain=critic_config.action_coembed_init_gain,
                act_fn_cls=act_fn_cls,
            )
            if critic_config.separate_observation_action_encoders
            else None
        )
        local_action_input_dim = (
            local_observation_input_dim + self.action_dim
            if self.observation_action_encoder is None
            else self.observation_action_encoder.output_dim
        )
        self.local_action_input_norm = (
            nn.LayerNorm(local_action_input_dim) if encoder_config.normalize_obs_inputs else nn.Identity()
        )
        action_coembed_hidden_dims = (
            [self.d_model]
            if critic_config.action_coembed_hidden_dims is None
            else [*critic_config.action_coembed_hidden_dims]
        )
        self.local_action_encoder = MLP(
            input_dim=local_action_input_dim + (self.global_encoder_input_dim if self.joint_obs_embedding else 0),
            hidden_dims=[*action_coembed_hidden_dims, self.d_model],
            end_with_act_fn=False,
            linear_init=make_init_linear_orthogonal(critic_config.action_coembed_init_gain),
            final_linear_init=make_init_linear_orthogonal(critic_config.action_coembed_output_init_gain),
            act_fn_cls=act_fn_cls,
        )

        self.global_input_norm = (
            nn.LayerNorm(self.global_encoder_input_dim)
            if encoder_config.normalize_obs_inputs and self.has_global_input
            else nn.Identity()
        )
        if self.has_global_input and not self.joint_obs_embedding:
            linear_init = make_init_linear_orthogonal(encoder_config.linear_init_gain)
            projection_linear_init = (
                linear_init
                if encoder_config.linear_projection_init_gain is None
                else make_init_linear_orthogonal(encoder_config.linear_projection_init_gain)
            )
            self.global_encoder = make_feedforward(
                input_dim=self.global_encoder_input_dim,
                output_dim=self.d_model,
                config=encoder_config.global_obs_encoder_config,
                linear_init=linear_init,
                output_linear_init=projection_linear_init,
                act_fn_cls=act_fn_cls,
            )
        else:
            self.global_encoder = None

        self.token_norm = nn.LayerNorm(self.d_model) if encoder_config.normalize_tokens else nn.Identity()
        prototype_layer = MATEncoderLayer(encoder_config)
        self.layers = nn.ModuleList([copy.deepcopy(prototype_layer) for _ in range(encoder_config.num_layers)])
        self.norm = nn.LayerNorm(
            self.d_model,
            eps=encoder_config.layer_norm_eps,
            bias=encoder_config.bias,
        )
        if encoder_config.transformer_ff_init_gain is not None:
            self._reinitialize_layers(feedforward_init_gain=encoder_config.transformer_ff_init_gain)

        self.agent_embeddings: nn.Parameter | None = None
        if encoder_config.add_agent_embeddings:
            self.agent_embeddings = nn.Parameter(
                torch.zeros(1, self.max_agents, self.d_model),
                requires_grad=True,
            )
            nn.init.orthogonal_(self.agent_embeddings)

    def forward(
        self,
        *,
        local_inputs: torch.Tensor,
        global_inputs: torch.Tensor,
        actions: torch.Tensor,
        hidden_local_vars: torch.Tensor | None,
        hidden_global_vars: torch.Tensor | None,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        n_agents = local_inputs.shape[1]
        if n_agents > self.max_agents:
            raise ValueError(f"Expected local_inputs second dim <= {self.max_agents}, got {n_agents}")

        observation_parts = [local_inputs]
        if self.hidden_local_vars_dim > 0:
            if hidden_local_vars is None:
                raise ValueError("hidden_local_vars must be provided when hidden_local_vars_dim > 0")
            observation_parts.append(hidden_local_vars)
        observation_inputs = (
            observation_parts[0] if len(observation_parts) == 1 else torch.cat(observation_parts, dim=-1)
        )
        token_inputs = (
            torch.cat((observation_inputs, actions), dim=-1)
            if self.observation_action_encoder is None
            else self.observation_action_encoder(observation_inputs, actions)
        )
        token_inputs = self.local_action_input_norm(token_inputs)
        if self.joint_obs_embedding:
            global_encoder_inputs = self._global_observation_inputs(global_inputs, hidden_global_vars)
            global_encoder_inputs = self.global_input_norm(global_encoder_inputs)
            token_inputs = torch.cat(
                (token_inputs, global_encoder_inputs.unsqueeze(1).expand(-1, n_agents, -1)),
                dim=-1,
            )
        tokens = self.local_action_encoder(token_inputs)
        if self.agent_embeddings is not None:
            tokens = tokens + self.agent_embeddings[:, :n_agents, :]

        if self.global_encoder is not None:
            global_encoder_inputs = self._global_observation_inputs(global_inputs, hidden_global_vars)
            global_tokens = self.global_encoder(self.global_input_norm(global_encoder_inputs))
            tokens = tokens + global_tokens.unsqueeze(1).expand(-1, n_agents, -1)
        tokens = self.token_norm(tokens)

        src_key_padding_mask = None if agent_mask is None else ~agent_mask
        for layer in self.layers:
            tokens = layer(tokens, src_key_padding_mask=src_key_padding_mask)
        return self.norm(tokens)

    def _global_observation_inputs(
        self,
        global_inputs: torch.Tensor,
        hidden_global_vars: torch.Tensor | None,
    ) -> torch.Tensor:
        global_parts = [global_inputs] if self.global_input_dim > 0 else []
        if self.hidden_global_vars_dim > 0:
            if hidden_global_vars is None:
                raise ValueError("hidden_global_vars must be provided when hidden_global_vars_dim > 0")
            global_parts.append(hidden_global_vars)
        return global_parts[0] if len(global_parts) == 1 else torch.cat(global_parts, dim=-1)

    def _reinitialize_layers(self, *, feedforward_init_gain: float) -> None:
        hidden_linear_init = make_init_linear_orthogonal(feedforward_init_gain)
        output_linear_init = make_init_linear_orthogonal(1.0)
        for layer in self.layers:
            if layer.self_attn is not None:
                reinitialize_multihead_attention(layer.self_attn)
            hidden_layers, output_layers = feedforward_linear_layers(layer.feedforward)
            for linear in hidden_layers:
                hidden_linear_init(linear)
            for linear in output_layers:
                output_linear_init(linear)


class TransformerTwinCritic(nn.Module):
    def __init__(
        self,
        *,
        n_agents: int,
        max_agents: int,
        local_input_dim: int,
        global_input_dim: int,
        hidden_local_vars_dim: int,
        hidden_global_vars_dim: int,
        action_dim: int,
        encoder_config: MATEncoderConfig,
        critic_config: TransformerCriticConfig,
        act_fn_cls: ActivationFactory,
        dropout: float,
    ) -> None:
        super().__init__()
        self.n_agents = int(n_agents)
        self.hidden_local_vars_dim = int(hidden_local_vars_dim)
        self.hidden_global_vars_dim = int(hidden_global_vars_dim)
        self.action_dim = int(action_dim)
        self.d_model = int(encoder_config.d_model)
        self.encoder_config = replace(
            encoder_config,
            act_fn_cls=act_fn_cls,
            dropout=dropout,
        )

        def build_encoder() -> ActionConditionedEncoder:
            return ActionConditionedEncoder(
                max_agents=max_agents,
                local_input_dim=local_input_dim,
                global_input_dim=global_input_dim,
                hidden_local_vars_dim=self.hidden_local_vars_dim,
                hidden_global_vars_dim=self.hidden_global_vars_dim,
                action_dim=self.action_dim,
                encoder_config=self.encoder_config,
                critic_config=critic_config,
                act_fn_cls=act_fn_cls,
            )

        self.encoder = build_encoder()
        self.encoder2 = build_encoder() if critic_config.independent_encoders else None
        self.nop_source_latent_dim = self.d_model * (2 if self.encoder2 is not None else 1)
        q_local_dim = self.d_model
        self.q1 = self._build_q_network(
            q_local_dim=q_local_dim,
            critic_config=critic_config,
            act_fn_cls=act_fn_cls,
        )
        self.q2 = self._build_q_network(
            q_local_dim=q_local_dim,
            critic_config=critic_config,
            act_fn_cls=act_fn_cls,
        )

    def encode(
        self,
        *,
        local_inputs: torch.Tensor,
        global_inputs: torch.Tensor,
        actions: torch.Tensor,
        hidden_local_vars: torch.Tensor | None,
        hidden_global_vars: torch.Tensor | None,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        primary_latents = self._encode_with(
            self.encoder,
            local_inputs=local_inputs,
            global_inputs=global_inputs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        if self.encoder2 is None:
            return primary_latents
        secondary_latents = self._encode_with(
            self.encoder2,
            local_inputs=local_inputs,
            global_inputs=global_inputs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        return self._combine_nop_source_latents(primary_latents, secondary_latents)

    def forward(
        self,
        *,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        actions: torch.Tensor,
        hidden_local_vars: torch.Tensor | None,
        hidden_global_vars: torch.Tensor | None,
        agent_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q1_latents = self._encode_with(
            self.encoder,
            local_inputs=local_obs,
            global_inputs=global_obs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )
        q2_latents = (
            q1_latents
            if self.encoder2 is None
            else self._encode_with(
                self.encoder2,
                local_inputs=local_obs,
                global_inputs=global_obs,
                actions=actions,
                hidden_local_vars=hidden_local_vars,
                hidden_global_vars=hidden_global_vars,
                agent_mask=agent_mask,
            )
        )
        return (
            self.q1(q1_latents, agent_mask=agent_mask),
            self.q2(q2_latents, agent_mask=agent_mask),
            self._combine_nop_source_latents(q1_latents, q2_latents),
        )

    def _combine_nop_source_latents(
        self,
        primary_latents: torch.Tensor,
        secondary_latents: torch.Tensor,
    ) -> torch.Tensor:
        if self.encoder2 is None:
            return primary_latents
        return torch.cat((primary_latents, secondary_latents), dim=-1)

    @staticmethod
    def _encode_with(
        encoder: ActionConditionedEncoder,
        *,
        local_inputs: torch.Tensor,
        global_inputs: torch.Tensor,
        actions: torch.Tensor,
        hidden_local_vars: torch.Tensor | None,
        hidden_global_vars: torch.Tensor | None,
        agent_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return encoder(
            local_inputs=local_inputs,
            global_inputs=global_inputs,
            actions=actions,
            hidden_local_vars=hidden_local_vars,
            hidden_global_vars=hidden_global_vars,
            agent_mask=agent_mask,
        )

    def _load_from_state_dict(
        self,
        state_dict: dict[str, Any],
        prefix: str,
        local_metadata: dict[str, Any],
        strict: bool,
        missing_keys: list[str],
        unexpected_keys: list[str],
        error_msgs: list[str],
    ) -> None:
        if self.encoder2 is not None:
            primary_prefix = f"{prefix}encoder."
            secondary_prefix = f"{prefix}encoder2."
            has_primary_encoder = any(key.startswith(primary_prefix) for key in state_dict)
            has_secondary_encoder = any(key.startswith(secondary_prefix) for key in state_dict)
            if has_primary_encoder and not has_secondary_encoder:
                for key, value in list(state_dict.items()):
                    if key.startswith(primary_prefix):
                        state_dict[f"{secondary_prefix}{key.removeprefix(primary_prefix)}"] = value
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    @staticmethod
    def _build_q_network(
        *,
        q_local_dim: int,
        critic_config: TransformerCriticConfig,
        act_fn_cls: ActivationFactory,
    ) -> DeepSetCritic:
        return DeepSetCritic(
            num_local_features=q_local_dim,
            local_projection_hidden_dims=[q_local_dim] * critic_config.n_local_projection_hidden_layers,
            value_regressor_hidden_dims=[q_local_dim] * critic_config.n_value_regressor_hidden_layers,
            act_fn_cls=act_fn_cls,
            local_projection_linear_init_gain=critic_config.local_projection_init_gain,
            value_regressor_linear_init_gain=critic_config.value_regressor_init_gain,
            value_head_linear_init_gain=critic_config.value_head_init_gain,
            use_popart=critic_config.use_popart,
            popart_beta=critic_config.popart_config.beta,
            popart_eps=critic_config.popart_config.eps,
            popart_min_std=critic_config.popart_config.min_std,
            popart_init_sigma=critic_config.popart_config.init_sigma,
        )
