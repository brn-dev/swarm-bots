from dataclasses import replace
from typing import Any

import torch
from torch import nn

from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.nn_components.activations import ActivationFactory
from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.feed_forward import MLPConfig


def make_value_head(config: Any, width: int, act_fn_cls: ActivationFactory) -> DeepSetCritic:
    return DeepSetCritic(
        num_local_features=width,
        local_projection_hidden_dims=(
            config.local_projection_hidden_dims
            if config.local_projection_hidden_dims is not None else [width] * config.n_local_projection_hidden_layers
        ),
        value_regressor_hidden_dims=(
            config.value_regressor_hidden_dims
            if config.value_regressor_hidden_dims is not None else [width] * config.n_value_regressor_hidden_layers
        ),
        num_global_features=0,
        act_fn_cls=act_fn_cls,
        context_in_elements=False,
        interaction_config=config.interaction_config,
        local_projection_linear_init_gain=config.local_projection_init_gain,
        value_regressor_linear_init_gain=config.value_regressor_init_gain,
        value_head_linear_init_gain=config.value_head_init_gain,
        use_popart=config.use_popart,
        popart_beta=config.popart_config.beta,
        popart_eps=config.popart_config.eps,
        popart_min_std=config.popart_config.min_std,
        popart_init_sigma=config.popart_config.init_sigma,
    )


def make_critic_encoder(
    config: MATEncoderConfig | None,
    actor_config: MATEncoderConfig,
    *,
    env: Any,
    max_agents: int,
) -> tuple[MATEncoderConfig, nn.Module]:
    from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig

    resolved = replace(
        actor_config if config is None else config,
        act_fn_cls=actor_config.act_fn_cls,
        dropout=actor_config.dropout,
    )
    if config is None:
        resolved = replace(
            resolved, use_agent_attention=True, local_obs_encoder_config=MLPConfig(hidden_dims=[resolved.d_model]),
        )
    encoder_class = RMATEncoder if isinstance(resolved, RMATEncoderConfig) else MATEncoder
    return resolved, encoder_class(
        config=resolved,
        max_agents=max_agents,
        local_obs_dim=env.local_obs_dim + env.hidden_local_vars_dim,
        global_obs_dim=env.global_obs_dim + env.hidden_global_vars_dim,
    )


def encode_critic_observations(
    encoder: nn.Module,
    local_obs: torch.Tensor,
    global_obs: torch.Tensor,
    hidden_local_vars: torch.Tensor | None,
    hidden_global_vars: torch.Tensor | None,
    agent_mask: torch.Tensor | None,
    **temporal_options: Any,
) -> Any:
    local_inputs = local_obs if hidden_local_vars is None else torch.cat((local_obs, hidden_local_vars), dim=-1)
    global_inputs = global_obs if hidden_global_vars is None else torch.cat((global_obs, hidden_global_vars), dim=-1)
    if agent_mask is not None:
        local_inputs = local_inputs.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
    result = encoder(local_inputs, global_inputs, agent_mask=agent_mask, **temporal_options)
    if isinstance(result, tuple):
        features, *state = result
        if agent_mask is not None:
            features = features.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
        return features, *state
    return result if agent_mask is None else result.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
