from dataclasses import replace
from functools import partial

import torch
from torch import nn

from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig, MATEncoderLayer
from swarmbots.learn.nn_components.deep_set import (
    DeepSetInteraction, DeepSetInteractionConfig, GatedDeepSetInteraction,
)
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal


class DeepSetEncoderLayer(MATEncoderLayer):
    """Retain the transformer block's FF and residuals with pooled set communication."""

    def __init__(self, config: MATEncoderConfig, interaction_config: DeepSetInteractionConfig) -> None:
        super().__init__(replace(config, use_agent_attention=False))
        interaction_type = GatedDeepSetInteraction if interaction_config.gated else DeepSetInteraction
        self.interaction = interaction_type(
            config.d_model,
            config=replace(interaction_config, local_update="residual", normalize=False),
            set_dim=1,
            pool_mode="mean",
            linear_init=make_init_linear_orthogonal(config.linear_init_gain),
            act_fn_cls=config.act_fn_cls,
        )

    def reinitialize_communication(self, gain: float) -> None:
        linear_init = make_init_linear_orthogonal(gain)
        for module in self.interaction.modules():
            if isinstance(module, nn.Linear):
                linear_init(module)

    def _self_attention_block(
        self,
        embeddings: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        key_padding_mask: torch.Tensor | None,
        is_causal: bool,
    ) -> torch.Tensor:
        if attention_mask is not None or is_causal:
            raise ValueError("Deep Set communication does not support pairwise or causal attention masks")
        element_mask = None if key_padding_mask is None else ~key_padding_mask
        return self.dropout1(self.interaction.message_update(embeddings, element_mask))


class DeepSetEncoder(MATEncoder):
    def __init__(
        self, config: MATEncoderConfig, interaction_config: DeepSetInteractionConfig, *,
        max_agents: int, local_obs_dim: int, global_obs_dim: int,
    ) -> None:
        super().__init__(
            config, max_agents=max_agents, local_obs_dim=local_obs_dim, global_obs_dim=global_obs_dim,
            layer_factory=partial(DeepSetEncoderLayer, interaction_config=interaction_config),
        )
        for layer in self.layers:
            layer.reinitialize_communication(config.linear_init_gain)
