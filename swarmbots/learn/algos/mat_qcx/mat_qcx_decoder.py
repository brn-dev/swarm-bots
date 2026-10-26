from dataclasses import dataclass

import torch
from torch import nn

from swarmbots.learn.algos.mat_qc_base_policy import MATQCBasePolicy
from swarmbots.learn.nn_components.activations import ActivationFactory, make_activation
from swarmbots.learn.nn_components.attention import (
    AttentionKeysValues,
    attention_with_projected_keys_values,
    project_attention_keys_values,
)
from swarmbots.learn.nn_components.nn_init import (
    init_transformer_feedforward,
    reinitialize_multihead_attention,
)


@dataclass(frozen=True)
class MATQCXDecoderConfig:
    d_model: int | None = None
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 128
    dropout: float = 0.0
    act_fn_cls: ActivationFactory = nn.GELU
    norm_first: bool = True
    layer_norm_eps: float = 1e-5
    bias: bool = True
    add_agent_embeddings: bool = False
    token_encoder_init_gain: float = 1.0
    token_encoder_projection_init_gain: float | None = 1.0
    actor_head_init_gain: float = 1.0
    transformer_ff_init_gain: float | None = 1.0
    query_encoder_hidden_dims: list[int] | None = None
    context_encoder_hidden_dims: list[int] | None = None
    action_encoder_dims: list[int] | None = None
    memory_dims: list[int] | None = None
    actor_head_hidden_dims: list[int] | None = None
    layer_token_encoder_end_with_act_fn: bool = True
    action_encoder_end_with_act_fn: bool = True
    memory_encoder_end_with_act_fn: bool = False
    normalize_context_input: bool = False
    normalize_action_input: bool = False
    normalize_memory_input: bool = False
    normalize_context_tokens: bool = True
    normalize_action_tokens: bool = False
    normalize_memory_tokens: bool = False
    normalize_actor_head_input: bool = False
    assume_agent_mask_is_active_prefix: bool = True


@dataclass
class MATQCXLayerCache:
    memory: AttentionKeysValues
    context: AttentionKeysValues | None = None
    previous_input: torch.Tensor | None = None


MATQCXDecoderCache = list[MATQCXLayerCache]


class MATQCXDecoderLayer(nn.Module):

    def __init__(
            self,
            *,
            action_d_model: int,
            d_model: int,
            memory_d_model: int,
            nhead: int,
            dim_feedforward: int,
            dropout: float,
            act_fn_cls: ActivationFactory,
            query_encoder_hidden_dims: list[int] | None,
            context_encoder_hidden_dims: list[int] | None,
            token_encoder_init_gain: float,
            token_encoder_projection_init_gain: float | None,
            layer_token_encoder_end_with_act_fn: bool,
            normalize_context_input: bool,
            normalize_context_tokens: bool,
            layer_norm_eps: float,
            norm_first: bool,
            bias: bool,
    ) -> None:
        super().__init__()
        self.query_encoder = self._build_optional_query_encoder(
            d_model=d_model,
            hidden_dims=query_encoder_hidden_dims,
            act_fn_cls=act_fn_cls,
            token_encoder_init_gain=token_encoder_init_gain,
            token_encoder_projection_init_gain=token_encoder_projection_init_gain,
            layer_token_encoder_end_with_act_fn=layer_token_encoder_end_with_act_fn,
        )
        self.context_input_norm = (
            nn.LayerNorm(d_model + action_d_model, eps=layer_norm_eps, bias=bias)
            if normalize_context_input
            else nn.Identity()
        )
        self.context_encoder = MATQCBasePolicy._build_token_encoder(
            input_dim=d_model + action_d_model,
            output_dim=d_model,
            hidden_dims=context_encoder_hidden_dims,
            act_fn_cls=act_fn_cls,
            linear_init_gain=token_encoder_init_gain,
            projection_init_gain=token_encoder_projection_init_gain,
            end_with_act_fn=layer_token_encoder_end_with_act_fn,
        )
        self.context_token_norm = (
            nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
            if normalize_context_tokens
            else nn.Identity()
        )

        self.query_context_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            batch_first=True,
            bias=bias,
        )
        self.memory_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
            kdim=memory_d_model,
            vdim=memory_d_model,
            batch_first=True,
            bias=bias,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias)

        self.norm_first = norm_first
        self.norm1 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm2 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.norm3 = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = make_activation(act_fn_cls, num_features=dim_feedforward)

    @staticmethod
    def _build_optional_query_encoder(
            *,
            d_model: int,
            hidden_dims: list[int] | None,
            act_fn_cls: ActivationFactory,
            token_encoder_init_gain: float,
            token_encoder_projection_init_gain: float | None,
            layer_token_encoder_end_with_act_fn: bool,
    ) -> nn.Module:
        if hidden_dims is None or len(hidden_dims) == 0:
            return nn.Identity()
        return MATQCBasePolicy._build_token_encoder(
            input_dim=d_model,
            output_dim=d_model,
            hidden_dims=hidden_dims,
            act_fn_cls=act_fn_cls,
            linear_init_gain=token_encoder_init_gain,
            projection_init_gain=token_encoder_projection_init_gain,
            end_with_act_fn=layer_token_encoder_end_with_act_fn,
        )

    def forward(
            self,
            x: torch.Tensor,
            action_tokens: torch.Tensor,
            memory_tokens: torch.Tensor,
            *,
            context_token_count: int | None,
            context_attention_mask: torch.Tensor | None,
            has_visible_context: torch.Tensor | None,
            context_key_padding_mask: torch.Tensor | None,
            memory_key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if context_token_count is None:
            context_input_tokens = x
            context_action_tokens = action_tokens
        else:
            context_input_tokens = x[:, :context_token_count, :]
            context_action_tokens = action_tokens[:, :context_token_count, :]
        context_tokens = self.context_token_norm(
            self.context_encoder(
                self.context_input_norm(torch.cat((context_input_tokens, context_action_tokens), dim=-1))
            )
        )

        if self.norm_first:
            query_tokens = self.norm1(self._encode_query_tokens(x))
            x = x + self._context_attn_block(
                query_tokens,
                context_tokens,
                context_attention_mask=context_attention_mask,
                has_visible_context=has_visible_context,
                context_key_padding_mask=context_key_padding_mask,
            )
            x = x + self._memory_attn_block(
                self.norm2(x),
                memory_tokens,
                memory_key_padding_mask=memory_key_padding_mask,
            )
            return x + self._ff_block(self.norm3(x))

        query_tokens = self._encode_query_tokens(x)
        x = self.norm1(
            x + self._context_attn_block(
                query_tokens,
                context_tokens,
                context_attention_mask=context_attention_mask,
                has_visible_context=has_visible_context,
                context_key_padding_mask=context_key_padding_mask,
            )
        )
        x = self.norm2(
            x + self._memory_attn_block(
                x,
                memory_tokens,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        )
        return self.norm3(x + self._ff_block(x))

    def _encode_query_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return self.query_encoder(x)

    def forward_cached_step(
            self,
            x: torch.Tensor,
            previous_action_token: torch.Tensor,
            cache: MATQCXLayerCache,
            *,
            context_mask: torch.Tensor | None,
            memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if cache.previous_input is not None:
            context_token = self.context_token_norm(self.context_encoder(self.context_input_norm(
                torch.cat((cache.previous_input, previous_action_token), dim=-1),
            )))
            keys_values = project_attention_keys_values(self.query_context_attn, context_token)
            cache.context = keys_values if cache.context is None else (
                torch.cat((cache.context[0], keys_values[0]), dim=2),
                torch.cat((cache.context[1], keys_values[1]), dim=2),
            )
        # The sampled action arrives afterward; its context uses this layer's
        # input, before attention, on the next decoding step.
        cache.previous_input = x

        query_tokens = self._encode_query_tokens(x)
        if self.norm_first:
            query_tokens = self.norm1(query_tokens)
        if cache.context is not None:
            context_output = attention_with_projected_keys_values(
                self.query_context_attn, query_tokens, cache.context, context_mask,
            )
            if context_mask is not None:
                context_output = context_output.masked_fill(~context_mask.any(dim=1, keepdim=True).unsqueeze(-1), 0.0)
            x = x + self.dropout1(context_output)
        if not self.norm_first:
            x = self.norm1(x)

        memory_query = self.norm2(x) if self.norm_first else x
        x = x + self.dropout2(attention_with_projected_keys_values(
            self.memory_attn, memory_query, cache.memory, memory_mask,
        ))
        if self.norm_first:
            return x + self._ff_block(self.norm3(x))
        x = self.norm2(x)
        return self.norm3(x + self._ff_block(x))

    def _context_attn_block(
            self,
            query_tokens: torch.Tensor,
            context_tokens: torch.Tensor,
            *,
            context_attention_mask: torch.Tensor | None,
            has_visible_context: torch.Tensor | None,
            context_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if context_tokens.shape[1] == 0:
            return torch.zeros_like(query_tokens)

        if (
                has_visible_context is not None
                and query_tokens.shape[1] > 1
                and context_attention_mask is not None
                and context_attention_mask.dim() == 2
        ):
            # A parallel causal mask always leaves query zero without prior context. Keep this
            # decision structural: branching on has_visible_context[:, 0].any() breaks full-graph compilation.
            attended_output = self.query_context_attn(
                query=query_tokens[:, 1:, :],
                key=context_tokens,
                value=context_tokens,
                attn_mask=context_attention_mask[1:, :],
                key_padding_mask=context_key_padding_mask,
                need_weights=False,
            )[0]
            attended_output = attended_output.masked_fill(~has_visible_context[:, 1:].unsqueeze(-1), 0.0)
            first_output = torch.zeros_like(query_tokens[:, :1, :])
            return self.dropout1(torch.cat((first_output, attended_output), dim=1))

        output = self.query_context_attn(
            query=query_tokens,
            key=context_tokens,
            value=context_tokens,
            attn_mask=context_attention_mask,
            key_padding_mask=context_key_padding_mask,
            need_weights=False,
        )[0]
        if has_visible_context is not None:
            output = output.masked_fill(~has_visible_context.unsqueeze(-1), 0.0)
        return self.dropout1(output)

    def _memory_attn_block(
            self,
            query_tokens: torch.Tensor,
            memory_tokens: torch.Tensor,
            *,
            memory_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        return self.dropout2(
            self.memory_attn(
                query=query_tokens,
                key=memory_tokens,
                value=memory_tokens,
                key_padding_mask=memory_key_padding_mask,
                need_weights=False,
            )[0]
        )

    def _ff_block(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout3(self.linear2(self.dropout(self.activation(self.linear1(x)))))


class MATQCXDecoder(nn.Module):

    def __init__(
            self,
            config: MATQCXDecoderConfig,
            *,
            max_agents: int,
            input_d_model: int,
            action_d_model: int,
            memory_d_model: int,
    ) -> None:
        super().__init__()
        if config.d_model is None:
            raise ValueError("MATQCXDecoderConfig.d_model must be set before constructing MATQCXDecoder")
        if config.num_layers <= 0:
            raise ValueError(f"MATQCXDecoderConfig.num_layers must be > 0, got {config.num_layers}")
        if config.add_agent_embeddings:
            raise ValueError("MATQCXDecoder does not support add_agent_embeddings yet")

        self.max_agents = max_agents
        self.input_d_model = input_d_model
        self.action_d_model = action_d_model
        self.d_model = config.d_model
        self.memory_d_model = memory_d_model
        self.nhead = config.nhead
        self.assume_agent_mask_is_active_prefix = config.assume_agent_mask_is_active_prefix
        self.input_projection = self._build_input_projection(
            input_d_model=input_d_model,
            d_model=self.d_model,
            act_fn_cls=config.act_fn_cls,
            token_encoder_init_gain=config.token_encoder_init_gain,
            token_encoder_projection_init_gain=config.token_encoder_projection_init_gain,
        )

        self.layers = nn.ModuleList(
            [
                MATQCXDecoderLayer(
                    action_d_model=action_d_model,
                    d_model=self.d_model,
                    memory_d_model=memory_d_model,
                    nhead=config.nhead,
                    dim_feedforward=config.dim_feedforward,
                    dropout=config.dropout,
                    act_fn_cls=config.act_fn_cls,
                    query_encoder_hidden_dims=config.query_encoder_hidden_dims,
                    context_encoder_hidden_dims=config.context_encoder_hidden_dims,
                    token_encoder_init_gain=config.token_encoder_init_gain,
                    token_encoder_projection_init_gain=config.token_encoder_projection_init_gain,
                    layer_token_encoder_end_with_act_fn=config.layer_token_encoder_end_with_act_fn,
                    normalize_context_input=config.normalize_context_input,
                    normalize_context_tokens=config.normalize_context_tokens,
                    layer_norm_eps=config.layer_norm_eps,
                    norm_first=config.norm_first,
                    bias=config.bias,
                )
                for layer_idx in range(config.num_layers)
            ]
        )
        self.norm = nn.LayerNorm(self.d_model, eps=config.layer_norm_eps, bias=config.bias)
        modules = tuple(self.modules())
        self._dropout_modules = tuple(module for module in modules if isinstance(module, (
            nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout, nn.FeatureAlphaDropout,
        )))
        self._randomized_activations = tuple(module for module in modules if isinstance(module, nn.RReLU))
        self._attention_modules = tuple(module for module in modules if isinstance(module, nn.MultiheadAttention))

        if config.transformer_ff_init_gain is not None:
            self._reinitialize_layers(feedforward_init_gain=config.transformer_ff_init_gain)

    @staticmethod
    def _build_input_projection(
            *,
            input_d_model: int,
            d_model: int,
            act_fn_cls: ActivationFactory,
            token_encoder_init_gain: float,
            token_encoder_projection_init_gain: float | None,
    ) -> nn.Module:
        if input_d_model == d_model:
            return nn.Identity()
        return MATQCBasePolicy._build_token_encoder(
            input_dim=input_d_model,
            output_dim=d_model,
            hidden_dims=None,
            act_fn_cls=act_fn_cls,
            linear_init_gain=token_encoder_init_gain,
            projection_init_gain=token_encoder_projection_init_gain,
            end_with_act_fn=False,
        )

    def forward(
            self,
            query_tokens: torch.Tensor,
            action_tokens: torch.Tensor,
            memory_tokens: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if memory_tokens.shape[1] > self.max_agents:
            raise ValueError(f"Expected memory_tokens second dim <= {self.max_agents}, got {memory_tokens.shape[1]}")
        n_queries = query_tokens.shape[1]
        n_contexts = action_tokens.shape[1]
        if n_contexts not in (n_queries, n_queries - 1):
            raise ValueError(
                f"Expected action_tokens second dim to be {n_queries} or {n_queries - 1}, got {n_contexts}"
            )

        context_attention_mask, has_visible_context, context_key_padding_mask = self._build_parallel_context_attention_mask(
            batch_size=query_tokens.shape[0],
            n_queries=n_queries,
            n_contexts=n_contexts,
            context_mask=agent_mask,
            device=query_tokens.device,
        )
        return self._decode(
            x=query_tokens,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            context_token_count=n_contexts,
            context_attention_mask=context_attention_mask,
            has_visible_context=has_visible_context,
            context_key_padding_mask=context_key_padding_mask,
            memory_mask=memory_mask,
        )

    def forward_step(
            self,
            action_tokens: torch.Tensor,
            query_token: torch.Tensor,
            memory_tokens: torch.Tensor,
            *,
            query_prefix_tokens: torch.Tensor | None = None,
            context_mask: torch.Tensor | None = None,
            query_prefix_mask: torch.Tensor | None = None,
            query_mask: torch.Tensor | None = None,
            memory_mask: torch.Tensor | None = None,
            cache: MATQCXDecoderCache | None = None,
    ) -> torch.Tensor:
        _ = query_prefix_mask, query_mask
        if query_prefix_tokens is None:
            raise ValueError("query_prefix_tokens must be provided for MATQCXDecoder.forward_step")
        if memory_tokens.shape[1] > self.max_agents:
            raise ValueError(
                f"Expected memory_tokens second dim <= {self.max_agents}, got {memory_tokens.shape[1]}"
            )
        if query_prefix_tokens.shape[1] != action_tokens.shape[1]:
            raise ValueError("query_prefix_tokens and action_tokens must have the same sequence length")

        if cache is not None:
            x = self.input_projection(query_token)
            for layer, layer_cache in zip(self.layers, cache):
                x = layer.forward_cached_step(
                    x, action_tokens[:, -1:, :], layer_cache,
                    context_mask=context_mask, memory_mask=memory_mask,
                )
            return self.norm(x)

        x = torch.cat((query_prefix_tokens, query_token), dim=1)
        combined_context_mask = context_mask

        context_attention_mask, has_visible_context, context_key_padding_mask = self._build_parallel_context_attention_mask(
            batch_size=query_token.shape[0],
            n_queries=x.shape[1],
            n_contexts=action_tokens.shape[1],
            context_mask=combined_context_mask,
            device=query_token.device,
        )
        return self._decode(
            x=x,
            action_tokens=action_tokens,
            memory_tokens=memory_tokens,
            context_token_count=action_tokens.shape[1],
            context_attention_mask=context_attention_mask,
            has_visible_context=has_visible_context,
            context_key_padding_mask=context_key_padding_mask,
            memory_mask=memory_mask,
        )[:, -1:, :]

    def create_step_cache(self, memory_tokens: torch.Tensor) -> MATQCXDecoderCache | None:
        # Prefix recomputation redraws activation noise, including modules nested
        # inside activation factories. Reusing that noise changes the joint policy.
        if any(module.training and module.p > 0.0 for module in self._dropout_modules) or any(
            module.training for module in self._randomized_activations
        ) or any(
            module.training and module.dropout > 0.0 for module in self._attention_modules
        ):
            return None
        return [
            MATQCXLayerCache(memory=project_attention_keys_values(layer.memory_attn, memory_tokens))
            for layer in self.layers
        ]

    def _decode(
            self,
            *,
            x: torch.Tensor,
            action_tokens: torch.Tensor,
            memory_tokens: torch.Tensor,
            context_token_count: int | None,
            context_attention_mask: torch.Tensor | None,
            has_visible_context: torch.Tensor | None,
            context_key_padding_mask: torch.Tensor | None,
            memory_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        memory_key_padding_mask = None if memory_mask is None else ~memory_mask
        x = self.input_projection(x)
        for layer in self.layers:
            x = layer(
                x=x,
                action_tokens=action_tokens,
                memory_tokens=memory_tokens,
                context_token_count=context_token_count,
                context_attention_mask=context_attention_mask,
                has_visible_context=has_visible_context,
                context_key_padding_mask=context_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
            )
        return self.norm(x)

    def _build_parallel_context_attention_mask(
            self,
            *,
            batch_size: int,
            n_queries: int,
            n_contexts: int,
            context_mask: torch.Tensor | None,
            device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if n_contexts == 0:
            return None, None, None

        if self._can_use_prefix_context_mask():
            has_visible_context = torch.arange(n_queries, device=device).unsqueeze(0) > 0
            has_visible_context = has_visible_context.expand(batch_size, n_queries)
            context_key_padding_mask = None
            if context_mask is not None:
                has_visible_context = has_visible_context & context_mask[:, :n_contexts].any(dim=1, keepdim=True)
                context_key_padding_mask = ~context_mask[:, :n_contexts]

            safe_visible_contexts = torch.tril(
                torch.ones(n_queries, n_contexts, dtype=torch.bool, device=device),
                diagonal=-1,
            )
            safe_visible_contexts[0, 0] = True
            return ~safe_visible_contexts, has_visible_context, context_key_padding_mask

        visible_contexts = torch.tril(
            torch.ones(n_queries, n_contexts, dtype=torch.bool, device=device),
            diagonal=-1,
        ).unsqueeze(0)
        if context_mask is not None:
            visible_contexts = visible_contexts & context_mask[:, None, :n_contexts]
        else:
            visible_contexts = visible_contexts.expand(batch_size, n_queries, n_contexts)

        attention_mask, has_visible_context = self._make_safe_context_attention_mask(visible_contexts)
        return attention_mask, has_visible_context, None

    def _build_step_context_attention_mask(
            self,
            *,
            batch_size: int,
            context_mask: torch.Tensor | None,
            n_contexts: int,
            device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None]:
        if n_contexts == 0:
            return None, None, None

        if self._can_use_prefix_context_mask():
            context_key_padding_mask = None
            if context_mask is None:
                has_visible_context = torch.ones(batch_size, 1, dtype=torch.bool, device=device)
            else:
                has_visible_context = context_mask[:, :n_contexts].any(dim=1, keepdim=True)
                context_key_padding_mask = ~context_mask[:, :n_contexts]
            return torch.zeros(1, n_contexts, dtype=torch.bool, device=device), has_visible_context, context_key_padding_mask

        if context_mask is None:
            visible_contexts = torch.ones(batch_size, 1, n_contexts, dtype=torch.bool, device=device)
        else:
            visible_contexts = context_mask[:, None, :n_contexts]

        attention_mask, has_visible_context = self._make_safe_context_attention_mask(visible_contexts)
        return attention_mask, has_visible_context, None

    def _can_use_prefix_context_mask(self) -> bool:
        return self.assume_agent_mask_is_active_prefix

    def _make_safe_context_attention_mask(
            self,
            visible_contexts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        has_visible_context = visible_contexts.any(dim=-1)
        safe_visible_contexts = visible_contexts.clone()
        safe_visible_contexts[:, :, 0] = safe_visible_contexts[:, :, 0] | ~has_visible_context
        attention_mask = ~safe_visible_contexts
        return attention_mask.repeat_interleave(self.nhead, dim=0), has_visible_context

    def _reinitialize_layers(self, *, feedforward_init_gain: float) -> None:
        for layer in self.layers:
            reinitialize_multihead_attention(layer.query_context_attn)
            reinitialize_multihead_attention(layer.memory_attn)
            init_transformer_feedforward(layer, gain=feedforward_init_gain)
