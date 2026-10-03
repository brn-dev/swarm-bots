from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MATInitGains:
    obs_encoder: float = 1.0
    obs_encoder_projection: float | None = 1.0
    encoder_transformer_ff: float | None = 1.0
    decoder_token_encoder: float = 1.0
    decoder_token_encoder_projection: float | None = 1.0
    decoder_transformer_ff: float | None = 1.0
    actor_head: float = 1.0
    action_net: float = 0.01
    critic_local_projection: float = 1.0
    critic_value_regressor: float = 1.0
    critic_value_head: float = 0.01


@dataclass(frozen=True)
class NOPInitGains:
    pre_transition: float = 1.0
    transition_coembed: float = 1.0
    transition_transformer_ff: float | None = 1.0
    transition_head: float = 0.01
    pre_predictors: float = 1.0
    predictors: float = 0.01


@dataclass(frozen=True)
class MATNormalizationConfig:
    normalize_obs_inputs: bool = False
    normalize_encoder_tokens: bool = False
    normalize_query_input: bool = False
    normalize_context_input: bool = False
    normalize_action_input: bool = False
    normalize_memory_input: bool = False
    normalize_query_tokens: bool = False
    normalize_context_tokens: bool = False
    normalize_action_tokens: bool = False
    normalize_memory_tokens: bool = False
    normalize_actor_head_input: bool = False
    normalize_prev_binary_actions: bool = False
