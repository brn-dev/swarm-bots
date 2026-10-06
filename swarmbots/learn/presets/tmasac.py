from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

from swarmbots.learn.algos.r_mat.temporal_sequence_model import (
    LSTMTemporalSequenceModel,
    LSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import ActorStateCriticInputConfig
from swarmbots.learn.algos.sac.sac_nop import SACNOPLatentSource
from swarmbots.learn.algos.xlstm.slstm import (
    SLSTMTemporalSequenceModel,
    SLSTMTemporalSequenceModelConfig,
)
from swarmbots.learn.nn_components.feed_forward import (
    GLUStackConfig,
    MLPConfig,
    SwiGLUConfig,
)

ACTOR_D_MODEL = 256

PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM = 344

TMASACVariant = Literal[
    "tmasac",
    "tmasac_dec",
    "tmasac_shared_encoder",
    "tmasac_swiglu",
    "tmasac_slstm",
    "tmasac_slstm_no_residual",
    "tmasac_slstm_shared_encoder",
    "tmasac_lstm",
    "tmasac_slstm_swiglu",
]


def make_tmasac_options(
    variant: TMASACVariant,
    *,
    include_slstm_memory_strength: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Resolve a TMASAC preset's architecture and optimizer/replay settings."""
    temporal_model_cls, temporal_model_config = _make_temporal_model_spec(variant=variant)
    is_recurrent = temporal_model_cls is not None
    recurrent_shared_encoder = variant == "tmasac_slstm_shared_encoder"
    shared_encoder = variant == "tmasac_shared_encoder" or recurrent_shared_encoder
    encoder_ff_config, actor_ff_config = _make_feedforward_configs(variant=variant)
    policy_options = {
        "policy_variant": "tmasac_recurrent" if is_recurrent else "tmasac_dec" if variant == "tmasac_dec" else "tmasac",
        "mat_encoder_transformer_ff_config": encoder_ff_config,
        "rmat_actor_d_model": ACTOR_D_MODEL if is_recurrent else None,
        "rmat_actor_transformer_ff_config": actor_ff_config if is_recurrent else None,
        "rmat_actor_inter_module_mlp": is_recurrent and not recurrent_shared_encoder,
        "tmasac_actor_state_critic_input_config": (
            ActorStateCriticInputConfig(
                projection_dim=ACTOR_D_MODEL,
                projection_hidden_dims=(ACTOR_D_MODEL,),
                include_slstm_memory_strength=include_slstm_memory_strength,
            )
            if is_recurrent and not recurrent_shared_encoder
            else None
        ),
        "tmasac_shared_encoder_num_layers": 2 if shared_encoder else None,
        "tmasac_recurrent_shared_encoder": recurrent_shared_encoder,
        "tmasac_actor_encoder_num_layers": 1 if shared_encoder else 2,
        "tmasac_critic_encoder_num_layers": 1 if shared_encoder else 2,
        "tmasac_nop_latent_source": (
            SACNOPLatentSource.SHARED_ENCODER if shared_encoder else SACNOPLatentSource.CRITIC
        ),
        "rmat_temporal_model_cls": temporal_model_cls,
        "rmat_temporal_model_config": temporal_model_config,
        "rmat_temporal_residual": variant.startswith("tmasac_slstm") and variant != "tmasac_slstm_no_residual",
        "rmat_temporal_layer_norm": False,
        "rmat_use_temporal_output_projection": not is_recurrent,
        "rmat_experimental_compile_lstm": variant == "tmasac_lstm",
    }
    algorithm_options: dict[str, Any] = {
        "learning_rate": 5e-5,
        "ent_coef_learning_rate": None,
        "ent_coef": "auto_0.05",
        "target_entropy": "auto_0.1",
    }
    if is_recurrent:
        algorithm_options.update(
            batch_size=16,
            buffer_capacity_per_env=1024,
            burn_in_steps=32,
            learning_steps=64,
            temporal_state_store_interval=16,
            temporal_state_storage_dtype=torch.float16,
        )
    return policy_options, algorithm_options


def _make_feedforward_configs(
    *,
    variant: TMASACVariant,
) -> tuple[MLPConfig | SwiGLUConfig, MLPConfig | SwiGLUConfig | None]:
    if variant in {
        "tmasac",
        "tmasac_dec",
        "tmasac_shared_encoder",
        "tmasac_slstm_shared_encoder",
    }:
        if variant == "tmasac_slstm_shared_encoder":
            config = MLPConfig(hidden_dims=[512, 512])
            return config, config
        return MLPConfig(hidden_dims=[512, 512]), None
    if variant == "tmasac_swiglu":
        return _make_stacked_swiglu_config(), None
    if variant in {
        "tmasac_slstm",
        "tmasac_slstm_no_residual",
        "tmasac_lstm",
    }:
        # Only the recurrent actor splits its feed-forward budget across the
        # two RMAT blocks. The ordinary critic still needs the full MAT block.
        return MLPConfig(hidden_dims=[512, 512]), MLPConfig(hidden_dims=[512])
    if variant == "tmasac_slstm_swiglu":
        return _make_stacked_swiglu_config(), SwiGLUConfig(hidden_dim=PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM)
    raise ValueError(f"Unknown TMASAC variant: {variant!r}")


def _make_temporal_model_spec(
    *,
    variant: TMASACVariant,
) -> tuple[type[nn.Module] | None, object | None]:
    if variant.startswith("tmasac_slstm"):
        return SLSTMTemporalSequenceModel, SLSTMTemporalSequenceModelConfig(num_heads=4)
    if variant == "tmasac_lstm":
        return LSTMTemporalSequenceModel, LSTMTemporalSequenceModelConfig()
    return None, None


def _make_stacked_swiglu_config() -> SwiGLUConfig:
    return SwiGLUConfig(
        hidden_dim=PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM,
        stacked=GLUStackConfig(
            n_layers=2,
            pre_norm=nn.LayerNorm,
            norm_first_layer=False,
            residual=True,
            residual_first_layer=False,
        ),
    )
