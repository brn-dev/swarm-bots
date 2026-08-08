from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch import nn

from experiments.mjw_experiment_common import (
    ContinuousActionDistVariant,
    MJWScenarioName,
)
from experiments.mjw_experiment_common import run_experiment as run_mjw_experiment
from swarmbots.learn.algos.sac.recurrent_tmasac_policy import (
    ActorStateCriticInputConfig,
)
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
EXPERIMENT_TOTAL_TIMESTEPS = 100_000_000

TMASACExperimentVariant = Literal[
    "tmasac_baseline",
    "tmasac_swiglu",
    "slstm_two_small_actor_state_critic",
    "slstm_two_small_swiglu_actor_state_critic",
]


def run_tmasac_experiment(
    *,
    experiment_run_name: str,
    scenario_name: MJWScenarioName,
    scenario_kwargs: dict[str, object],
    variant: TMASACExperimentVariant,
    entrypoint_path: Path,
    continuous_action_dist: ContinuousActionDistVariant = "gumbel_softmax_sign_magnitude_beta",
) -> None:
    is_recurrent = variant.startswith("slstm_")
    mat_transformer_ff_config, actor_transformer_ff_config = _make_feedforward_configs(
        variant=variant,
    )

    run_mjw_experiment(
        num_envs=1024,
        rollout_steps_per_env=1,
        variant_name=(
            variant
            if continuous_action_dist == "gumbel_softmax_sign_magnitude_beta"
            else f"{variant}_{continuous_action_dist}"
        ),
        entrypoint_path=entrypoint_path,
        continuous_action_dist=continuous_action_dist,
        policy_variant="r_tmasac" if is_recurrent else "tmasac",
        mat_add_agent_embeddings=False,
        mat_encoder_transformer_ff_config=mat_transformer_ff_config,
        nop_add_agent_embeddings_transition_model=False,
        use_nop=True,
        experiment_run_name=experiment_run_name,
        scenario_name=scenario_name,
        scenario_kwargs=scenario_kwargs,
        total_timesteps=EXPERIMENT_TOTAL_TIMESTEPS,
        sac_learning_rate=5e-5,
        sac_ent_coef_learning_rate=None,
        sac_ent_coef="auto_0.05",
        sac_target_entropy="auto_0.1",
        sac_batch_size=16 if is_recurrent else None,
        sac_buffer_capacity_per_env=1024 if is_recurrent else None,
        sac_recurrent_burn_in_steps=32,
        sac_recurrent_learning_steps=64,
        sac_temporal_state_store_interval=16,
        sac_temporal_state_storage_dtype=torch.float16 if is_recurrent else None,
        rmat_actor_d_model=ACTOR_D_MODEL if is_recurrent else None,
        rmat_actor_transformer_ff_config=(
            actor_transformer_ff_config if is_recurrent else None
        ),
        rmat_actor_inter_module_mlp=is_recurrent,
        r_tmasac_actor_state_critic_input_config=(
            ActorStateCriticInputConfig(
                projection_dim=ACTOR_D_MODEL,
                projection_hidden_dims=(ACTOR_D_MODEL,),
            )
            if is_recurrent
            else None
        ),
        rmat_temporal_model_cls=SLSTMTemporalSequenceModel if is_recurrent else None,
        rmat_temporal_model_config=(
            SLSTMTemporalSequenceModelConfig(num_heads=4) if is_recurrent else None
        ),
        rmat_temporal_residual=False,
        rmat_temporal_layer_norm=False,
        rmat_use_temporal_output_projection=not is_recurrent,
    )


def _make_feedforward_configs(
    *,
    variant: TMASACExperimentVariant,
) -> tuple[MLPConfig | SwiGLUConfig, MLPConfig | SwiGLUConfig | None]:
    if variant == "tmasac_baseline":
        return MLPConfig(hidden_dims=[512, 512]), None
    if variant == "tmasac_swiglu":
        return _make_stacked_swiglu_config(), None
    if variant == "slstm_two_small_actor_state_critic":
        return MLPConfig(hidden_dims=[512]), MLPConfig(hidden_dims=[512])
    if variant == "slstm_two_small_swiglu_actor_state_critic":
        swiglu_config = SwiGLUConfig(hidden_dim=PARAMETER_MATCHED_SWIGLU_HIDDEN_DIM)
        return swiglu_config, swiglu_config
    raise ValueError(f"Unknown TMASAC experiment variant: {variant!r}")


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
