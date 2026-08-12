from __future__ import annotations

from pathlib import Path
from typing import Literal

from experiments.mjw_experiment_common import MJWScenarioName, PolicyVariant
from experiments.mjw_experiment_common import run_experiment as run_mjw_experiment
from experiments.tmasac_experiment_common import (
    TMASACExperimentVariant,
    run_tmasac_experiment,
)

EXPERIMENT_TOTAL_TIMESTEPS = 100_000_000

ThesisAlgorithmVariant = Literal[
    "mappo",
    "mat_qcx",
    "mat_ind",
    "mat_orig",
    "tmasac_baseline",
    "slstm_two_small_actor_state_critic",
]

PPO_POLICY_VARIANTS: dict[ThesisAlgorithmVariant, PolicyVariant] = {
    "mappo": "mat_ind",
    "mat_qcx": "mat_qcx",
    "mat_ind": "mat_ind",
    "mat_orig": "mat_orig",
}
TMASAC_VARIANTS = {
    "tmasac_baseline",
    "slstm_two_small_actor_state_critic",
}


def run_thesis_experiment(
    *,
    experiment_run_name: str,
    scenario_name: MJWScenarioName,
    scenario_kwargs: dict[str, object],
    variant: ThesisAlgorithmVariant,
    entrypoint_path: Path,
) -> None:
    if variant in TMASAC_VARIANTS:
        run_tmasac_experiment(
            experiment_run_name=experiment_run_name,
            scenario_name=scenario_name,
            scenario_kwargs=scenario_kwargs,
            variant=_as_tmasac_variant(variant),
            entrypoint_path=entrypoint_path,
        )
        return

    run_mjw_experiment(
        num_envs=1024,
        rollout_steps_per_env=4,
        variant_name=variant,
        entrypoint_path=entrypoint_path,
        policy_variant=PPO_POLICY_VARIANTS[variant],
        mat_add_agent_embeddings=False,
        mat_use_agent_attention=variant != "mappo",
        nop_add_agent_embeddings_transition_model=False,
        use_nop=True,
        use_transition_obs=False,
        experiment_run_name=experiment_run_name,
        scenario_name=scenario_name,
        scenario_kwargs=scenario_kwargs,
        total_timesteps=EXPERIMENT_TOTAL_TIMESTEPS,
    )


def _as_tmasac_variant(variant: ThesisAlgorithmVariant) -> TMASACExperimentVariant:
    if variant == "tmasac_baseline":
        return variant
    if variant == "slstm_two_small_actor_state_critic":
        return variant
    raise ValueError(f"Not a TMASAC thesis variant: {variant!r}")
