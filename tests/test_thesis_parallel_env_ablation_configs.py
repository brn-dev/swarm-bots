from pathlib import Path
from unittest.mock import patch

import pytest

from experiments import thesis_experiment_common
from experiments.thesis_parallel_env_ablation_po_wall_medium import plot_results
from experiments.thesis_parallel_env_ablation_po_wall_medium.scripts import common


@pytest.mark.parametrize("algorithm_variant", ("mat_ind", "mat_qcx"))
@pytest.mark.parametrize(
    ("num_envs", "rollout_steps_per_env"),
    common.PARALLEL_ENV_CONFIGS,
)
def test_parallel_env_variants_keep_rollout_batch_size_fixed(
    algorithm_variant: common.ParallelEnvAlgorithmVariant,
    num_envs: int,
    rollout_steps_per_env: int,
) -> None:
    entrypoint_path = Path(__file__)

    with patch.object(thesis_experiment_common, "run_mjw_experiment") as run:
        common.run_experiment(
            algorithm_variant=algorithm_variant,
            num_envs=num_envs,
            rollout_steps_per_env=rollout_steps_per_env,
            entrypoint_path=entrypoint_path,
        )

    run.assert_called_once_with(
        num_envs=num_envs,
        rollout_steps_per_env=rollout_steps_per_env,
        variant_name=f"{algorithm_variant}_{num_envs}x{rollout_steps_per_env}",
        entrypoint_path=entrypoint_path,
        continuous_action_dist="sign_magnitude_beta",
        policy_variant=algorithm_variant,
        mat_add_agent_embeddings=False,
        mat_use_agent_attention=True,
        nop_add_agent_embeddings_transition_model=False,
        use_nop=True,
        use_transition_obs=False,
        experiment_run_name="thesis_parallel_env_ablation_po_wall_medium",
        scenario_name="wall",
        scenario_kwargs=common.SCENARIO_KWARGS,
        total_timesteps=100_000_000,
    )
    assert num_envs * rollout_steps_per_env == common.ROLLOUT_BATCH_SIZE


def test_parallel_env_ablation_has_one_entrypoint_per_variant() -> None:
    scripts_dir = Path(common.__file__).parent
    expected_entrypoints = {
        f"run_{algorithm_variant}_{num_envs}x{rollout_steps_per_env}.py"
        for algorithm_variant in ("mat_ind", "mat_qcx")
        for num_envs, rollout_steps_per_env in common.PARALLEL_ENV_CONFIGS
    }

    assert {path.name for path in scripts_dir.glob("run_*.py")} == expected_entrypoints


def test_parallel_env_ablation_plot_includes_all_variants() -> None:
    assert plot_results.EXPERIMENT_RUN_DIR.name == common.EXPERIMENT_RUN_NAME
    assert set(plot_results.GROUP_ORDER) == {
        f"{algorithm_variant}_{num_envs}x{rollout_steps_per_env}"
        for algorithm_variant in ("mat_ind", "mat_qcx")
        for num_envs, rollout_steps_per_env in common.PARALLEL_ENV_CONFIGS
    }
