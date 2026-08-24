from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.tmasac_experiment_common import run_tmasac_experiment
from experiments.thesis_experiment_common import run_thesis_ppo_experiment
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import (
    MJWPreConnectedUnitLocationsConfig,
)
from swarmbots.scenario_presets.scenario_presets_kwargs import (
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    make_scenario_kwargs,
)

EXPERIMENT_RUN_NAME = "thesis_mjw_po_wall_disconnected_finetune"
ADDITIONAL_TIMESTEPS = 20_000_000
EVALUATION_MILESTONES = (50.0, 100.0)
EVALUATION_RECORDING_EPISODES = 0
LIVE_RECORDING_SCHEDULE: dict[float, int] = {}
LONG_ADDITIONAL_TIMESTEPS = 50_000_000
LONG_EXPERIMENT_RUN_NAME = "thesis_mjw_po_wall_disconnected_finetune_50m"


def _make_disconnected_pool(*, pool_seed_start: int) -> MJWPreConnectedUnitLocationsConfig:
    return MJWPreConnectedUnitLocationsConfig(
        num_units=5,
        num_unit_probs={4: 1.0, 5: 1.0},
        max_radius=1.5,
        unconnected_prob=1.0,
        z_pos=0.5,
        pool_seeds=tuple(range(pool_seed_start, pool_seed_start + 50)),
    )


SCENARIO_KWARGS = make_scenario_kwargs(
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    {
        "continuous_connector_actions": True,
        "unit_start_locations": _make_disconnected_pool(pool_seed_start=42_000),
    },
)
EVALUATION_SCENARIO_KWARGS = make_scenario_kwargs(
    PO_WALL_MEDIUM_SCENARIO_KWARGS,
    {
        "continuous_connector_actions": True,
        "unit_start_locations": _make_disconnected_pool(pool_seed_start=3_000_000),
    },
)


def run_experiment(*, checkpoint_path: Path, entrypoint_path: Path) -> None:
    checkpoint_path = _resolve_checkpoint(checkpoint_path)

    run_tmasac_experiment(
        experiment_run_name=EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        evaluation_scenario_kwargs=EVALUATION_SCENARIO_KWARGS,
        evaluation_milestones=EVALUATION_MILESTONES,
        evaluation_recording_episodes=EVALUATION_RECORDING_EPISODES,
        live_recording_schedule=LIVE_RECORDING_SCHEDULE,
        variant="tmasac_baseline",
        variant_name="tmasac_disconnected_finetune",
        entrypoint_path=entrypoint_path,
        load_path=checkpoint_path,
        additional_timesteps=ADDITIONAL_TIMESTEPS,
    )


def run_tmasac_50m(*, checkpoint_path: Path, entrypoint_path: Path) -> None:
    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    run_tmasac_experiment(
        experiment_run_name=LONG_EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        evaluation_scenario_kwargs=EVALUATION_SCENARIO_KWARGS,
        evaluation_milestones=EVALUATION_MILESTONES,
        evaluation_recording_episodes=EVALUATION_RECORDING_EPISODES,
        live_recording_schedule=LIVE_RECORDING_SCHEDULE,
        variant="tmasac_baseline",
        variant_name="tmasac_baseline",
        entrypoint_path=entrypoint_path,
        load_path=checkpoint_path,
        additional_timesteps=LONG_ADDITIONAL_TIMESTEPS,
    )


def run_tmasac_no_connectors_50m(
    *,
    checkpoint_path: Path,
    entrypoint_path: Path,
) -> None:
    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    run_tmasac_experiment(
        experiment_run_name=LONG_EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        evaluation_scenario_kwargs=EVALUATION_SCENARIO_KWARGS,
        evaluation_milestones=EVALUATION_MILESTONES,
        evaluation_recording_episodes=EVALUATION_RECORDING_EPISODES,
        live_recording_schedule=LIVE_RECORDING_SCHEDULE,
        variant="tmasac_baseline",
        variant_name="tmasac_no_connectors",
        entrypoint_path=entrypoint_path,
        load_path=checkpoint_path,
        additional_timesteps=LONG_ADDITIONAL_TIMESTEPS,
        disable_connector_actions=True,
        migrate_removed_connector_actions=True,
    )


def run_mat_qcx_50m(*, checkpoint_path: Path, entrypoint_path: Path) -> None:
    checkpoint_path = _resolve_checkpoint(checkpoint_path)
    run_thesis_ppo_experiment(
        experiment_run_name=LONG_EXPERIMENT_RUN_NAME,
        scenario_name="wall",
        scenario_kwargs=SCENARIO_KWARGS,
        evaluation_scenario_kwargs=EVALUATION_SCENARIO_KWARGS,
        evaluation_milestones=EVALUATION_MILESTONES,
        evaluation_recording_episodes=EVALUATION_RECORDING_EPISODES,
        live_recording_schedule=LIVE_RECORDING_SCHEDULE,
        variant="mat_qcx",
        variant_name="mat_qcx",
        entrypoint_path=entrypoint_path,
        load_path=checkpoint_path,
        additional_timesteps=LONG_ADDITIONAL_TIMESTEPS,
    )


def _resolve_checkpoint(checkpoint_path: Path) -> Path:
    checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    return checkpoint_path
