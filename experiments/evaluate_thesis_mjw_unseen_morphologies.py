from __future__ import annotations

import argparse
import csv
import json
import re
import statistics
import sys
from collections.abc import Callable, Iterable, Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Sequence, cast

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_OUTPUT_PATH = (
    REPO_ROOT / "experiments" / "thesis_mjw_unseen_morphologies" / "results" / "evaluation.json"
)
DEFAULT_UNIT_COUNTS = tuple(range(2, 11))
POOL_SEED_UNIT_STRIDE = 100_000
RESULT_SCHEMA_VERSION = 1
FINAL_CHECKPOINT_PATTERN = re.compile(r"^model_(?P<steps>\d+)_steps_final\.pt$")
BEST_CHECKPOINT_NAME = "model_best.pt"

TargetKey = Literal["po_wall_tmasac", "find_opening_slstm_tmasac"]


@dataclass(frozen=True)
class EvaluationTarget:
    key: TargetKey
    display_name: str
    scenario_name: Literal["wall", "find_opening"]
    tmasac_variant: Literal["tmasac_baseline", "slstm_two_small_actor_state_critic"]
    checkpoint_group_dirs: tuple[Path, ...]
    include_slstm_memory_strength: bool = False


@dataclass(frozen=True)
class EvaluationConfig:
    unit_counts: tuple[int, ...]
    pool_size: int
    episodes: int
    max_parallel_envs: int
    pool_seed_base: int
    rollout_seed: int
    deterministic: bool
    episode_length: int
    unconnected_prob: float = 0.0
    disable_connector_actions: bool = False


@dataclass(frozen=True)
class MetricSummary:
    mean: float
    std: float
    min: float
    max: float


TARGETS: dict[TargetKey, EvaluationTarget] = {
    "po_wall_tmasac": EvaluationTarget(
        key="po_wall_tmasac",
        display_name="TMASAC on PO-Wall (Medium)",
        scenario_name="wall",
        tmasac_variant="tmasac_baseline",
        checkpoint_group_dirs=(
            REPO_ROOT / "runs" / "thesis_mjw_po_wall_medium" / "tmasac_baseline",
            REPO_ROOT
            / "runs"
            / "mjw_po_wall_medium_1024x1_tmasac"
            / "tmasac_lr=5e-5_bigger_mlps",
        ),
    ),
    "find_opening_slstm_tmasac": EvaluationTarget(
        key="find_opening_slstm_tmasac",
        display_name="TMASAC + sLSTM on Find-Opening",
        scenario_name="find_opening",
        tmasac_variant="slstm_two_small_actor_state_critic",
        checkpoint_group_dirs=(
            REPO_ROOT
            / "runs"
            / "thesis_mjw_find_opening"
            / "slstm_two_small_actor_state_critic",
            REPO_ROOT
            / "runs"
            / "mjw_find_opening_tmasac"
            / "slstm_two_small_actor_state_critic",
        ),
        include_slstm_memory_strength=True,
    ),
}


def _best_checkpoint_timesteps(checkpoint_path: Path) -> int:
    metadata_path = Path(f"{checkpoint_path}.json")
    if not metadata_path.is_file():
        return -1
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    return int(metadata.get("n_total_timesteps", -1))


def discover_evaluation_checkpoints(group_dirs: Sequence[Path]) -> list[Path]:
    checkpoints: list[Path] = []
    for group_dir in group_dirs:
        if not group_dir.is_dir():
            continue
        for run_dir in sorted(path for path in group_dir.iterdir() if path.is_dir()):
            models_dir = run_dir / "models"
            candidates = [
                (int(match.group("steps")), path)
                for path in models_dir.glob("*_final.pt")
                if (match := FINAL_CHECKPOINT_PATTERN.fullmatch(path.name)) is not None
            ]
            if candidates:
                checkpoints.append(max(candidates, key=lambda item: (item[0], item[1].name))[1].resolve())
                continue

            best_candidates = list((models_dir / "best").rglob(BEST_CHECKPOINT_NAME))
            if best_candidates:
                checkpoints.append(
                    max(
                        best_candidates,
                        key=lambda path: (_best_checkpoint_timesteps(path), str(path)),
                    ).resolve()
                )
    return sorted(set(checkpoints))


def make_pool_seeds(*, pool_seed_base: int, unit_count: int, pool_size: int) -> tuple[int, ...]:
    if pool_size > POOL_SEED_UNIT_STRIDE:
        raise ValueError(f"pool_size must not exceed {POOL_SEED_UNIT_STRIDE}")
    start = pool_seed_base + (unit_count - min(DEFAULT_UNIT_COUNTS)) * POOL_SEED_UNIT_STRIDE
    return tuple(range(start, start + pool_size))


def resolve_num_envs(*, episodes: int, max_parallel_envs: int) -> int:
    if episodes > max_parallel_envs:
        raise ValueError(
            "One-episode-per-environment evaluation requires --max-parallel-envs "
            f"({max_parallel_envs}) to be at least --episodes ({episodes})"
        )
    return episodes


def summarize_values(values: Sequence[float]) -> MetricSummary:
    if not values:
        raise ValueError("Cannot summarize an empty metric")
    return MetricSummary(
        mean=float(statistics.fmean(values)),
        std=float(statistics.pstdev(values)),
        min=float(min(values)),
        max=float(max(values)),
    )


def summarize_episode_metrics(metrics: dict[str, list[float]]) -> dict[str, object]:
    successes = metrics["success"]
    return {
        "episode_count": len(successes),
        "success_count": int(sum(successes)),
        "success_rate_percent": 100.0 * statistics.fmean(successes),
        "episode_return": asdict(summarize_values(metrics["episode_return"])),
        "episode_length": asdict(summarize_values(metrics["episode_length"])),
        "progress_reward": asdict(summarize_values(metrics["progress_reward"])),
        "guidance_reward": asdict(summarize_values(metrics["guidance_reward"])),
    }


def _target_scenario_kwargs(
    target: EvaluationTarget,
    *,
    unit_count: int,
    pool_seeds: tuple[int, ...],
    unconnected_prob: float = 0.0,
) -> dict[str, object]:
    from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig

    unit_start_locations = MJWPreConnectedUnitLocationsConfig(
        num_units=unit_count,
        num_unit_probs=None,
        unconnected_prob=unconnected_prob,
        max_radius=1.5,
        z_pos=0.5,
        pool_seeds=pool_seeds,
    )
    if target.scenario_name == "find_opening":
        return {
            "continuous_connector_actions": True,
            "unit_start_locations": unit_start_locations,
        }

    from swarmbots.scenario_presets.scenario_presets_kwargs import (
        PO_WALL_MEDIUM_SCENARIO_KWARGS,
        make_scenario_kwargs,
    )

    return make_scenario_kwargs(
        PO_WALL_MEDIUM_SCENARIO_KWARGS,
        {
            "continuous_connector_actions": True,
            "unit_start_locations": unit_start_locations,
        },
    )


def _build_env_and_policy(
    *,
    target: EvaluationTarget,
    unit_count: int,
    pool_seeds: tuple[int, ...],
    num_envs: int,
    episode_length: int,
    device: Any,
    unconnected_prob: float = 0.0,
    disable_policy_connector_actions: bool = False,
) -> tuple[Any, Any]:
    import torch
    from torch import nn

    from experiments.mjw_experiment_common import (
        _make_base_policy,
        configure_float32_matmul_precision,
        make_vector_env,
        set_actuator_gsde_init_joint_stds,
        wrap_vec_env,
    )
    from experiments.tmasac_experiment_common import (
        ACTOR_D_MODEL,
        _make_feedforward_configs,
        _make_temporal_model_spec,
    )
    from experiments.transformer_policy_common import (
        MATInitGains,
        MATNormalizationConfig,
        NOPInitGains,
    )
    from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderSelfAttentionMode
    from swarmbots.learn.algos.sac.recurrent_tmasac_policy import ActorStateCriticInputConfig
    from swarmbots.learn.algos.sac.sac_nop import SACNOPLatentSource
    from swarmbots.learn.swarmbots_obs_indices import build_obs_indices

    configure_float32_matmul_precision()
    scenario_kwargs = _target_scenario_kwargs(
        target,
        unit_count=unit_count,
        pool_seeds=pool_seeds,
        unconnected_prob=unconnected_prob,
    )
    vector_env = make_vector_env(
        episode_length=episode_length,
        num_envs=num_envs,
        first_episode_lengths=None,
        settle_initial_reset=True,
        device=device,
        scenario_name=target.scenario_name,
        scenario_kwargs=scenario_kwargs,
    )
    env_settings = vector_env.get_settings()
    obs_indices = build_obs_indices(
        env_settings=env_settings,
        local_obs_dim=int(vector_env.single_observation_space["local_obs"].shape[-1]),
        global_obs_dim=int(vector_env.single_observation_space["global_obs"].shape[-1]),
        hidden_local_vars_dim=int(vector_env.single_observation_space["hidden_local_vars"].shape[-1]),
        hidden_global_vars_dim=int(vector_env.single_observation_space["hidden_global_vars"].shape[-1]),
    )
    env = wrap_vec_env(
        vector_env=vector_env,
        obs_indices=obs_indices,
        gamma=0.99,
        use_popart=False,
        rollout_device=device,
        disable_connector_actions=disable_policy_connector_actions,
    )

    temporal_model_cls, temporal_model_config = _make_temporal_model_spec(
        variant=target.tmasac_variant,
    )
    is_recurrent = temporal_model_cls is not None
    mat_transformer_ff_config, actor_transformer_ff_config = _make_feedforward_configs(
        variant=target.tmasac_variant,
    )
    mat_init_gains = MATInitGains()
    policy = _make_base_policy(
        env=env,
        policy_variant="r_tmasac" if is_recurrent else "tmasac",
        enc_d_model=256,
        enc_nhead=4,
        dec_d_model=128,
        dec_nhead=2,
        use_popart=False,
        popart_beta=5e-4,
        popart_init_sigma=0.65,
        compile_policy_modules=True,
        policy_compile_mode="default",
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
        initial_stickiness=0.25,
        gsde_init_stds=[0.25, 0.30],
        mat_add_agent_embeddings=False,
        mat_use_agent_attention=True,
        mat_decoder_self_attention_mode=MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
        act_fn_cls=nn.GELU,
        mat_init_gains=mat_init_gains,
        nop_init_gains=NOPInitGains(),
        mat_normalization=MATNormalizationConfig(),
        use_nop=True,
        nop_add_agent_embeddings_transition_model=False,
        nop_skip_first_transition_for_critic=True,
        compile_world_model_modules=True,
        world_model_loss_coef=0.1,
        world_model_num_next_steps=3,
        transition_model_d_model=128,
        transition_model_nhead=2,
        mat_encoder_transformer_ff_config=mat_transformer_ff_config,
        rmat_actor_d_model=ACTOR_D_MODEL if is_recurrent else None,
        rmat_actor_transformer_ff_config=actor_transformer_ff_config if is_recurrent else None,
        rmat_actor_inter_module_mlp=is_recurrent,
        r_tmasac_actor_state_critic_input_config=(
            ActorStateCriticInputConfig(
                projection_dim=ACTOR_D_MODEL,
                projection_hidden_dims=(ACTOR_D_MODEL,),
                include_slstm_memory_strength=target.include_slstm_memory_strength,
            )
            if is_recurrent
            else None
        ),
        tmasac_actor_encoder_num_layers=2,
        tmasac_critic_encoder_num_layers=2,
        tmasac_nop_latent_source=SACNOPLatentSource.CRITIC,
        obs_indices=obs_indices,
        rmat_temporal_model_cls=temporal_model_cls,
        rmat_temporal_model_config=temporal_model_config,
        rmat_temporal_residual=False,
        rmat_temporal_layer_norm=False,
        rmat_use_temporal_output_projection=not is_recurrent,
        rmat_experimental_compile_lstm=False,
        assume_agent_mask_is_active_prefix=True,
        bernoulli_initial_prob=0.8,
    )
    actuators_per_limb = env.actuators_dim // env.connectors_dim
    set_actuator_gsde_init_joint_stds(
        policy=policy,
        actuators_per_limb=actuators_per_limb,
        joint_stds=[0.25, 0.30],
    )
    policy.to(device)
    return env, policy


def _load_checkpoint(*, checkpoint_path: Path, env: Any, policy: Any) -> None:
    from swarmbots.learn.checkpointing import (
        apply_env_state,
        extract_env_state,
        extract_policy_state_dict,
        freeze_env_normalization,
        load_checkpoint,
    )

    checkpoint = load_checkpoint(checkpoint_path, map_location="cpu")
    policy_state_dict = _align_torch_compile_state_dict_keys(
        extract_policy_state_dict(checkpoint),
        target_keys=policy.state_dict().keys(),
    )
    policy.load_state_dict(policy_state_dict, strict=True)
    apply_env_state(env, extract_env_state(checkpoint))
    freeze_env_normalization(env)


def _align_torch_compile_state_dict_keys(
    state_dict: Mapping[str, Any],
    *,
    target_keys: Iterable[str],
) -> dict[str, Any]:
    from swarmbots.learn.checkpointing import align_torch_compile_state_dict_keys

    return align_torch_compile_state_dict_keys(state_dict, target_keys=target_keys)


def evaluate_policy(
    *,
    env: Any,
    policy: Any,
    episode_count: int,
    deterministic: bool,
    rollout_seed: int,
    progress_description: str | None = None,
    show_progress: bool = True,
    on_reset: Callable[[], None] | None = None,
    disable_connector_actions: bool = False,
) -> dict[str, object]:
    import torch
    from tqdm.auto import tqdm

    from swarmbots.learn.rollout_utils import append_episode_infos

    if episode_count != env.num_envs:
        raise ValueError(
            "One-episode-per-environment evaluation requires episode_count "
            f"({episode_count}) to equal env.num_envs ({env.num_envs})"
        )
    metrics = {
        "episode_return": [],
        "episode_length": [],
        "progress_reward": [],
        "guidance_reward": [],
        "success": [],
    }
    with tqdm(
        total=episode_count,
        desc=progress_description or "Episodes",
        unit="episode",
        dynamic_ncols=True,
        disable=not show_progress,
    ) as progress:
        policy.eval()
        torch.manual_seed(rollout_seed)
        obs, _ = env.reset(seed=rollout_seed)
        if on_reset is not None:
            on_reset()
        previous_actions = None
        if policy.requires_previous_actions():
            previous_actions = torch.zeros(
                (env.num_envs, env.n_agents, env.action_space.total_agent_action_dim),
                device=obs["local_obs"].device,
                dtype=obs["local_obs"].dtype,
            )
        temporal_state = policy.initial_temporal_state(
            batch_size=env.num_envs,
            n_agents=env.n_agents,
            device=obs["local_obs"].device,
            dtype=obs["local_obs"].dtype,
        )
        episode_start_mask = torch.ones(
            (env.num_envs,),
            device=obs["local_obs"].device,
            dtype=torch.bool,
        )
        completed_envs = [False] * env.num_envs

        while len(metrics["success"]) < episode_count:
            with torch.no_grad():
                actions, temporal_state = policy.act_with_temporal_state(
                    local_obs=obs["local_obs"],
                    global_obs=obs["global_obs"],
                    hidden_local_vars=obs["hidden_local_vars"],
                    hidden_global_vars=obs["hidden_global_vars"],
                    agent_mask=obs.get("agent_mask"),
                    previous_actions=previous_actions,
                    deterministic=deterministic,
                    temporal_state=temporal_state,
                    episode_start_mask=episode_start_mask,
                )
            if disable_connector_actions:
                actions = actions.clone()
                actions[..., env.actuators_dim :] = -1.0
            obs, _rewards, terminations, truncations, infos = env.step(actions)
            dones = torch.logical_or(terminations, truncations)
            completed_env_indices = torch.nonzero(dones, as_tuple=False).flatten().tolist()
            completed_episode_infos: list[dict[str, Any]] = []
            append_episode_infos(
                episode_infos=completed_episode_infos,
                infos=infos,
                dones=dones,
            )
            if len(completed_episode_infos) != len(completed_env_indices):
                raise RuntimeError(
                    "Episode statistics count does not match the number of completed environments"
                )
            episode_start_mask = dones.to(device=obs["local_obs"].device, dtype=torch.bool)
            if previous_actions is not None:
                previous_actions = actions.detach().masked_fill(dones[:, None, None], 0.0)

            accepted_episode_infos = []
            for env_idx, episode_info in zip(
                completed_env_indices,
                completed_episode_infos,
                strict=True,
            ):
                if completed_envs[env_idx]:
                    continue
                completed_envs[env_idx] = True
                accepted_episode_infos.append(episode_info)

            for episode_info in accepted_episode_infos:
                metrics["episode_return"].append(float(episode_info["r"]))
                metrics["episode_length"].append(float(episode_info["l"]))
                metrics["progress_reward"].append(float(episode_info["progress_reward"]))
                metrics["guidance_reward"].append(float(episode_info["guidance_reward"]))
                metrics["success"].append(float(episode_info["success"]))
            progress.update(len(accepted_episode_infos))
            if accepted_episode_infos:
                success_rate = 100.0 * statistics.fmean(metrics["success"])
                progress.set_postfix_str(f"success={success_rate:.1f}%", refresh=False)

    return summarize_episode_metrics(metrics)


def _checkpoint_overrides(args: argparse.Namespace, target_key: TargetKey) -> list[Path] | None:
    raw_paths = (
        args.po_wall_checkpoint
        if target_key == "po_wall_tmasac"
        else args.find_opening_checkpoint
    )
    if raw_paths is None:
        return None
    checkpoints = [path.resolve() for path in raw_paths]
    invalid = [path for path in checkpoints if not path.is_file()]
    if invalid:
        raise FileNotFoundError(f"Checkpoint files do not exist: {invalid}")
    return sorted(set(checkpoints))


def resolve_target_checkpoints(args: argparse.Namespace, target: EvaluationTarget) -> list[Path]:
    overrides = _checkpoint_overrides(args, target.key)
    checkpoints = (
        overrides
        if overrides is not None
        else discover_evaluation_checkpoints(target.checkpoint_group_dirs)
    )
    if not checkpoints:
        flag = "--po-wall-checkpoint" if target.key == "po_wall_tmasac" else "--find-opening-checkpoint"
        searched = ", ".join(str(path) for path in target.checkpoint_group_dirs)
        raise FileNotFoundError(
            f"No final or best checkpoints found for {target.display_name}. Searched: {searched}. "
            f"Pass one or more {flag} paths explicitly."
        )
    return checkpoints


def _job_key(*, target_key: str, checkpoint_path: Path, unit_count: int) -> tuple[str, str, int]:
    return target_key, str(checkpoint_path.resolve()), unit_count


def _checkpoint_run_id(checkpoint_path: Path) -> str:
    models_dir = next((parent for parent in checkpoint_path.parents if parent.name == "models"), None)
    if models_dir is not None:
        return models_dir.parent.name
    return checkpoint_path.stem


def _write_results(output_path: Path, payload: dict[str, object]) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_suffix(f"{output_path.suffix}.tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
        encoding="utf-8",
    )
    temporary_path.replace(output_path)
    _write_csv(output_path.with_suffix(".csv"), payload.get("results", []))


def _write_csv(output_path: Path, raw_results: object) -> None:
    results = raw_results if isinstance(raw_results, list) else []
    fieldnames = [
        "target",
        "checkpoint",
        "unit_count",
        "pool_size",
        "episodes",
        "success_count",
        "success_rate_percent",
        "episode_return_mean",
        "episode_return_std",
        "progress_reward_mean",
        "progress_reward_std",
        "guidance_reward_mean",
        "guidance_reward_std",
        "episode_length_mean",
        "episode_length_std",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        for result in results:
            if not isinstance(result, dict):
                continue
            summary = result["summary"]
            writer.writerow(
                {
                    "target": result["target"],
                    "checkpoint": result["checkpoint"],
                    "unit_count": result["unit_count"],
                    "pool_size": result["pool_size"],
                    "episodes": summary["episode_count"],
                    "success_count": summary["success_count"],
                    "success_rate_percent": summary["success_rate_percent"],
                    "episode_return_mean": summary["episode_return"]["mean"],
                    "episode_return_std": summary["episode_return"]["std"],
                    "progress_reward_mean": summary["progress_reward"]["mean"],
                    "progress_reward_std": summary["progress_reward"]["std"],
                    "guidance_reward_mean": summary["guidance_reward"]["mean"],
                    "guidance_reward_std": summary["guidance_reward"]["std"],
                    "episode_length_mean": summary["episode_length"]["mean"],
                    "episode_length_std": summary["episode_length"]["std"],
                }
            )


def _serialize_config(config: EvaluationConfig) -> dict[str, object]:
    serialized = asdict(config)
    serialized["unit_counts"] = list(config.unit_counts)
    return serialized


def _load_resume_payload(
    output_path: Path,
    config: EvaluationConfig,
    *,
    resolved_num_envs: int,
) -> dict[str, object]:
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != RESULT_SCHEMA_VERSION:
        raise ValueError(f"Cannot resume {output_path}: unsupported or missing result schema version")
    payload_config = payload.get("config")
    if isinstance(payload_config, dict) and "unconnected_prob" not in payload_config:
        payload_config = {**payload_config, "unconnected_prob": 0.0}
    if isinstance(payload_config, dict) and "disable_connector_actions" not in payload_config:
        payload_config = {**payload_config, "disable_connector_actions": False}
    if payload_config != _serialize_config(config):
        raise ValueError(f"Cannot resume {output_path}: its evaluation config differs from this invocation")
    if payload.get("resolved_num_envs") != resolved_num_envs:
        raise ValueError(f"Cannot resume {output_path}: its resolved environment count differs from this invocation")
    if not isinstance(payload.get("results"), list):
        raise ValueError(f"Cannot resume {output_path}: results is not a list")
    return payload


def _selected_target_keys(raw_target: str) -> tuple[TargetKey, ...]:
    if raw_target == "all":
        return "po_wall_tmasac", "find_opening_slstm_tmasac"
    return (cast(TargetKey, raw_target),)


def _parse_args(
    argv: Sequence[str] | None = None,
    *,
    default_output_path: Path = DEFAULT_OUTPUT_PATH,
    morphology_description: str = "unseen, fully pre-connected morphologies",
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            f"Evaluate thesis TMASAC checkpoints on {morphology_description} "
            "containing 2 through 10 units, falling back to a run's best checkpoint when final is absent."
        ),
    )
    parser.add_argument(
        "--target",
        choices=("all", *TARGETS),
        default="all",
        help="Model/scenario target to evaluate (default: both).",
    )
    parser.add_argument("--unit-counts", nargs="+", type=int, default=DEFAULT_UNIT_COUNTS)
    parser.add_argument("--pool-size", type=int, default=50)
    parser.add_argument(
        "--episodes",
        type=int,
        default=512,
        help="Episodes and parallel environment lanes per unit-count/checkpoint (default: 512).",
    )
    parser.add_argument(
        "--max-parallel-envs",
        type=int,
        default=512,
        help="Safety cap for parallel lanes; must be at least --episodes (default: 512).",
    )
    parser.add_argument("--pool-seed-base", type=int, default=1_000_000)
    parser.add_argument("--rollout-seed", type=int, default=2_000_000)
    parser.add_argument("--episode-length", type=int, default=512)
    parser.add_argument("--stochastic", action="store_true", help="Sample policy actions instead of using modes.")
    parser.add_argument("--cuda_idx", "--cuda-idx", "--gpu", type=int, default=None)
    parser.add_argument("--output", type=Path, default=default_output_path)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable episode progress bars.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--po-wall-checkpoint", type=Path, action="append")
    parser.add_argument("--find-opening-checkpoint", type=Path, action="append")
    parser.add_argument(
        "--disable-connector-actions",
        action="store_true",
        help="Force every connector action to -1 so modules cannot form or retain connections.",
    )
    return parser.parse_args(argv)


def _validate_args(
    args: argparse.Namespace,
    *,
    unconnected_prob: float = 0.0,
) -> EvaluationConfig:
    if not 0.0 <= unconnected_prob <= 1.0:
        raise ValueError("unconnected_prob must be in [0, 1]")
    unit_counts = tuple(dict.fromkeys(args.unit_counts))
    if not unit_counts or any(count < 2 or count > 20 for count in unit_counts):
        raise ValueError("--unit-counts must contain values in [2, 20]")
    if args.pool_size <= 0:
        raise ValueError("--pool-size must be positive")
    if args.pool_size > POOL_SEED_UNIT_STRIDE:
        raise ValueError(f"--pool-size must not exceed {POOL_SEED_UNIT_STRIDE}")
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.max_parallel_envs <= 0:
        raise ValueError("--max-parallel-envs must be positive")
    if args.episode_length <= 0:
        raise ValueError("--episode-length must be positive")
    return EvaluationConfig(
        unit_counts=unit_counts,
        pool_size=args.pool_size,
        episodes=args.episodes,
        max_parallel_envs=args.max_parallel_envs,
        pool_seed_base=args.pool_seed_base,
        rollout_seed=args.rollout_seed,
        deterministic=not args.stochastic,
        episode_length=args.episode_length,
        unconnected_prob=unconnected_prob,
        disable_connector_actions=args.disable_connector_actions,
    )


def main(
    argv: Sequence[str] | None = None,
    *,
    unconnected_prob: float = 0.0,
    default_output_path: Path = DEFAULT_OUTPUT_PATH,
    morphology_description: str = "unseen, fully pre-connected morphologies",
) -> int:
    args = _parse_args(
        argv,
        default_output_path=default_output_path,
        morphology_description=morphology_description,
    )
    config = _validate_args(args, unconnected_prob=unconnected_prob)
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive")
    targets = [TARGETS[key] for key in _selected_target_keys(args.target)]
    checkpoints_by_target = {
        target.key: resolve_target_checkpoints(args, target)
        for target in targets
    }
    num_envs = resolve_num_envs(
        episodes=config.episodes,
        max_parallel_envs=config.max_parallel_envs,
    )
    job_count = sum(len(checkpoints_by_target[target.key]) for target in targets) * len(config.unit_counts)
    print(
        f"Planned {job_count} evaluations: {config.episodes} episodes each, "
        f"{config.pool_size} unseen morphologies per unit count, {num_envs} parallel envs."
    )
    for target in targets:
        print(f"{target.display_name}: {len(checkpoints_by_target[target.key])} checkpoint(s)")
        for checkpoint in checkpoints_by_target[target.key]:
            fallback_label = " [best fallback]" if checkpoint.name == BEST_CHECKPOINT_NAME else ""
            print(f"  {checkpoint}{fallback_label}")
    if args.dry_run:
        return 0

    output_path = args.output.resolve()
    csv_output_path = output_path.with_suffix(".csv")
    existing_outputs = [path for path in (output_path, csv_output_path) if path.exists()]
    if existing_outputs and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Refusing to replace existing result files: {existing_outputs}. "
            "Pass --resume or --overwrite."
        )

    import torch
    import warp as wp

    if not torch.cuda.is_available():
        raise RuntimeError("MJW evaluation requires CUDA")
    if args.cuda_idx is not None:
        if args.cuda_idx < 0 or args.cuda_idx >= torch.cuda.device_count():
            raise ValueError(f"Invalid CUDA device index: {args.cuda_idx}")
        torch.cuda.set_device(args.cuda_idx)
        wp.set_device(f"cuda:{args.cuda_idx}")
    device = torch.device("cuda")

    if args.resume and output_path.is_file():
        payload = _load_resume_payload(output_path, config, resolved_num_envs=num_envs)
    else:
        payload = {
            "schema_version": RESULT_SCHEMA_VERSION,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "config": _serialize_config(config),
            "resolved_num_envs": num_envs,
            "results": [],
        }
    results = payload["results"]
    assert isinstance(results, list)
    completed_jobs = {
        _job_key(
            target_key=result["target"],
            checkpoint_path=Path(result["checkpoint"]),
            unit_count=int(result["unit_count"]),
        )
        for result in results
        if isinstance(result, dict)
    }

    for target in targets:
        checkpoints = checkpoints_by_target[target.key]
        for unit_count in config.unit_counts:
            pending_checkpoints = [
                checkpoint
                for checkpoint in checkpoints
                if _job_key(
                    target_key=target.key,
                    checkpoint_path=checkpoint,
                    unit_count=unit_count,
                )
                not in completed_jobs
            ]
            if not pending_checkpoints:
                continue
            # Static-shape compilation intentionally specializes each morphology. Reset Dynamo's
            # per-code-object cache so those expected specializations do not accumulate to its
            # global recompile limit across the 2--10 unit sweep.
            torch.compiler.reset()
            pool_seeds = make_pool_seeds(
                pool_seed_base=config.pool_seed_base,
                unit_count=unit_count,
                pool_size=config.pool_size,
            )
            # Automatic episode resets settle normally. Only the explicit initial settled reset is
            # one-shot per env instance, so use a fresh env to give every checkpoint the same start.
            for checkpoint in pending_checkpoints:
                print(f"Building {target.display_name}, {unit_count} units for {checkpoint}...")
                env, policy = _build_env_and_policy(
                    target=target,
                    unit_count=unit_count,
                    pool_seeds=pool_seeds,
                    num_envs=num_envs,
                    episode_length=config.episode_length,
                    device=device,
                    unconnected_prob=config.unconnected_prob,
                )
                try:
                    print(f"Evaluating {checkpoint}...")
                    _load_checkpoint(checkpoint_path=checkpoint, env=env, policy=policy)
                    summary = evaluate_policy(
                        env=env,
                        policy=policy,
                        episode_count=config.episodes,
                        deterministic=config.deterministic,
                        rollout_seed=config.rollout_seed,
                        progress_description=f"{target.display_name}, {unit_count} units",
                        show_progress=not args.no_progress,
                        disable_connector_actions=config.disable_connector_actions,
                    )
                    result = {
                        "target": target.key,
                        "checkpoint": str(checkpoint),
                        "run_id": _checkpoint_run_id(checkpoint),
                        "unit_count": unit_count,
                        "pool_size": config.pool_size,
                        "pool_seed_range_inclusive": [pool_seeds[0], pool_seeds[-1]],
                        "summary": summary,
                    }
                    results.append(result)
                    completed_jobs.add(
                        _job_key(
                            target_key=target.key,
                            checkpoint_path=checkpoint,
                            unit_count=unit_count,
                        )
                    )
                    _write_results(output_path, payload)
                    print(
                        f"  success={summary['success_rate_percent']:.2f}% "
                        f"return={summary['episode_return']['mean']:.3f}"
                    )
                finally:
                    env.close()

    payload["completed_at"] = datetime.now(timezone.utc).isoformat()
    _write_results(output_path, payload)
    print(f"Wrote {output_path} and {output_path.with_suffix('.csv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
