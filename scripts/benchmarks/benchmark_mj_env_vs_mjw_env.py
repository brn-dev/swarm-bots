from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import torch
from gymnasium.vector import AutoresetMode
from loguru import logger

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from swarmbots.learn.env_wrappers.worker_pool_async_vector_env import WorkerPoolAsyncVectorEnv
from swarmbots.mj_env.scenarios.scenario_presets import default_wall as default_mj_wall
from swarmbots.mj_env.swarm.homogeneous_swarm import PreConnectedUnitLocationsConfig
from swarmbots.mj_env.swarm_bots_env import SwarmBotsEnv
from swarmbots.mjw_env import MJWSwarmBotsVectorEnv
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_wall as default_mjw_wall
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


BackendName = Literal["mj_env", "mjw_env"]


@dataclass(frozen=True)
class BenchmarkConfig:
    num_envs: int
    warmup_steps: int
    measured_steps: int
    reset_repeats: int
    action_pool_size: int
    episode_length: int
    mj_workers: int
    mj_copy: bool
    connector_prob: float
    seed: int


@dataclass(frozen=True)
class BenchmarkResult:
    backend: BackendName
    num_envs: int
    create_seconds: float
    reset_repeats: int
    reset_seconds: float
    reset_envs_per_second: float
    measured_steps: int
    step_seconds: float
    step_envs_per_second: float
    done_count: int
    mj_workers: int | None
    mj_copy: bool | None
    device: str


@dataclass(frozen=True)
class BenchmarkFailure:
    backend: BackendName
    num_envs: int
    phase: str
    error_type: str
    error: str


def make_mj_unit_start_locations(pool_seeds: tuple[int, ...]) -> PreConnectedUnitLocationsConfig:
    return PreConnectedUnitLocationsConfig(
        num_units=5,
        num_unit_probs={
            4: 1.0,
            5: 1.0,
        },
        max_radius=1.5,
        unconnected_prob=0.02,
        z_pos=0.5,
        pool_seeds=pool_seeds,
    )


def make_mjw_unit_start_locations(pool_seeds: tuple[int, ...]) -> MJWPreConnectedUnitLocationsConfig:
    return MJWPreConnectedUnitLocationsConfig(
        num_units=5,
        num_unit_probs={
            4: 1.0,
            5: 1.0,
        },
        max_radius=1.5,
        unconnected_prob=0.02,
        z_pos=0.5,
        pool_seeds=pool_seeds,
    )


def make_mj_env_fn(
    *,
    episode_length: int,
    unit_start_locations: PreConnectedUnitLocationsConfig,
) -> Callable[[], SwarmBotsEnv]:
    def _init() -> SwarmBotsEnv:
        scenario = default_mj_wall(
            first_wall_distance=1.0,
            unit_start_locations=unit_start_locations,
            quantize_connection_twist=8,
        )
        return SwarmBotsEnv(
            scenario=scenario,
            episode_length=episode_length,
            render_mode=None,
            camera=0,
        )

    return _init


def create_mj_env(config: BenchmarkConfig) -> WorkerPoolAsyncVectorEnv:
    unit_start_locations = make_mj_unit_start_locations(pool_seeds=tuple(range(42_000, 42_005)))
    env_fns = [
        make_mj_env_fn(
            episode_length=config.episode_length,
            unit_start_locations=unit_start_locations,
        )
        for _ in range(config.num_envs)
    ]
    return WorkerPoolAsyncVectorEnv(
        env_fns,
        num_workers=config.mj_workers,
        autoreset_mode=AutoresetMode.SAME_STEP,
        copy=config.mj_copy,
    )


def create_mjw_env(config: BenchmarkConfig) -> MJWSwarmBotsVectorEnv:
    if not torch.cuda.is_available():
        raise RuntimeError("mjw_env benchmark requires CUDA.")

    scenario = default_mjw_wall(
        first_wall_distance=1.0,
        unit_start_locations=make_mjw_unit_start_locations(pool_seeds=tuple(range(42_000, 42_005))),
        quantize_connection_twist=8,
    )
    return MJWSwarmBotsVectorEnv(
        scenario=scenario,
        num_envs=config.num_envs,
        episode_length=config.episode_length,
        device=torch.device("cuda"),
    )


def synchronize_env(env: Any) -> None:
    device = getattr(env, "device", None)
    if isinstance(device, torch.device) and device.type == "cuda":
        torch.cuda.synchronize(device)


def count_true(values: np.ndarray | torch.Tensor) -> int:
    if isinstance(values, torch.Tensor):
        return int(values.sum().item())
    return int(np.asarray(values, dtype=np.int64).sum())


def build_numpy_action_pool(
    *,
    num_envs: int,
    actuators_shape: tuple[int, ...],
    connectors_shape: tuple[int, ...],
    pool_size: int,
    connector_prob: float,
    seed: int,
) -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(seed)
    return [
        {
            "actuators": rng.uniform(
                low=-1.0,
                high=1.0,
                size=(num_envs, *actuators_shape),
            ).astype(np.float32),
            "connectors": (rng.random(size=(num_envs, *connectors_shape)) < connector_prob),
        }
        for _ in range(pool_size)
    ]


def build_torch_action_pool(
    *,
    num_envs: int,
    actuators_shape: tuple[int, ...],
    connectors_shape: tuple[int, ...],
    pool_size: int,
    connector_prob: float,
    seed: int,
    device: torch.device,
) -> list[dict[str, torch.Tensor]]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    return [
        {
            "actuators": (torch.rand((num_envs, *actuators_shape), device=device, generator=generator) * 2.0 - 1.0),
            "connectors": torch.rand(
                (num_envs, *connectors_shape),
                device=device,
                generator=generator,
            ) < connector_prob,
        }
        for _ in range(pool_size)
    ]


def build_action_pool(env: Any, config: BenchmarkConfig) -> list[dict[str, Any]]:
    actuators_shape = tuple(env.single_action_space["actuators"].shape)
    connectors_shape = tuple(env.single_action_space["connectors"].shape)
    if getattr(env, "action_backend", None) == "torch":
        return build_torch_action_pool(
            num_envs=config.num_envs,
            actuators_shape=actuators_shape,
            connectors_shape=connectors_shape,
            pool_size=config.action_pool_size,
            connector_prob=config.connector_prob,
            seed=config.seed,
            device=env.device,
        )
    return build_numpy_action_pool(
        num_envs=config.num_envs,
        actuators_shape=actuators_shape,
        connectors_shape=connectors_shape,
        pool_size=config.action_pool_size,
        connector_prob=config.connector_prob,
        seed=config.seed,
    )


def measure_resets(
    *,
    env: Any,
    repeats: int,
    seed: int,
) -> tuple[float, float]:
    synchronize_env(env)
    start = time.perf_counter()
    for repeat_idx in range(repeats):
        env.reset(seed=seed + repeat_idx * 1000)
    synchronize_env(env)
    elapsed = time.perf_counter() - start
    envs_per_second = (repeats * env.num_envs) / elapsed
    return elapsed, envs_per_second


def measure_steps(
    *,
    env: Any,
    action_pool: list[dict[str, Any]],
    warmup_steps: int,
    measured_steps: int,
    seed: int,
) -> tuple[float, float, int]:
    env.reset(seed=seed)
    for step_idx in range(warmup_steps):
        env.step(action_pool[step_idx % len(action_pool)])

    synchronize_env(env)
    start = time.perf_counter()
    done_count = 0
    for step_idx in range(measured_steps):
        _, _, terminations, truncations, _ = env.step(action_pool[step_idx % len(action_pool)])
        done_count += count_true(terminations)
        done_count += count_true(truncations)
    synchronize_env(env)
    elapsed = time.perf_counter() - start
    envs_per_second = (measured_steps * env.num_envs) / elapsed
    return elapsed, envs_per_second, done_count


def benchmark_backend(
    *,
    backend: BackendName,
    config: BenchmarkConfig,
) -> BenchmarkResult:
    create_start = time.perf_counter()
    env = create_mj_env(config) if backend == "mj_env" else create_mjw_env(config)
    synchronize_env(env)
    create_seconds = time.perf_counter() - create_start

    try:
        action_pool = build_action_pool(env, config)
        reset_seconds, reset_envs_per_second = measure_resets(
            env=env,
            repeats=config.reset_repeats,
            seed=config.seed,
        )
        step_seconds, step_envs_per_second, done_count = measure_steps(
            env=env,
            action_pool=action_pool,
            warmup_steps=config.warmup_steps,
            measured_steps=config.measured_steps,
            seed=config.seed + 77_000,
        )
        return BenchmarkResult(
            backend=backend,
            num_envs=config.num_envs,
            create_seconds=create_seconds,
            reset_repeats=config.reset_repeats,
            reset_seconds=reset_seconds,
            reset_envs_per_second=reset_envs_per_second,
            measured_steps=config.measured_steps,
            step_seconds=step_seconds,
            step_envs_per_second=step_envs_per_second,
            done_count=done_count,
            mj_workers=config.mj_workers if backend == "mj_env" else None,
            mj_copy=config.mj_copy if backend == "mj_env" else None,
            device=str(getattr(env, "device", "cpu")),
        )
    finally:
        env.close()


def format_speedup(results: list[BenchmarkResult], *, num_envs: int) -> str:
    per_backend = {result.backend: result for result in results if result.num_envs == num_envs}
    if "mj_env" not in per_backend or "mjw_env" not in per_backend:
        return "-"
    baseline = per_backend["mj_env"].step_envs_per_second
    accelerated = per_backend["mjw_env"].step_envs_per_second
    return f"{accelerated / baseline:.2f}x"


def print_results_table(
    results: list[BenchmarkResult],
    failures: list[BenchmarkFailure],
    *,
    requested_env_counts: list[int],
) -> None:
    headers = [
        "status",
        "backend",
        "envs",
        "create_s",
        "reset_env/s",
        "step_env/s",
        "done",
        "workers",
        "copy",
        "device",
        "step_speedup_vs_mj",
        "error",
    ]
    rows: list[list[str]] = []
    for result in results:
        rows.append(
            [
                "ok",
                result.backend,
                str(result.num_envs),
                f"{result.create_seconds:.2f}",
                f"{result.reset_envs_per_second:,.0f}",
                f"{result.step_envs_per_second:,.0f}",
                str(result.done_count),
                "-" if result.mj_workers is None else str(result.mj_workers),
                "-" if result.backend != "mj_env" else str(result.mj_copy).lower(),
                result.device,
                format_speedup(results, num_envs=result.num_envs),
                "",
            ]
        )
    for failure in failures:
        rows.append(
            [
                "fail",
                failure.backend,
                str(failure.num_envs),
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                "-",
                f"{failure.phase}: {failure.error_type}: {failure.error}",
            ]
        )

    widths = [
        max(len(header), *(len(row[col_idx]) for row in rows))
        for col_idx, header in enumerate(headers)
    ]
    print(" | ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(value.ljust(widths[idx]) for idx, value in enumerate(row)))

    missing_speedups = [
        num_envs
        for num_envs in requested_env_counts
        if len([result for result in results if result.num_envs == num_envs]) < 2
    ]
    if missing_speedups:
        logger.warning(f"Skipped speedup comparison for env counts without both backends: {missing_speedups}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark raw wall-env throughput for mj_env versus mjw_env.",
    )
    parser.add_argument(
        "--backends",
        nargs="+",
        choices=["mj_env", "mjw_env"],
        default=["mjw_env"],
        help="Benchmarked backends.",
    )
    parser.add_argument(
        "--num-envs",
        nargs="+",
        type=int,
        default=[256, 2048, 16384],
        help="Vector-env sizes to benchmark.",
    )
    parser.add_argument("--mj-workers", type=int, default=23, help="Worker count for mj_env.")
    parser.add_argument(
        "--mj-copy",
        choices=["true", "false"],
        default="false",
        help="Whether mj_env should deepcopy batched observations in the parent process.",
    )
    parser.add_argument("--warmup-steps", type=int, default=64, help="Warmup steps before timing step throughput.")
    parser.add_argument("--steps", type=int, default=256, help="Measured steps for throughput.")
    parser.add_argument("--reset-repeats", type=int, default=8, help="Number of full-reset timing iterations.")
    parser.add_argument("--action-pool-size", type=int, default=16, help="Number of prebuilt random actions to cycle.")
    parser.add_argument(
        "--episode-length",
        type=int,
        default=100_000,
        help="Episode length used for the benchmark to avoid truncation noise.",
    )
    parser.add_argument(
        "--connector-prob",
        type=float,
        default=0.15,
        help="Activation probability for connector actions in the random action pool.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Base RNG seed for actions and resets.")
    parser.add_argument("--json-out", type=Path, default=None, help="Optional path for machine-readable benchmark output.")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if any(num_envs <= 0 for num_envs in args.num_envs):
        raise ValueError("--num-envs values must be positive.")
    if args.mj_workers <= 0:
        raise ValueError("--mj-workers must be positive.")
    if args.warmup_steps < 0 or args.steps <= 0 or args.reset_repeats <= 0 or args.action_pool_size <= 0:
        raise ValueError("warmup/steps/reset-repeats/action-pool-size must be positive, with warmup >= 0.")
    if not 0.0 <= args.connector_prob <= 1.0:
        raise ValueError("--connector-prob must be in [0, 1].")


def format_failure_message(error: Exception) -> str:
    message = str(error)
    if isinstance(error, BrokenPipeError):
        return f"{message}. A worker died during startup; inspect worker stderr for the underlying error."
    return message


def main() -> None:
    logger.remove()
    logger.add(
        sys.stderr,
        colorize=True,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <5}</level> | <level>{message}</level>",
    )

    args = parse_args()
    validate_args(args)

    results: list[BenchmarkResult] = []
    failures: list[BenchmarkFailure] = []
    for num_envs in args.num_envs:
        for backend in args.backends:
            config = BenchmarkConfig(
                num_envs=num_envs,
                warmup_steps=args.warmup_steps,
                measured_steps=args.steps,
                reset_repeats=args.reset_repeats,
                action_pool_size=args.action_pool_size,
                episode_length=args.episode_length,
                mj_workers=args.mj_workers,
                mj_copy=args.mj_copy == "true",
                connector_prob=args.connector_prob,
                seed=args.seed,
            )
            logger.info(
                f"Benchmarking {backend} with num_envs={num_envs}, "
                f"warmup_steps={config.warmup_steps}, measured_steps={config.measured_steps}"
            )
            try:
                results.append(benchmark_backend(backend=backend, config=config))
            except Exception as error:
                failures.append(
                    BenchmarkFailure(
                        backend=backend,
                        num_envs=num_envs,
                        phase="benchmark",
                        error_type=type(error).__name__,
                        error=format_failure_message(error),
                    )
                )
                logger.error(f"Benchmark failed for {backend} with num_envs={num_envs}: {format_failure_message(error)}")

    print_results_table(results, failures, requested_env_counts=args.num_envs)

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "args": {
                key: str(value) if isinstance(value, Path) else value
                for key, value in vars(args).items()
            },
            "results": [asdict(result) for result in results],
            "failures": [asdict(failure) for failure in failures],
        }
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        logger.info(f"Wrote benchmark JSON to {args.json_out}")


if __name__ == "__main__":
    main()
