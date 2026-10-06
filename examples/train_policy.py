"""Train any public preset, evaluate its actor, and record complete episodes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from swarmbots import evaluate_policy, record_policy
from swarmbots.learn import as_benchmark_policy, list_variants, train
from swarmbots.learn.logging_levels import LOGGING_LEVELS


def positive_int(value: str) -> int:
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def make_parser(
    *,
    description: str = __doc__,
    variants: tuple[str, ...] | None = None,
    default_variant: str = "mappo",
    run_dir_help: str = "defaults to runs/<benchmark-id>/<variant>",
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("benchmark_id", nargs="?", default="SwarmBots-WallEasy-v0")
    allowed_variants = list_variants(include_hidden=True) if variants is None else variants
    def variant_choice(value: str) -> str:
        if value not in allowed_variants:
            raise argparse.ArgumentTypeError(f"Unknown variant: {value}")
        return value
    parser.add_argument("--variant", type=variant_choice, default=default_variant, metavar="VARIANT",
                        help="core presets: " + ", ".join(list_variants() if variants is None else variants))
    parser.add_argument("--model-scale", default="5M NOP1M", help="fixed tiers: 2.5M, 5M (default), 10M; legacy uses custom widths")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--num-envs", type=positive_int, default=256, help="parallel training worlds")
    parser.add_argument("--total-timesteps", type=positive_int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=42, help="training seed")
    parser.add_argument("--eval-seed", type=int, default=1000)
    parser.add_argument("--eval-envs", type=positive_int, default=256, help="one evaluation episode per world")
    parser.add_argument("--video-episodes", type=positive_int, default=2)
    parser.add_argument("--video-width", type=positive_int, default=640)
    parser.add_argument("--video-height", type=positive_int, default=480)
    parser.add_argument(
        "--episode-length", type=positive_int, help="override the task's control-step limit in all stages"
    )
    parser.add_argument(
        "--nop", action=argparse.BooleanOptionalAction, default=None, help="override the preset's NOP setting"
    )
    parser.add_argument("--compile", action="store_true", help="compile policy and world-model modules")
    parser.add_argument("--run-dir", type=Path, help=run_dir_help)
    parser.add_argument("--console-log-level", choices=LOGGING_LEVELS, default="minimal")
    parser.add_argument("--persistent-log-level", choices=LOGGING_LEVELS, default="full", help="CSV/W&B metric preset")
    return parser


def run_experiment(
    args: argparse.Namespace,
    *,
    policy_kwargs: dict[str, Any] | None = None,
    algorithm_kwargs: dict[str, Any] | None = None,
) -> None:
    run_dir = args.run_dir or Path("runs") / args.benchmark_id / args.variant

    # train() saves metrics/checkpoints and closes the training simulator. The
    # returned trainer retains its actor and observation-normalization statistics.
    trainer = train(
        args.benchmark_id,
        args.variant,
        total_timesteps=args.total_timesteps,
        run_dir=run_dir,
        num_envs=args.num_envs,
        device=args.device,
        seed=args.seed,
        episode_length=args.episode_length,
        use_nop=args.nop,
        model_scale=None if args.model_scale == "legacy" else args.model_scale,
        compile_modules=args.compile,
        policy_kwargs=policy_kwargs,
        algorithm_kwargs=algorithm_kwargs,
        learn_kwargs={
            "log_interval": 10,
            "logging_console_level": args.console_log_level,
            "logging_persistence_level": args.persistent_log_level,
        },
    )

    # The adapter freezes normalization, chooses deterministic actions, and owns
    # recurrent state. Evaluation runs in a separate environment with a new seed.
    result = evaluate_policy(
        as_benchmark_policy(trainer),
        args.benchmark_id,
        num_envs=args.eval_envs,
        num_episodes=args.eval_envs,
        device=args.device,
        seed=args.eval_seed,
        episode_length=args.episode_length,
        action_mode="deterministic",
        policy_metadata={
            "variant": args.variant,
            "training_seed": args.seed,
            "training_timesteps": trainer.n_total_timesteps,
            "settings": trainer.policy.get_hyper_parameters(),
        },
    )
    evaluation_path = run_dir / "evaluation.json"
    evaluation_path.write_text(json.dumps(result.to_dict(), indent=2, allow_nan=False) + "\n", encoding="utf-8")
    print(f"Evaluation: mean_return={result.mean_return:.4f}, success_rate={result.success_rate}")
    print(f"Saved evaluation to {evaluation_path}")

    # Use a fresh adapter so recording starts with fresh recurrent/previous-action
    # state. record_policy() closes its simulator and waits for the MP4 writers.
    video_folder = record_policy(
        as_benchmark_policy(trainer),
        args.benchmark_id,
        video_folder=run_dir / "videos",
        video_name_prefix=args.variant,
        num_episodes=args.video_episodes,
        device=args.device,
        seed=args.eval_seed,
        episode_length=args.episode_length,
        width=args.video_width,
        height=args.video_height,
    )
    print(f"Saved videos to {video_folder}")
    print(f"Saved training checkpoints under {run_dir / 'models'}")


def main() -> None:
    run_experiment(make_parser().parse_args())


if __name__ == "__main__":
    main()
