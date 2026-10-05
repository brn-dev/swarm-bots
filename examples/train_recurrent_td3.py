"""Train, evaluate, and record TD3 with a recurrent actor and optional recurrent critic."""

from pathlib import Path

if __package__:
    from .train_policy import make_parser, run_experiment
else:
    from train_policy import make_parser, run_experiment


def main() -> None:
    parser = make_parser(
        description=__doc__,
        variants=("matd3_mlp", "matd3_deepset", "tmatd3", "tmatd3_dec"),
        default_variant="tmatd3",
        run_dir_help="defaults to runs/<benchmark-id>/<variant>_recurrent_actor (or _recurrent_critic)",
    )
    parser.add_argument(
        "--recurrent-critic",
        action="store_true",
        help="give the transformer critic its own recurrent history (TMATD3 presets only)",
    )
    args = parser.parse_args()
    if args.recurrent_critic and args.variant not in ("tmatd3", "tmatd3_dec"):
        parser.error("--recurrent-critic requires --variant tmatd3 or tmatd3_dec")
    if args.run_dir is None:
        critic_suffix = "recurrent_critic" if args.recurrent_critic else "recurrent_actor"
        args.run_dir = Path("runs") / args.benchmark_id / f"{args.variant}_{critic_suffix}"

    # A feed-forward critic receives detached actor state by default. A recurrent
    # critic instead builds its own history from replay observations and actions.
    policy_kwargs = {
        "td3_recurrent_actor": True,
        "td3_recurrent_critic": args.recurrent_critic,
    }
    algorithm_kwargs = {
        "burn_in_steps": 32,
        "learning_steps": 64,
        "temporal_state_store_interval": 16,
    }

    # Reuse the train -> evaluate -> record workflow in train_policy.py. Both
    # inference stages use the trained actor and its frozen normalization directly.
    run_experiment(args, policy_kwargs=policy_kwargs, algorithm_kwargs=algorithm_kwargs)


if __name__ == "__main__":
    main()
