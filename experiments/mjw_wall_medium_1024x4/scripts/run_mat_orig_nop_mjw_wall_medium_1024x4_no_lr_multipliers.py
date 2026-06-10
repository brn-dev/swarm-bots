from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_orig_no_lr_multipliers",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_orig",
        mat_decoder_lr_multiplier=1.0,
        mat_query_context_lr_multiplier=1.0,
    )


if __name__ == "__main__":
    main()
