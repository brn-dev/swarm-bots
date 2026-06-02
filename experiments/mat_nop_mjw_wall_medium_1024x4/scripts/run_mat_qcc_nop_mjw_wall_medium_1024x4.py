from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_qcc",
        entrypoint_path=Path(__file__).resolve(),
        mat_decoder_lr_multiplier=0.5,
        mat_query_context_lr_multiplier=0.5,
    )


if __name__ == "__main__":
    main()
