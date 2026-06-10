from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="mat_orig_with_projection_lr_multiplier",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_orig",
        mat_encoder_decoder_projection_lr_multiplier=0.25,
    )


if __name__ == "__main__":
    main()
