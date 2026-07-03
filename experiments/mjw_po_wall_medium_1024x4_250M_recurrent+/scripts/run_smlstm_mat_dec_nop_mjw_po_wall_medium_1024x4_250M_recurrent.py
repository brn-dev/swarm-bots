from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="smlstm_mat_dec",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="r_mat_dec",
        temporal_model_variant="smlstm",
    )


if __name__ == "__main__":
    main()

