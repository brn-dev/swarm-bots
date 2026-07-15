from pathlib import Path

from common import run_experiment


def main() -> None:
    run_experiment(
        variant_name="tmasac_gumbel_smb",
        entrypoint_path=Path(__file__).resolve(),
        continuous_action_dist="gumbel_softmax_sign_magnitude_beta",
    )


if __name__ == "__main__":
    main()
