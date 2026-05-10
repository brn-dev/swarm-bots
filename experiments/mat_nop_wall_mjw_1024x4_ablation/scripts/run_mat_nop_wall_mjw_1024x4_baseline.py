from pathlib import Path

from common import run_ablation


def main() -> None:
    run_ablation(
        variant_name="sticky_lr_beta_nop",
        entrypoint_path=Path(__file__).resolve(),
    )


if __name__ == "__main__":
    main()
