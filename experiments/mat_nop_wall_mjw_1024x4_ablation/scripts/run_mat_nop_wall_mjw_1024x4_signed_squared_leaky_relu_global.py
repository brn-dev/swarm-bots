from pathlib import Path

from common import run_ablation
from swarmbots.learn.nn_components.activations import NegativeSlopeMode, SignedSquaredLeakyReluFactory


def main() -> None:
    run_ablation(
        variant_name="sticky_lr_beta_nop_signed_squared_leaky_relu_global",
        entrypoint_path=Path(__file__).resolve(),
        act_fn_cls=SignedSquaredLeakyReluFactory(
            negative_slope_mode=NegativeSlopeMode.GLOBAL,
        ),
    )


if __name__ == "__main__":
    main()
