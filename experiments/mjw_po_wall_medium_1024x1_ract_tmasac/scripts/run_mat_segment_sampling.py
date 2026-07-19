from pathlib import Path

from common import run_experiment


if __name__ == "__main__":
    run_experiment(
        variant_name="mat_segment_sampling",
        entrypoint_path=Path(__file__),
        temporal_model_variant="mat",
        mlp_layout=None,
    )
