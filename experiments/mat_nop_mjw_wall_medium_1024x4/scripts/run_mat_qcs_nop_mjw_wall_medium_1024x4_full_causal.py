from pathlib import Path

from common import run_experiment
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderSelfAttentionMode


def main() -> None:
    run_experiment(
        variant_name="mat_qcs_full_causal",
        entrypoint_path=Path(__file__).resolve(),
        policy_variant="mat_qcs",
        mat_decoder_self_attention_mode=MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
        mat_decoder_lr_multiplier=0.5,
        mat_query_context_lr_multiplier=0.5,
    )


if __name__ == "__main__":
    main()
