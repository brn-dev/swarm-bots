from pathlib import Path

from common import MATDecoderSelfAttentionMode, run_ablation


def main() -> None:
    run_ablation(
        variant_name="sticky_lr_beta_nop_context_tokens_only_no_agent_embeddings",
        entrypoint_path=Path(__file__).resolve(),
        mat_decoder_self_attention_mode=MATDecoderSelfAttentionMode.CONTEXT_TOKENS_ONLY,
        mat_add_agent_embeddings=False,
    )


if __name__ == "__main__":
    main()
