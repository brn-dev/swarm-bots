from pathlib import Path

from common import MATInitGains, MATNormalizationConfig, NOPInitGains, run_ablation


def main() -> None:
    hidden_init_gain = 1.5
    transformer_stack_init_gain = 1.5
    output_init_gain = 0.01

    run_ablation(
        variant_name="sticky_lr_beta_nop_proper_init",
        entrypoint_path=Path(__file__).resolve(),
        mat_init_gains=MATInitGains(
            obs_encoder=hidden_init_gain,
            obs_encoder_projection=1.0,
            encoder_transformer_ff=transformer_stack_init_gain,
            decoder_token_encoder=hidden_init_gain,
            decoder_token_encoder_projection=1.0,
            decoder_transformer_ff=transformer_stack_init_gain,
            actor_head=hidden_init_gain,
            action_net=output_init_gain,
            critic_local_projection=hidden_init_gain,
            critic_value_regressor=hidden_init_gain,
            critic_value_head=hidden_init_gain,
        ),
        nop_init_gains=NOPInitGains(
            pre_transition=hidden_init_gain,
            transition_coembed=hidden_init_gain,
            transition_transformer_ff=transformer_stack_init_gain,
            transition_head=output_init_gain,
            pre_predictors=hidden_init_gain,
            predictors=output_init_gain,
        ),
        mat_normalization=MATNormalizationConfig(
            normalize_obs_inputs=False,
            normalize_encoder_tokens=True,
            normalize_query_input=True,
            normalize_context_input=False,
            normalize_memory_input=True,
            normalize_query_tokens=True,
            normalize_context_tokens=True,
            normalize_memory_tokens=True,
            normalize_actor_head_input=True,
            normalize_prev_binary_actions=True,
        ),
    )


if __name__ == "__main__":
    main()
