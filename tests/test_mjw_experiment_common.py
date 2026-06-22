from gymnasium import spaces
from torch import nn

from experiments.mjw_experiment_common import (
    MATInitGains,
    MATNormalizationConfig,
    _make_base_policy,
    _make_mat_parameter_lr_multipliers,
)
from swarmbots.learn.algos.mat_qcs.mat_qcs_decoder import MATQCSDecoderSelfAttentionMode
from swarmbots.learn.algos.mat_qcx.mat_qcx_policy import MATQCXPolicy
from swarmbots.learn.hybrid_action_space import HybridActionSpace


class _DummyEnv:
    n_agents = 3
    local_obs_dim = 4
    global_obs_dim = 2
    hidden_local_vars_dim = 0
    hidden_global_vars_dim = 0
    action_space = HybridActionSpace(
        {
            "cont": spaces.Box(low=-1.0, high=1.0, shape=(n_agents, 2), dtype=float),
            "disc": spaces.MultiBinary((n_agents, 1)),
        }
    )


def test_mat_qcs_and_qcc_lr_multipliers_do_not_include_qcx_action_modules() -> None:
    expected_prefixes = {
        "decoder",
        "query_input_norm",
        "query_encoder",
        "query_token_norm",
        "context_input_norm",
        "context_encoder",
        "context_token_norm",
        "memory_input_norm",
        "memory_encoder",
        "memory_token_norm",
        "agent_embeddings_decoder",
    }

    for policy_variant in ("mat_qcs", "mat_qcc"):
        multipliers = _make_mat_parameter_lr_multipliers(
            policy_variant=policy_variant,
            mat_decoder_lr_multiplier=0.25,
        )

        assert set(multipliers) == expected_prefixes
        assert set(multipliers.values()) == {0.25}


def test_mat_qcx_lr_multipliers_include_decoder_and_memory_modules() -> None:
    multipliers = _make_mat_parameter_lr_multipliers(
        policy_variant="mat_qcx",
        mat_decoder_lr_multiplier=0.25,
    )

    assert set(multipliers) == {
        "decoder",
        "action_input_norm",
        "action_encoder",
        "action_token_norm",
        "memory_input_norm",
        "memory_encoder",
        "memory_token_norm",
    }
    assert set(multipliers.values()) == {0.25}


def test_mat_lr_multipliers_can_include_actor_head_modules() -> None:
    expected_qcs_qcc_prefixes = {
        "decoder",
        "query_input_norm",
        "query_encoder",
        "query_token_norm",
        "context_input_norm",
        "context_encoder",
        "context_token_norm",
        "memory_input_norm",
        "memory_encoder",
        "memory_token_norm",
        "agent_embeddings_decoder",
        "actor_head_input_norm",
        "actor_head",
    }

    for policy_variant in ("mat_qcs", "mat_qcc"):
        multipliers = _make_mat_parameter_lr_multipliers(
            policy_variant=policy_variant,
            mat_decoder_lr_multiplier=0.25,
            include_actor_head_lr_multiplier=True,
        )

        assert set(multipliers) == expected_qcs_qcc_prefixes
        assert set(multipliers.values()) == {0.25}

    qcx_multipliers = _make_mat_parameter_lr_multipliers(
        policy_variant="mat_qcx",
        mat_decoder_lr_multiplier=0.25,
        include_actor_head_lr_multiplier=True,
    )

    assert set(qcx_multipliers) == {
        "decoder",
        "action_input_norm",
        "action_encoder",
        "action_token_norm",
        "memory_input_norm",
        "memory_encoder",
        "memory_token_norm",
        "actor_head_input_norm",
        "actor_head",
    }
    assert set(qcx_multipliers.values()) == {0.25}

    mat_dec_multipliers = _make_mat_parameter_lr_multipliers(
        policy_variant="mat_dec",
        mat_decoder_lr_multiplier=0.25,
        include_actor_head_lr_multiplier=True,
    )

    assert mat_dec_multipliers == {"actor_head": 0.25}


def test_make_base_policy_constructs_mat_qcx_variant() -> None:
    policy = _make_base_policy(
        env=_DummyEnv(),
        policy_variant="mat_qcx",
        enc_d_model=8,
        enc_nhead=1,
        dec_d_model=8,
        dec_nhead=1,
        use_popart=False,
        popart_beta=0.1,
        popart_init_sigma=1.0,
        compile_policy_modules=False,
        policy_compile_mode="default",
        continuous_action_dist="sign_magnitude_beta",
        initial_stickiness=0.25,
        gsde_init_stds=[1.0, 1.0],
        mat_add_agent_embeddings=False,
        mat_decoder_self_attention_mode=MATQCSDecoderSelfAttentionMode.FULL_CAUSAL,
        act_fn_cls=nn.GELU,
        mat_init_gains=MATInitGains(),
        mat_normalization=MATNormalizationConfig(
            normalize_action_input=True,
            normalize_action_tokens=True,
        ),
        assume_agent_mask_is_active_prefix=False,
    )

    assert isinstance(policy, MATQCXPolicy)
    assert isinstance(policy.action_input_norm, nn.LayerNorm)
    assert isinstance(policy.action_token_norm, nn.LayerNorm)
    assert isinstance(policy.decoder.layers[0].context_token_norm, nn.LayerNorm)


def test_mat_lr_multiplier_one_disables_parameter_overrides() -> None:
    multipliers = _make_mat_parameter_lr_multipliers(
        policy_variant="mat_qcx",
        mat_decoder_lr_multiplier=1.0,
    )

    assert multipliers == {}
