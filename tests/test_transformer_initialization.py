import math

import torch
from torch import nn

from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig
from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
    TransformerTransitionModelConfig,
)


def test_mat_encoder_keeps_default_cloned_transformer_initialization() -> None:
    torch.manual_seed(123)
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    first_layer = encoder.encoder.layers[0]
    second_layer = encoder.encoder.layers[1]
    torch.testing.assert_close(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    torch.testing.assert_close(first_layer.linear1.weight, second_layer.linear1.weight)


def test_mat_encoder_proper_init_reinitializes_cloned_transformer_layers() -> None:
    torch.manual_seed(123)
    encoder = MATEncoder(
        MATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            transformer_ff_init_gain=1.5,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    first_layer = encoder.encoder.layers[0]
    second_layer = encoder.encoder.layers[1]
    assert not torch.equal(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    assert not torch.equal(first_layer.linear1.weight, second_layer.linear1.weight)

    expected_linear1_norm = 1.5 * math.sqrt(first_layer.linear1.weight.shape[1])
    torch.testing.assert_close(
        torch.linalg.vector_norm(first_layer.linear1.weight),
        torch.tensor(expected_linear1_norm),
        rtol=1e-5,
        atol=1e-6,
    )


def test_transition_model_keeps_default_cloned_transformer_initialization() -> None:
    torch.manual_seed(123)
    model = TransformerTransitionModel(
        TransformerTransitionModelConfig(
            n_agents=4,
            latent_dim=8,
            action_dim=3,
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            act_fn_cls=nn.GELU,
        )
    )

    first_layer = model.encoder.layers[0]
    second_layer = model.encoder.layers[1]
    torch.testing.assert_close(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    torch.testing.assert_close(first_layer.linear1.weight, second_layer.linear1.weight)


def test_transition_model_proper_init_reinitializes_cloned_transformer_layers() -> None:
    torch.manual_seed(123)
    model = TransformerTransitionModel(
        TransformerTransitionModelConfig(
            n_agents=4,
            latent_dim=8,
            action_dim=3,
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            act_fn_cls=nn.GELU,
            transformer_ff_init_gain=1.5,
        )
    )

    first_layer = model.encoder.layers[0]
    second_layer = model.encoder.layers[1]
    assert not torch.equal(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
    assert not torch.equal(first_layer.linear1.weight, second_layer.linear1.weight)


def test_rmat_encoder_uses_mat_init_gains() -> None:
    torch.manual_seed(123)
    encoder = RMATEncoder(
        RMATEncoderConfig(
            d_model=8,
            nhead=2,
            num_layers=2,
            dim_feedforward=16,
            linear_init_gain=1.5,
            linear_projection_init_gain=1.0,
            transformer_ff_init_gain=1.5,
        ),
        max_agents=4,
        local_obs_dim=5,
        global_obs_dim=0,
    )

    assert isinstance(encoder.local_obs_encoder, nn.Linear)
    expected_projection_norm = math.sqrt(encoder.local_obs_encoder.weight.shape[1])
    torch.testing.assert_close(
        torch.linalg.vector_norm(encoder.local_obs_encoder.weight),
        torch.tensor(expected_projection_norm),
        rtol=1e-5,
        atol=1e-6,
    )

    first_layer = encoder.layers[0].inter_agent_attention_encoder.layers[0]
    second_layer = encoder.layers[1].inter_agent_attention_encoder.layers[0]
    assert not torch.equal(first_layer.self_attn.in_proj_weight, second_layer.self_attn.in_proj_weight)
