from typing import Callable

from torch import nn

LinearInitialization = Callable[[nn.Linear], nn.Linear]
DEFAULT_ORTHOGONAL_GAIN = 1.0


def init_linear_orthogonal(module: nn.Linear, gain: float = DEFAULT_ORTHOGONAL_GAIN) -> nn.Linear:
    nn.init.orthogonal_(module.weight, gain=gain)
    if module.bias is not None:
        nn.init.constant_(module.bias, 0)
    return module


def make_init_linear_orthogonal(gain: float) -> LinearInitialization:
    gain = float(gain)

    def init(module: nn.Linear) -> nn.Linear:
        return init_linear_orthogonal(module, gain=gain)

    return init


def init_transformer_feedforward(layer: nn.Module, *, gain: float) -> None:
    if hasattr(layer, "feedforward"):
        from swarmbots.learn.nn_components.feed_forward import feedforward_linear_layers

        hidden_layers, output_layers = feedforward_linear_layers(layer.feedforward)
    else:
        hidden_layers, output_layers = [layer.linear1], [layer.linear2]
    for linear in hidden_layers:
        init_linear_orthogonal(linear, gain=gain)
    for linear in output_layers:
        init_linear_orthogonal(linear, gain=1.0)


def reinitialize_multihead_attention(module: nn.MultiheadAttention) -> None:
    if module.in_proj_weight is not None:
        nn.init.xavier_uniform_(module.in_proj_weight)
    else:
        assert module.q_proj_weight is not None
        assert module.k_proj_weight is not None
        assert module.v_proj_weight is not None
        nn.init.xavier_uniform_(module.q_proj_weight)
        nn.init.xavier_uniform_(module.k_proj_weight)
        nn.init.xavier_uniform_(module.v_proj_weight)

    if module.in_proj_bias is not None:
        nn.init.constant_(module.in_proj_bias, 0.0)
    if module.out_proj.bias is not None:
        nn.init.constant_(module.out_proj.bias, 0.0)
    if module.bias_k is not None:
        nn.init.xavier_normal_(module.bias_k)
    if module.bias_v is not None:
        nn.init.xavier_normal_(module.bias_v)
    nn.init.xavier_uniform_(module.out_proj.weight)


def reinitialize_transformer_stack(
        stack: nn.Module,
        *,
        feedforward_init_gain: float,
) -> None:
    for layer in stack.layers:
        for module in layer.modules():
            if isinstance(module, nn.MultiheadAttention):
                reinitialize_multihead_attention(module)

        init_transformer_feedforward(layer, gain=feedforward_init_gain)
