from torch import nn

from swarmbots.learn.nn_components.activations import ActivationFactory, make_activation
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal


class MLP(nn.Sequential):

    def __init__(
            self,
            input_dim: int,
            hidden_dims: list[int],
            end_with_act_fn: bool,
            start_with_act_fn: bool = False,
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fn_cls: ActivationFactory = nn.Tanh,
    ):
        assert len(hidden_dims) > 0

        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        dims = [input_dim, *hidden_dims]
        n_layers = len(dims) - 1

        modules: list[nn.Module] = []

        if start_with_act_fn:
            modules.append(make_activation(act_fn_cls, num_features=input_dim))

        for i in range(n_layers):
            linear = nn.Linear(dims[i], dims[i + 1])
            linear_init(linear)
            modules.append(linear)

            if i < n_layers - 1 or end_with_act_fn:
                modules.append(make_activation(act_fn_cls, num_features=dims[i + 1]))

        super().__init__(*modules)
