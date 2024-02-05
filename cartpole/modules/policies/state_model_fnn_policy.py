import torch
import torch.nn as nn
import torch.nn.init as nn_init
import torch.nn.functional as F

from ..self_normalizing_fnn import SelfNormalizingFNN


class StateModelFnnPolicy(nn.Module):

    def __init__(
            self,
            action_size: int,
            state_size: int,
            in_state_embedding_sizes: list[int],
            action_pred_hidden_sizes: list[int],
            state_pred_hidden_sizes: list[int]
    ):
        super().__init__()

        self.in_state_embedding = SelfNormalizingFNN(
            input_size=state_size,
            hidden_sizes=in_state_embedding_sizes[:-1],
            output_size=in_state_embedding_sizes[-1]
        )

        self.action_regression = SelfNormalizingFNN(
            input_size=in_state_embedding_sizes[-1],
            hidden_sizes=action_pred_hidden_sizes,
            output_size=action_size
        )
        nn_init.xavier_normal_(self.action_regression.snn[-1].weight)  # apply xavier initialization for tanh

        self.state_regression = SelfNormalizingFNN(
            input_size=in_state_embedding_sizes[-1] + action_size,
            hidden_sizes=state_pred_hidden_sizes,
            output_size=state_size
        )

    def forward(self, in_state: torch.Tensor):
        in_state_embedded = self.in_state_embedding.forward(in_state)

        action_pred = self.action_regression.forward(in_state_embedded)
        action_pred = F.tanh(action_pred)

        state_pred = self.state_regression.forward(torch.cat([in_state_embedded, action_pred], dim=-1))

        return action_pred, state_pred

