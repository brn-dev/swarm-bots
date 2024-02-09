import torch
import torch.nn as nn
import torch.nn.functional as F

from ..self_normalizing_fnn import SelfNormalizingFNN


class StateModelFnnPolicy(nn.Module):

    def __init__(
            self,
            action_size: int,
            state_size: int,
            hidden_sizes: list[int],
    ):
        super().__init__()

        self.action_size = action_size
        self.state_size = state_size

        self.fnn = SelfNormalizingFNN(
            input_size=state_size,
            hidden_sizes=hidden_sizes,
            output_size=state_size + action_size
        )
        nn.init.xavier_normal_(self.fnn.snn[-1].weight)  # apply xavier initialization for tanh


    def forward(self, in_state: torch.Tensor):
        out = self.fnn.forward(in_state)

        action_pred = out[-self.action_size:]
        action_pred = F.tanh(action_pred)

        state_pred = out[:self.state_size]

        return action_pred, state_pred

