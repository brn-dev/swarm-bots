import torch
from torch import nn

from ..self_normalizing_fnn import SelfNormalizingFNN

class CrossAgentAttentionWithStatePredictionPolicy(nn.Module):

    def __init__(
            self,
            action_size: int,
            state_size: int,
            in_state_embedding_sizes: list[int],
            attention_layers: int,
            attention_nhead: int,
            attention_dim_feedforward: int,
            action_pred_hidden_sizes: list[int],
            state_pred_hidden_sizes: list[int]
    ):
        super().__init__()

        embedded_state_size = in_state_embedding_sizes[-1]

        self.state_embedding = SelfNormalizingFNN(
            input_size=state_size,
            hidden_sizes=in_state_embedding_sizes[:-1],
            output_size=embedded_state_size
        )

        self.cross_agent_attention = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                embedded_state_size,
                attention_nhead,
                attention_dim_feedforward,
            ),
            attention_layers
        )

        self.action_regression = SelfNormalizingFNN(
            input_size=embedded_state_size,
            hidden_sizes=action_pred_hidden_sizes,
            output_size=action_size
        )
        nn.init.xavier_normal_(self.action_regression.snn[-1].weight)  # apply xavier initialization for tanh

        self.next_state_regression = SelfNormalizingFNN(
            input_size=embedded_state_size + action_size,
            hidden_sizes=state_pred_hidden_sizes,
            output_size=state_size
        )

    def forward(self, agent_states):
        # nr_agents, state_size = agent_states.shape

        agent_states = self.state_embedding.forward(agent_states)
        agent_states = self.cross_agent_attention.forward(agent_states)

        action_pred = self.action_regression.forward(agent_states)
        action_pred = nn.functional.tanh(action_pred)

        state_pred = self.next_state_regression.forward(torch.cat([agent_states, action_pred], dim=-1))

        return action_pred, state_pred
