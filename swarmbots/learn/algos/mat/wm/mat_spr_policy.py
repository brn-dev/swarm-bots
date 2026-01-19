from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import ContinuousActionDistConfig
from swarmbots.learn.algos.mat.mat_policy import MATPolicy
from swarmbots.learn.algos.world_modeling.spr_mixin import SPRMixin
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper


class MATSPRPolicy(MATPolicy, SPRMixin):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            d_model: int = 64,
            d_model_decoder: int | None = None,
            nhead_encoder: int = 4,
            nhead_decoder: int = 4,
            num_layers_encoder: int = 2,
            num_layers_decoder: int = 2,
            dim_feedforward_encoder: int = 128,
            dim_feedforward_decoder: int = 128,
            dropout: float = 0.0,
            cross_attn_first: bool = False,
            n_critic_local_projection_hidden_layers: int = 1,
            n_critic_value_regressor_hidden_layers: int = 2,
            actor_head_hidden_dims: list[int] | None = None,
            act_fn_cls=nn.ReLU,
            continuous_config: ContinuousActionDistConfig | list[ContinuousActionDistConfig | None] | None = None,
            bernoulli_initial_prob: float | None = None,
            add_agent_embeddings_encoder: bool = True,
            add_agent_embeddings_decoder: bool = True,
    ):
        super().__init__(
            env=env,
            d_model=d_model,
            d_model_decoder=d_model_decoder,
            nhead_encoder=nhead_encoder,
            nhead_decoder=nhead_decoder,
            num_layers_encoder=num_layers_encoder,
            num_layers_decoder=num_layers_decoder,
            dim_feedforward_encoder=dim_feedforward_encoder,
            dim_feedforward_decoder=dim_feedforward_decoder,
            dropout=dropout,
            cross_attn_first=cross_attn_first,
            n_critic_local_projection_hidden_layers=n_critic_local_projection_hidden_layers,
            n_critic_value_regressor_hidden_layers=n_critic_value_regressor_hidden_layers,
            actor_head_hidden_dims=actor_head_hidden_dims,
            act_fn_cls=act_fn_cls,
            continuous_config=continuous_config,
            bernoulli_initial_prob=bernoulli_initial_prob,
            add_agent_embeddings_encoder=add_agent_embeddings_encoder,
            add_agent_embeddings_decoder=add_agent_embeddings_decoder,
        )
        self.setup_modules(...)  # todo

    @property
    def online_encoder(self) -> nn.Module:
        return self.encoder