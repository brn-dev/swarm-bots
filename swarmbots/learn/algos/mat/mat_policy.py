from typing import Any, Optional

import torch
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import (
    HybridActionDistribution,
    ContinuousActionDistConfig,
    serialize_continuous_action_dist_configs,
)
from swarmbots.learn.algos.mat.mat_decoder import MATDecoder
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder
from swarmbots.learn.algos.ppo.ppo import AGENTS_DIM
from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.deep_set import DeepSetCritic
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class MATPolicy(BasePPOPolicy):

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
        super().__init__()

        self.n_agents: int = env.n_agents
        self.local_obs_dim: int = env.local_obs_dim
        self.global_obs_dim: int = env.global_obs_dim
        self.has_global_obs = env.global_obs_dim > 0
        self.add_agent_embeddings_encoder: bool = add_agent_embeddings_encoder
        self.add_agent_embeddings_decoder: bool = add_agent_embeddings_decoder

        self.d_model_encoder = d_model
        self.d_model_decoder = d_model if d_model_decoder is None else d_model_decoder

        self.encoder = MATEncoder(
            n_agents=env.n_agents,
            local_obs_dim=env.local_obs_dim,
            global_obs_dim=env.global_obs_dim,
            d_model=self.d_model_encoder,
            num_layers=num_layers_encoder,
            bias=True,
            norm_first=True,
            layer_norm_eps=1e-5,
            activation=act_fn_cls(),
            dropout=dropout,
            dim_feedforward=dim_feedforward_encoder,
            nhead=nhead_encoder,
            output_norm=nn.LayerNorm(self.d_model_encoder),
            enable_nested_tensor=False,
            add_agent_embeddings=self.add_agent_embeddings_encoder,
        )

        self.agent_embeddings_decoder: nn.Parameter | None = None
        if self.add_agent_embeddings_decoder:
            self.agent_embeddings_decoder = nn.Parameter(
                torch.zeros(1, env.n_agents - 1, self.d_model_decoder), requires_grad=True
            )
            nn.init.orthogonal_(self.agent_embeddings_decoder)

        self.action_encoder = nn.Linear(env.action_space.total_agent_action_dim, self.d_model_decoder)
        init_linear_orthogonal(self.action_encoder)

        self.sos_token = nn.Parameter(torch.zeros(1, 1, self.d_model_decoder), requires_grad=True)
        nn.init.orthogonal_(self.sos_token)

        self.decoder = MATDecoder(
            n_agents=env.n_agents,
            num_layers=num_layers_decoder,
            bias=True,
            norm_first=True,
            cross_attn_first=cross_attn_first,
            layer_norm_eps=1e-5,
            activation=act_fn_cls(),
            dropout=dropout,
            dim_feedforward=dim_feedforward_decoder,
            nhead=nhead_decoder,
            d_model=self.d_model_decoder,
            memory_d_model=self.d_model_encoder,
            output_norm=nn.LayerNorm(self.d_model_decoder)
        )

        if actor_head_hidden_dims is not None and len(actor_head_hidden_dims) > 0:
            self.actor_head = MLP(
                input_dim=self.d_model_decoder,
                hidden_dims=actor_head_hidden_dims,
                end_with_act_fn=True,
                linear_init=init_linear_orthogonal,
                act_fn_cls=act_fn_cls
            )
            latent_pi_dim = actor_head_hidden_dims[-1]
        else:
            self.actor_head = nn.Identity()
            latent_pi_dim = self.d_model_decoder

        self.action_dist = HybridActionDistribution(
            latent_dim=latent_pi_dim,
            action_space=env.action_space,
            continuous_config=continuous_config,
            bernoulli_initial_prob=bernoulli_initial_prob,
        )

        self.critic = DeepSetCritic(
            num_local_features=self.d_model_encoder,
            local_projection_hidden_dims=[self.d_model_encoder] * n_critic_local_projection_hidden_layers,
            value_regressor_hidden_dims=[self.d_model_encoder] * n_critic_value_regressor_hidden_layers,
            num_global_features=0,
            set_dim=AGENTS_DIM,
            pool_mode="mean",
            linear_init=init_linear_orthogonal,
            act_fn_cls=act_fn_cls,
        )

        serialized_continuous_config = serialize_continuous_action_dist_configs(continuous_config)

        self.hyper_parameters = {
            "d_model_encoder": self.d_model_encoder,
            "d_model_decoder": self.d_model_decoder,
            "nhead_encoder": nhead_encoder,
            "nhead_decoder": nhead_decoder,
            "num_layers_encoder": num_layers_encoder,
            "num_layers_decoder": num_layers_decoder,
            "dim_feedforward_encoder": dim_feedforward_encoder,
            "dim_feedforward_decoder": dim_feedforward_decoder,
            "dropout": dropout,
            "cross_attn_first_decoder": cross_attn_first,
            "n_critic_local_projection_hidden_layers": n_critic_local_projection_hidden_layers,
            "n_critic_value_regressor_hidden_layers": n_critic_value_regressor_hidden_layers,
            "actor_head_hidden_dims": actor_head_hidden_dims,
            "act_fn_cls": act_fn_cls.__name__,
            "continuous_config": serialized_continuous_config,
            "bernoulli_initial_prob": bernoulli_initial_prob,
            "add_agent_embeddings_encoder": add_agent_embeddings_encoder,
            "add_agent_embeddings_decoder": add_agent_embeddings_decoder,
        }

    def get_hyper_parameters(self) -> dict[str, Any]:
        return self.hyper_parameters

    def _encode_obs(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(local_obs=local_obs, global_obs=global_obs)

    def _generate_actions(
        self,
        augmented_observations: torch.Tensor,
        batch_size: int,
        deterministic: bool = False,
        return_log_probs: bool = False
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Autoregressively generate actions based on augmented observations (encoder output)
        :return: (actions, Optional[log_probs])
        """
        actions_list = []
        log_probs_list = []

        decoder_input = self.sos_token.expand(batch_size, 1, -1)

        for i in range(self.n_agents):
            seq_len = decoder_input.shape[1]
            tgt_mask = self.decoder.tgt_mask[:seq_len, :seq_len]

            out = self.decoder.decoder(
                tgt=decoder_input,
                memory=augmented_observations,
                tgt_mask=tgt_mask
            )

            latent_pi = self.actor_head(out[:, -1:, :])

            if return_log_probs:
                action, log_prob = self.action_dist.get_actions_with_log_probs(latent_pi, deterministic, agent=i)
                log_probs_list.append(log_prob)
            else:
                action = self.action_dist.update_latent_features(latent_pi).get_actions(deterministic, agent=i)

            actions_list.append(action)

            if i < self.n_agents - 1:
                action_emb = self.action_encoder(action)
                if self.agent_embeddings_decoder is not None:
                    action_emb = action_emb + self.agent_embeddings_decoder[:, i, :]
                decoder_input = torch.cat([decoder_input, action_emb], dim=AGENTS_DIM)

        actions = torch.cat(actions_list, dim=AGENTS_DIM)
        if return_log_probs:
            return actions, torch.cat(log_probs_list, dim=AGENTS_DIM)
        return actions, None

    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            deterministic: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        augmented_observations = self._encode_obs(local_obs, global_obs)
        actions, log_probs = self._generate_actions(
            augmented_observations=augmented_observations,
            batch_size=local_obs.shape[0],
            deterministic=deterministic,
            return_log_probs=True,
        )
        values = self.critic(augmented_observations)

        return actions, log_probs, values

    def evaluate_actions(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_observations = self._encode_obs(local_obs, global_obs)
        
        action_embeddings = self.action_encoder(actions[:, :-1, :])
        if self.agent_embeddings_decoder is not None:
            action_embeddings = action_embeddings + self.agent_embeddings_decoder
        sos_expanded = self.sos_token.expand(actions.shape[0], 1, -1)
        
        shifted_actions = torch.cat([sos_expanded, action_embeddings], dim=1)
        
        latent_pi = self.actor_head(self.decoder(shifted_actions, augmented_observations))

        self.action_dist.update_latent_features(latent_pi)
        log_probs = self.action_dist.log_prob(actions)
        entropies = self.action_dist.entropy()

        values = self.critic(augmented_observations)
        return log_probs, entropies, values

    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            deterministic: bool = False
    ) -> torch.Tensor:
        augmented_observations = self._encode_obs(local_obs, global_obs)
        actions, _ = self._generate_actions(
            augmented_observations=augmented_observations,
            batch_size=local_obs.shape[0],
            deterministic=deterministic,
            return_log_probs=False,
        )
        return actions
