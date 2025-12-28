from typing import Optional

import torch
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import HybridActionDistribution
from swarmbots.learn.algos.mat.mat_decoder import MATDecoder
from swarmbots.learn.algos.mat.mat_deepset_critic import MATDeepSetCritic
from swarmbots.learn.algos.mat.mat_encoder import MATEncoder
from swarmbots.learn.algos.ppo.ppo import AGENTS_DIM
from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.mlp import MLP
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


class MATPolicy(BasePPOPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            d_model: int = 64,
            nhead_encoder: int = 4,
            nhead_decoder: int = 4,
            num_layers_encoder: int = 2,
            num_layers_decoder: int = 2,
            dim_feedforward_encoder: int = 128,
            dim_feedforward_decoder: int = 128,
            dropout: float = 0.0,
            latent_pi_dim_per_agent: int = 64,
            base_std: float = 1.0,
            n_critic_local_projection_hidden_layers: int = 1,
            n_critic_value_regressor_hidden_layers: int = 2,
    ):
        BasePolicy.__init__(self)

        self.n_agents: int = env.n_agents
        self.local_obs_dim: int = env.local_obs_dim
        self.global_obs_dim: int = env.global_obs_dim
        self.has_global_obs = env.global_obs_dim > 0
        self.latent_pi_dim = latent_pi_dim_per_agent
        self.d_model = d_model

        self.agent_embeddings = nn.Parameter(torch.zeros(1, env.n_agents, d_model), requires_grad=True)
        nn.init.orthogonal_(self.agent_embeddings)

        self.local_obs_encoder = MLP(
            input_dim=self.local_obs_dim,
            hidden_dims=[d_model],
            end_with_act_fn=False,
            linear_init=init_linear_orthogonal,
            act_fn_cls=nn.ReLU
        )
        
        if self.has_global_obs:
            self.global_obs_encoder = MLP(
                input_dim=self.global_obs_dim,
                hidden_dims=[d_model],
                end_with_act_fn=False,
                linear_init=init_linear_orthogonal,
                act_fn_cls=nn.ReLU
            )
        else:
            self.global_obs_encoder = None

        self.action_encoder = MLP(
            input_dim=env.action_space.total_agent_action_dim,
            hidden_dims=[d_model],
            end_with_act_fn=False,
            linear_init=init_linear_orthogonal,
            act_fn_cls=nn.ReLU
        )

        self.sos_token = nn.Parameter(torch.zeros(1, 1, d_model), requires_grad=True)
        nn.init.orthogonal_(self.sos_token)

        self.encoder = MATEncoder(
            n_agents=env.n_agents,
            num_layers=num_layers_encoder,
            bias=True,
            norm_first=False,
            layer_norm_eps=1e-5,
            activation=nn.ReLU(),
            dropout=dropout,
            dim_feedforward=dim_feedforward_encoder,
            nhead=nhead_encoder,
            d_model=d_model,
            output_norm=nn.LayerNorm(d_model)
        )

        self.decoder = MATDecoder(
            n_agents=env.n_agents,
            num_layers=num_layers_decoder,
            bias=True,
            norm_first=False,
            layer_norm_eps=1e-5,
            activation=nn.ReLU(),
            dropout=dropout,
            dim_feedforward=dim_feedforward_decoder,
            nhead=nhead_decoder,
            d_model=d_model,
            output_norm=nn.LayerNorm(d_model)
        )

        if d_model != latent_pi_dim_per_agent:
            self.policy_head = MLP(
                input_dim=d_model,
                hidden_dims=[latent_pi_dim_per_agent],
                end_with_act_fn=False,
                linear_init=init_linear_orthogonal,
                act_fn_cls=nn.ReLU
            )
        else:
            self.policy_head = nn.Identity()

        self.action_dist = HybridActionDistribution(
            latent_dim=latent_pi_dim_per_agent,
            action_space=env.action_space,
            base_std=base_std,
        )

        self.critic = MATDeepSetCritic(
            n_agents=env.n_agents,
            augmented_observations_dim=d_model,
            local_projection_hidden_dims=[d_model] * n_critic_local_projection_hidden_layers,
            value_regressor_hidden_dims=[d_model] * n_critic_value_regressor_hidden_layers,
        )

    def _encode_obs(self, local_obs: torch.Tensor, global_obs: torch.Tensor) -> torch.Tensor:
        local_embeddings = self.local_obs_encoder(local_obs) + self.agent_embeddings

        if self.has_global_obs:
            global_embeddings = self.global_obs_encoder(global_obs)
            expanded_global_embeddings = global_embeddings.unsqueeze(1).expand(-1, self.n_agents, -1)
            local_embeddings = local_embeddings + expanded_global_embeddings
        
        return self.encoder(local_embeddings)

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

            last_out = out[:, -1:, :]
            latent_pi = self.policy_head(last_out)

            if return_log_probs:
                action, log_prob = self.action_dist.get_actions_with_log_probs(latent_pi, deterministic)
                log_probs_list.append(log_prob)
            else:
                action = self.action_dist.update_latent_features(latent_pi).get_actions(deterministic)

            actions_list.append(action)

            if i < self.n_agents - 1:
                action_emb = self.action_encoder(action)
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
        
        action_embeddings = self.action_encoder(actions)
        sos_expanded = self.sos_token.expand(actions.shape[0], 1, -1)
        
        shifted_actions = torch.cat([sos_expanded, action_embeddings[:, :-1, :]], dim=1)
        
        latent_pi = self.decoder(shifted_actions, augmented_observations)
        latent_pi = self.policy_head(latent_pi)
        
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
