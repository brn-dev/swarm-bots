from typing import Any

import torch
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import ContinuousActionDistConfig
from swarmbots.learn.algos.mat.mat_policy import MATPolicy
from swarmbots.learn.algos.ppo.wm.ppo_wm import PPOWMPolicyMixin
from swarmbots.learn.algos.world_modeling.spr_mixin import SPRMixin
from swarmbots.learn.algos.world_modeling.transformer_transition_model import TransformerTransitionModel
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.mlp import MLP


class MATSPRPolicy(MATPolicy, SPRMixin, PPOWMPolicyMixin):

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
            act_fn_cls: type[nn.Module] = nn.ReLU,
            local_obs_encoder_hidden_dims: list[int] | None = None,
            global_obs_encoder_hidden_dims: list[int] | None = None,
            action_encoder_hidden_dims: list[int] | None = None,
            continuous_config: ContinuousActionDistConfig | list[ContinuousActionDistConfig | None] | None = None,
            bernoulli_initial_prob: float | None = None,
            add_agent_embeddings_encoder: bool = True,
            add_agent_embeddings_decoder: bool = True,
            max_agents: int | None = None,
            d_model_transition_model: int = 128,
            nhead_transition_model: int = 4,
            num_layers_transition_model: int = 2,
            dim_feedforward_transition_model: int = 256,
            add_agent_embeddings_transition_model: bool = False,
            transition_model_coembed_hidden_dims: list[int] | None = None,
            transition_model_head_hidden_dims: list[int] | None = None,
            spr_projection_dims: list[int] | None = None,
            spr_predictor_hidden_dims: list[int] | None = None,
            residual_predictor: bool = True
    ) -> None:
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
            local_obs_encoder_hidden_dims=local_obs_encoder_hidden_dims,
            global_obs_encoder_hidden_dims=global_obs_encoder_hidden_dims,
            action_encoder_hidden_dims=action_encoder_hidden_dims,
            continuous_config=continuous_config,
            bernoulli_initial_prob=bernoulli_initial_prob,
            add_agent_embeddings_encoder=add_agent_embeddings_encoder,
            add_agent_embeddings_decoder=add_agent_embeddings_decoder,
            max_agents=max_agents,
        )
        if spr_projection_dims is None:
            projection_dims = [self.d_model_encoder]
        else:
            projection_dims = spr_projection_dims

        if spr_predictor_hidden_dims is None:
            predictor_hidden_dims = [projection_dims[-1]]
        else:
            predictor_hidden_dims = spr_predictor_hidden_dims + [projection_dims[-1]]

        self.setup_spr(
            transition_model=TransformerTransitionModel(
                n_agents=self.n_agents,
                latent_dim=self.d_model_encoder,
                action_dim=env.action_space.total_agent_action_dim,
                d_model=d_model_transition_model,
                nhead=nhead_transition_model,
                num_layers=num_layers_transition_model,
                dim_feedforward=dim_feedforward_transition_model,
                dropout=dropout,
                act_fn_cls=act_fn_cls,
                add_agent_embeddings=add_agent_embeddings_transition_model,
                predict_delta=True,
                coembed_mlp_hidden_dims=transition_model_coembed_hidden_dims,
                head_mlp_hidden_dims=transition_model_head_hidden_dims,
            ),
            projection=MLP(
                input_dim=self.d_model_encoder,
                hidden_dims=[*projection_dims],
                end_with_act_fn=False,
                act_fn_cls=act_fn_cls,
            ),
            predictor=MLP(
                input_dim=projection_dims[-1],
                hidden_dims=[*predictor_hidden_dims],
                end_with_act_fn=False,
                act_fn_cls=act_fn_cls,
            ),
            residual_predictor=residual_predictor
        )

        self.hyper_parameters.update(
            {
                "d_model_transition_model": d_model_transition_model,
                "nhead_transition_model": nhead_transition_model,
                "num_layers_transition_model": num_layers_transition_model,
                "dim_feedforward_transition_model": dim_feedforward_transition_model,
                "add_agent_embeddings_transition_model": add_agent_embeddings_transition_model,
                "transition_model_coembed_hidden_dims": transition_model_coembed_hidden_dims,
                "transition_model_head_hidden_dims": transition_model_head_hidden_dims,
                "spr_projection_dims": projection_dims,
                "spr_predictor_hidden_dims": predictor_hidden_dims,
            }
        )

    @property
    def online_encoder(self) -> nn.Module:
        return self.encoder

    def evaluate_actions_and_world_model(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            next_local_obs: torch.Tensor,
            next_global_obs: torch.Tensor,
            next_validity_mask: torch.Tensor,
            agent_mask: torch.Tensor | None = None,
            wm_agent_mask: torch.Tensor | None = None,
            wm_loss_agent_mask: torch.Tensor | None = None,
            hidden_vars: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        if actions.ndim == 4:
            policy_actions = actions[:, 0]
        else:
            policy_actions = actions
        policy_local_obs = local_obs
        policy_global_obs = global_obs

        augmented_observations, log_probs, entropies, values = self._evaluate_actions(
            local_obs=policy_local_obs,
            global_obs=policy_global_obs,
            actions=policy_actions,
            hidden_vars=hidden_vars,
            agent_mask=agent_mask,
        )

        spr_loss = self.compute_spr_loss(
            online_local_latents=augmented_observations,
            next_local_obs=next_local_obs,
            next_global_obs=next_global_obs,
            actions=actions,
            agent_mask=wm_agent_mask,
            loss_agent_mask=wm_loss_agent_mask,
            time_mask=next_validity_mask,
        )

        return log_probs, entropies, values, spr_loss, {}

    def update_world_model_targets(self, tau: float) -> None:
        self.update_spr_targets(tau)
