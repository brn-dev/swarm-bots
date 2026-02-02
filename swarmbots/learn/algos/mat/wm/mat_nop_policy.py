import torch
from torch import nn

from swarmbots.learn.action_dists.hybrid_action_dist import ContinuousActionDistConfig
from swarmbots.learn.algos.mat.mat_policy import MATPolicy
from swarmbots.learn.algos.ppo.wm.ppo_wm import PPOWMPolicyMixin
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import NextObsPredMixin
from swarmbots.learn.algos.world_modeling.transformer_transition_model import TransformerTransitionModel
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.mlp import MLP


class MATNOPPolicy(MATPolicy, NextObsPredMixin, PPOWMPolicyMixin):

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
            d_model_transition_model: int = 128,
            nhead_transition_model: int = 4,
            num_layers_transition_model: int = 2,
            dim_feedforward_transition_model: int = 256,
            add_agent_embeddings_transition_model: bool = False,
            transition_model_coembed_hidden_dims: list[int] | None = None,
            transition_model_head_hidden_dims: list[int] | None = None,
            transition_model_predict_delta: bool = True,
            local_scalar_target_indices: list[int] | None = None,
            local_angle_target_indices: list[int] | None = None,
            local_rot6d_target_indices: list[int] | None = None,
            local_binary_target_indices: list[int] | None = None,
            wm_predictor_hidden_dims: list[int] | None = None,
            scalar_loss_fn: nn.Module | None = None,
            binary_loss_fn: nn.Module | None = None,
            scalar_loss_weight: float = 1.0,
            angle_loss_weight: float = 1.0,
            rot6d_loss_weight: float = 1.0,
            binary_loss_weight: float = 1.0,
            pre_transition_dims: list[int] | None = None,
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
        )
        predictor_hidden_dims = wm_predictor_hidden_dims or []

        if local_scalar_target_indices is not None and scalar_loss_fn is None:
            scalar_loss_fn = nn.MSELoss(reduction="none")
        if local_binary_target_indices is not None and binary_loss_fn is None:
            binary_loss_fn = nn.BCEWithLogitsLoss(reduction="none")

        pre_transition_transform = None
        wm_latent_dim = self.d_model_encoder
        if pre_transition_dims is not None and len(pre_transition_dims) > 0:
            pre_transition_transform = MLP(
                input_dim=self.d_model_encoder,
                hidden_dims=[*pre_transition_dims],
                end_with_act_fn=False,
                act_fn_cls=act_fn_cls,
            )
            wm_latent_dim = pre_transition_dims[-1]

        local_scalars_predictor = self._build_predictor(
            input_dim=wm_latent_dim,
            output_dim=len(local_scalar_target_indices) if local_scalar_target_indices is not None else 0,
            hidden_dims=predictor_hidden_dims,
            act_fn_cls=act_fn_cls,
        )
        local_angles_predictor = self._build_predictor(
            input_dim=wm_latent_dim,
            output_dim=(len(local_angle_target_indices) * 2) if local_angle_target_indices is not None else 0,
            hidden_dims=predictor_hidden_dims,
            act_fn_cls=act_fn_cls,
        )
        local_rot6ds_predictor = self._build_predictor(
            input_dim=wm_latent_dim,
            output_dim=(len(local_rot6d_target_indices) * 6) if local_rot6d_target_indices is not None else 0,
            hidden_dims=predictor_hidden_dims,
            act_fn_cls=act_fn_cls,
        )
        local_binaries_predictor = self._build_predictor(
            input_dim=wm_latent_dim,
            output_dim=len(local_binary_target_indices) if local_binary_target_indices is not None else 0,
            hidden_dims=predictor_hidden_dims,
            act_fn_cls=act_fn_cls,
        )

        self.setup_modules(
            transition_model=TransformerTransitionModel(
                n_agents=self.n_agents,
                latent_dim=wm_latent_dim,
                action_dim=env.action_space.total_agent_action_dim,
                d_model=d_model_transition_model,
                nhead=nhead_transition_model,
                num_layers=num_layers_transition_model,
                dim_feedforward=dim_feedforward_transition_model,
                dropout=dropout,
                act_fn_cls=act_fn_cls,
                add_agent_embeddings=add_agent_embeddings_transition_model,
                predict_delta=transition_model_predict_delta,
                coembed_mlp_hidden_dims=transition_model_coembed_hidden_dims,
                head_mlp_hidden_dims=transition_model_head_hidden_dims,
            ),
            pre_transition_transform=pre_transition_transform,
            local_scalar_target_indices=local_scalar_target_indices,
            local_angle_target_indices=local_angle_target_indices,
            local_rot6d_target_indices=local_rot6d_target_indices,
            local_binary_target_indices=local_binary_target_indices,
            scalar_loss_fn=scalar_loss_fn,
            binary_loss_fn=binary_loss_fn,
            local_scalars_predictor=local_scalars_predictor,
            local_angles_predictor=local_angles_predictor,
            local_rot6ds_predictor=local_rot6ds_predictor,
            local_binaries_predictor=local_binaries_predictor,
            scalar_loss_weight=scalar_loss_weight,
            angle_loss_weight=angle_loss_weight,
            rot6d_loss_weight=rot6d_loss_weight,
            binary_loss_weight=binary_loss_weight,
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
                "transition_model_predict_delta": transition_model_predict_delta,
                "local_scalar_target_indices": local_scalar_target_indices,
                "local_angle_target_indices": local_angle_target_indices,
                "local_rot6d_target_indices": local_rot6d_target_indices,
                "local_binary_target_indices": local_binary_target_indices,
                "wm_predictor_hidden_dims": predictor_hidden_dims,
                "scalar_loss_weight": scalar_loss_weight,
                "angle_loss_weight": angle_loss_weight,
                "rot6d_loss_weight": rot6d_loss_weight,
                "binary_loss_weight": binary_loss_weight,
                "pre_transition_dims": pre_transition_dims,
            }
        )

    def evaluate_actions_and_world_model(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            next_local_obs: torch.Tensor,
            next_global_obs: torch.Tensor,
            next_validity_mask: torch.Tensor,
            hidden_vars: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        )
        next_obs_pred_loss = self.compute_next_obs_pred_loss(
            local_latents=augmented_observations,
            next_local_obs=next_local_obs,
            actions=actions,
            time_mask=next_validity_mask,
        )

        return log_probs, entropies, values, next_obs_pred_loss

    def update_world_model_targets(self, tau: float) -> None:
        pass

    def _build_predictor(
            self,
            *,
            input_dim: int,
            output_dim: int,
            hidden_dims: list[int],
            act_fn_cls: type[nn.Module],
    ) -> nn.Module | None:
        if output_dim <= 0:
            return None
        return MLP(
            input_dim=input_dim,
            hidden_dims=[*hidden_dims, output_dim] if hidden_dims else [output_dim],
            end_with_act_fn=False,
            act_fn_cls=act_fn_cls,
        )
