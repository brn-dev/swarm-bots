from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.algos.mat.mat_policy import MATPolicy
from swarmbots.learn.algos.mat.mat_policy import MATPolicyConfig
from swarmbots.learn.algos.mat.mat_policy import serialize_mat_policy_config
from swarmbots.learn.algos.ppo.wm.ppo_wm import PPOWMPolicyMixin
from swarmbots.learn.algos.world_modeling.spr_mixin import SPRMixin, serialize_spr_world_model_config
from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
    TransformerTransitionModelConfig,
)
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.mlp import MLP


@dataclass(frozen=True)
class MATSPRWorldModelConfig:
    d_model_transition_model: int = 128
    nhead_transition_model: int = 4
    num_layers_transition_model: int = 2
    dim_feedforward_transition_model: int = 256
    add_agent_embeddings_transition_model: bool = False
    transition_model_coembed_hidden_dims: list[int] | None = None
    transition_model_head_hidden_dims: list[int] | None = None
    spr_projection_dims: list[int] | None = None
    spr_predictor_hidden_dims: list[int] | None = None
    residual_predictor: bool = True
    spr_loss_weight: float = 1.0


@dataclass(frozen=True)
class MATSPRPolicyConfig:
    mat_policy_config: MATPolicyConfig = field(default_factory=MATPolicyConfig)
    world_model_config: MATSPRWorldModelConfig = field(default_factory=MATSPRWorldModelConfig)


class MATSPRPolicy(MATPolicy, SPRMixin, PPOWMPolicyMixin):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MATSPRPolicyConfig = MATSPRPolicyConfig(),
    ) -> None:
        super().__init__(env=env, config=config.mat_policy_config)
        world_model_config = config.world_model_config
        if world_model_config.spr_loss_weight < 0:
            raise ValueError(f"spr_loss_weight must be >= 0, got {world_model_config.spr_loss_weight}")
        self.spr_loss_weight = world_model_config.spr_loss_weight
        if world_model_config.spr_projection_dims is None:
            projection_dims = [self.d_model_encoder]
        else:
            projection_dims = world_model_config.spr_projection_dims

        if world_model_config.spr_predictor_hidden_dims is None:
            predictor_hidden_dims = [projection_dims[-1]]
        else:
            predictor_hidden_dims = world_model_config.spr_predictor_hidden_dims + [projection_dims[-1]]

        self.setup_spr(
            transition_model=TransformerTransitionModel(config=TransformerTransitionModelConfig(
                n_agents=self.n_agents,
                latent_dim=self.d_model_encoder,
                action_dim=env.action_space.total_agent_action_dim,
                d_model=world_model_config.d_model_transition_model,
                nhead=world_model_config.nhead_transition_model,
                num_layers=world_model_config.num_layers_transition_model,
                dim_feedforward=world_model_config.dim_feedforward_transition_model,
                dropout=config.mat_policy_config.dropout,
                act_fn_cls=config.mat_policy_config.act_fn_cls,
                add_agent_embeddings=world_model_config.add_agent_embeddings_transition_model,
                predict_delta=True,
                coembed_mlp_hidden_dims=world_model_config.transition_model_coembed_hidden_dims,
                head_mlp_hidden_dims=world_model_config.transition_model_head_hidden_dims,
            )),
            projection=MLP(
                input_dim=self.d_model_encoder,
                hidden_dims=[*projection_dims],
                end_with_act_fn=False,
                act_fn_cls=config.mat_policy_config.act_fn_cls,
            ),
            predictor=MLP(
                input_dim=projection_dims[-1],
                hidden_dims=[*predictor_hidden_dims],
                end_with_act_fn=False,
                act_fn_cls=config.mat_policy_config.act_fn_cls,
            ),
            residual_predictor=world_model_config.residual_predictor
        )

        transition_model_config = TransformerTransitionModelConfig(
            n_agents=self.n_agents,
            latent_dim=self.d_model_encoder,
            action_dim=env.action_space.total_agent_action_dim,
            d_model=world_model_config.d_model_transition_model,
            nhead=world_model_config.nhead_transition_model,
            num_layers=world_model_config.num_layers_transition_model,
            dim_feedforward=world_model_config.dim_feedforward_transition_model,
            dropout=config.mat_policy_config.dropout,
            act_fn_cls=config.mat_policy_config.act_fn_cls,
            add_agent_embeddings=world_model_config.add_agent_embeddings_transition_model,
            predict_delta=True,
            coembed_mlp_hidden_dims=world_model_config.transition_model_coembed_hidden_dims,
            head_mlp_hidden_dims=world_model_config.transition_model_head_hidden_dims,
        )
        self.hyper_parameters["mat_spr_policy_config"] = {
            "mat_policy_config": serialize_mat_policy_config(config.mat_policy_config),
            "world_model_config": serialize_spr_world_model_config(
                transition_model_config=transition_model_config,
                spr_projection_dims=projection_dims,
                spr_predictor_hidden_dims=predictor_hidden_dims,
                residual_predictor=world_model_config.residual_predictor,
                spr_loss_weight=world_model_config.spr_loss_weight,
            ),
        }

    def get_grad_norms(self) -> dict[str, float]:
        grad_norms = super().get_grad_norms()
        grad_norms.update(
            {
                "wm_transition_model": self._module_grad_norm(self.transition_model),
                "wm_online_projection": self._module_grad_norm(self.online_projection),
                "wm_predictor": self._module_grad_norm(self.predictor),
                "wm_target_encoder": self._module_grad_norm(self.target_encoder),
                "wm_target_projection": self._module_grad_norm(self.target_projection),
            }
        )
        return grad_norms

    def update_loss_weights(self, **weights: float) -> None:
        if not weights:
            return

        remaining_weights = dict(weights)
        spr_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("spr_loss_weight", "spr_loss", "spr"),
        )
        if spr_weight is not None:
            alias, value = spr_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.spr_loss_weight = value
            self.hyper_parameters["mat_spr_policy_config"]["world_model_config"]["spr_loss_weight"] = value

        super().update_loss_weights(**remaining_weights)

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
            previous_actions: torch.Tensor | None = None,
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
        dict[str, torch.Tensor],
        dict[str, Any],
    ]:
        if actions.ndim == 4:
            policy_actions = actions[:, 0]
        else:
            policy_actions = actions
        policy_local_obs = local_obs
        policy_global_obs = global_obs

        augmented_observations, log_probs, values, extra_losses, extra_loss_metrics = self._evaluate_actions(
            local_obs=policy_local_obs,
            global_obs=policy_global_obs,
            actions=policy_actions,
            hidden_vars=hidden_vars,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            action_splitter=action_splitter,
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

        spr_loss_weighted = self.spr_loss_weight * spr_loss
        merged_extra_losses = dict(extra_losses)
        merged_extra_losses["world_model"] = spr_loss_weighted
        merged_extra_loss_metrics = dict(extra_loss_metrics)
        merged_extra_loss_metrics["spr_loss"] = spr_loss.item()
        merged_extra_loss_metrics["spr_loss_weighted"] = spr_loss_weighted.item()

        return (
            log_probs,
            values,
            {},
            merged_extra_losses,
            merged_extra_loss_metrics,
        )

    def update_world_model_targets(self, tau: float) -> None:
        self.update_spr_targets(tau)
