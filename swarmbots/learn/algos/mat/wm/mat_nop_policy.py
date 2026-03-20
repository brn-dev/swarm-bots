from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.algos.mat.mat_policy import MATPolicy
from swarmbots.learn.algos.mat.mat_policy import MATPolicyConfig
from swarmbots.learn.algos.mat.mat_policy import serialize_mat_policy_config
from swarmbots.learn.algos.ppo.wm.ppo_wm import PPOWMPolicyMixin
from swarmbots.learn.algos.world_modeling.next_obs_pred_mixin import (
    NextObsPredMixin,
    NextObsPredConfig,
    PredictDeltaMode,
    serialize_next_obs_pred_world_model_config,
)
from swarmbots.learn.algos.world_modeling.transformer_transition_model import (
    TransformerTransitionModel,
    TransformerTransitionModelConfig,
)
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.nn_components.mlp import MLP


@dataclass(frozen=True)
class MATNOPWorldModelConfig:
    d_model_transition_model: int = 128
    nhead_transition_model: int = 4
    num_layers_transition_model: int = 2
    dim_feedforward_transition_model: int = 256
    add_agent_embeddings_transition_model: bool = False
    transition_model_coembed_hidden_dims: list[int] | None = None
    transition_model_head_hidden_dims: list[int] | None = None
    transition_model_predict_delta: bool = True
    wm_pre_transition_dims: list[int] | None = None
    wm_pre_predictors_dims: list[int] | None = None
    wm_scalar_predictor_hidden_dims: list[int] | None = None
    wm_angle_predictor_hidden_dims: list[int] | None = None
    wm_rot6d_predictor_hidden_dims: list[int] | None = None
    wm_binary_predictor_hidden_dims: list[int] | None = None
    scalar_loss_fn: str | nn.Module | None = None
    next_obs_pred_config: NextObsPredConfig = field(default_factory=NextObsPredConfig)


@dataclass(frozen=True)
class MATNOPPolicyConfig:
    mat_policy_config: MATPolicyConfig = field(default_factory=MATPolicyConfig)
    world_model_config: MATNOPWorldModelConfig = field(default_factory=MATNOPWorldModelConfig)


class MATNOPPolicy(MATPolicy, NextObsPredMixin, PPOWMPolicyMixin):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MATNOPPolicyConfig = MATNOPPolicyConfig(),
    ) -> None:
        super().__init__(env=env, config=config.mat_policy_config)
        world_model_config = config.world_model_config
        next_obs_pred_config = world_model_config.next_obs_pred_config

        scalar_loss_fn = world_model_config.scalar_loss_fn
        if not isinstance(scalar_loss_fn, nn.Module):
            scalar_loss_name = world_model_config.scalar_loss_fn or "mse"
            scalar_loss_name = scalar_loss_name.lower()
            if scalar_loss_name == "mse":
                scalar_loss_fn = nn.MSELoss(reduction="none")
            elif scalar_loss_name == "smooth_l1":
                scalar_loss_fn = nn.SmoothL1Loss(reduction="none")
            else:
                raise ValueError(f"Unknown scalar_loss {scalar_loss_name!r}")

        pre_transition_transform = None
        wm_latent_dim = self.d_model_encoder
        if world_model_config.wm_pre_transition_dims is not None and len(world_model_config.wm_pre_transition_dims) > 0:
            pre_transition_transform = MLP(
                input_dim=self.d_model_encoder,
                hidden_dims=[*world_model_config.wm_pre_transition_dims],
                end_with_act_fn=False,
                act_fn_cls=config.mat_policy_config.act_fn_cls,
            )
            wm_latent_dim = world_model_config.wm_pre_transition_dims[-1]

        pre_predictors_transform = None
        wm_pre_predictors_dim = wm_latent_dim
        if world_model_config.wm_pre_predictors_dims is not None and len(world_model_config.wm_pre_predictors_dims) > 0:
            pre_predictors_transform = MLP(
                input_dim=wm_latent_dim,
                hidden_dims=[*world_model_config.wm_pre_predictors_dims],
                end_with_act_fn=False,
                act_fn_cls=config.mat_policy_config.act_fn_cls,
            )
            wm_pre_predictors_dim = world_model_config.wm_pre_predictors_dims[-1]

        angle_output_multiplier = 1 if next_obs_pred_config.predict_delta else 2
        rot6d_output_multiplier = 3 if next_obs_pred_config.predict_delta else 6

        local_scalars_predictor = self._build_predictor(
            input_dim=wm_pre_predictors_dim,
            output_dim=(
                len(next_obs_pred_config.local_scalar_target_indices)
                if next_obs_pred_config.local_scalar_target_indices is not None
                else 0
            ),
            hidden_dims=world_model_config.wm_scalar_predictor_hidden_dims,
            act_fn_cls=config.mat_policy_config.act_fn_cls,
        )
        local_angles_predictor = self._build_predictor(
            input_dim=wm_pre_predictors_dim,
            output_dim=(
                len(next_obs_pred_config.local_angle_target_indices) * angle_output_multiplier
                if next_obs_pred_config.local_angle_target_indices is not None
                else 0
            ),
            hidden_dims=world_model_config.wm_angle_predictor_hidden_dims,
            act_fn_cls=config.mat_policy_config.act_fn_cls,
        )
        local_rot6ds_predictor = self._build_predictor(
            input_dim=wm_pre_predictors_dim,
            output_dim=(
                len(next_obs_pred_config.local_rot6d_target_indices) * rot6d_output_multiplier
                if next_obs_pred_config.local_rot6d_target_indices is not None
                else 0
            ),
            hidden_dims=world_model_config.wm_rot6d_predictor_hidden_dims,
            act_fn_cls=config.mat_policy_config.act_fn_cls,
        )
        local_binaries_predictor = self._build_predictor(
            input_dim=wm_pre_predictors_dim,
            output_dim=(
                len(next_obs_pred_config.local_binary_target_indices)
                if next_obs_pred_config.local_binary_target_indices is not None
                else 0
            ),
            hidden_dims=world_model_config.wm_binary_predictor_hidden_dims,
            act_fn_cls=config.mat_policy_config.act_fn_cls,
        )

        transition_model_config = TransformerTransitionModelConfig(
            n_agents=self.n_agents,
            latent_dim=wm_latent_dim,
            action_dim=env.action_space.total_agent_action_dim,
            d_model=world_model_config.d_model_transition_model,
            nhead=world_model_config.nhead_transition_model,
            num_layers=world_model_config.num_layers_transition_model,
            dim_feedforward=world_model_config.dim_feedforward_transition_model,
            dropout=config.mat_policy_config.dropout,
            act_fn_cls=config.mat_policy_config.act_fn_cls,
            add_agent_embeddings=world_model_config.add_agent_embeddings_transition_model,
            predict_delta=world_model_config.transition_model_predict_delta,
            coembed_mlp_hidden_dims=world_model_config.transition_model_coembed_hidden_dims,
            head_mlp_hidden_dims=world_model_config.transition_model_head_hidden_dims,
        )
        self.setup_next_obs_pred(
            transition_model=TransformerTransitionModel(config=transition_model_config),
            config=next_obs_pred_config,
            pre_transition_transform=pre_transition_transform,
            pre_predictors_transform=pre_predictors_transform,
            scalar_loss_fn=scalar_loss_fn,
            local_scalars_predictor=local_scalars_predictor,
            local_angles_predictor=local_angles_predictor,
            local_rot6ds_predictor=local_rot6ds_predictor,
            local_binaries_predictor=local_binaries_predictor,
        )

        self.hyper_parameters["mat_nop_policy_config"] = {
            "mat_policy_config": serialize_mat_policy_config(config.mat_policy_config),
            "world_model_config": serialize_next_obs_pred_world_model_config(
                transition_model_config=transition_model_config,
                wm_pre_transition_dims=world_model_config.wm_pre_transition_dims,
                wm_pre_predictors_dims=world_model_config.wm_pre_predictors_dims,
                wm_scalar_predictor_hidden_dims=world_model_config.wm_scalar_predictor_hidden_dims,
                wm_angle_predictor_hidden_dims=world_model_config.wm_angle_predictor_hidden_dims,
                wm_rot6d_predictor_hidden_dims=world_model_config.wm_rot6d_predictor_hidden_dims,
                wm_binary_predictor_hidden_dims=world_model_config.wm_binary_predictor_hidden_dims,
                scalar_loss_fn=scalar_loss_fn,
                next_obs_pred_config=next_obs_pred_config,
            ),
        }

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
            action_splitter=action_splitter,
        )
        next_obs_pred_loss, nop_loss_metrics = self.compute_next_obs_pred_loss(
            local_latents=augmented_observations,
            next_local_obs=next_local_obs,
            actions=actions,
            local_obs=local_obs,
            agent_mask=wm_agent_mask,
            loss_agent_mask=wm_loss_agent_mask,
            time_mask=next_validity_mask,
        )

        merged_extra_losses = dict(extra_losses)
        merged_extra_losses["world_model"] = next_obs_pred_loss

        return (
            log_probs,
            values,
            nop_loss_metrics,
            merged_extra_losses,
            extra_loss_metrics,
        )

    def update_world_model_targets(self, tau: float) -> None:
        pass

    def get_grad_norms(self) -> dict[str, float]:
        grad_norms = super().get_grad_norms()
        grad_norms.update(
            {
                "wm_pre_transition_transform": self._module_grad_norm(self.pre_transition_transform),
                "wm_transition_model": self._module_grad_norm(self.transition_model),
                "wm_pre_predictors_transform": self._module_grad_norm(self.pre_predictors_transform),
                "wm_local_scalars_predictor": self._module_grad_norm(self.local_scalars_predictor),
                "wm_local_angles_predictor": self._module_grad_norm(self.local_angles_predictor),
                "wm_local_rot6ds_predictor": self._module_grad_norm(self.local_rot6ds_predictor),
                "wm_local_binaries_predictor": self._module_grad_norm(self.local_binaries_predictor),
            }
        )
        return grad_norms

    def update_loss_weights(self, **weights: float) -> None:
        if not weights:
            return

        remaining_weights = dict(weights)

        scalar_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("scalar_loss_weight", "scalar"),
        )
        if scalar_weight is not None:
            alias, value = scalar_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.scalar_loss_weight = value
            self.hyper_parameters["mat_nop_policy_config"]["world_model_config"]["next_obs_pred_config"]["scalar_loss_weight"] = value

        angle_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("angle_loss_weight", "angle"),
        )
        if angle_weight is not None:
            alias, value = angle_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.angle_loss_weight = value
            self.hyper_parameters["mat_nop_policy_config"]["world_model_config"]["next_obs_pred_config"]["angle_loss_weight"] = value

        rot6d_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("rot6d_loss_weight", "rot6d"),
        )
        if rot6d_weight is not None:
            alias, value = rot6d_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.rot6d_loss_weight = value
            self.hyper_parameters["mat_nop_policy_config"]["world_model_config"]["next_obs_pred_config"]["rot6d_loss_weight"] = value

        binary_weight = self._pop_loss_weight_alias(
            remaining_weights,
            aliases=("binary_loss_weight", "binary"),
        )
        if binary_weight is not None:
            alias, value = binary_weight
            if value < 0:
                raise ValueError(f"{alias} must be >= 0, got {value}")
            self.binary_loss_weight = value
            self.hyper_parameters["mat_nop_policy_config"]["world_model_config"]["next_obs_pred_config"]["binary_loss_weight"] = value

        super().update_loss_weights(**remaining_weights)

    @staticmethod
    def _build_predictor(
            input_dim: int,
            output_dim: int,
            hidden_dims: list[int] | None,
            act_fn_cls: type[nn.Module],
    ) -> nn.Module | None:
        if output_dim <= 0:
            return None
        return MLP(
            input_dim=input_dim,
            hidden_dims=[*(hidden_dims or []), output_dim] if hidden_dims else [output_dim],
            end_with_act_fn=False,
            act_fn_cls=act_fn_cls,
        )
