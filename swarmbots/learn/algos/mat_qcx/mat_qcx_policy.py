from dataclasses import dataclass, field, replace
from typing import Any

import torch
from torch import nn

from swarmbots.learn.algos.mat_qcs.mat_qcs_policy import MATQCSPolicy, MATQCSPolicyConfig
from swarmbots.learn.algos.mat_qcx.mat_qcx_decoder import MATQCXDecoder, MATQCXDecoderConfig
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.serialization_utils import serialize_dataclass


@dataclass(frozen=True)
class MATQCXPolicyConfig(MATQCSPolicyConfig):
    decoder_config: MATQCXDecoderConfig = field(default_factory=MATQCXDecoderConfig)


class MATQCXPolicy(MATQCSPolicy):

    def __init__(
            self,
            env: BaseLearnEnvWrapper,
            config: MATQCXPolicyConfig = MATQCXPolicyConfig(),
    ) -> None:
        super().__init__(env=env, config=config)

        self.action_input_norm = (
            nn.LayerNorm(self.agent_action_dim)
            if config.decoder_config.normalize_action_input
            else nn.Identity()
        )
        if config.decoder_config.project_actions:
            self.action_encoder = self._build_token_encoder(
                input_dim=self.agent_action_dim,
                output_dim=self.d_model_decoder,
                hidden_dims=config.decoder_config.action_encoder_hidden_dims,
                act_fn_cls=config.act_fn_cls,
                linear_init_gain=config.decoder_config.token_encoder_init_gain,
                projection_init_gain=config.decoder_config.token_encoder_projection_init_gain,
                end_with_act_fn=config.decoder_config.action_encoder_end_with_act_fn,
            )
        else:
            if self.agent_action_dim != self.d_model_decoder:
                raise ValueError(
                    "MATQCXDecoderConfig.project_actions=False requires agent action dim "
                    f"({self.agent_action_dim}) to match decoder d_model ({self.d_model_decoder})"
                )
            self.action_encoder = nn.Identity()
        self.action_token_norm = (
            nn.LayerNorm(self.d_model_decoder)
            if config.decoder_config.normalize_action_tokens
            else nn.Identity()
        )

        if self.config.compile_modules:
            self.action_input_norm = self._compile_module(self.action_input_norm)
            self.action_encoder = self._compile_module(self.action_encoder)
            self.action_token_norm = self._compile_module(self.action_token_norm)

    def get_hyper_parameters(self) -> dict[str, Any]:
        hyper_parameters = super().get_hyper_parameters()["mat_qcs_policy_config"]
        hyper_parameters["decoder_config"] = serialize_dataclass(self.decoder_config)
        return {
            "mat_qcx_policy_config": hyper_parameters,
        }

    def _build_decoder_config(self) -> MATQCXDecoderConfig:
        return replace(
            self.config.decoder_config,
            d_model=self.d_model_decoder,
            act_fn_cls=self.config.act_fn_cls,
            dropout=self.config.dropout,
        )

    def _build_query_input_norm(self) -> nn.Module:
        return nn.Identity()

    def _build_query_encoder(self) -> nn.Module:
        return nn.Identity()

    def _build_query_token_norm(self) -> nn.Module:
        return nn.Identity()

    def _build_context_input_norm(self) -> nn.Module:
        return nn.Identity()

    def _build_context_encoder(self) -> nn.Module:
        return nn.Identity()

    def _build_context_token_norm(self) -> nn.Module:
        return nn.Identity()

    def _build_memory_encoder(self) -> tuple[nn.Module, int]:
        memory_dims = self.config.decoder_config.memory_dims
        if memory_dims is None:
            return nn.Identity(), self.d_model_encoder
        if len(memory_dims) == 0:
            raise ValueError("decoder_config.memory_dims must be None or contain at least one dimension")
        return (
            self._build_encoder_from_dims(
                input_dim=self.d_model_encoder,
                dims=memory_dims,
                act_fn_cls=self.config.act_fn_cls,
                linear_init_gain=self.config.decoder_config.token_encoder_init_gain,
                projection_init_gain=self.config.decoder_config.token_encoder_projection_init_gain,
                end_with_act_fn=self.config.decoder_config.memory_encoder_end_with_act_fn,
            ),
            memory_dims[-1],
        )

    def _build_decoder(
            self,
            decoder_config: MATQCXDecoderConfig,
    ) -> nn.Module:
        return MATQCXDecoder(
            config=decoder_config,
            max_agents=self.max_agents,
            input_d_model=self.d_model_encoder,
            action_d_model=self.d_model_decoder,
            memory_d_model=self.memory_d_model,
        )

    def _encode_query_tokens(
            self,
            augmented_observations: torch.Tensor,
            *,
            agent_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = agent_embeddings
        return augmented_observations

    def _encode_context_tokens(
            self,
            augmented_observations: torch.Tensor,
            actions: torch.Tensor,
            *,
            agent_embeddings: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = augmented_observations, agent_embeddings
        return self.action_token_norm(self.action_encoder(self.action_input_norm(actions)))

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "encoder": self._module_grad_norm(self.encoder),
            "action_input_norm": self._module_grad_norm(self.action_input_norm),
            "action_encoder": self._module_grad_norm(self.action_encoder),
            "action_token_norm": self._module_grad_norm(self.action_token_norm),
            "memory_input_norm": self._module_grad_norm(self.memory_input_norm),
            "memory_encoder": self._module_grad_norm(self.memory_encoder),
            "memory_token_norm": self._module_grad_norm(self.memory_token_norm),
            "decoder": self._module_grad_norm(self.decoder),
            "actor_head_input_norm": self._module_grad_norm(self.actor_head_input_norm),
            "actor_head": self._module_grad_norm(self.actor_head),
            "action_dist": self._module_grad_norm(self.action_dist),
            "critic": self._module_grad_norm(self.critic),
            "total": self._module_grad_norm(self),
        }
