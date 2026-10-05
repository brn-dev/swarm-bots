"""Detached recurrent actor-state inputs for feed-forward off-policy critics."""

from dataclasses import dataclass
from typing import Any

import torch

from swarmbots.learn.algos.off_policy.joint_critic import JointQNetwork
from swarmbots.learn.algos.off_policy.transformer_critic import TransformerTwinCritic
from swarmbots.learn.algos.r_mat.temporal_sequence_model import LSTMTemporalSequenceModel
from swarmbots.learn.algos.xlstm.slstm import SLSTMTemporalSequenceModel
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import make_init_linear_orthogonal


@dataclass(frozen=True)
class ActorStateCriticInputConfig:
    projection_dim: int | None = None
    projection_hidden_dims: tuple[int, ...] | None = None
    include_slstm_memory_strength: bool = False
    init_gain: float = 1.0
    output_init_gain: float = 1.0


def actor_state_input_dim(temporal_model, d_model: int, config: ActorStateCriticInputConfig) -> int:
    if isinstance(temporal_model, LSTMTemporalSequenceModel):
        return 2 * d_model
    if isinstance(temporal_model, SLSTMTemporalSequenceModel):
        return (3 if config.include_slstm_memory_strength else 2) * d_model
    raise TypeError(
        "Actor-state critic input supports only LSTMTemporalSequenceModel and "
        f"SLSTMTemporalSequenceModel, got {type(temporal_model).__name__}."
    )


def actor_state_critic_input(
    temporal_model, last_layer_state: Any, config: ActorStateCriticInputConfig
) -> torch.Tensor:
    if isinstance(temporal_model, LSTMTemporalSequenceModel):
        hidden_state, cell_state = last_layer_state
        return torch.cat((hidden_state[..., -1, :], cell_state[..., -1, :]), dim=-1).detach()
    if isinstance(temporal_model, SLSTMTemporalSequenceModel):
        hidden_state, cell_state, normalizer_state, stabilizer_state = last_layer_state
        safe_normalizer_state = normalizer_state.clamp_min(1.0)
        normalized_memory = cell_state / safe_normalizer_state
        if not config.include_slstm_memory_strength:
            return torch.cat((hidden_state, normalized_memory), dim=-1).detach()
        log_memory_strength = torch.log(safe_normalizer_state) + stabilizer_state
        bounded_memory_strength = torch.nn.functional.softsign(log_memory_strength)
        return torch.cat((hidden_state, normalized_memory, bounded_memory_strength), dim=-1).detach()
    # Use the same unsupported-model diagnostic as dimension resolution.
    actor_state_input_dim(temporal_model, 1, config)
    raise AssertionError("Unreachable")


class _ActorStateInputMixin:
    def _init_actor_state_encoder(self, *, input_dim, projection_dim, config, act_fn_cls) -> None:
        hidden_dims = (projection_dim,) if config.projection_hidden_dims is None else config.projection_hidden_dims
        self.actor_state_encoder = MLP(
            input_dim=input_dim,
            hidden_dims=[*hidden_dims, projection_dim],
            end_with_act_fn=False,
            linear_init=make_init_linear_orthogonal(config.init_gain),
            final_linear_init=make_init_linear_orthogonal(config.output_init_gain),
            act_fn_cls=act_fn_cls,
        )

    def _append_actor_state(self, local_inputs: torch.Tensor, actor_state: torch.Tensor) -> torch.Tensor:
        projected_state = self.actor_state_encoder(actor_state.detach())
        return torch.cat((local_inputs, projected_state), dim=-1)


class ActorStateTransformerTwinCritic(_ActorStateInputMixin, TransformerTwinCritic):
    def __init__(
        self,
        *,
        actor_state_input_dim: int,
        actor_state_config: ActorStateCriticInputConfig,
        actor_state_default_projection_dim: int,
        **kwargs: Any,
    ) -> None:
        projection_dim = (
            actor_state_default_projection_dim
            if actor_state_config.projection_dim is None
            else int(actor_state_config.projection_dim)
        )
        local_input_dim = int(kwargs.pop("local_input_dim"))
        super().__init__(local_input_dim=local_input_dim + projection_dim, **kwargs)
        self._init_actor_state_encoder(
            input_dim=actor_state_input_dim,
            projection_dim=projection_dim,
            config=actor_state_config,
            act_fn_cls=kwargs["act_fn_cls"],
        )

    def encode(self, *, actor_state: torch.Tensor, local_inputs: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        return super().encode(local_inputs=self._append_actor_state(local_inputs, actor_state), **kwargs)

    def forward(self, *, actor_state: torch.Tensor, local_obs: torch.Tensor, **kwargs: Any):
        return super().forward(local_obs=self._append_actor_state(local_obs, actor_state), **kwargs)


class ActorStateJointQNetwork(_ActorStateInputMixin, JointQNetwork):
    def __init__(
        self,
        *,
        actor_state_input_dim: int,
        actor_state_config: ActorStateCriticInputConfig,
        actor_state_default_projection_dim: int,
        **kwargs: Any,
    ) -> None:
        projection_dim = (
            actor_state_default_projection_dim
            if actor_state_config.projection_dim is None
            else int(actor_state_config.projection_dim)
        )
        local_obs_dim = int(kwargs.pop("local_obs_dim"))
        super().__init__(local_obs_dim=local_obs_dim + projection_dim, **kwargs)
        self._init_actor_state_encoder(
            input_dim=actor_state_input_dim,
            projection_dim=projection_dim,
            config=actor_state_config,
            act_fn_cls=kwargs["act_fn_cls"],
        )

    def forward_with_latents(self, *, actor_state: torch.Tensor, local_obs: torch.Tensor, **kwargs: Any):
        return super().forward_with_latents(local_obs=self._append_actor_state(local_obs, actor_state), **kwargs)

    def forward(self, *, actor_state: torch.Tensor, local_obs: torch.Tensor, **kwargs: Any):
        values, _latents = self.forward_with_latents(actor_state=actor_state, local_obs=local_obs, **kwargs)
        return values
