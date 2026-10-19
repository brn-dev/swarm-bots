from dataclasses import dataclass, field

import torch
from torch import nn

from swarmbots.learn.algos.mappo.mappo_encoder import MAPPOEncoderConfig
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import LinearInitialization, make_init_linear_orthogonal
from swarmbots.learn.nn_components.residual_mlp import ResidualMLP, ResidualMLPConfig


@dataclass(frozen=True)
class MAPPOActorConfig(MAPPOEncoderConfig):
    actor_head_hidden_dims: list[int] = field(default_factory=list)
    latent_pi_dim: int = 64
    actor_head_init_gain: float = 1.0
    residual_mlp_config: ResidualMLPConfig | None = None


class MAPPOActor(nn.Module):

    def __init__(
            self,
            local_latent_dim: int,
            config: MAPPOActorConfig,
            linear_init: LinearInitialization | None = None,
            act_fun_class: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        actor_act_fun_class = config.act_fun_class if act_fun_class is None else act_fun_class
        self.local_latent_dim = local_latent_dim
        self.latent_pi_dim = config.latent_pi_dim
        self.hidden_dims = list(config.actor_head_hidden_dims)
        self.act_fun_class = actor_act_fun_class
        head_initialization = make_init_linear_orthogonal(config.actor_head_init_gain)

        self.mlp = ResidualMLP(
            local_latent_dim, config.latent_pi_dim, config.residual_mlp_config,
            act_fn_cls=actor_act_fun_class,
            linear_init=head_initialization if linear_init is None else linear_init,
            output_linear_init=head_initialization,
        ) if config.residual_mlp_config is not None else MLP(
            input_dim=local_latent_dim,
            hidden_dims=[*config.actor_head_hidden_dims, config.latent_pi_dim],
            end_with_act_fn=True,
            linear_init=head_initialization if linear_init is None else linear_init,
            final_linear_init=head_initialization,
            act_fn_cls=actor_act_fun_class,
        )

    def forward(self, local_latents: torch.Tensor) -> torch.Tensor:
        return self.mlp(local_latents)
