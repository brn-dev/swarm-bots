from dataclasses import dataclass, field

import torch
from torch import nn

from swarmbots.learn.algos.mappo.mappo_encoder import MAPPOEncoderConfig
from swarmbots.learn.nn_components.feed_forward import MLP
from swarmbots.learn.nn_components.nn_init import LinearInitialization, init_linear_orthogonal, make_init_linear_orthogonal


@dataclass(frozen=True)
class MAPPOActorConfig(MAPPOEncoderConfig):
    actor_head_hidden_dims: list[int] = field(default_factory=list)
    latent_pi_dim: int = 64
    actor_head_init_gain: float = 1.0


class MAPPOActor(nn.Module):

    def __init__(
            self,
            local_latent_dim: int,
            config: MAPPOActorConfig,
            linear_init: LinearInitialization = init_linear_orthogonal,
            act_fun_class: type[nn.Module] | None = None,
    ) -> None:
        super().__init__()
        actor_act_fun_class = config.act_fun_class if act_fun_class is None else act_fun_class
        self.local_latent_dim = local_latent_dim
        self.latent_pi_dim = config.latent_pi_dim
        self.hidden_dims = list(config.actor_head_hidden_dims)
        self.act_fun_class = actor_act_fun_class

        self.mlp = MLP(
            input_dim=local_latent_dim,
            hidden_dims=[*config.actor_head_hidden_dims, config.latent_pi_dim],
            end_with_act_fn=True,
            linear_init=linear_init,
            final_linear_init=make_init_linear_orthogonal(config.actor_head_init_gain),
            act_fn_cls=actor_act_fun_class,
        )

    def forward(self, local_latents: torch.Tensor) -> torch.Tensor:
        return self.mlp(local_latents)
