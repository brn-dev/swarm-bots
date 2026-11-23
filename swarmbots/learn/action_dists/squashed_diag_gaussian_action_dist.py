from dataclasses import dataclass
from typing import Any

from swarmbots.learn.action_dists.action_dist import ActionNetInitialization
from swarmbots.learn.action_dists.diag_gaussian_action_dist import DiagGaussianActionDist
from swarmbots.learn.nn_components.nn_init import init_linear_orthogonal


@dataclass(frozen=True)
class SquashedDiagGaussianConfig:
    std: float
    std_learnable: bool


class SquashedDiagGaussianActionDist(DiagGaussianActionDist):

    def __init__(
            self,
            latent_dim: int,
            action_dim: int,
            std: float,
            std_learnable: bool,
            action_net_initialization: ActionNetInitialization = init_linear_orthogonal,
    ) -> None:
        super().__init__(
            latent_dim=latent_dim,
            action_dim=action_dim,
            std=std,
            std_learnable=std_learnable,
            action_net_initialization=action_net_initialization,
        )
        self.squash_output = True

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "squash_output": True,
        }

    @property
    def compile_friendly(self) -> bool:
        return True
