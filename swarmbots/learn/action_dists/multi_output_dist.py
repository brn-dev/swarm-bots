from typing import Union, Optional

import torch
from gymnasium import spaces

from swarmbots.learn.action_dists.action_dist import ActionDist, ActionNetInitialization, AGENT_ACTIONS_DIM
from swarmbots.learn.action_dists.bernoulli_action_dist import BernoulliActionDist
from swarmbots.learn.action_dists.predicted_std_action_dist import PredictedStdActionDist
from swarmbots.learn.multi_output_utils import get_action_dim

"""
https://github.com/adysonmaia/sb3-plus/blob/main/sb3_plus/mimo/distributions.py
"""
class MultiOutputDistribution(ActionDist):
    """
    Distribution to a multi outputs represented as a Dict or Tuple action space

    """

    def __init__(
            self,
            latent_dim: int,
            action_space: Union[spaces.Dict, spaces.Tuple],
            action_net_initialization: ActionNetInitialization | None,
    ):
        super().__init__(
            latent_dim=latent_dim,
            action_dim=get_action_dim(action_space),
            action_net_initialization=action_net_initialization
        )
        self.action_space = action_space
        list_spaces = action_space.spaces.values() if isinstance(action_space, spaces.Dict) else action_space.spaces

        self.distributions: list[ActionDist] = [make_proba_distribution(latent_dim, s) for s in list_spaces]
        self.action_dims = [get_action_dim(s) for s in list_spaces]

    def update_latent_features(self, latent_pi: torch.Tensor):
        for dist in self.distributions:
            dist.update_latent_features(latent_pi)

    def sample(self) -> torch.Tensor:
        return torch.cat([dist.sample() for dist in self.distributions], dim=AGENT_ACTIONS_DIM)

    def mode(self) -> torch.Tensor:
        return torch.cat([dist.mode() for dist in self.distributions], dim=AGENT_ACTIONS_DIM)

    def log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        split_actions = torch.split(actions, self.action_dims, dim=AGENT_ACTIONS_DIM)
        return torch.stack(
            [dist.log_prob(action) for dist, action in zip(self.distributions, split_actions, strict=True)],
            dim=-1
        ).sum(dim=-1)

    def entropy(self) -> Optional[torch.Tensor]:
        entropies = [dist.entropy() for dist in self.distributions]
        if None in entropies:
            return None
        return torch.stack(entropies, dim=-1).sum(dim=-1)



def make_proba_distribution(
        latent_dim: int,
        action_space: spaces.Space,
) -> ActionDist:
    if isinstance(action_space, spaces.Box):
        # todo range
        return PredictedStdActionDist(
            latent_dim=latent_dim,
            action_dim=action_space.shape[-1],
            base_std=1.0,
            squash_output=True,
            action_net_initialization=None,
            log_std_net_initialization=None,
        )
    elif isinstance(action_space, spaces.MultiBinary):
        return BernoulliActionDist(
            latent_dim=latent_dim,
            action_dim=action_space.shape[-1],
            action_net_initialization=None,
        )
    else:
        raise NotImplementedError
