from typing import Optional

import torch

from swarmbots.learn.algos.mat.mat_policy import MATPolicy
from swarmbots.learn.algos.ppo.ppo import PPO
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper


class MAT(PPO):

    def __init__(
            self,
            policy: MATPolicy,
            env: BaseLearnEnvWrapper,
            learning_rate: float = 3e-4,
            n_episodes_per_rollout: int = 64,
            max_episode_length: int = 1000,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: Optional[float] = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            max_grad_norm: float = 0.5,
            target_kl: Optional[float] = None,
            device: str | torch.device = "auto",
    ):
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            n_episodes_per_rollout=n_episodes_per_rollout,
            max_episode_length=max_episode_length,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            ent_coef=ent_coef,
            vf_coef=vf_coef,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            device=device,
        )
