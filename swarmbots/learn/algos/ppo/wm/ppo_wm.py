import abc
from typing import Any, Literal

import torch
import torch.nn as nn
from loguru import logger

from swarmbots.learn.algos.ppo.ppo import PPO, PPOLearningRate
from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisode
from swarmbots.learn.algos.ppo.wm.ppo_wm_sampler import PPOWMSampler, PPOWMSamples
from swarmbots.learn.env_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEResetMode


class PPOWMPolicyMixin(abc.ABC):

    @abc.abstractmethod
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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]]:
        """
        :return: log_probs, entropies, values, world_model_loss, metrics
        """
        raise NotImplementedError()

    @abc.abstractmethod
    def update_world_model_targets(self, tau: float) -> None:
        raise NotImplementedError()


class PPOWM(PPO[PPOWMSamples, PPOWMSampler]):

    policy: BasePPOPolicy | PPOWMPolicyMixin
    world_model_num_next_steps: int
    world_model_loss_coef: float
    world_model_target_tau: float | None

    def __init__(
            self,
            policy: BasePPOPolicy | PPOWMPolicyMixin,
            env: BaseLearnEnvWrapper,
            learning_rate: PPOLearningRate = 3e-4,
            n_episodes_per_rollout: int = 64,
            max_episode_length: int = 1000,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: float | None = None,
            normalize_advantage: bool = True,
            ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            value_loss_fn: nn.Module | None = None,
            max_grad_norm: float = 0.5,
            target_kl: float | None = None,
            gsde_reset_mode: GSDEResetMode | None = None,
            agent_logprob_reduction: Literal["sum", "mean"] | None = None,
            train_device: str | torch.device = "auto",
            rollout_device: str | torch.device = "cpu",
            world_model_num_next_steps: int = 1,
            world_model_loss_coef: float = 1.0,
            world_model_target_tau: float | None = None,
    ) -> None:
        if world_model_num_next_steps < 1:
            raise ValueError(f"world_model_num_next_steps must be >= 1, got {world_model_num_next_steps}")
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
            value_loss_fn=value_loss_fn,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            gsde_reset_mode=gsde_reset_mode,
            agent_logprob_reduction=agent_logprob_reduction,
            train_device=train_device,
            rollout_device=rollout_device,
        )
        self.world_model_num_next_steps = world_model_num_next_steps
        self.world_model_loss_coef = world_model_loss_coef
        self.world_model_target_tau = world_model_target_tau

    def _make_sampler(self, episodes: list[PPOEpisode]) -> PPOWMSampler:
        return PPOWMSampler(
            episodes=episodes,
            num_next_steps=self.world_model_num_next_steps
        )

    def compute_loss(
            self,
            batch: PPOWMSamples,
    ) -> tuple[torch.Tensor, float, dict[str, Any]]:
        log_probs, entropies, values, world_model_loss, wm_loss_metrics = self.policy.evaluate_actions_and_world_model(
            local_obs=batch.local_obs,
            global_obs=batch.global_obs,
            actions=batch.actions,
            next_local_obs=batch.next_local_obs,
            next_global_obs=batch.next_global_obs,
            next_validity_mask=batch.next_validity_mask,
            agent_mask=batch.agent_mask,
            wm_agent_mask=batch.wm_agent_mask,
            wm_loss_agent_mask=batch.wm_loss_agent_mask,
            hidden_vars=batch.hidden_vars,
        )

        ppo_loss, approx_kl_div, metrics = self.compute_ppo_loss(
            batch=batch,
            entropies=entropies,
            log_probs=log_probs,
            values=values,
        )

        loss = ppo_loss + self.world_model_loss_coef * world_model_loss

        metrics.update(wm_loss_metrics)

        metrics['wm_loss'] = world_model_loss.item()
        metrics['wm_loss_scaled'] = (self.world_model_loss_coef * world_model_loss).item()

        return loss, approx_kl_div, metrics

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            **super().get_hyper_parameters(),
            "world_model_num_next_steps": self.world_model_num_next_steps,
            "world_model_loss_coef": self.world_model_loss_coef,
            "world_model_target_tau": self.world_model_target_tau,
        }

    def _execute_command(
            self,
            cmd: str,
            params: str,
            extra_run_metadata: dict[str, Any] | None
    ) -> bool:
        if cmd in {"set_wm_loss_coef", "set_world_model_loss_coef", "wm_loss_coef"}:
            coef = float(params)
            logger.warning(f"Setting world_model_loss_coef to {coef}")
            self.world_model_loss_coef = coef
            return True
        return super()._execute_command(cmd, params, extra_run_metadata)

    def _after_optimizer_step(self) -> None:
        if self.world_model_target_tau is None:
            return
        self.policy.update_world_model_targets(self.world_model_target_tau)
