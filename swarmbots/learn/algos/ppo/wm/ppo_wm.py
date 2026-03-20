import abc
from typing import Any, Literal, Callable

import torch
import torch.nn as nn
from loguru import logger

from swarmbots.learn.action_dists.action_dist import ActionMetricsSplitterInput
from swarmbots.learn.algos.ppo.ppo import PPO, PPOLearningRate, PPORolloutMode, WholeEpisodesRolloutMode
from swarmbots.learn.algos.ppo.ppo_policy import BasePPOPolicy
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisode
from swarmbots.learn.algos.ppo.wm.ppo_wm_sampler import PPOWMSampler, PPOWMSamples
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.gsde_reset import GSDEResetMode
from swarmbots.learn.losses import LossDict, LossMetrics


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
            action_splitter: ActionMetricsSplitterInput = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        dict[str, Any],
        LossDict,
        LossMetrics,
    ]:
        """
        :return: log_probs, values, world_model_metrics, extra_losses, extra_loss_metrics
        extra_losses must include key "world_model" (unscaled).
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
            rollout_mode: PPORolloutMode = WholeEpisodesRolloutMode(6),
            max_episode_length: int = 1000,
            batch_size: int = 64,
            n_epochs: int = 10,
            gamma: float = 0.99,
            gae_lambda: float = 0.95,
            clip_range: float = 0.2,
            clip_range_vf: float | None = None,
            normalize_advantage: bool = True,
            mc_ent_coef: float = 0.0,
            vf_coef: float = 0.5,
            value_loss_fn: nn.Module | None = None,
            max_grad_norm: float = 2.0,
            target_kl: float | None = None,
            gsde_reset_mode: GSDEResetMode | None = None,
            agent_logprob_reduction: Literal["sum", "mean"] | None = None,
            train_device: str | torch.device = "auto",
            rollout_device: str | torch.device = "cpu",
            world_model_num_next_steps: int = 1,
            world_model_loss_coef: float = 1.0,
            world_model_target_tau: float | None = None,
            metrics_action_splitters: list[Callable[[torch.Tensor], dict[str, torch.Tensor]] | None] | None = None,
            use_popart: bool = False,
    ) -> None:
        if world_model_num_next_steps < 1:
            raise ValueError(f"world_model_num_next_steps must be >= 1, got {world_model_num_next_steps}")
        super().__init__(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            rollout_mode=rollout_mode,
            max_episode_length=max_episode_length,
            batch_size=batch_size,
            n_epochs=n_epochs,
            gamma=gamma,
            gae_lambda=gae_lambda,
            clip_range=clip_range,
            clip_range_vf=clip_range_vf,
            normalize_advantage=normalize_advantage,
            mc_ent_coef=mc_ent_coef,
            vf_coef=vf_coef,
            value_loss_fn=value_loss_fn,
            max_grad_norm=max_grad_norm,
            target_kl=target_kl,
            gsde_reset_mode=gsde_reset_mode,
            agent_logprob_reduction=agent_logprob_reduction,
            train_device=train_device,
            rollout_device=rollout_device,
            use_popart=use_popart,
            metrics_action_splitters=metrics_action_splitters,
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
        (
            log_probs,
            values,
            wm_loss_metrics,
            extra_losses,
            extra_loss_metrics,
        ) = self.policy.evaluate_actions_and_world_model(
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
            action_splitter=self.metrics_action_splitters,
        )

        loss, approx_kl_div, metrics = self.compute_ppo_loss(
            batch=batch,
            log_probs=log_probs,
            values=values,
        )

        if "world_model" not in extra_losses:
            raise ValueError('evaluate_actions_and_world_model must return "world_model" in extra_losses')

        world_model_loss = extra_losses["world_model"]
        extra_losses["world_model"] = self.world_model_loss_coef * world_model_loss

        reduced_extra_losses = self._reduce_extra_losses(batch, extra_losses)
        loss = loss + torch.stack(tuple(reduced_extra_losses.values())).sum()
        metrics.update({f"{name}_loss_scaled": value.item() for name, value in reduced_extra_losses.items()})
        metrics.update(extra_loss_metrics)
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
        if cmd in {"set_wm_num_next_steps", "set_world_model_num_next_steps", "wm_num_next_steps"}:
            num_next_steps = int(params)
            if num_next_steps < 1:
                raise ValueError(f"world_model_num_next_steps must be >= 1, got {num_next_steps}")
            logger.warning(f"Setting world_model_num_next_steps to {num_next_steps}")
            self.world_model_num_next_steps = num_next_steps
            return True
        if cmd in {"set_wm_target_tau", "set_world_model_target_tau", "wm_target_tau"}:
            param = params.strip().lower()
            if param in {"none", "null", ""}:
                logger.warning("Disabling world_model_target_tau")
                self.world_model_target_tau = None
                return True
            tau = float(params)
            if not (0.0 < tau <= 1.0):
                raise ValueError(f"world_model_target_tau must be in (0, 1], got {tau}")
            logger.warning(f"Setting world_model_target_tau to {tau}")
            self.world_model_target_tau = tau
            return True
        return super()._execute_command(cmd, params, extra_run_metadata)

    def _after_optimizer_step(self) -> None:
        if self.world_model_target_tau is None:
            return
        self.policy.update_world_model_targets(self.world_model_target_tau)
