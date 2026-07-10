from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Any

import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.action_dists.reparameterized_sign_magnitude_kumaraswamy_action_dist import (
    ReparameterizedSignMagnitudeKumaraswamyActionDist,
)
from swarmbots.learn.algos.off_policy.replay_buffer import OffPolicyReplayBatch, OffPolicyReplayEpisodeSegmentBatch
from swarmbots.learn.algos.sac import BaseSACPolicy, SAC
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv


@dataclass(frozen=True)
class TargetCase:
    name: str
    target_action: float
    expected_sign: int


@dataclass(frozen=True)
class CaseResult:
    name: str
    mode_mean: float
    sample_mean: float
    sample_abs_mean: float
    positive_fraction: float
    negative_fraction: float
    near_zero_fraction: float
    final_actor_loss: float
    final_q_pi: float


@dataclass(frozen=True)
class PassThresholds:
    sign_fraction: float
    signed_sample_mean: float
    zero_mode_abs: float
    zero_sample_abs_mean: float


class MockCriticRSMKSACPolicy(BaseSACPolicy):
    def __init__(
            self,
            *,
            n_agents: int,
            action_dim: int,
            target_action: float,
            q_scale: float,
    ) -> None:
        super().__init__()
        self.n_agents = n_agents
        self.action_dim = action_dim
        self.target_action = float(target_action)
        self.q_scale = float(q_scale)
        self.action_dist = ReparameterizedSignMagnitudeKumaraswamyActionDist(
            latent_dim=1,
            action_dim=action_dim,
            action_net_initialization=None,
            initial_positive_prob=0.5,
            negative_a=2.0,
            negative_b=2.5,
            positive_a=2.0,
            positive_b=2.5,
        )
        self.critic_bias = torch.nn.Parameter(torch.tensor(0.0))

    def action_log_prob(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            deterministic: bool = False,
            use_rsample: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (global_obs, hidden_local_vars, hidden_global_vars)
        latent = local_obs.new_zeros((*local_obs.shape[:2], 1))
        actions, log_probs = self.action_dist.get_actions_with_log_probs(
            latent,
            deterministic=deterministic,
            use_rsample=use_rsample,
        )
        if agent_mask is not None:
            actions = actions.masked_fill(~agent_mask.unsqueeze(-1), 0.0)
            log_probs = log_probs.masked_fill(~agent_mask, 0.0)
        return actions, log_probs

    def q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (local_obs, global_obs, hidden_local_vars, hidden_global_vars)
        target = torch.full_like(actions, self.target_action)
        q_per_action = -self.q_scale * (actions - target).square()
        if agent_mask is not None:
            q_per_action = q_per_action * agent_mask.unsqueeze(-1).to(dtype=q_per_action.dtype)
        q_value = q_per_action.sum(dim=(1, 2)) + self.critic_bias
        return q_value, q_value

    def target_q_values(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            actions: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        _ = (global_obs, actions, hidden_local_vars, hidden_global_vars, agent_mask)
        target_q = local_obs.new_zeros(local_obs.shape[0])
        return target_q, target_q

    def actor_parameters(self) -> list[torch.nn.Parameter]:
        return list(self.action_dist.parameters())

    def critic_parameters(self) -> list[torch.nn.Parameter]:
        return [self.critic_bias]

    def compute_actor_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        _ = batch
        return None, {}

    def compute_critic_nop_loss(
            self,
            batch: OffPolicyReplayBatch | OffPolicyReplayEpisodeSegmentBatch,
    ) -> tuple[torch.Tensor | None, dict[str, Any]]:
        _ = batch
        return None, {}

    def polyak_update_targets(self, tau: float) -> None:
        _ = tau

    def get_hyper_parameters(self) -> dict[str, Any]:
        return {
            "target_action": self.target_action,
            "q_scale": self.q_scale,
            "action_dist": self.action_dist.get_hyper_parameters(),
        }

    def get_grad_norms(self) -> dict[str, float]:
        return {
            "actor": self._grad_norm_from_parameters(self.actor_parameters()),
            "critic": self._parameter_grad_norm(self.critic_bias),
        }

    def update_loss_weights(self, **weights: float) -> None:
        if weights:
            raise ValueError(f"Unknown weights given: {weights}")


def make_env() -> SwarmBotsLearnEnvWrapper:
    vector_env = SyncVectorEnv(
        [
            lambda: TestingSwarmBotsEnv(
                n_agents=1,
                n_local_obs=1,
                n_global_obs=1,
                actuators_dim=1,
                connectors_dim=1,
                n_hidden_local_vars=1,
                n_hidden_global_vars=1,
                continuous_connector_actions=True,
            )
        ],
        autoreset_mode=AutoresetMode.SAME_STEP,
    )
    return SwarmBotsLearnEnvWrapper(vector_env)


def make_batch(env: SwarmBotsLearnEnvWrapper, *, batch_size: int, device: torch.device) -> OffPolicyReplayBatch:
    n_agents = env.n_agents
    action_dim = env.action_space.total_agent_action_dim
    return OffPolicyReplayBatch(
        local_obs=torch.zeros(batch_size, n_agents, env.local_obs_dim, device=device),
        global_obs=torch.zeros(batch_size, env.global_obs_dim, device=device),
        hidden_local_vars=torch.zeros(batch_size, n_agents, env.hidden_local_vars_dim, device=device),
        hidden_global_vars=torch.zeros(batch_size, env.hidden_global_vars_dim, device=device),
        agent_mask=None,
        actions=torch.zeros(batch_size, n_agents, action_dim, device=device),
        rewards=torch.zeros(batch_size, device=device),
        terminations=torch.ones(batch_size, dtype=torch.bool, device=device),
        truncations=torch.zeros(batch_size, dtype=torch.bool, device=device),
        previous_actions=None,
        next_local_obs=torch.zeros(batch_size, n_agents, env.local_obs_dim, device=device),
        next_global_obs=torch.zeros(batch_size, env.global_obs_dim, device=device),
        next_hidden_local_vars=torch.zeros(batch_size, n_agents, env.hidden_local_vars_dim, device=device),
        next_hidden_global_vars=torch.zeros(batch_size, env.hidden_global_vars_dim, device=device),
        next_agent_mask=None,
    )


def train_case(
        target_case: TargetCase,
        *,
        steps: int,
        batch_size: int,
        learning_rate: float,
        q_scale: float,
        eval_samples: int,
        device: torch.device,
) -> CaseResult:
    env = make_env()
    try:
        policy = MockCriticRSMKSACPolicy(
            n_agents=env.n_agents,
            action_dim=env.action_space.total_agent_action_dim,
            target_action=target_case.target_action,
            q_scale=q_scale,
        )
        algo = SAC(
            policy=policy,
            env=env,
            learning_rate=learning_rate,
            buffer_capacity_per_env=batch_size,
            learning_starts=0,
            batch_size=batch_size,
            gamma=0.0,
            ent_coef=0.0,
            target_entropy=0.0,
            max_grad_norm=None,
            train_device=device,
            rollout_device=device,
            replay_storage_device=device,
        )
        batch = make_batch(env, batch_size=batch_size, device=device)
        last_metrics: dict[str, float] = {}
        for update_idx in range(steps):
            last_metrics, _actor_grad_norm, _critic_grad_norm = algo._train_step(
                batch,
                global_update_idx=update_idx,
            )

        with torch.no_grad():
            eval_latent = torch.zeros(eval_samples, env.n_agents, 1, device=device)
            policy.action_dist.update_latent_features(eval_latent)
            mode = policy.action_dist.mode()
            samples = policy.action_dist.sample()

        return CaseResult(
            name=target_case.name,
            mode_mean=mode.mean().item(),
            sample_mean=samples.mean().item(),
            sample_abs_mean=samples.abs().mean().item(),
            positive_fraction=(samples > 0.0).to(torch.float32).mean().item(),
            negative_fraction=(samples < 0.0).to(torch.float32).mean().item(),
            near_zero_fraction=(samples.abs() < 0.05).to(torch.float32).mean().item(),
            final_actor_loss=last_metrics["actor_loss"],
            final_q_pi=last_metrics["q_pi"],
        )
    finally:
        env.close()


def assert_case_passed(result: CaseResult, target_case: TargetCase, thresholds: PassThresholds) -> None:
    if target_case.expected_sign > 0:
        if (
                result.positive_fraction < thresholds.sign_fraction
                or result.sample_mean < thresholds.signed_sample_mean
        ):
            raise AssertionError(f"{result.name} did not learn the positive side: {result}")
        return
    if target_case.expected_sign < 0:
        if (
                result.negative_fraction < thresholds.sign_fraction
                or result.sample_mean > -thresholds.signed_sample_mean
        ):
            raise AssertionError(f"{result.name} did not learn the negative side: {result}")
        return
    if (
            abs(result.mode_mean) > thresholds.zero_mode_abs
            or result.sample_abs_mean > thresholds.zero_sample_abs_mean
    ):
        raise AssertionError(f"{result.name} did not learn near-zero actions: {result}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train RSMK with SAC actor updates against a mock quadratic critic.",
    )
    parser.add_argument("--steps", type=int, default=2500)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=5e-3)
    parser.add_argument("--q-scale", type=float, default=20.0)
    parser.add_argument("--eval-samples", type=int, default=10_000)
    parser.add_argument("--sign-fraction-threshold", type=float, default=0.9)
    parser.add_argument("--signed-sample-mean-threshold", type=float, default=0.35)
    parser.add_argument("--zero-mode-abs-threshold", type=float, default=0.13)
    parser.add_argument("--zero-sample-abs-mean-threshold", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-assert", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    target_cases = [
        TargetCase(name="positive", target_action=0.75, expected_sign=1),
        TargetCase(name="negative", target_action=-0.75, expected_sign=-1),
        TargetCase(name="zero", target_action=0.0, expected_sign=0),
    ]
    thresholds = PassThresholds(
        sign_fraction=args.sign_fraction_threshold,
        signed_sample_mean=args.signed_sample_mean_threshold,
        zero_mode_abs=args.zero_mode_abs_threshold,
        zero_sample_abs_mean=args.zero_sample_abs_mean_threshold,
    )

    for target_case in target_cases:
        result = train_case(
            target_case,
            steps=args.steps,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            q_scale=args.q_scale,
            eval_samples=args.eval_samples,
            device=device,
        )
        print(
            f"{result.name:>8}: "
            f"mode_mean={result.mode_mean:+.4f}, "
            f"sample_mean={result.sample_mean:+.4f}, "
            f"abs_mean={result.sample_abs_mean:.4f}, "
            f"pos_frac={result.positive_fraction:.3f}, "
            f"neg_frac={result.negative_fraction:.3f}, "
            f"near_zero_frac={result.near_zero_fraction:.3f}, "
            f"actor_loss={result.final_actor_loss:+.4f}, "
            f"q_pi={result.final_q_pi:+.4f}"
        )
        if not args.no_assert:
            assert_case_passed(result, target_case, thresholds)


if __name__ == "__main__":
    main()
