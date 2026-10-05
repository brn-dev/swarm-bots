from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.rollout_utils import initial_previous_actions
from swarmbots.learn.temporal_state import clone_detach_temporal_state


class BenchmarkPolicy:
    """Adapt a trainer's actor and frozen normalization to benchmark evaluation.

    Recurrent state and previous actions belong to this adapter. Create a fresh
    adapter for each evaluation call; the trainer's actor is shared, so training
    and evaluation must not run concurrently.
    """

    def __init__(self, policy: BasePolicy, env: BaseLearnEnvWrapper, *, deterministic: bool = True) -> None:
        self.policy = policy
        self.deterministic = deterministic
        normalizers = []
        while hasattr(env, "env"):
            if isinstance(env, TorchFeatureWiseObsNormWrapper) and env.obs_key in ("local_obs", "global_obs"):
                normalizers.append(env)
            if isinstance(env, SwarmBotsLearnEnvWrapper):
                self.action_env = env
            env = env.env
        self.normalizers = tuple(reversed(normalizers))
        self.temporal_state: Any = None
        self.previous_actions: torch.Tensor | None = None
        action_dist = getattr(self.policy, "action_dist", None)
        self.distribution_state = () if action_dist is None else tuple(None for _ in action_dist.distributions)
        self.initialized = False

    @torch.no_grad()
    def __call__(
        self,
        observations: Mapping[str, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> Mapping[str, torch.Tensor]:
        # The evaluator may pass privileged fields, but they must never reach the actor.
        actor_obs = {key: observations[key] for key in ("local_obs", "global_obs", "agent_mask")}
        for normalizer in self.normalizers:
            was_updating = normalizer.update_running_mean
            normalizer.update_running_mean = False
            try:
                actor_obs = normalizer.observations(actor_obs)
            finally:
                normalizer.update_running_mean = was_updating
        local_obs = actor_obs["local_obs"]
        if not self.initialized:
            self.temporal_state = self.policy.initial_temporal_state(
                local_obs.shape[0],
                local_obs.shape[1],
                device=local_obs.device,
                dtype=local_obs.dtype,
            )
            self.previous_actions = initial_previous_actions(
                obs=actor_obs,
                n_agent_actions=self.action_env.action_space.total_agent_action_dim,
                policy=self.policy,
            )
            self.initialized = True
        if self.previous_actions is not None:
            self.previous_actions[episode_starts] = 0
        module_training_modes = {module: module.training for module in self.policy.modules()}
        action_dist = getattr(self.policy, "action_dist", None)
        training_distribution_state = None if action_dist is None else action_dist.get_temporal_correlation_state()
        if action_dist is not None:
            action_dist.set_temporal_correlation_state(self.distribution_state)
        self.policy.eval()
        try:
            if action_dist is not None:
                action_dist.reset_temporal_correlations_on_ep_start(episode_starts)
            if not self.deterministic and self.policy.gsde_enabled:
                self.policy.action_dist.reset_temporal_correlations_on_step(batch_shape=tuple(local_obs.shape[:-1]))
            actions, temporal_state = self.policy.act_with_temporal_state(
                **actor_obs,
                previous_actions=self.previous_actions,
                deterministic=self.deterministic,
                temporal_state=self.temporal_state,
                episode_start_mask=episode_starts,
            )
            if action_dist is not None:
                self.distribution_state = clone_detach_temporal_state(action_dist.get_temporal_correlation_state())
        finally:
            if action_dist is not None:
                action_dist.set_temporal_correlation_state(training_distribution_state)
            for module, was_training in module_training_modes.items():
                module.training = was_training
        self.temporal_state = clone_detach_temporal_state(temporal_state)
        if self.previous_actions is not None:
            self.previous_actions = actions.clone()
        return self.action_env._actions_to_env(actions)


def as_benchmark_policy(algorithm: BaseAlgorithm, *, deterministic: bool = True) -> BenchmarkPolicy:
    """Return a policy accepted by ``swarmbots.evaluate_policy``."""
    return BenchmarkPolicy(algorithm.policy, algorithm.env, deterministic=deterministic)
