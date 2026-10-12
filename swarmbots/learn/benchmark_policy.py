from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import torch
from torch.nn import functional as F

from swarmbots.learn.algos.base_algorithm import BaseAlgorithm
from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import BaseLearnEnvWrapper
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.env_wrappers.torch_transition_obs_wrapper import TorchTransitionObsWrapper, _step_transition_tensors
from swarmbots.learn.gsde_reset import (
    GSDEResetMode,
    GSDEEpisodeResetMode,
    GSDEProbabilityResetMode,
    build_gsde_step_reset_mask,
    resolve_gsde_reset_mode,
)
from swarmbots.learn.rollout_utils import initial_previous_actions
from swarmbots.learn.temporal_state import clone_detach_temporal_state


@dataclass
class _TransitionObservationState:
    local_obs: torch.Tensor
    global_obs: torch.Tensor | None
    actions: torch.Tensor


class BenchmarkPolicy:
    """Adapt a trainer's actor and frozen normalization to benchmark evaluation.

    Recurrent state and previous actions belong to this adapter. Create a fresh
    adapter for each evaluation call; the trainer's actor is shared, so training
    and evaluation must not run concurrently.
    """

    def __init__(
        self,
        policy: BasePolicy,
        env: BaseLearnEnvWrapper,
        *,
        deterministic: bool = True,
        scenario_id: int | None = None,
        gsde_reset_mode: GSDEResetMode | None = None,
    ) -> None:
        self.policy = policy
        self.deterministic = deterministic
        self.scenario_id = scenario_id
        self.gsde_reset_mode = resolve_gsde_reset_mode(
            gsde_enabled=policy.gsde_enabled,
            reset_mode=GSDEEpisodeResetMode() if gsde_reset_mode is None else gsde_reset_mode,
        )
        self.rollout_step_idx = 0
        self.has_scenario_id = env.has_scenario_id
        if scenario_id is not None:
            if not self.has_scenario_id:
                raise ValueError("scenario_id selection requires a scenario-conditioned training environment.")
            if not 0 <= scenario_id < len(env.scenario_names):
                raise ValueError(f"scenario_id must be in [0, {len(env.scenario_names)}), got {scenario_id}.")
        observation_transforms = []
        while hasattr(env, "env"):
            if isinstance(env, TorchFeatureWiseObsNormWrapper) and env.obs_key in ("local_obs", "global_obs"):
                observation_transforms.append(env)
            if isinstance(env, TorchTransitionObsWrapper):
                observation_transforms.append(env)
            if isinstance(env, SwarmBotsLearnEnvWrapper):
                self.action_env = env
            env = env.env
        self.observation_transforms = tuple(reversed(observation_transforms))
        self.transition_states: dict[TorchTransitionObsWrapper, _TransitionObservationState] = {}
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
        scenario_ids = observations.get("scenario_id")
        if scenario_ids is None and self.scenario_id is not None:
            scenario_ids = torch.full_like(episode_starts, self.scenario_id, dtype=torch.long)
        if self.has_scenario_id:
            if scenario_ids is None:
                raise ValueError(
                    "Scenario-conditioned benchmark policies require observations['scenario_id'] "
                    "or an explicit scenario_id selection."
                )
            global_obs = actor_obs["global_obs"]
            padding = self.action_env.global_obs_dim - global_obs.shape[-1]
            if padding > 0:
                actor_obs["global_obs"] = F.pad(global_obs, (0, padding))
        if scenario_ids is not None:
            actor_obs["scenario_ids"] = scenario_ids
        for transform in self.observation_transforms:
            if isinstance(transform, TorchFeatureWiseObsNormWrapper):
                was_updating = transform.update_running_mean
                transform.update_running_mean = False
                try:
                    actor_obs = transform.observations(actor_obs)
                finally:
                    transform.update_running_mean = was_updating
            else:
                actor_obs = self._transition_observations(transform, actor_obs, episode_starts)
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
            step_reset_mask = None
            if not self.deterministic and self.gsde_reset_mode is not None:
                if self.rollout_step_idx == 0:
                    action_dist.reset_temporal_correlations_on_step(batch_shape=tuple(local_obs.shape[:-1]))
                else:
                    step_reset_mask = build_gsde_step_reset_mask(
                        gsde_reset_mode=self.gsde_reset_mode,
                        rollout_step_idx=self.rollout_step_idx,
                        batch_shape=tuple(local_obs.shape[:-1]),
                        rollout_device=local_obs.device,
                    )
            if action_dist is not None:
                action_dist.reset_temporal_correlations_on_ep_start(episode_starts)
            if step_reset_mask is not None:
                if isinstance(self.gsde_reset_mode, GSDEProbabilityResetMode):
                    action_dist.reset_temporal_correlations_on_step(mask=step_reset_mask)
                else:
                    action_dist.reset_temporal_correlations_on_step(batch_shape=tuple(local_obs.shape[:-1]))
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
        self.rollout_step_idx += 1
        for transform, state in self.transition_states.items():
            state.actions = transform._actions_to_features(actions, dtype=state.local_obs.dtype).clone()
        if self.previous_actions is not None:
            self.previous_actions = actions.clone()
        return self.action_env._actions_to_env(actions)

    def _transition_observations(
        self,
        wrapper: TorchTransitionObsWrapper,
        observations: dict[str, torch.Tensor],
        episode_starts: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        local_obs = observations["local_obs"]
        global_obs = observations.get("global_obs")
        state = self.transition_states.get(wrapper)
        if state is None:
            state = _TransitionObservationState(
                local_obs=torch.zeros_like(local_obs),
                global_obs=None if global_obs is None else torch.zeros_like(global_obs),
                actions=local_obs.new_zeros((*local_obs.shape[:-1], wrapper._n_action_features)),
            )
        stacked_local, _, _, stacked_global, _ = _step_transition_tensors(
            local_obs, state.local_obs, state.actions, episode_starts,
            global_obs, state.global_obs, wrapper.new_obs_first,
        )
        self.transition_states[wrapper] = _TransitionObservationState(
            local_obs=local_obs.clone(),
            global_obs=None if global_obs is None else global_obs.clone(),
            actions=state.actions,
        )
        transformed = dict(observations)
        transformed["local_obs"] = stacked_local
        if stacked_global is not None:
            transformed["global_obs"] = stacked_global
        return transformed


def as_benchmark_policy(
    algorithm: BaseAlgorithm,
    *,
    deterministic: bool = True,
    scenario_id: int | None = None,
    gsde_reset_mode: GSDEResetMode | None = None,
) -> BenchmarkPolicy:
    """Adapt an actor for benchmark evaluation, selecting a scenario when needed.

    ``scenario_id`` identifies a task in a scenario-conditioned training environment
    when benchmark observations omit their own ``scenario_id``. Supplied observation
    IDs take precedence. Single-task global observations are padded before normalization.
    Stochastic gSDE actions inherit the trainer's reset schedule unless explicitly overridden.
    """
    return BenchmarkPolicy(
        algorithm.policy, algorithm.env, deterministic=deterministic, scenario_id=scenario_id,
        gsde_reset_mode=(
            getattr(algorithm, "gsde_reset_mode", None) if gsde_reset_mode is None else gsde_reset_mode
        ),
    )
