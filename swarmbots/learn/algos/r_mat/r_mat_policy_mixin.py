from typing import Optional

import torch

from swarmbots.learn.algos.mat.independent_encoders import encode_critic_observations
from swarmbots.learn.base_policy import PolicyCompileConfig, policy_inference_step
from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderState
from swarmbots.learn.algos.r_mat.r_ppo_wm_sampler import RPPOWMSampler, RPPOWMSamplerConfig
from swarmbots.learn.algos.r_mat.r_ppo_wm_sampler import RPPOWMSamples

RMATPolicyState = dict[str, RMATEncoderState]


class RMATPolicyMixin:
    config: PolicyCompileConfig
    actor_encoder: RMATEncoder
    critic_encoder: RMATEncoder
    hidden_local_vars_dim: int
    hidden_global_vars_dim: int

    def _generate_rmat_actions(
            self,
            *,
            augmented_observations: torch.Tensor,
            agent_mask: torch.Tensor | None,
            previous_actions: torch.Tensor | None,
            deterministic: bool,
            return_log_probs: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        raise NotImplementedError()

    @policy_inference_step
    def forward(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        augmented_observations, next_state = self._encode_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=None,
            episode_start_mask=None,
        )
        actions, log_probs = self._generate_rmat_actions(
            augmented_observations=augmented_observations,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=True,
        )
        critic_observations, _ = self._encode_critic_observations(
            local_obs=local_obs, global_obs=global_obs, agent_mask=agent_mask,
            temporal_state=next_state, episode_start_mask=None,
            hidden_local_vars=hidden_local_vars, hidden_global_vars=hidden_global_vars,
        )
        values = self.critic(
            critic_observations,
            agent_mask=agent_mask,
        )
        assert log_probs is not None
        return actions, log_probs, values

    @policy_inference_step
    def predict_values(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = previous_actions
        critic_observations, _ = self._encode_critic_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=None,
            episode_start_mask=None,
            hidden_local_vars=hidden_local_vars, hidden_global_vars=hidden_global_vars,
        )
        return self.critic(
            critic_observations,
            agent_mask=agent_mask,
        )

    @policy_inference_step
    def act(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
    ) -> torch.Tensor:
        _ = hidden_local_vars
        _ = hidden_global_vars
        augmented_observations, _ = self._encode_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=None,
            episode_start_mask=None,
        )
        actions, _ = self._generate_rmat_actions(
            augmented_observations=augmented_observations,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=False,
        )
        return actions

    def make_sampler(
            self,
            episodes: list[PPOEpisodeSegment],
            config: RPPOWMSamplerConfig,
    ) -> RPPOWMSampler:
        return RPPOWMSampler(
            episodes=episodes,
            config=config,
            requires_previous_actions=self.requires_previous_actions(),
        )

    def supports_rollout_batch_sampler(self, config: object) -> bool:
        _ = config
        return False

    def initial_temporal_state(
            self,
            batch_size: int,
            n_agents: int,
            *,
            device: torch.device,
            dtype: torch.dtype,
    ) -> RMATPolicyState:
        return {
            role: encoder.initial_state(
                batch_size=batch_size, n_agents=n_agents, device=device, dtype=dtype,
            )
            for role, encoder in (("actor", self.actor_encoder), ("critic", self.critic_encoder))
        }

    @policy_inference_step
    def forward_with_temporal_state(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            *,
            temporal_state: RMATPolicyState | None = None,
            episode_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, RMATPolicyState]:
        augmented_observations, next_state = self._encode_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=temporal_state,
            episode_start_mask=episode_start_mask,
        )
        actions, log_probs = self._generate_rmat_actions(
            augmented_observations=augmented_observations,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=True,
        )
        critic_observations, next_state = self._encode_critic_observations(
            local_obs=local_obs, global_obs=global_obs, agent_mask=agent_mask,
            temporal_state=next_state, episode_start_mask=episode_start_mask,
            hidden_local_vars=hidden_local_vars, hidden_global_vars=hidden_global_vars,
        )
        values = self.critic(
            critic_observations,
            agent_mask=agent_mask,
        )
        assert log_probs is not None
        return actions, log_probs, values, next_state

    @policy_inference_step
    def predict_values_with_temporal_state(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            *,
            temporal_state: RMATPolicyState | None = None,
            episode_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RMATPolicyState]:
        _ = previous_actions
        critic_observations, next_state = self._encode_critic_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=temporal_state,
            episode_start_mask=episode_start_mask,
            hidden_local_vars=hidden_local_vars, hidden_global_vars=hidden_global_vars,
        )
        values = self.critic(
            critic_observations,
            agent_mask=agent_mask,
        )
        return values, next_state

    @policy_inference_step
    def act_with_temporal_state(
            self,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            hidden_local_vars: torch.Tensor | None = None,
            hidden_global_vars: torch.Tensor | None = None,
            agent_mask: torch.Tensor | None = None,
            scenario_ids: torch.Tensor | None = None,
            previous_actions: torch.Tensor | None = None,
            deterministic: bool = False,
            *,
            temporal_state: RMATPolicyState | None = None,
            episode_start_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, RMATPolicyState]:
        _ = hidden_local_vars
        _ = hidden_global_vars
        _ = scenario_ids
        augmented_observations, next_state = self._encode_observations(
            local_obs=local_obs,
            global_obs=global_obs,
            agent_mask=agent_mask,
            temporal_state=temporal_state,
            episode_start_mask=episode_start_mask,
        )
        actions, _ = self._generate_rmat_actions(
            augmented_observations=augmented_observations,
            agent_mask=agent_mask,
            previous_actions=previous_actions,
            deterministic=deterministic,
            return_log_probs=False,
        )
        return actions, next_state

    def _encode_observations(
            self,
            *,
            local_obs: torch.Tensor,
            global_obs: torch.Tensor,
            agent_mask: torch.Tensor | None,
            temporal_state: RMATPolicyState | None,
            episode_start_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, RMATPolicyState]:
        if temporal_state is None:
            temporal_state = self.initial_temporal_state(
                batch_size=local_obs.shape[0],
                n_agents=local_obs.shape[-2],
                device=local_obs.device,
                dtype=local_obs.dtype,
            )
        observations, actor_state = self.actor_encoder(
            local_obs,
            global_obs,
            agent_mask=agent_mask,
            initial_state=temporal_state["actor"],
            reset_mask=episode_start_mask,
        )
        return observations, {**temporal_state, "actor": actor_state}

    def _encode_critic_observations(
        self,
        *,
        local_obs: torch.Tensor,
        global_obs: torch.Tensor,
        hidden_local_vars: torch.Tensor | None,
        hidden_global_vars: torch.Tensor | None,
        agent_mask: torch.Tensor | None,
        temporal_state: RMATPolicyState | None,
        episode_start_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, RMATPolicyState]:
        if temporal_state is None:
            temporal_state = self.initial_temporal_state(
                batch_size=local_obs.shape[0], n_agents=local_obs.shape[-2],
                device=local_obs.device, dtype=local_obs.dtype,
            )
        observations, critic_state = encode_critic_observations(
            self.critic_encoder, local_obs, global_obs, hidden_local_vars, hidden_global_vars, agent_mask,
            initial_state=temporal_state["critic"], reset_mask=episode_start_mask,
        )
        return observations, {**temporal_state, "critic": critic_state}

    def _encode_training_observations(self, batch: RPPOWMSamples) -> tuple[torch.Tensor, torch.Tensor]:
        states = batch.initial_temporal_state
        actor_observations, _ = self.actor_encoder(
            batch.local_obs, batch.global_obs, agent_mask=batch.agent_mask,
            time_mask=batch.time_mask, initial_state=None if states is None else states["actor"],
            reset_mask=batch.episode_start_mask,
        )
        critic_observations, _ = encode_critic_observations(
            self.critic_encoder, batch.local_obs, batch.global_obs,
            batch.hidden_local_vars, batch.hidden_global_vars, batch.agent_mask,
            time_mask=batch.time_mask, initial_state=None if states is None else states["critic"],
            reset_mask=batch.episode_start_mask,
        )
        return actor_observations, critic_observations


def flatten_time_agent_mask(
        *,
        agent_mask: torch.Tensor | None,
        time_mask: torch.Tensor,
        n_agents: int,
) -> torch.Tensor:
    if agent_mask is None:
        return time_mask.unsqueeze(-1).expand(*time_mask.shape, n_agents).reshape(-1, n_agents)
    return (agent_mask & time_mask.unsqueeze(-1)).reshape(-1, n_agents)
