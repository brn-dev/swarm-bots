"""SAC with a decentralized actor and an MLP or Deep Set team critic."""

from dataclasses import dataclass, field
from typing import Any

from swarmbots.learn.algos.off_policy.joint_critic import JointCriticConfig, JointTwinCritic
from swarmbots.learn.algos.sac.tmasac_actor_heads import TMASACActorHeadConfig, TMASACActorHeadKind
from swarmbots.learn.algos.sac.tmasac_policy import TMASACPolicy, TMASACPolicyConfig
from swarmbots.learn.serialization_utils import serialize_dataclass


@dataclass(frozen=True)
class MASACPolicyConfig(TMASACPolicyConfig):
    actor_head_config: TMASACActorHeadConfig = field(
        default_factory=lambda: TMASACActorHeadConfig(kind=TMASACActorHeadKind.DECENTRALIZED)
    )
    joint_critic_config: JointCriticConfig = field(default_factory=JointCriticConfig)


class MASACPolicy(TMASACPolicy):
    config: MASACPolicyConfig

    def __init__(self, env, config: MASACPolicyConfig = MASACPolicyConfig()) -> None:
        if config.nop_config.enabled and config.joint_critic_config.kind == "mlp":
            raise ValueError("MLP critics do not support NOP; use a Deep Set critic or set use_nop=False")
        if config.share_observation_encoder or config.shared_encoder_config is not None:
            raise ValueError("MASAC baselines use separate actor and critic observation inputs")
        super().__init__(env, config)

    def _build_critic(self, *, local_input_dim: int, global_input_dim: int) -> JointTwinCritic:
        return JointTwinCritic(
            n_agents=self.n_agents,
            local_obs_dim=local_input_dim,
            global_obs_dim=global_input_dim,
            hidden_local_vars_dim=self.critic_hidden_local_vars_dim,
            hidden_global_vars_dim=self.critic_hidden_global_vars_dim,
            action_dim=self.agent_action_dim,
            config=self.config.joint_critic_config,
            act_fn_cls=self.config.act_fn_cls,
        )

    def _critic_module(self) -> JointTwinCritic:
        return getattr(self.critic, "_orig_mod", self.critic)

    def get_hyper_parameters(self) -> dict[str, Any]:
        settings = super().get_hyper_parameters()["tmasac_policy_config"]
        settings["joint_critic_config"] = serialize_dataclass(self.config.joint_critic_config)
        return {"masac_policy_config": settings}
