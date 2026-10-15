from dataclasses import replace
from typing import Any

from torch import nn

from swarmbots.learn.algos.mat.mat_encoder import MATEncoder, MATEncoderConfig


def make_critic_encoder(
    config: MATEncoderConfig | None,
    actor_config: MATEncoderConfig,
    *,
    env: Any,
    max_agents: int,
) -> tuple[MATEncoderConfig, nn.Module]:
    from swarmbots.learn.algos.r_mat.r_mat_encoder import RMATEncoder, RMATEncoderConfig

    resolved = replace(
        actor_config if config is None else config,
        act_fn_cls=actor_config.act_fn_cls,
        dropout=actor_config.dropout,
    )
    encoder_class = RMATEncoder if isinstance(resolved, RMATEncoderConfig) else MATEncoder
    return resolved, encoder_class(
        config=resolved,
        max_agents=max_agents,
        local_obs_dim=env.local_obs_dim,
        global_obs_dim=env.global_obs_dim,
    )
