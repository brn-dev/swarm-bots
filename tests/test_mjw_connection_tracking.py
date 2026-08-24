from __future__ import annotations

import torch

from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv


def test_connection_episode_stats_only_include_active_units() -> None:
    env = object.__new__(MJWSwarmBotsVectorEnv)
    env.successful_connection_counts = torch.tensor(
        [
            [2, 4, 99],
            [5, 1, 3],
        ]
    )
    env.units_active_mask = torch.tensor(
        [
            [True, True, False],
            [True, True, True],
        ]
    )
    dones = torch.tensor([True, False])
    infos: dict[str, object] = {}

    MJWSwarmBotsVectorEnv._inject_connection_episode_stats(
        env,
        infos=infos,
        dones=dones,
    )

    episode = infos["episode"]
    assert isinstance(episode, dict)
    assert torch.equal(infos["_episode"], dones)
    assert torch.equal(
        episode["successful_connections_per_unit_mean"],
        torch.tensor([3.0, 0.0]),
    )
    assert torch.equal(
        episode["successful_connections_per_unit_std"],
        torch.tensor([1.0, 0.0]),
    )
    assert torch.equal(
        episode["successful_connections_per_unit_min"],
        torch.tensor([2.0, 0.0]),
    )
    assert torch.equal(
        episode["successful_connections_per_unit_max"],
        torch.tensor([4.0, 0.0]),
    )


def test_connection_tracker_counts_reconnections_for_each_participating_unit() -> None:
    env = object.__new__(MJWSwarmBotsVectorEnv)
    env.partner_unit = torch.full((1, 3, 2), -1, dtype=torch.long)
    env._connection_active_before_update = torch.empty_like(env.partner_unit, dtype=torch.bool)
    env._connection_active_after_update = torch.empty_like(env.partner_unit, dtype=torch.bool)
    env.successful_connection_counts = torch.zeros((1, 3), dtype=torch.int64)
    env._successful_connection_count_update = torch.empty_like(env.successful_connection_counts)

    torch.ge(env.partner_unit, 0, out=env._connection_active_before_update)
    env.partner_unit[0, 0, 0] = 1
    env.partner_unit[0, 1, 1] = 0
    MJWSwarmBotsVectorEnv._record_successful_connections(env)

    env.partner_unit.fill_(-1)
    torch.ge(env.partner_unit, 0, out=env._connection_active_before_update)
    env.partner_unit[0, 0, 1] = 2
    env.partner_unit[0, 2, 0] = 0
    MJWSwarmBotsVectorEnv._record_successful_connections(env)

    assert env.successful_connection_counts.tolist() == [[2, 1, 1]]
