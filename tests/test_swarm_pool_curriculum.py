from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from swarmbots.mj_env.swarm.homogeneous_swarm import HomogeneousSwarm, PreConnectedUnitLocationsConfig
from swarmbots.mjw_env.scenarios.base_mjw_scenario import (
    BaseMJWScenarioRuntime,
    MJWRuntimeBindings,
)
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig, MJWHomogeneousSwarm


def test_mj_preconnected_swarm_active_pool_size_uses_seed_prefix() -> None:
    prefixed_swarm = HomogeneousSwarm(
        PreConnectedUnitLocationsConfig(
            num_units=5,
            num_unit_probs={4: 1.0, 5: 1.0},
            max_radius=1.5,
            unconnected_prob=0.02,
            z_pos=0.5,
            pool_seeds=(42_000, 42_001, 42_002),
            active_pool_size=1,
        )
    )
    single_seed_swarm = HomogeneousSwarm(
        PreConnectedUnitLocationsConfig(
            num_units=5,
            num_unit_probs={4: 1.0, 5: 1.0},
            max_radius=1.5,
            unconnected_prob=0.02,
            z_pos=0.5,
            pool_seeds=(42_000,),
            active_pool_size=1,
        )
    )

    prefixed_locations, prefixed_quats, prefixed_connections = prefixed_swarm._generate_preconnected_swarm(
        np.random.default_rng(123)
    )
    single_seed_locations, single_seed_quats, single_seed_connections = single_seed_swarm._generate_preconnected_swarm(
        np.random.default_rng(123)
    )

    assert prefixed_locations == single_seed_locations
    assert prefixed_quats == single_seed_quats
    np.testing.assert_allclose(prefixed_connections.twist_angles, single_seed_connections.twist_angles)
    np.testing.assert_array_equal(prefixed_connections.connections, single_seed_connections.connections)


@dataclass
class _FakeScenario:
    swarm: Any
    swarm_start_x: float = 0.0
    swarm_start_y: float = 0.0
    randomize_initial_swarm_z_rotation: bool = False


class _RuntimeForTest(BaseMJWScenarioRuntime):
    @property
    def global_obs(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def hidden_local_obs(self) -> torch.Tensor:
        raise NotImplementedError()

    @property
    def hidden_global_obs(self) -> torch.Tensor:
        raise NotImplementedError()

    def sample_reset_batch(self, *, n_reset: int, rng: torch.Generator) -> Any:
        raise NotImplementedError()

    def select_reset_batch(self, *, reset_batch: Any, mask: torch.Tensor) -> Any:
        raise NotImplementedError()

    def apply_reset_batch(self, *, world_idx: torch.Tensor, reset_batch: Any) -> None:
        raise NotImplementedError()

    def build_cpu_reset_specs(self, *, reset_batch: Any) -> list[Any]:
        raise NotImplementedError()

    def settle_cpu_reset_specs(self, *, specs: list[Any]) -> list[Any]:
        raise NotImplementedError()

    def apply_settled_reset_batch(self, *, world_idx: torch.Tensor, snapshots: list[Any]) -> None:
        raise NotImplementedError()

    def compute_step_rewards(self, *, stable_mask: torch.Tensor) -> Any:
        raise NotImplementedError()


def test_mjw_reset_sampling_respects_active_pool_size() -> None:
    swarm = MJWHomogeneousSwarm(
        MJWPreConnectedUnitLocationsConfig(
            num_units=5,
            num_unit_probs={4: 1.0, 5: 1.0},
            max_radius=1.5,
            unconnected_prob=0.02,
            z_pos=0.5,
            pool_seeds=tuple(range(42_000, 42_010)),
            active_pool_size=3,
        )
    )
    bindings = MJWRuntimeBindings(
        device=torch.device("cpu"),
        wp_device=None,
        num_envs=8,
        model=None,
        data=None,
        metadata=None,
        pool=type("Pool", (), {"size": swarm.get_pool_size()})(),
        pool_eq_active=torch.empty(0),
        inactive_unit_positions=torch.empty(0),
        qpos=torch.empty(0),
        qvel=torch.empty(0),
        ctrl=torch.empty(0),
        eq_active=torch.empty(0),
        mocap_pos=torch.empty(0),
        mocap_quat=torch.empty(0),
        xpos=torch.empty(0),
        xquat=torch.empty(0),
        xmat=torch.empty(0),
        time=torch.empty(0),
        qacc_warmstart=torch.empty(0),
        act=torch.empty(0),
        units_active_mask=torch.empty(0),
        partner_unit=torch.empty(0),
        partner_connector=torch.empty(0),
        connection_twist_idx=torch.empty(0),
        disconnect_potentials=torch.empty(0),
        current_step=torch.empty(0),
        is_first_episode=torch.empty(0),
        base_qpos=torch.empty(0),
        base_mocap_pos=torch.empty(0),
        base_mocap_quat=torch.empty(0),
        unit_qpos_adr=torch.empty(0),
        qpos_wp=None,
        pool_active_mask_wp=None,
        pool_positions_wp=None,
        pool_quats_wp=None,
        inactive_unit_positions_wp=None,
        unit_qpos_adr_wp=None,
    )
    runtime = _RuntimeForTest(scenario=_FakeScenario(swarm=swarm), bindings=bindings)
    rng = torch.Generator(device="cpu")
    rng.manual_seed(123)

    reset_batch = runtime._sample_common_reset_batch(n_reset=128, rng=rng)

    assert int(reset_batch.pool_idx.min().item()) >= 0
    assert int(reset_batch.pool_idx.max().item()) < 3
