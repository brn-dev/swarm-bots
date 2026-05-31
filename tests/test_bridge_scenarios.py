from __future__ import annotations

from types import SimpleNamespace

import mujoco
import torch

from swarmbots.mj_env.float_or_dist_params import UniformDistParams
from swarmbots.mj_env.scenarios.scenario_presets import default_bridge as default_mj_bridge
from swarmbots.mjw_env.mjw_model_metadata import build_model_metadata
from swarmbots.mjw_env.scenarios.mjw_bridge_runtime import (
    BridgeMJWScenarioRuntime,
    _compute_bridge_reward_kernel,
)
from swarmbots.mjw_env.scenarios.mjw_bridge_scenario import MJWBridgeRuntimeMetadata
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_bridge as default_mjw_bridge
from swarmbots.scenario_presets.scenario_presets_kwargs import BRIDGE_SCENARIO_KWARGS


def test_bridge_presets_expose_bridge_x_as_hidden_global_obs_in_both_backends() -> None:
    mj_scenario = default_mj_bridge(reset_settle_time=0.0)
    mjw_scenario = default_mjw_bridge()
    state, connections = mj_scenario.reset_scenario(mj_scenario.dummy_model, mj_scenario.dummy_data, settle=False)
    mj_obs = mj_scenario.get_obs(mj_scenario.dummy_model, mj_scenario.dummy_data, state, connections)

    assert mj_obs["global_obs"].shape == (0,)
    assert mj_obs["hidden_global_vars"].shape == (1,)
    assert mj_scenario.fell_off_bridge_reward == BRIDGE_SCENARIO_KWARGS["fell_off_bridge_reward"]
    assert mjw_scenario.get_single_observation_space()["global_obs"].shape == (0,)
    assert mjw_scenario.get_single_observation_space()["hidden_global_vars"].shape == (1,)
    assert mjw_scenario.fell_off_bridge_reward == BRIDGE_SCENARIO_KWARGS["fell_off_bridge_reward"]


def test_bridge_mjw_runtime_metadata_hook_owns_bridge_mocap_id() -> None:
    scenario = default_mjw_bridge()
    host_model = scenario.build_model()

    generic_metadata = build_model_metadata(host_model, scenario)
    runtime_metadata = scenario.build_runtime_metadata(host_model=host_model)

    bridge_body_id = mujoco.mj_name2id(host_model, mujoco.mjtObj.mjOBJ_BODY, "Bridge")
    assert isinstance(runtime_metadata, MJWBridgeRuntimeMetadata)
    assert runtime_metadata.bridge_mocap_id == int(host_model.body_mocapid[bridge_body_id])
    assert not hasattr(generic_metadata, "bridge_mocap_id")


def test_bridge_mjw_reset_sampling_respects_bridge_x_distribution() -> None:
    runtime = object.__new__(BridgeMJWScenarioRuntime)
    device = torch.device("cpu")
    runtime.bindings = SimpleNamespace(device=device)
    runtime.scenario = SimpleNamespace(
        bridge_x=UniformDistParams(low=-0.25, high=0.25),
        swarm=SimpleNamespace(get_active_pool_size=lambda: 2, max_unit_extent=0.2),
        swarm_start_x=0.0,
        swarm_start_y=0.0,
        randomize_initial_swarm_z_rotation=False,
    )

    samples = runtime.sample_reset_batch(n_reset=32, rng=torch.Generator(device=device).manual_seed(123))

    assert samples.bridge_x.shape == (32,)
    assert torch.all(samples.bridge_x >= -0.25)
    assert torch.all(samples.bridge_x <= 0.25)


def test_bridge_mjw_reward_kernel_adds_fall_penalty_and_termination() -> None:
    unit_y = torch.tensor([[1.4, 1.0], [0.3, 0.4]], dtype=torch.float32)
    unit_z = torch.tensor([[0.2, -1.2], [0.2, -1.2]], dtype=torch.float32)
    stable_mask = torch.tensor([True, False], dtype=torch.bool)
    active_mask = torch.tensor([[True, True], [True, True]], dtype=torch.bool)
    partner_unit = torch.full((2, 2, 1), -1, dtype=torch.long)
    progress = torch.tensor([1.0, 0.0], dtype=torch.float32)

    new_progress, progress_reward, guidance_reward, fell_off_bridge, fall_reward, _active_below_threshold = (
        _compute_bridge_reward_kernel(
            unit_y,
            unit_z,
            stable_mask,
            active_mask,
            partner_unit,
            progress,
            -1.0,
            -2.0,
            0.5,
            1.0,
            -0.25,
            2.0,
        )
    )

    assert torch.allclose(new_progress, torch.tensor([1.2, 0.0]))
    assert torch.allclose(progress_reward, torch.tensor([0.1, 0.0]))
    assert torch.allclose(guidance_reward, torch.tensor([-0.5, -0.5]))
    assert torch.equal(fell_off_bridge, torch.tensor([True, False]))
    assert torch.allclose(fall_reward, torch.tensor([-2.0, 0.0]))
