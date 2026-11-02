from typing import Any
from unittest.mock import patch

import numpy as np
import pytest
import torch
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.env_wrappers.learn_wrappers.base_learn_env_wrapper import TorchObs
from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.env_wrappers.learn_wrappers.torch_env_wrapper import TorchEnvWrapper
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv as _TestingSwarmBotsEnv


CUDA = pytest.param("cuda", marks=(
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
))


def _make_env(device: str = "cpu") -> SwarmBotsLearnEnvWrapper:
    return SwarmBotsLearnEnvWrapper(
        SyncVectorEnv([
            lambda: _TestingSwarmBotsEnv(
                n_agents=2, n_local_obs=2, n_global_obs=1, actuators_dim=1, connectors_dim=1,
                n_hidden_local_vars=1, n_hidden_global_vars=1,
            ),
        ], autoreset_mode=AutoresetMode.SAME_STEP),
        device=device,
    )


class _ReusableOutputWrapper(TorchEnvWrapper):
    def __init__(self, env: SwarmBotsLearnEnvWrapper, *, replace_in_place: bool = False) -> None:
        super().__init__(env)
        self._output = torch.empty((1, 2, 2), device=self.device)
        self._replace_in_place = replace_in_place

    def observations(self, observations: TorchObs) -> TorchObs:
        output = self._output[0] if observations["local_obs"].ndim == 2 else self._output
        output.copy_(observations["local_obs"] + 1)
        if self._replace_in_place:
            observations["local_obs"] = output
            return observations
        return {**observations, "local_obs": output}


@pytest.mark.parametrize("replace_in_place", [False, True])
def test_terminal_transforms_snapshot_reused_outputs_and_preserve_unchanged_snapshots(replace_in_place: bool) -> None:
    env = _make_env()
    try:
        wrapper = _ReusableOutputWrapper(env, replace_in_place=replace_in_place)
        terminal_obs, _ = env.reset()
        terminal_snapshot = {key: value.clone() for key, value in terminal_obs.items()}
        infos = {"final_obs": terminal_snapshot, "_final_obs": torch.tensor([True])}
        transformed = wrapper._transform_infos(infos)["final_obs"]
        wrapper.observations({key: value + 5 for key, value in terminal_obs.items()})

        torch.testing.assert_close(transformed["local_obs"], torch.ones(1, 2, 2))
        torch.testing.assert_close(terminal_snapshot["local_obs"], torch.zeros(1, 2, 2))
        for key in terminal_snapshot.keys() - {"local_obs"}:
            assert transformed[key] is terminal_snapshot[key]
    finally:
        env.close()


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("has_terminal", [False, True])
def test_tensor_terminal_normalization_chain_stays_on_device_and_preserves_snapshots(
    device: str, has_terminal: bool,
) -> None:
    env = _make_env(device)
    try:
        terminal_obs, _ = env.reset()
        raw_snapshot = {key: value.clone() for key, value in terminal_obs.items()}
        infos: dict[str, Any] = {
            "final_obs": raw_snapshot, "_final_obs": torch.tensor([has_terminal], device=device),
        }
        wrappers = []
        wrapped = env
        for key in terminal_obs:
            wrapped = TorchFeatureWiseObsNormWrapper(
                wrapped, obs_key=key, scalar_feature_indices=[0], quaternion_indices=[],
                compile_tensor_operations=False,
            )
            wrappers.append(wrapped)
        for wrapper in wrappers:
            previous_obs = infos["final_obs"]
            with (
                patch.object(torch.Tensor, "__bool__", side_effect=AssertionError("terminal tensors must stay on device")),
                patch.object(torch.Tensor, "item", side_effect=AssertionError("terminal tensors must stay on device")),
                patch.object(torch.Tensor, "cpu", side_effect=AssertionError("terminal tensors must stay on device")),
            ):
                infos = wrapper._transform_infos(infos)
            for key in terminal_obs.keys() - {wrapper.obs_key}:
                assert infos["final_obs"][key] is previous_obs[key]
            assert wrapper.obs_rms is not None
            assert wrapper.obs_rms.count.item() == pytest.approx(1e-4)

        retained_obs = {key: value.clone() for key, value in infos["final_obs"].items()}
        for wrapper in wrappers:
            wrapper.observations({key: value + 20 for key, value in terminal_obs.items()})
        for key, expected in retained_obs.items():
            torch.testing.assert_close(infos["final_obs"][key], expected)
            torch.testing.assert_close(raw_snapshot[key], terminal_obs[key])
    finally:
        env.close()


def test_object_terminal_entries_transform_only_selected_lanes() -> None:
    env = _make_env()
    try:
        wrapper = _ReusableOutputWrapper(env)
        terminal_obs, _ = env.reset()
        entry = {key: value[0].numpy().copy() for key, value in terminal_obs.items()}
        infos = {"final_obs": np.array([entry], dtype=object), "_final_obs": np.array([True])}
        transformed = wrapper._transform_infos(infos)
        wrapper.observations({key: value + 5 for key, value in terminal_obs.items()})
        torch.testing.assert_close(transformed["final_obs"][0]["local_obs"], torch.ones(2, 2))
        np.testing.assert_array_equal(entry["local_obs"], np.zeros((2, 2)))

        absent = {"final_obs": np.array([None], dtype=object), "_final_obs": np.array([False])}
        assert wrapper._transform_infos(absent)["final_obs"][0] is None
    finally:
        env.close()
