import unittest
from types import SimpleNamespace

import numpy as np
import torch
from gymnasium import spaces
from gymnasium.vector import AutoresetMode, SyncVectorEnv

from swarmbots.learn.env_wrappers.learn_wrappers.swarm_bots_learn_env_wrapper import SwarmBotsLearnEnvWrapper
from swarmbots.learn.testing_env import TestingSwarmBotsEnv
from swarmbots.mj_env.scenarios.scenario_presets import default_move_to as default_mj_move_to
from swarmbots.mj_env.swarm.swarm_config import SwarmConfig
from swarmbots.mj_env.swarm.swarm_connections import SwarmConnections
from swarmbots.mj_env.swarm.unit_config import UNIT_CONFIG_TETRAHEDRON_ZX
from swarmbots.mjw_env.mjw_env_tensor_ops import (
    MJWActionLayout,
    MJWEnvTensorOperations,
    MJWObservationLayout,
    build_mjw_env_tensor_operations,
)
from swarmbots.mjw_env.scenarios.mjw_scenario_presets import default_move_to as default_mjw_move_to
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv


def _make_connections() -> SwarmConnections:
    config = SwarmConfig(
        num_units=2,
        unit_config=UNIT_CONFIG_TETRAHEDRON_ZX[:1],
        connection_torquescale=1.0,
    )
    connections = SwarmConnections(config)
    connections.connect(0, 0, 1, 0, twist_angle=0.0)
    return connections


def _make_action_tensor_operations(*, continuous_connectors: bool) -> MJWEnvTensorOperations:
    return build_mjw_env_tensor_operations(
        observation_layout=MJWObservationLayout(
            num_envs=1,
            num_agents=2,
            num_connectors=1,
            free_joint_position=slice(0, 3),
            free_joint_rotation=slice(3, 7),
            hinge=slice(7, 9),
            qvel=slice(9, 17),
            connector=slice(17, 22),
            connector_position=slice(22, 25),
            use_rot6d=False,
            include_connector_positions=False,
        ),
        action_layout=MJWActionLayout(
            num_envs=1,
            continuous_connectors=continuous_connectors,
        ),
        compile_operations=False,
        compile_mode="default",
    )


class ContinuousConnectorActionTests(unittest.TestCase):
    def test_numpy_continuous_connector_updates_match_binary_extremes(self) -> None:
        cases = [
            (np.array([[1.0], [1.0]], dtype=np.float32), 3.0, 1.0),
            (np.array([[-1.0], [1.0]], dtype=np.float32), 3.0, 4.0),
            (np.array([[-1.0], [-1.0]], dtype=np.float32), 3.0, 5.0),
            (np.array([[0.5], [1.0]], dtype=np.float32), 3.0, 2.0),
            (np.array([[-0.25], [1.0]], dtype=np.float32), 3.0, 3.25),
        ]

        for actions, initial_potential, expected_potential in cases:
            with self.subTest(actions=actions.tolist()):
                connections = _make_connections()
                connections.disconnect_potentials[:] = initial_potential

                deactivation_mask = connections.update_disconnect_potentials_continuous(
                    connections.get_is_active_mask(),
                    actions,
                    disconnect_potential_threshold=10.0,
                )

                np.testing.assert_allclose(
                    connections.disconnect_potentials,
                    np.full((2, 1), expected_potential, dtype=np.float32),
                )
                self.assertFalse(deactivation_mask.any())

    def test_numpy_continuous_connector_disconnects_at_threshold(self) -> None:
        connections = _make_connections()

        deactivation_mask = connections.update_disconnect_potentials_continuous(
            connections.get_is_active_mask(),
            np.array([[-1.0], [-1.0]], dtype=np.float32),
            disconnect_potential_threshold=2.0,
        )

        self.assertTrue(bool(deactivation_mask[0, 0]))
        self.assertTrue(bool(deactivation_mask[1, 0]))

    def test_learn_wrapper_preserves_continuous_connector_actions(self) -> None:
        vector_env = SyncVectorEnv(
            [
                lambda: TestingSwarmBotsEnv(
                    n_agents=2,
                    n_local_obs=3,
                    n_global_obs=2,
                    actuators_dim=1,
                    connectors_dim=1,
                    continuous_connector_actions=True,
                )
            ],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        env = SwarmBotsLearnEnvWrapper(vector_env)

        self.assertIsInstance(env.action_space["connectors"], spaces.Box)
        action_dict = env._actions_to_env_dict(torch.tensor([[[0.25, -0.75], [0.5, 0.125]]]))

        self.assertEqual(action_dict["connectors"].dtype, np.float32)
        np.testing.assert_allclose(action_dict["connectors"], np.array([[[-0.75], [0.125]]], dtype=np.float32))

    def test_learn_wrapper_default_connector_actions_stay_binary(self) -> None:
        vector_env = SyncVectorEnv(
            [
                lambda: TestingSwarmBotsEnv(
                    n_agents=2,
                    n_local_obs=3,
                    n_global_obs=2,
                    actuators_dim=1,
                    connectors_dim=1,
                )
            ],
            autoreset_mode=AutoresetMode.SAME_STEP,
        )
        env = SwarmBotsLearnEnvWrapper(vector_env)

        self.assertIsInstance(env.action_space["connectors"], spaces.MultiBinary)
        action_dict = env._actions_to_env_dict(torch.tensor([[[0.25, 0.5], [0.5, 0.75]]]))

        self.assertEqual(action_dict["connectors"].dtype, np.bool_)
        np.testing.assert_array_equal(
            action_dict["connectors"],
            np.array([[[False], [True]]], dtype=np.bool_),
        )

    def test_scenario_presets_expose_continuous_connector_action_spaces(self) -> None:
        mj_scenario = default_mj_move_to(
            reset_settle_time=0.0,
            continuous_connector_actions=True,
        )
        mjw_scenario = default_mjw_move_to(continuous_connector_actions=True)

        for connector_space in (
            mj_scenario.get_action_space()["connectors"],
            mjw_scenario.get_single_action_space()["connectors"],
        ):
            self.assertIsInstance(connector_space, spaces.Box)
            self.assertEqual(connector_space.dtype, np.float32)
            np.testing.assert_allclose(connector_space.low, -1.0)
            np.testing.assert_allclose(connector_space.high, 1.0)

        self.assertTrue(mj_scenario.get_settings()["continuous_connector_actions"])
        self.assertTrue(mjw_scenario.get_settings()["continuous_connector_actions"])

    def test_scenario_presets_default_connector_action_spaces_stay_binary(self) -> None:
        mj_scenario = default_mj_move_to(reset_settle_time=0.0)
        mjw_scenario = default_mjw_move_to()

        for connector_space in (
            mj_scenario.get_action_space()["connectors"],
            mjw_scenario.get_single_action_space()["connectors"],
        ):
            self.assertIsInstance(connector_space, spaces.MultiBinary)

        self.assertFalse(mj_scenario.get_settings()["continuous_connector_actions"])
        self.assertFalse(mjw_scenario.get_settings()["continuous_connector_actions"])

    def test_mjw_default_apply_actions_uses_binary_connector_path(self) -> None:
        calls: list[tuple[str, torch.Tensor]] = []
        env = SimpleNamespace()
        env.num_envs = 1
        env.units_active_mask = torch.tensor([[True, False]])
        env._ctrl = torch.empty((1, 0), dtype=torch.float32)
        env._ctrl_flat_indices = torch.empty((0,), dtype=torch.long)
        env.scenario = SimpleNamespace(actuator_strength=1.0)
        env._continuous_connector_actions = False
        env._tensor_operations = _make_action_tensor_operations(continuous_connectors=False)
        env._try_connect = lambda connector_action: calls.append(("connect", connector_action.clone()))
        env._disconnect = lambda connector_action: calls.append(("disconnect", connector_action.clone()))
        env._disconnect_continuous = lambda connector_action: calls.append(("continuous", connector_action.clone()))

        MJWSwarmBotsVectorEnv._apply_actions(
            env,
            actuators=torch.empty((1, 2, 0), dtype=torch.float32),
            connectors=torch.tensor([[[True], [True]]]),
        )

        self.assertEqual([name for name, _action in calls], ["connect", "disconnect"])
        for _name, connector_action in calls:
            self.assertEqual(connector_action.dtype, torch.bool)
            torch.testing.assert_close(connector_action, torch.tensor([[[True], [False]]]))

    def test_mjw_continuous_disconnect_updates_potential_and_disconnects(self) -> None:
        env = SimpleNamespace()
        env.num_envs = 1
        env._n_agents = 2
        env._n_connectors = 1
        env._n_total_connectors = 2
        env.scenario = SimpleNamespace(disconnect_potential_threshold=2.0)
        env.partner_unit = torch.tensor([[[1], [0]]], dtype=torch.long)
        env.partner_connector = torch.tensor([[[0], [0]]], dtype=torch.long)
        env.connection_twist_idx = torch.zeros_like(env.partner_unit)
        env.disconnect_potentials = torch.zeros((1, 2, 1), dtype=torch.float32)
        env._disconnect_update = torch.empty_like(env.disconnect_potentials)
        env._unit_indices = torch.arange(2, dtype=torch.long).view(1, 2, 1)
        env._connector_indices = torch.zeros((1, 1, 1), dtype=torch.long)
        env._eq_indices = torch.zeros((2, 1, 2, 1, 1), dtype=torch.long)
        env._eq_active = torch.ones((1, 1), dtype=torch.int32)

        MJWSwarmBotsVectorEnv._disconnect_continuous(env, torch.tensor([[[-0.25], [1.0]]], dtype=torch.float32))
        torch.testing.assert_close(env.disconnect_potentials, torch.full((1, 2, 1), 0.25))
        self.assertEqual(int(env._eq_active[0, 0].item()), 1)

        MJWSwarmBotsVectorEnv._disconnect_continuous(env, torch.tensor([[[-1.0], [-1.0]]], dtype=torch.float32))
        self.assertEqual(int(env._eq_active[0, 0].item()), 0)
        self.assertTrue(torch.equal(env.partner_unit, torch.full_like(env.partner_unit, -1)))
        torch.testing.assert_close(env.disconnect_potentials, torch.zeros_like(env.disconnect_potentials))


if __name__ == "__main__":
    unittest.main()
