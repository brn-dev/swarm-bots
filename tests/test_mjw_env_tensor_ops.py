import math
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import torch

import swarmbots.mjw_env.mjw_env_tensor_ops as mjw_env_tensor_ops
from swarmbots.mjw_env.mjw_env_tensor_ops import (
    MJWActionLayout,
    MJWEnvTensorOperations,
    MJWObservationLayout,
    build_mjw_env_tensor_operations,
    should_compile_mjw_env_tensor_operations_by_default,
)
from swarmbots.mjw_env.mjw_swarm_bots_vector_env import MJWSwarmBotsVectorEnv


_REAL_TORCH_COMPILE = torch.compile


def _compile_with_aot_eager(
    function: Callable[..., Any],
    **kwargs: Any,
) -> Callable[..., Any]:
    return _REAL_TORCH_COMPILE(
        function,
        backend="aot_eager",
        fullgraph=kwargs["fullgraph"],
        dynamic=kwargs["dynamic"],
    )


def _observation_layout(
    *,
    use_rot6d: bool,
    include_connector_positions: bool,
) -> MJWObservationLayout:
    rotation_dim = 6 if use_rot6d else 4
    free_joint_rotation = slice(3, 3 + rotation_dim)
    hinge = slice(free_joint_rotation.stop, free_joint_rotation.stop + 4)
    qvel = slice(hinge.stop, hinge.stop + 8)
    connector = slice(qvel.stop, qvel.stop + 10)
    connector_position = slice(connector.stop, connector.stop + 6)
    return MJWObservationLayout(
        num_envs=2,
        num_agents=2,
        num_connectors=2,
        free_joint_position=slice(0, 3),
        free_joint_rotation=free_joint_rotation,
        hinge=hinge,
        qvel=qvel,
        connector=connector,
        connector_position=connector_position,
        use_rot6d=use_rot6d,
        include_connector_positions=include_connector_positions,
    )


def _action_layout(*, continuous_connectors: bool) -> MJWActionLayout:
    return MJWActionLayout(
        num_envs=2,
        continuous_connectors=continuous_connectors,
    )


def _build_operations(
    *,
    use_rot6d: bool = False,
    include_connector_positions: bool = False,
    continuous_connectors: bool = False,
    compile_operations: bool = False,
) -> MJWEnvTensorOperations:
    return build_mjw_env_tensor_operations(
        observation_layout=_observation_layout(
            use_rot6d=use_rot6d,
            include_connector_positions=include_connector_positions,
        ),
        action_layout=_action_layout(continuous_connectors=continuous_connectors),
        compile_operations=compile_operations,
        compile_mode="default",
    )


def _observation_inputs(layout: MJWObservationLayout) -> tuple[torch.Tensor, ...]:
    local_obs_dim = (
        layout.connector_position.stop
        if layout.include_connector_positions
        else layout.connector.stop
    )
    local_obs = torch.full((2, 2, local_obs_dim), -99.0)

    identity_quat = [1.0, 0.0, 0.0, 0.0]
    z_90_quat = [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]
    unit_qpos = torch.tensor([
        [1.0, 2.0, 3.0, *identity_quat, 0.0, math.pi / 2],
        [4.0, 5.0, 6.0, *z_90_quat, math.pi, -math.pi / 2],
        [7.0, 8.0, 9.0, *identity_quat, math.pi / 4, -math.pi / 4],
        [10.0, 11.0, 12.0, *z_90_quat, math.pi / 6, math.pi / 3],
    ]).reshape(2, 18)
    qpos_padding = torch.full((2, 2), 123.0)
    qpos = torch.cat((qpos_padding[:, :1], unit_qpos, qpos_padding[:, 1:]), dim=1)
    qpos_flat_indices = torch.arange(1, 19, dtype=torch.long)

    unit_qvel = torch.arange(32, dtype=torch.float32).reshape(2, 16) / 10.0
    qvel = torch.cat((torch.full((2, 2), -123.0), unit_qvel), dim=1)
    qvel_flat_indices = torch.arange(2, 18, dtype=torch.long)

    partner_unit = torch.tensor([
        [[1, -1], [0, 0]],
        [[-1, 1], [0, -1]],
    ])
    connection_twist_idx = torch.tensor([
        [[0, -1], [-1, 1]],
        [[-1, 2], [2, -1]],
    ])
    disconnect_potentials = torch.tensor([
        [[0.25, 0.0], [0.5, 0.75]],
        [[0.0, 1.0], [1.25, 0.0]],
    ])
    twist_values = torch.tensor([0.0, math.pi / 2, math.pi])

    xpos = torch.arange(2 * 7 * 3, dtype=torch.float32).reshape(2, 7, 3)
    connector_body_indices = torch.tensor([1, 3, 4, 6], dtype=torch.long)
    return (
        local_obs,
        qpos,
        qvel,
        partner_unit,
        connection_twist_idx,
        disconnect_potentials,
        twist_values,
        xpos,
        qpos_flat_indices,
        qvel_flat_indices,
        connector_body_indices,
    )


def _clone_inputs(inputs: tuple[torch.Tensor, ...]) -> tuple[torch.Tensor, ...]:
    return tuple(value.clone() for value in inputs)


class MJWEnvTensorOperationsTests(unittest.TestCase):
    def test_build_local_obs_populates_every_quaternion_layout_field(self) -> None:
        layout = _observation_layout(use_rot6d=False, include_connector_positions=True)
        operations = _build_operations(include_connector_positions=True)
        inputs = _observation_inputs(layout)
        local_obs = inputs[0]

        result = operations.build_local_obs(*inputs)

        self.assertEqual(result.data_ptr(), local_obs.data_ptr())
        torch.testing.assert_close(
            result[:, :, layout.free_joint_position],
            torch.tensor([
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
                [[7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
            ]),
        )
        expected_quats = torch.tensor([
            [[1.0, 0.0, 0.0, 0.0], [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]],
            [[1.0, 0.0, 0.0, 0.0], [math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5)]],
        ])
        torch.testing.assert_close(result[:, :, layout.free_joint_rotation], expected_quats)

        expected_hinges = torch.tensor([
            [[[0.0, 1.0], [1.0, 0.0]], [[0.0, -1.0], [-1.0, 0.0]]],
            [[[math.sqrt(0.5), math.sqrt(0.5)], [-math.sqrt(0.5), math.sqrt(0.5)]],
             [[0.5, math.sqrt(0.75)], [math.sqrt(0.75), 0.5]]],
        ])
        torch.testing.assert_close(
            result[:, :, layout.hinge].view(2, 2, 2, 2),
            expected_hinges,
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            result[:, :, layout.qvel],
            inputs[2][:, inputs[9]].reshape(2, 2, 8),
        )
        torch.testing.assert_close(
            result[:, :, layout.connector_position],
            inputs[7][:, inputs[10]].reshape(2, 2, 6),
        )

    def test_build_local_obs_populates_rot6d_and_connector_state_semantics(self) -> None:
        layout = _observation_layout(use_rot6d=True, include_connector_positions=False)
        operations = _build_operations(use_rot6d=True)
        inputs = _observation_inputs(layout)

        result = operations.build_local_obs(*inputs)

        expected_rot6d = torch.tensor([
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]],
            [[1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, -1.0, 0.0, 0.0]],
        ])
        torch.testing.assert_close(
            result[:, :, layout.free_joint_rotation],
            expected_rot6d,
            atol=1e-6,
            rtol=1e-6,
        )
        connector_obs = result[:, :, layout.connector].view(2, 2, 2, 5)
        expected_active = inputs[3] >= 0
        torch.testing.assert_close(connector_obs[..., 0], (~expected_active).float())
        torch.testing.assert_close(connector_obs[..., 1], expected_active.float())
        valid_twist = expected_active & (inputs[4] >= 0)
        twist_values = inputs[6][inputs[4].clamp_min(0)]
        torch.testing.assert_close(
            connector_obs[..., 2],
            torch.sin(twist_values) * valid_twist.float(),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(
            connector_obs[..., 3],
            torch.cos(twist_values) * valid_twist.float(),
            atol=1e-6,
            rtol=1e-6,
        )
        torch.testing.assert_close(connector_obs[..., 4], inputs[5])

    def test_build_local_obs_does_not_read_connector_positions_when_disabled(self) -> None:
        layout = _observation_layout(use_rot6d=False, include_connector_positions=False)
        operations = _build_operations()
        inputs = list(_observation_inputs(layout))
        inputs[7] = torch.full_like(inputs[7], float("nan"))

        result = operations.build_local_obs(*inputs)

        self.assertTrue(torch.isfinite(result).all())
        self.assertEqual(result.shape[-1], layout.connector.stop)

    def test_prepare_binary_actions_masks_inactive_agents_and_writes_scaled_controls(self) -> None:
        operations = _build_operations()
        actuators = torch.tensor([
            [[1.0, -1.0], [0.5, 0.25]],
            [[-0.5, 0.75], [1.0, 1.0]],
        ])
        connectors = torch.tensor([
            [[True, False], [True, True]],
            [[False, True], [True, False]],
        ])
        active_mask = torch.tensor([[True, False], [False, True]])
        ctrl = torch.full((2, 6), -7.0)
        ctrl_indices = torch.tensor([1, 2, 4, 5])

        connector_action = operations.prepare_actions(
            actuators,
            connectors,
            active_mask,
            ctrl,
            ctrl_indices,
            2.5,
        )

        torch.testing.assert_close(
            ctrl[:, ctrl_indices],
            torch.tensor([[2.5, -2.5, 0.0, 0.0], [0.0, 0.0, 2.5, 2.5]]),
        )
        torch.testing.assert_close(ctrl[:, [0, 3]], torch.full((2, 2), -7.0))
        torch.testing.assert_close(
            connector_action,
            torch.tensor([
                [[True, False], [False, False]],
                [[False, False], [True, False]],
            ]),
        )

    def test_prepare_continuous_actions_uses_disconnect_sentinel_for_inactive_agents(self) -> None:
        operations = _build_operations(continuous_connectors=True)
        actuators = torch.empty((2, 2, 0))
        connectors = torch.tensor([
            [[0.5, -0.25], [1.0, 0.0]],
            [[-0.5, 0.75], [0.25, -1.0]],
        ])
        active_mask = torch.tensor([[True, False], [False, True]])
        ctrl = torch.empty((2, 0))

        connector_action = operations.prepare_actions(
            actuators,
            connectors,
            active_mask,
            ctrl,
            torch.empty((0,), dtype=torch.long),
            2.5,
        )

        torch.testing.assert_close(
            connector_action,
            torch.tensor([
                [[0.5, -0.25], [-1.0, -1.0]],
                [[-1.0, -1.0], [0.25, -1.0]],
            ]),
        )

    def test_prepare_actions_uses_the_current_actuator_strength(self) -> None:
        operations = _build_operations()
        actuators = torch.ones((2, 2, 2))
        connectors = torch.zeros((2, 2, 2), dtype=torch.bool)
        active_mask = torch.ones((2, 2), dtype=torch.bool)
        ctrl = torch.zeros((2, 4))
        ctrl_indices = torch.arange(4)

        operations.prepare_actions(
            actuators,
            connectors,
            active_mask,
            ctrl,
            ctrl_indices,
            3.0,
        )

        torch.testing.assert_close(ctrl, torch.full((2, 4), 3.0))

    def test_mask_error_observations_zeros_only_unstable_rows_in_place(self) -> None:
        operations = _build_operations()
        local_obs = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        global_obs = torch.arange(6, dtype=torch.float32).reshape(2, 3)
        hidden_local_obs = torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)
        hidden_global_obs = torch.empty((2, 0))
        stable_copies = tuple(value[0].clone() for value in (
            local_obs,
            global_obs,
            hidden_local_obs,
            hidden_global_obs,
        ))

        outputs = operations.mask_error_observations(
            local_obs,
            global_obs,
            hidden_local_obs,
            hidden_global_obs,
            torch.tensor([False, True]),
        )

        for output, original, stable_copy in zip(
            outputs,
            (local_obs, global_obs, hidden_local_obs, hidden_global_obs),
            stable_copies,
            strict=True,
        ):
            self.assertEqual(output.data_ptr(), original.data_ptr())
            torch.testing.assert_close(output[0], stable_copy)
            self.assertTrue((output[1] == 0).all())

    def test_compiled_operations_match_eager_for_every_static_layout_variant(self) -> None:
        torch.manual_seed(123)
        for use_rot6d in (False, True):
            for include_connector_positions in (False, True):
                for continuous_connectors in (False, True):
                    with self.subTest(
                        use_rot6d=use_rot6d,
                        include_connector_positions=include_connector_positions,
                        continuous_connectors=continuous_connectors,
                    ):
                        with patch.object(
                            mjw_env_tensor_ops.torch,
                            "compile",
                            side_effect=_compile_with_aot_eager,
                        ):
                            compiled = _build_operations(
                                use_rot6d=use_rot6d,
                                include_connector_positions=include_connector_positions,
                                continuous_connectors=continuous_connectors,
                                compile_operations=True,
                            )
                        eager = _build_operations(
                            use_rot6d=use_rot6d,
                            include_connector_positions=include_connector_positions,
                            continuous_connectors=continuous_connectors,
                        )
                        layout = _observation_layout(
                            use_rot6d=use_rot6d,
                            include_connector_positions=include_connector_positions,
                        )
                        eager_inputs = _observation_inputs(layout)
                        compiled_inputs = _clone_inputs(eager_inputs)
                        torch.testing.assert_close(
                            compiled.build_local_obs(*compiled_inputs),
                            eager.build_local_obs(*eager_inputs),
                        )

                        actuators = torch.randn(2, 2, 2)
                        connectors = (
                            torch.randn(2, 2, 2)
                            if continuous_connectors
                            else torch.randint(0, 2, (2, 2, 2), dtype=torch.bool)
                        )
                        active_mask = torch.tensor([[True, False], [True, True]])
                        eager_ctrl = torch.zeros(2, 4)
                        compiled_ctrl = eager_ctrl.clone()
                        ctrl_indices = torch.arange(4)
                        eager_connector_actions = eager.prepare_actions(
                            actuators,
                            connectors,
                            active_mask,
                            eager_ctrl,
                            ctrl_indices,
                            2.5,
                        )
                        compiled_connector_actions = compiled.prepare_actions(
                            actuators,
                            connectors,
                            active_mask,
                            compiled_ctrl,
                            ctrl_indices,
                            2.5,
                        )
                        torch.testing.assert_close(compiled_ctrl, eager_ctrl)
                        torch.testing.assert_close(compiled_connector_actions, eager_connector_actions)

                        eager_obs = [torch.randn(2, 3, 4), torch.randn(2, 2), torch.randn(2, 3, 1), torch.empty(2, 0)]
                        compiled_obs = [value.clone() for value in eager_obs]
                        unstable_mask = torch.tensor([True, False])
                        eager_masked = eager.mask_error_observations(*eager_obs, unstable_mask)
                        compiled_masked = compiled.mask_error_observations(*compiled_obs, unstable_mask)
                        for eager_value, compiled_value in zip(eager_masked, compiled_masked, strict=True):
                            torch.testing.assert_close(compiled_value, eager_value)

    def test_compile_builder_uses_static_full_graphs(self) -> None:
        with patch.object(
            mjw_env_tensor_ops.torch,
            "compile",
            side_effect=lambda function, **_kwargs: function,
        ) as compile_mock:
            build_mjw_env_tensor_operations(
                observation_layout=_observation_layout(
                    use_rot6d=False,
                    include_connector_positions=False,
                ),
                action_layout=_action_layout(continuous_connectors=False),
                compile_operations=True,
                compile_mode="reduce-overhead",
            )

        self.assertEqual(compile_mock.call_count, 3)
        self.assertEqual(
            [call.args[0].__name__ for call in compile_mock.call_args_list],
            ["build_local_obs", "prepare_actions", "_mask_error_observations"],
        )
        for compile_call in compile_mock.call_args_list:
            self.assertEqual(compile_call.kwargs, {
                "mode": "reduce-overhead",
                "fullgraph": True,
                "dynamic": False,
            })

    def test_compile_configuration_is_validated(self) -> None:
        kwargs = {
            "observation_layout": _observation_layout(
                use_rot6d=False,
                include_connector_positions=False,
            ),
            "action_layout": _action_layout(continuous_connectors=False),
            "compile_operations": True,
        }
        with patch.object(mjw_env_tensor_ops.torch, "compile", None):
            with self.assertRaisesRegex(RuntimeError, "torch.compile support"):
                build_mjw_env_tensor_operations(**kwargs, compile_mode="default")
        with self.assertRaisesRegex(ValueError, "compile_mode"):
            build_mjw_env_tensor_operations(**kwargs, compile_mode="")

    def test_default_compilation_is_enabled_only_for_cuda(self) -> None:
        self.assertFalse(should_compile_mjw_env_tensor_operations_by_default(torch.device("cpu")))
        self.assertTrue(should_compile_mjw_env_tensor_operations_by_default(torch.device("cuda")))

    def test_environment_methods_delegate_to_tensor_operations_without_compiling_dict_assembly(self) -> None:
        layout = _observation_layout(use_rot6d=False, include_connector_positions=False)
        operations = _build_operations()
        inputs = _observation_inputs(layout)
        global_obs = torch.randn(2, 3)
        hidden_local_obs = torch.randn(2, 2, 1)
        hidden_global_obs = torch.randn(2, 2)
        units_active_mask = torch.tensor([[True, False], [True, True]])
        runtime = SimpleNamespace(
            global_obs=global_obs,
            hidden_local_obs=hidden_local_obs,
            hidden_global_obs=hidden_global_obs,
        )
        env = SimpleNamespace(
            _tensor_operations=operations,
            _local_obs=inputs[0],
            _qpos=inputs[1],
            _qvel=inputs[2],
            partner_unit=inputs[3],
            connection_twist_idx=inputs[4],
            disconnect_potentials=inputs[5],
            _twist_values=inputs[6],
            _xpos=inputs[7],
            _qpos_flat_indices=inputs[8],
            _qvel_flat_indices=inputs[9],
            _connector_body_indices=inputs[10],
            _scenario_runtime=runtime,
            units_active_mask=units_active_mask,
        )

        obs = MJWSwarmBotsVectorEnv._build_obs(env)

        self.assertEqual(obs["local_obs"].data_ptr(), inputs[0].data_ptr())
        self.assertEqual(obs["global_obs"].data_ptr(), global_obs.data_ptr())
        self.assertEqual(obs["hidden_local_vars"].data_ptr(), hidden_local_obs.data_ptr())
        self.assertEqual(obs["hidden_global_vars"].data_ptr(), hidden_global_obs.data_ptr())
        self.assertEqual(obs["agent_mask"].data_ptr(), units_active_mask.data_ptr())

        masked_obs = MJWSwarmBotsVectorEnv._apply_error_obs(
            env,
            obs,
            torch.tensor([False, True]),
        )
        for key in ("local_obs", "global_obs", "hidden_local_vars", "hidden_global_vars"):
            self.assertTrue((masked_obs[key][1] == 0).all())
        torch.testing.assert_close(masked_obs["agent_mask"], units_active_mask)


if __name__ == "__main__":
    unittest.main()
