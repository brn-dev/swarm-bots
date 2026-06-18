import unittest

from swarmbots.learn.swarmbots_obs_indices import build_obs_indices


def _env_settings(
    *,
    unit_types: list[str],
    include_connectors_xpos_in_obs: bool,
    include_connectors_xquat_in_obs: bool,
    quat_rot6d_representation: bool,
    scenario_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    scenario: dict[str, object] = {
        "swarm": {
            "config": {
                "unit_config": [{"type": unit_type} for unit_type in unit_types],
            },
        },
        "include_connectors_xpos_in_obs": include_connectors_xpos_in_obs,
        "include_connectors_xquat_in_obs": include_connectors_xquat_in_obs,
        "quat_rot6d_representation": quat_rot6d_representation,
    }
    if scenario_overrides is not None:
        scenario.update(scenario_overrides)
    return {"scenario": scenario}


class ObsIndicesTests(unittest.TestCase):
    def test_quaternion_local_layout_indexes_all_normalized_feature_groups(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["LimbType.xy", "LimbType.zx", "LimbType.xyz"],
                include_connectors_xpos_in_obs=True,
                include_connectors_xquat_in_obs=True,
                quat_rot6d_representation=False,
            ),
            local_obs_dim=70,
            global_obs_dim=0,
            hidden_local_vars_dim=4,
            hidden_global_vars_dim=2,
        )

        self.assertEqual(
            obs_indices.local_scalar_indices,
            [0, 1, 2, *range(21, 34), *range(49, 58), 38, 43, 48],
        )
        self.assertEqual(obs_indices.local_angle_indices, [7, 9, 11, 13, 15, 17, 19, 36, 41, 46])
        self.assertEqual(obs_indices.local_quaternion_indices, [3, 58, 62, 66])
        self.assertEqual(obs_indices.local_rot6d_indices, [])
        self.assertEqual(obs_indices.local_binary_indices, [34, 39, 44])
        self.assertEqual(obs_indices.hidden_local_vars_scalar_indices, [])
        self.assertEqual(obs_indices.hidden_local_vars_quaternion_indices, [])
        self.assertEqual(obs_indices.hidden_global_vars_scalar_indices, [0, 1])

    def test_rot6d_local_layout_keeps_rotations_out_of_scalar_normalization(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["xy", "xyz"],
                include_connectors_xpos_in_obs=False,
                include_connectors_xquat_in_obs=False,
                quat_rot6d_representation=True,
            ),
            local_obs_dim=40,
            global_obs_dim=0,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=1,
        )

        self.assertEqual(obs_indices.local_scalar_indices, [0, 1, 2, *range(19, 30), 34, 39])
        self.assertEqual(obs_indices.local_angle_indices, [9, 11, 13, 15, 17, 32, 37])
        self.assertEqual(obs_indices.local_rot6d_indices, [3])
        self.assertEqual(obs_indices.local_quaternion_indices, [])
        self.assertEqual(obs_indices.local_binary_indices, [30, 35])
        self.assertEqual(obs_indices.global_scalar_indices, [])
        self.assertEqual(obs_indices.global_rot6d_indices, [])
        self.assertEqual(obs_indices.hidden_global_vars_scalar_indices, [0])

    def test_move_to_goal_global_obs_is_treated_as_scalars(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["xy"],
                include_connectors_xpos_in_obs=True,
                include_connectors_xquat_in_obs=False,
                quat_rot6d_representation=True,
                scenario_overrides={"goal": object()},
            ),
            local_obs_dim=29,
            global_obs_dim=2,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=0,
        )

        self.assertEqual(obs_indices.global_scalar_indices, [0, 1])
        self.assertEqual(obs_indices.global_rot6d_indices, [])
        self.assertEqual(obs_indices.global_quaternion_indices, [])

    def test_payload_global_obs_only_normalizes_position_not_rot6d_orientation(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["xy"],
                include_connectors_xpos_in_obs=True,
                include_connectors_xquat_in_obs=False,
                quat_rot6d_representation=True,
                scenario_overrides={"payload_shape": "box"},
            ),
            local_obs_dim=29,
            global_obs_dim=9,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=0,
        )

        self.assertEqual(obs_indices.global_scalar_indices, [0, 1, 2])
        self.assertEqual(obs_indices.global_rot6d_indices, [3])
        self.assertEqual(obs_indices.global_quaternion_indices, [])

    def test_move_to_dual_payload_adapter_normalizes_both_payload_positions(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["xy"],
                include_connectors_xpos_in_obs=True,
                include_connectors_xquat_in_obs=False,
                quat_rot6d_representation=True,
                scenario_overrides={"global_obs_adapter": "move_to_dual_payload", "num_payloads": 2},
            ),
            local_obs_dim=29,
            global_obs_dim=18,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=0,
        )

        self.assertEqual(obs_indices.global_scalar_indices, [0, 1, 2, 9, 10, 11])
        self.assertEqual(obs_indices.global_rot6d_indices, [])
        self.assertEqual(obs_indices.global_quaternion_indices, [])

    def test_move_to_payload_adapter_normalizes_payload_position_only(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["xy"],
                include_connectors_xpos_in_obs=True,
                include_connectors_xquat_in_obs=False,
                quat_rot6d_representation=True,
                scenario_overrides={"global_obs_adapter": "move_to_payload"},
            ),
            local_obs_dim=29,
            global_obs_dim=9,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=0,
        )

        self.assertEqual(obs_indices.global_scalar_indices, [0, 1, 2])
        self.assertEqual(obs_indices.global_rot6d_indices, [])
        self.assertEqual(obs_indices.global_quaternion_indices, [])

    def test_climb_goal_global_obs_is_xyz_scalars(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["xy"],
                include_connectors_xpos_in_obs=True,
                include_connectors_xquat_in_obs=False,
                quat_rot6d_representation=True,
                scenario_overrides={"global_obs_layout": "climb_goal_xyz"},
            ),
            local_obs_dim=29,
            global_obs_dim=3,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=0,
        )

        self.assertEqual(obs_indices.global_scalar_indices, [0, 1, 2])
        self.assertEqual(obs_indices.global_rot6d_indices, [])
        self.assertEqual(obs_indices.global_quaternion_indices, [])

    def test_dual_payload_global_obs_normalizes_both_payload_positions(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["xy"],
                include_connectors_xpos_in_obs=True,
                include_connectors_xquat_in_obs=False,
                quat_rot6d_representation=True,
                scenario_overrides={"payload_shape": "box", "num_payloads": 2},
            ),
            local_obs_dim=29,
            global_obs_dim=18,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=0,
        )

        self.assertEqual(obs_indices.global_scalar_indices, [0, 1, 2, 9, 10, 11])
        self.assertEqual(obs_indices.global_rot6d_indices, [3, 12])
        self.assertEqual(obs_indices.global_quaternion_indices, [])

    def test_multi_payload_goal_global_obs_indexes_payload_records(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["xy"],
                include_connectors_xpos_in_obs=True,
                include_connectors_xquat_in_obs=False,
                quat_rot6d_representation=True,
                scenario_overrides={
                    "payload_shape": "box",
                    "global_obs_layout": "multi_payload_goal",
                    "num_payloads": 3,
                },
            ),
            local_obs_dim=29,
            global_obs_dim=36,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=0,
        )

        self.assertEqual(
            obs_indices.global_scalar_indices,
            [1, 2, 3, 10, 11, 13, 14, 15, 22, 23, 25, 26, 27, 34, 35],
        )
        self.assertEqual(obs_indices.global_rot6d_indices, [4, 16, 28])
        self.assertEqual(obs_indices.global_quaternion_indices, [])

    def test_hidden_dual_payload_pose_only_normalizes_positions(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["xy"],
                include_connectors_xpos_in_obs=True,
                include_connectors_xquat_in_obs=False,
                quat_rot6d_representation=True,
                scenario_overrides={
                    "payload_shape": "box",
                    "num_payloads": 2,
                    "payload_pos_observable": False,
                },
            ),
            local_obs_dim=29,
            global_obs_dim=0,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=18,
        )

        self.assertEqual(obs_indices.global_scalar_indices, [])
        self.assertEqual(obs_indices.global_rot6d_indices, [])
        self.assertEqual(obs_indices.hidden_global_vars_scalar_indices, [0, 1, 2, 9, 10, 11])
        self.assertEqual(obs_indices.hidden_global_vars_quaternion_indices, [])

    def test_find_opening_hidden_globals_only_normalize_opening_x(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["xy"],
                include_connectors_xpos_in_obs=True,
                include_connectors_xquat_in_obs=False,
                quat_rot6d_representation=True,
                scenario_overrides={
                    "scenario_type": "find_opening",
                    "opening_x": 0.0,
                    "wall_exploration_cell_count": 10,
                },
            ),
            local_obs_dim=29,
            global_obs_dim=0,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=11,
        )

        self.assertEqual(obs_indices.hidden_global_vars_scalar_indices, [0])
        self.assertEqual(obs_indices.hidden_global_vars_quaternion_indices, [])

    def test_find_opening_legacy_settings_keep_visited_cells_unnormalized(self) -> None:
        obs_indices = build_obs_indices(
            env_settings=_env_settings(
                unit_types=["xy"],
                include_connectors_xpos_in_obs=True,
                include_connectors_xquat_in_obs=False,
                quat_rot6d_representation=True,
                scenario_overrides={
                    "opening_x": 0.0,
                    "wall_exploration_cell_count": 4,
                },
            ),
            local_obs_dim=29,
            global_obs_dim=0,
            hidden_local_vars_dim=0,
            hidden_global_vars_dim=5,
        )

        self.assertEqual(obs_indices.hidden_global_vars_scalar_indices, [0])
        self.assertEqual(obs_indices.hidden_global_vars_quaternion_indices, [])

    def test_unknown_non_empty_global_obs_layout_must_be_added_explicitly(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported non-empty global_obs layout"):
            build_obs_indices(
                env_settings=_env_settings(
                    unit_types=["xy"],
                    include_connectors_xpos_in_obs=True,
                    include_connectors_xquat_in_obs=False,
                    quat_rot6d_representation=True,
                ),
                local_obs_dim=29,
                global_obs_dim=3,
                hidden_local_vars_dim=0,
                hidden_global_vars_dim=0,
            )

    def test_local_obs_dim_mismatch_raises_with_expected_layout_context(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unexpected local_obs_dim"):
            build_obs_indices(
                env_settings=_env_settings(
                    unit_types=["xy"],
                    include_connectors_xpos_in_obs=True,
                    include_connectors_xquat_in_obs=False,
                    quat_rot6d_representation=True,
                ),
                local_obs_dim=32,
                global_obs_dim=0,
                hidden_local_vars_dim=0,
                hidden_global_vars_dim=0,
            )


if __name__ == "__main__":
    unittest.main()
