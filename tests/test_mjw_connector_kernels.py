import torch
import warp as wp

from swarmbots.mjw_env.mjw_kernels import (
    apply_connection_candidates,
    update_binary_connector_disconnections,
    update_continuous_connector_disconnections,
)


def _warp(tensor: torch.Tensor) -> wp.array:
    return wp.from_torch(tensor)


def _connection_state() -> tuple[torch.Tensor, ...]:
    partner_unit = torch.tensor([[[1], [0]]], dtype=torch.int64)
    partner_connector = torch.zeros_like(partner_unit)
    connection_twist_idx = torch.zeros_like(partner_unit)
    disconnect_potentials = torch.zeros((1, 2, 1), dtype=torch.float32)
    eq_active = torch.ones((1, 4), dtype=torch.bool)
    eq_indices = torch.tensor([0, 1, 2, 3, 0, 1, 2, 3], dtype=torch.int64)
    return (
        partner_unit,
        partner_connector,
        connection_twist_idx,
        disconnect_potentials,
        eq_active,
        eq_indices,
    )


def test_apply_connection_candidates_accepts_only_mutual_pairs() -> None:
    wp.init()
    best_partner_idx = torch.tensor([[1, 0, -1]], dtype=torch.int32)
    best_twist_idx = torch.tensor([[1, 1, -1]], dtype=torch.int32)
    connector_unit_idx = torch.tensor([0, 1, 2], dtype=torch.int32)
    partner_unit = torch.full((1, 3, 1), -1, dtype=torch.int64)
    partner_connector = torch.full_like(partner_unit, -1)
    connection_twist_idx = torch.full_like(partner_unit, -1)
    disconnect_potentials = torch.ones((1, 3, 1), dtype=torch.float32)
    eq_active = torch.zeros((1, 18), dtype=torch.bool)
    eq_indices = torch.arange(18, dtype=torch.int64)

    wp.launch(
        kernel=apply_connection_candidates,
        dim=(1, 3),
        inputs=[
            _warp(best_partner_idx),
            _warp(best_twist_idx),
            _warp(connector_unit_idx),
            1,
            3,
            2,
            _warp(eq_indices),
        ],
        outputs=[
            _warp(partner_unit),
            _warp(partner_connector),
            _warp(connection_twist_idx),
            _warp(disconnect_potentials),
            _warp(eq_active),
        ],
        device="cpu",
    )

    assert torch.equal(partner_unit, torch.tensor([[[1], [0], [-1]]]))
    assert torch.equal(connection_twist_idx, torch.tensor([[[1], [1], [-1]]]))
    assert torch.equal(disconnect_potentials, torch.tensor([[[0.0], [0.0], [1.0]]]))
    assert torch.equal(torch.nonzero(eq_active), torch.tensor([[0, 3]]))


def test_binary_disconnect_kernel_accumulates_both_endpoint_requests() -> None:
    wp.init()
    state = _connection_state()
    partner_unit, partner_connector, twist_idx, potentials, eq_active, eq_indices = state

    wp.launch(
        kernel=update_binary_connector_disconnections,
        dim=(1, 2, 1),
        inputs=[
            _warp(torch.tensor([[[False], [True]]], dtype=torch.bool)),
            2.0,
            1,
            2,
            2,
            _warp(eq_indices),
        ],
        outputs=[
            _warp(partner_unit),
            _warp(partner_connector),
            _warp(twist_idx),
            _warp(potentials),
            _warp(eq_active),
        ],
        device="cpu",
    )
    torch.testing.assert_close(potentials, torch.ones_like(potentials))

    wp.launch(
        kernel=update_binary_connector_disconnections,
        dim=(1, 2, 1),
        inputs=[
            _warp(torch.zeros((1, 2, 1), dtype=torch.bool)),
            2.0,
            1,
            2,
            2,
            _warp(eq_indices),
        ],
        outputs=[
            _warp(partner_unit),
            _warp(partner_connector),
            _warp(twist_idx),
            _warp(potentials),
            _warp(eq_active),
        ],
        device="cpu",
    )

    assert torch.equal(partner_unit, torch.full_like(partner_unit, -1))
    assert torch.equal(partner_connector, torch.full_like(partner_connector, -1))
    assert torch.equal(twist_idx, torch.full_like(twist_idx, -1))
    assert torch.equal(potentials, torch.zeros_like(potentials))
    assert not bool(eq_active[0, 2:4].any())


def test_continuous_disconnect_kernel_matches_pairwise_potential_rule() -> None:
    wp.init()
    state = _connection_state()
    partner_unit, partner_connector, twist_idx, potentials, eq_active, eq_indices = state

    wp.launch(
        kernel=update_continuous_connector_disconnections,
        dim=(1, 2, 1),
        inputs=[
            _warp(torch.tensor([[[-0.25], [1.0]]], dtype=torch.float32)),
            2.0,
            1,
            2,
            2,
            _warp(eq_indices),
        ],
        outputs=[
            _warp(partner_unit),
            _warp(partner_connector),
            _warp(twist_idx),
            _warp(potentials),
            _warp(eq_active),
        ],
        device="cpu",
    )

    torch.testing.assert_close(potentials, torch.full_like(potentials, 0.25))
    assert bool(eq_active[0, 2:4].all())
