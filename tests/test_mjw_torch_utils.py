from __future__ import annotations

import math

import pytest
import torch

from swarmbots.mjw_env.mjw_torch_utils import copy_masked_rows_


CUDA = pytest.param("cuda", marks=(
    pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA"),
))


@pytest.mark.parametrize("device", ["cpu", CUDA])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64, torch.int64, torch.bool])
@pytest.mark.parametrize("row_shape", [(), (3,), (2, 3)])
@pytest.mark.parametrize("selected_rows", [[], [False, False, False], [True, True, True], [True, False, True]])
def test_copy_masked_rows_preserves_unselected_values_and_existing_storage(
    device: str, dtype: torch.dtype, row_shape: tuple[int, ...], selected_rows: list[bool],
) -> None:
    shape = (len(selected_rows), *row_shape)
    source_values = torch.arange(math.prod(shape), dtype=torch.float64).reshape(shape)
    destination_values = -source_values - 1.0
    if dtype is torch.bool:
        source_values = source_values.remainder(2).bool()
        destination_values = ~source_values
    elif dtype.is_floating_point:
        for row, selected in enumerate(selected_rows):
            if not selected:
                source_values[row] = torch.nan
        if selected_rows:
            destination_values[0] = torch.nan
    source_values = source_values.to(dtype=dtype)
    destination_values = destination_values.to(dtype=dtype)
    expected = destination_values.clone()
    for row, selected in enumerate(selected_rows):
        if selected:
            expected[row] = source_values[row]

    destination_storage = torch.empty((*shape, 2), dtype=dtype, device=device)
    destination = destination_storage[..., 0]
    destination.copy_(destination_values)
    source_storage = torch.empty_like(destination_storage)
    source = source_storage[..., 1]
    source.copy_(source_values)
    mask = torch.tensor(selected_rows, dtype=torch.bool, device=device)
    original_pointer, original_stride = destination.data_ptr(), destination.stride()

    copy_masked_rows_(destination, source, mask)

    torch.testing.assert_close(destination.cpu(), expected, equal_nan=True)
    torch.testing.assert_close(source.cpu(), source_values, equal_nan=True)
    assert destination.data_ptr() == original_pointer
    assert destination.stride() == original_stride
    assert destination.dtype is dtype


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_copy_masked_rows_can_replay_with_a_changed_cuda_mask() -> None:
    destination = torch.zeros((3, 2), device="cuda")
    source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], device="cuda")
    mask = torch.tensor([True, False, True], device="cuda")
    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        copy_masked_rows_(destination, source, mask)
    torch.cuda.current_stream().wait_stream(warmup_stream)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        copy_masked_rows_(destination, source, mask)

    destination.fill_(-1.0)
    graph.replay()
    torch.testing.assert_close(destination.cpu(), torch.tensor([[1.0, 2.0], [-1.0, -1.0], [5.0, 6.0]]))

    mask.copy_(torch.tensor([False, True, False], device="cuda"))
    source[0] = torch.nan
    source[2] = torch.nan
    graph.replay()
    torch.testing.assert_close(destination.cpu(), torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]))
