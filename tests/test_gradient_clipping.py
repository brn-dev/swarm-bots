import pytest
import torch

from swarmbots.learn.base_policy import BasePolicy
from swarmbots.learn.optimizers import clip_grad_norm_


DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)]


def _parameters(gradients: list[torch.Tensor]) -> list[torch.nn.Parameter]:
    parameters = [torch.nn.Parameter(torch.zeros_like(gradient)) for gradient in gradients]
    for parameter, gradient in zip(parameters, gradients, strict=True):
        parameter.grad = gradient.clone()
    return parameters


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64])
@pytest.mark.parametrize("max_norm", [None, 2.0])
@pytest.mark.parametrize("error_if_nonfinite", [False, True])
def test_large_finite_gradients_have_finite_norm_and_preserve_direction_when_clipped(
    device: str, dtype: torch.dtype, max_norm: float | None, error_if_nonfinite: bool,
) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    magnitude = 10_000.0 if dtype is torch.float16 else 1e34
    gradients = [
        torch.tensor([magnitude, -magnitude], dtype=dtype, device=device),
        torch.tensor([magnitude / 2], dtype=dtype, device=device),
    ]
    parameters = _parameters(gradients)
    expected_norm = torch.linalg.vector_norm(torch.cat([gradient.double() for gradient in gradients]))

    norm = clip_grad_norm_(parameters, max_norm, error_if_nonfinite=error_if_nonfinite)

    torch.testing.assert_close(norm, expected_norm, rtol=1e-12, atol=0)
    if max_norm is None:
        for parameter, gradient in zip(parameters, gradients, strict=True):
            torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)
    else:
        clipped = torch.cat([parameter.grad.double() for parameter in parameters])
        assert torch.isfinite(clipped).all() and torch.count_nonzero(clipped) == clipped.numel()
        torch.testing.assert_close(clipped.norm(), clipped.new_tensor(max_norm), rtol=0.01, atol=0)
        torch.testing.assert_close(clipped / clipped.norm(), torch.cat([
            gradient.double() for gradient in gradients
        ]) / expected_norm, rtol=0.01, atol=0)
    assert all(parameter.grad.dtype == dtype for parameter in parameters)


@pytest.mark.parametrize("max_norm", [None, 0.5, 20.0])
def test_mixed_dtypes_and_missing_gradients_share_one_norm(max_norm: float | None) -> None:
    gradients = [torch.tensor([3.0], dtype=torch.float32), torch.tensor([4.0], dtype=torch.float64)]
    parameters = _parameters(gradients) + [torch.nn.Parameter(torch.ones(3))]

    norm = clip_grad_norm_(iter(parameters), max_norm)

    torch.testing.assert_close(norm, torch.tensor(5.0, dtype=torch.float64))
    assert parameters[-1].grad is None
    coefficient = 1.0 if max_norm is None else min(max_norm / (5.0 + 1e-6), 1.0)
    for parameter, gradient in zip(parameters, gradients):
        torch.testing.assert_close(parameter.grad, gradient * coefficient)


def test_empty_and_zero_gradients_are_noops() -> None:
    parameter = torch.nn.Parameter(torch.zeros(2))
    assert clip_grad_norm_([parameter], 2.0) == 0.0
    assert parameter.grad is None
    parameter.grad = torch.zeros_like(parameter)
    assert clip_grad_norm_([parameter], 2.0) == 0.0
    assert not parameter.grad.any()


@pytest.mark.parametrize("device", DEVICES)
def test_policy_norm_diagnostics_are_read_only_and_accumulate_in_float64(device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    gradients = [
        torch.tensor([3e20, -4e20], dtype=torch.float32, device=device),
        torch.tensor([1e21], dtype=torch.float64, device=device),
    ]
    parameters = _parameters(gradients)
    module = torch.nn.ParameterList(parameters)
    expected = torch.cat([gradient.double() for gradient in gradients]).norm().item()

    assert BasePolicy._grad_norm_from_parameters(iter(parameters)) == pytest.approx(expected, rel=1e-12)
    assert BasePolicy._module_grad_norm(module) == pytest.approx(expected, rel=1e-12)
    for parameter, gradient in zip(parameters, gradients, strict=True):
        assert BasePolicy._parameter_grad_norm(parameter) == pytest.approx(gradient.double().norm().item(), rel=1e-12)
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0)


def test_policy_norm_diagnostics_without_gradients_return_zero() -> None:
    assert BasePolicy._grad_norm_from_parameters(iter(())) == 0.0
    assert BasePolicy._module_grad_norm(None) == 0.0
    assert BasePolicy._parameter_grad_norm(None) == 0.0
    assert BasePolicy._parameter_grad_norm(torch.nn.Parameter(torch.ones(2))) == 0.0


@pytest.mark.parametrize("value", [float("inf"), float("nan")])
def test_genuinely_nonfinite_gradients_remain_visible(value: float) -> None:
    parameters = _parameters([torch.tensor([value, 1.0])])

    norm = clip_grad_norm_(parameters, 2.0)

    assert not torch.isfinite(norm)
    assert not torch.isfinite(parameters[0].grad).all()


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("max_norm", [None, 2.0])
@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_nonfinite_rejection_preserves_every_gradient(device: str, max_norm: float | None, value: float) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    gradients = [torch.tensor([value, 1.0], device=device), torch.tensor([3.0, 4.0], device=device)]
    parameters = _parameters(gradients)

    with pytest.raises(RuntimeError, match="non-finite"):
        clip_grad_norm_(parameters, max_norm, error_if_nonfinite=True)

    for parameter, gradient in zip(parameters, gradients, strict=True):
        torch.testing.assert_close(parameter.grad, gradient, rtol=0, atol=0, equal_nan=True)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
def test_compiled_cuda_clipping_keeps_large_finite_gradients_and_needs_no_scalar_reads() -> None:
    torch.compiler.reset()
    try:
        parameters = _parameters([torch.tensor([1e34, -1e34], device="cuda")])
        compiled_clip = torch.compile(clip_grad_norm_, fullgraph=True, dynamic=False)

        for _ in range(2):
            parameters[0].grad.fill_(1e34)
            norm = compiled_clip(parameters, 2.0)
            assert torch.isfinite(norm)
            torch.testing.assert_close(parameters[0].grad, torch.full_like(parameters[0], 2 ** 0.5))
    finally:
        torch.compiler.reset()
