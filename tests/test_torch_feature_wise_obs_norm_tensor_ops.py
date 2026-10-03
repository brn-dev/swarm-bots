import unittest
from collections.abc import Callable
from typing import Any
from unittest.mock import patch

import torch

import swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_tensor_ops as obs_norm_tensor_ops
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_tensor_ops import (
    NormalizedObservations,
    build_obs_norm_tensor_operation,
    should_compile_obs_norm_tensor_operations_by_default,
)


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


def _run_operation(
    *,
    obs: torch.Tensor,
    statistics_mask: torch.Tensor,
    scalar_indices: list[int],
    quaternion_starts: list[int] | None = None,
    running_mean: torch.Tensor | None = None,
    running_var: torch.Tensor | None = None,
    running_count: torch.Tensor | None = None,
    eps: float = 0.0,
    update_statistics: bool = True,
    operation: Callable[..., NormalizedObservations] | None = None,
) -> NormalizedObservations:
    scalar_indices_tensor = torch.tensor(scalar_indices, device=obs.device, dtype=torch.long)
    if quaternion_starts:
        quaternion_slices = torch.tensor(
            [[start + offset for offset in range(4)] for start in quaternion_starts],
            device=obs.device,
            dtype=torch.long,
        )
    else:
        quaternion_slices = torch.empty((0, 4), device=obs.device, dtype=torch.long)

    statistics_shape = (len(scalar_indices),)
    if running_mean is None:
        running_mean = torch.zeros(statistics_shape, device=obs.device, dtype=torch.float64)
    if running_var is None:
        running_var = torch.ones(statistics_shape, device=obs.device, dtype=torch.float64)
    if running_count is None:
        running_count = torch.zeros((), device=obs.device, dtype=torch.float64)

    if operation is None:
        operation = build_obs_norm_tensor_operation(
            compile_operation=False,
            compile_mode="default",
        )
    return operation(
        obs,
        statistics_mask,
        scalar_indices_tensor,
        quaternion_slices,
        running_mean,
        running_var,
        running_count,
        eps,
        update_statistics,
    )


class TorchFeatureWiseObsNormTensorOpsTests(unittest.TestCase):
    def test_mask_excludes_non_finite_samples_across_envs_and_agents(self) -> None:
        obs = torch.tensor(
            [
                [
                    [1.0, 10.0, 100.0],
                    [float("nan"), float("inf"), float("-inf")],
                    [3.0, 30.0, 300.0],
                ],
                [
                    [float("inf"), float("nan"), float("-inf")],
                    [5.0, 50.0, 500.0],
                    [float("-inf"), float("inf"), float("nan")],
                ],
            ],
            dtype=torch.float32,
        )
        mask = torch.tensor(
            [[True, False, True], [False, True, False]],
            dtype=torch.bool,
        )

        _normalized, mean, variance, count = _run_operation(
            obs=obs,
            statistics_mask=mask,
            scalar_indices=[0, 1, 2],
        )

        torch.testing.assert_close(
            mean,
            torch.tensor([3.0, 30.0, 300.0], dtype=torch.float64),
        )
        torch.testing.assert_close(
            variance,
            torch.tensor([8.0 / 3.0, 800.0 / 3.0, 80_000.0 / 3.0], dtype=torch.float64),
        )
        torch.testing.assert_close(count, torch.tensor(3.0, dtype=torch.float64))

    def test_all_masked_non_finite_samples_preserve_running_statistics(self) -> None:
        obs = torch.tensor(
            [[[float("nan"), float("inf")], [float("-inf"), float("nan")]]],
            dtype=torch.float32,
        )
        mean = torch.tensor([2.0, 20.0], dtype=torch.float64)
        variance = torch.tensor([4.0, 25.0], dtype=torch.float64)
        count = torch.tensor(7.0, dtype=torch.float64)

        _normalized, next_mean, next_variance, next_count = _run_operation(
            obs=obs,
            statistics_mask=torch.zeros((1, 2), dtype=torch.bool),
            scalar_indices=[0, 1],
            running_mean=mean,
            running_var=variance,
            running_count=count,
        )

        torch.testing.assert_close(next_mean, mean)
        torch.testing.assert_close(next_variance, variance)
        torch.testing.assert_close(next_count, count)

    def test_batch_moments_merge_with_existing_running_statistics(self) -> None:
        obs = torch.tensor(
            [[[4.0, 40.0], [999.0, 999.0], [6.0, 60.0]]],
            dtype=torch.float32,
        )

        _normalized, mean, variance, count = _run_operation(
            obs=obs,
            statistics_mask=torch.tensor([[True, False, True]], dtype=torch.bool),
            scalar_indices=[0, 1],
            running_mean=torch.tensor([2.0, 20.0], dtype=torch.float64),
            running_var=torch.tensor([1.0, 100.0], dtype=torch.float64),
            running_count=torch.tensor(2.0, dtype=torch.float64),
        )

        torch.testing.assert_close(mean, torch.tensor([3.5, 35.0], dtype=torch.float64))
        torch.testing.assert_close(variance, torch.tensor([3.25, 325.0], dtype=torch.float64))
        torch.testing.assert_close(count, torch.tensor(4.0, dtype=torch.float64))

    def test_normalization_uses_updated_statistics_for_every_sample(self) -> None:
        obs = torch.tensor(
            [[[1.0, 50.0], [3.0, 70.0], [100.0, 90.0]]],
            dtype=torch.float32,
        )
        original_obs = obs.clone()

        normalized, mean, variance, _count = _run_operation(
            obs=obs,
            statistics_mask=torch.tensor([[True, True, False]], dtype=torch.bool),
            scalar_indices=[0],
            eps=0.0,
        )

        expected = original_obs.clone()
        expected[..., 0] = torch.tensor([[-1.0, 1.0, 98.0]])
        torch.testing.assert_close(normalized, expected)
        torch.testing.assert_close(obs, original_obs)
        torch.testing.assert_close(mean, torch.tensor([2.0], dtype=torch.float64))
        torch.testing.assert_close(variance, torch.tensor([1.0], dtype=torch.float64))
        self.assertNotEqual(normalized.data_ptr(), obs.data_ptr())

    def test_disabled_updates_use_existing_statistics_and_leave_them_unchanged(self) -> None:
        obs = torch.tensor(
            [[4.0, 8.0], [0.0, 20.0]],
            dtype=torch.float32,
        )
        mean = torch.tensor([2.0, 10.0], dtype=torch.float64)
        variance = torch.tensor([4.0, 25.0], dtype=torch.float64)
        count = torch.tensor(11.0, dtype=torch.float64)

        normalized, next_mean, next_variance, next_count = _run_operation(
            obs=obs,
            statistics_mask=torch.tensor([True, True], dtype=torch.bool),
            scalar_indices=[0, 1],
            running_mean=mean,
            running_var=variance,
            running_count=count,
            update_statistics=False,
        )

        torch.testing.assert_close(
            normalized,
            torch.tensor([[1.0, -0.4], [-1.0, 2.0]], dtype=torch.float32),
        )
        torch.testing.assert_close(next_mean, mean)
        torch.testing.assert_close(next_variance, variance)
        torch.testing.assert_close(next_count, count)

    def test_rank_two_observations_reduce_only_the_batch_dimension(self) -> None:
        obs = torch.tensor(
            [[1.0, 10.0, -1.0], [3.0, 30.0, -2.0], [5.0, 50.0, -3.0]],
            dtype=torch.float32,
        )

        normalized, mean, variance, count = _run_operation(
            obs=obs,
            statistics_mask=torch.tensor([True, False, True], dtype=torch.bool),
            scalar_indices=[0, 1],
        )

        torch.testing.assert_close(mean, torch.tensor([3.0, 30.0], dtype=torch.float64))
        torch.testing.assert_close(variance, torch.tensor([4.0, 400.0], dtype=torch.float64))
        torch.testing.assert_close(count, torch.tensor(2.0, dtype=torch.float64))
        torch.testing.assert_close(normalized[:, 2], obs[:, 2])

    def test_multiple_quaternion_slices_are_canonicalized_without_scalar_features(self) -> None:
        obs = torch.tensor(
            [
                [
                    [-1.0, 2.0, 3.0, 4.0, 99.0, -0.5, -1.0, -2.0, -3.0],
                    [1.0, 2.0, 3.0, 4.0, 88.0, 0.5, -1.0, -2.0, -3.0],
                ],
            ],
            dtype=torch.float32,
        )

        normalized, mean, variance, count = _run_operation(
            obs=obs,
            statistics_mask=torch.ones((1, 2), dtype=torch.bool),
            scalar_indices=[],
            quaternion_starts=[0, 5],
            update_statistics=False,
        )

        expected = torch.tensor(
            [
                [
                    [1.0, -2.0, -3.0, -4.0, 99.0, 0.5, 1.0, 2.0, 3.0],
                    [1.0, 2.0, 3.0, 4.0, 88.0, 0.5, -1.0, -2.0, -3.0],
                ],
            ],
            dtype=torch.float32,
        )
        torch.testing.assert_close(normalized, expected)
        self.assertEqual(mean.numel(), 0)
        self.assertEqual(variance.numel(), 0)
        torch.testing.assert_close(count, torch.tensor(0.0, dtype=torch.float64))

    def test_compiled_and_eager_operations_match_with_masked_non_finite_values(self) -> None:
        obs = torch.tensor(
            [
                [
                    [1.0, -1.0, 2.0, 3.0, 4.0],
                    [float("nan"), 1.0, 2.0, 3.0, 4.0],
                    [5.0, -2.0, 1.0, 0.0, 3.0],
                ],
            ],
            dtype=torch.float32,
        )
        eager = build_obs_norm_tensor_operation(
            compile_operation=False,
            compile_mode="default",
        )

        build_obs_norm_tensor_operation.cache_clear()
        try:
            with patch.object(obs_norm_tensor_ops.torch, "compile", side_effect=_compile_with_aot_eager):
                compiled = build_obs_norm_tensor_operation(
                    compile_operation=True,
                    compile_mode="test-aot-eager",
                )
                cases = (
                    ("mixed mask", torch.tensor([[True, False, True]]), True),
                    ("all masked", torch.zeros((1, 3), dtype=torch.bool), True),
                    ("updates disabled", torch.tensor([[True, False, True]]), False),
                )
                for name, mask, update_statistics in cases:
                    with self.subTest(name=name):
                        eager_result = _run_operation(
                            obs=obs,
                            statistics_mask=mask,
                            scalar_indices=[0],
                            quaternion_starts=[1],
                            eps=1e-6,
                            update_statistics=update_statistics,
                            operation=eager,
                        )
                        compiled_result = _run_operation(
                            obs=obs,
                            statistics_mask=mask,
                            scalar_indices=[0],
                            quaternion_starts=[1],
                            eps=1e-6,
                            update_statistics=update_statistics,
                            operation=compiled,
                        )
                        for eager_value, compiled_value in zip(eager_result, compiled_result, strict=True):
                            torch.testing.assert_close(eager_value, compiled_value, equal_nan=True)
        finally:
            build_obs_norm_tensor_operation.cache_clear()

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA compilation")
    def test_default_compiled_cuda_operation_excludes_masked_non_finite_values(self) -> None:
        device = torch.device("cuda:0")
        operation = build_obs_norm_tensor_operation(
            compile_operation=True,
            compile_mode="reduce-overhead",
        )
        obs = torch.tensor(
            [[[1.0, 10.0], [3.0, 30.0], [float("nan"), float("inf")]]],
            device=device,
        )

        _normalized, mean, variance, count = _run_operation(
            obs=obs,
            statistics_mask=torch.tensor([[True, True, False]], device=device),
            scalar_indices=[0, 1],
            operation=operation,
        )
        torch.cuda.synchronize()

        torch.testing.assert_close(mean.cpu(), torch.tensor([2.0, 20.0], dtype=torch.float64))
        torch.testing.assert_close(variance.cpu(), torch.tensor([1.0, 100.0], dtype=torch.float64))
        torch.testing.assert_close(count.cpu(), torch.tensor(2.0, dtype=torch.float64))

        persistent_mean = mean.clone()
        persistent_variance = variance.clone()
        persistent_count = count.clone()
        _normalized, frozen_mean, frozen_variance, frozen_count = _run_operation(
            obs=obs,
            statistics_mask=torch.zeros((1, 3), device=device, dtype=torch.bool),
            scalar_indices=[0, 1],
            running_mean=persistent_mean,
            running_var=persistent_variance,
            running_count=persistent_count,
            update_statistics=False,
            operation=operation,
        )
        torch.cuda.synchronize()
        torch.testing.assert_close(frozen_mean, persistent_mean)
        torch.testing.assert_close(frozen_variance, persistent_variance)
        torch.testing.assert_close(frozen_count, persistent_count)

    def test_builder_rejects_compilation_without_torch_compile(self) -> None:
        build_obs_norm_tensor_operation.cache_clear()
        try:
            with patch.object(obs_norm_tensor_ops.torch, "compile", None):
                with self.assertRaisesRegex(RuntimeError, "torch.compile support"):
                    build_obs_norm_tensor_operation(
                        compile_operation=True,
                        compile_mode="default",
                    )
        finally:
            build_obs_norm_tensor_operation.cache_clear()

    def test_builder_rejects_empty_compile_mode(self) -> None:
        with self.assertRaisesRegex(ValueError, "compile_mode must be non-empty"):
            build_obs_norm_tensor_operation(
                compile_operation=True,
                compile_mode="",
            )

    def test_default_compilation_selection_requires_cuda_and_torch_compile(self) -> None:
        self.assertFalse(
            should_compile_obs_norm_tensor_operations_by_default(torch.device("cpu")),
        )
        expected_cuda_default = hasattr(torch, "compile") and callable(torch.compile)
        self.assertEqual(
            should_compile_obs_norm_tensor_operations_by_default(torch.device("cuda:0")),
            expected_cuda_default,
        )


if __name__ == "__main__":
    unittest.main()
