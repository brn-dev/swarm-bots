import unittest

import numpy as np
import torch

from swarmbots.learn.tensor_conversion import (
    normalize_reset_mask_options,
    to_backend_array,
    to_numpy_array,
    to_torch_tensor,
)


class _BrokenDlpackArray:
    def __init__(self, fallback_value: np.ndarray) -> None:
        self.fallback_value = fallback_value

    def __dlpack__(self):
        raise RuntimeError("broken dlpack")

    def __array__(self, dtype=None):
        return np.asarray(self.fallback_value, dtype=dtype)


class TensorConversionTests(unittest.TestCase):
    def test_to_torch_tensor_preserves_existing_tensor_device_and_converts_dtype(self) -> None:
        source = torch.arange(4, dtype=torch.float64)

        converted = to_torch_tensor(source, device="cpu", dtype=torch.float32)

        self.assertEqual(converted.device, torch.device("cpu"))
        self.assertEqual(converted.dtype, torch.float32)
        self.assertTrue(torch.equal(converted, source.to(torch.float32)))

    def test_to_numpy_array_detaches_tensor_and_converts_dtype_without_aliasing_grad(self) -> None:
        source = torch.arange(4, dtype=torch.float32, requires_grad=True)

        converted = to_numpy_array(source, dtype=np.float64)

        self.assertEqual(converted.dtype, np.float64)
        np.testing.assert_array_equal(converted, np.arange(4, dtype=np.float64))

    def test_broken_dlpack_falls_back_to_array_protocol(self) -> None:
        source = _BrokenDlpackArray(np.array([1, 0, 1], dtype=np.int64))

        tensor = to_torch_tensor(source, device="cpu", dtype=torch.bool)
        array = to_numpy_array(source, dtype=np.bool_)

        self.assertTrue(torch.equal(tensor, torch.tensor([True, False, True])))
        np.testing.assert_array_equal(array, np.array([True, False, True]))

    def test_to_backend_array_normalizes_supported_backends_and_rejects_unknown_backend(self) -> None:
        tensor = torch.tensor([1.0, 2.0])

        numpy_value = to_backend_array(tensor, backend="numpy", dtype=np.float64)
        torch_value = to_backend_array(np.array([1, 2], dtype=np.int64), backend="torch", dtype=torch.float32)

        self.assertEqual(numpy_value.dtype, np.float64)
        self.assertEqual(torch_value.dtype, torch.float32)
        self.assertTrue(torch.equal(torch_value, torch.tensor([1.0, 2.0])))
        with self.assertRaisesRegex(ValueError, "Unsupported backend"):
            to_backend_array(tensor, backend="jax")

    def test_normalize_reset_mask_options_copies_options_and_flattens_bool_mask(self) -> None:
        original_options = {
            "reset_mask": torch.tensor([[1], [0], [1]]),
            "unchanged": "value",
        }

        normalized = normalize_reset_mask_options(original_options, num_envs=3)

        self.assertIsNot(normalized, original_options)
        assert normalized is not None
        self.assertEqual(normalized["unchanged"], "value")
        np.testing.assert_array_equal(normalized["reset_mask"], np.array([True, False, True]))
        self.assertEqual(normalized["reset_mask"].dtype, np.bool_)

    def test_normalize_reset_mask_options_rejects_wrong_mask_length(self) -> None:
        with self.assertRaisesRegex(ValueError, "Expected reset_mask shape"):
            normalize_reset_mask_options({"reset_mask": [True, False]}, num_envs=3)


if __name__ == "__main__":
    unittest.main()
