import unittest

import torch

from swarmbots.learn.torch_running_mean_std import TorchRunningMeanStd


class TorchRunningMeanStdTests(unittest.TestCase):
    def test_incremental_updates_match_single_batch_update(self) -> None:
        batch = torch.tensor(
            [
                [1.0, 10.0],
                [3.0, 30.0],
                [5.0, 50.0],
                [7.0, 70.0],
            ],
            dtype=torch.float32,
        )
        one_shot = TorchRunningMeanStd(shape=(2,), initial_count=0.0)
        incremental = TorchRunningMeanStd(shape=(2,), initial_count=0.0)

        one_shot.update(batch)
        incremental.update(batch[:1])
        incremental.update(batch[1:3])
        incremental.update(batch[3:])

        torch.testing.assert_close(incremental.mean, one_shot.mean)
        torch.testing.assert_close(incremental.var, one_shot.var)
        torch.testing.assert_close(incremental.count, one_shot.count)

    def test_scalar_update_accepts_zero_dimensional_tensor(self) -> None:
        rms = TorchRunningMeanStd(shape=(), initial_count=0.0)

        rms.update(torch.tensor(3.0))

        torch.testing.assert_close(rms.mean, torch.tensor(3.0, dtype=torch.float64))
        torch.testing.assert_close(rms.var, torch.tensor(0.0, dtype=torch.float64))
        torch.testing.assert_close(rms.count, torch.tensor(1.0, dtype=torch.float64))

    def test_initial_count_behaves_like_a_neutral_prior(self) -> None:
        rms = TorchRunningMeanStd(shape=(), initial_count=1.0)

        rms.update(torch.tensor([2.0]))

        torch.testing.assert_close(rms.mean, torch.tensor(1.0, dtype=torch.float64))
        torch.testing.assert_close(rms.var, torch.tensor(1.5, dtype=torch.float64))
        torch.testing.assert_close(rms.count, torch.tensor(2.0, dtype=torch.float64))

    def test_empty_batch_update_is_noop(self) -> None:
        rms = TorchRunningMeanStd(shape=(2,), initial_count=0.0)
        rms.update(torch.tensor([[1.0, 2.0]], dtype=torch.float32))
        mean = rms.mean.clone()
        var = rms.var.clone()
        count = rms.count.clone()

        rms.update(torch.empty((0, 2), dtype=torch.float32))

        torch.testing.assert_close(rms.mean, mean)
        torch.testing.assert_close(rms.var, var)
        torch.testing.assert_close(rms.count, count)

    def test_to_moves_all_state_tensors(self) -> None:
        rms = TorchRunningMeanStd(shape=(2,), device="cpu")

        rms.to("cpu")

        self.assertEqual(rms.mean.device, torch.device("cpu"))
        self.assertEqual(rms.var.device, torch.device("cpu"))
        self.assertEqual(rms.count.device, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
