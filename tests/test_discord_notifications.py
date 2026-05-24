import unittest
from pathlib import Path
from types import SimpleNamespace

from swarmbots.learn.discord_notifications import _format_training_run_finished_message


class DiscordNotificationTests(unittest.TestCase):
    def test_finished_message_uses_final_return_ema(self) -> None:
        algorithm = SimpleNamespace(
            n_total_timesteps=123,
            _last_return_ema=None,
            _final_return_ema=12.5,
            _best_return_ema=13.0,
        )

        message = _format_training_run_finished_message(
            run_name="test-run",
            status="finished",
            run_dir=Path("runs/test-run"),
            total_timesteps=123,
            algorithm=algorithm,
            error=None,
        )

        self.assertIn("machine: ", message)
        self.assertIn("final_ep_rew_ema: 12.5", message)
        self.assertIn("best_ep_rew_ema: 13", message)


if __name__ == "__main__":
    unittest.main()
