import unittest
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

from swarmbots.learn import discord_notifications
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

    def test_mjw_simulation_instability_notification_is_sent_once_per_run(self) -> None:
        with (
            patch.dict(
                "os.environ",
                {discord_notifications.DISCORD_WEBHOOK_ENV_VAR: "https://example.test/webhook"},
            ),
            patch.object(discord_notifications, "_mjw_warning_notification_keys", set()),
            patch.object(discord_notifications, "send_discord_message", return_value=True) as send_message,
        ):
            first_result = discord_notifications.notify_mjw_simulation_instability_once(
                scenario_name="TestScenario",
                num_envs=1024,
                unstable_world_indices=[3, 17],
            )
            second_result = discord_notifications.notify_mjw_simulation_instability_once(
                scenario_name="TestScenario",
                num_envs=1024,
                unstable_world_indices=[24],
            )

        self.assertTrue(first_result)
        self.assertFalse(second_result)
        send_message.assert_called_once()
        content = send_message.call_args.kwargs["content"]
        self.assertIn("simulation instability detected", content)
        self.assertIn("unstable worlds: 2 / 1024 (3, 17)", content)


if __name__ == "__main__":
    unittest.main()
