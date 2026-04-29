import unittest
from dataclasses import fields
import shutil
import sys

import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeSegment
from swarmbots.learn.algos.world_modeling.ppo_wm_sampler import PPOWMSampler, PPOWMSamplerConfig
from swarmbots.learn.algos.world_modeling.wm_sampler_helper import (
    WMEpisodeWindows,
    build_wm_episode_windows,
    build_wm_episode_windows_batch,
)


def _make_episode(
        *,
        base_value: float,
        num_steps: int,
        with_agent_mask: bool,
) -> PPOEpisodeSegment:
    n_agents = 3
    n_local_obs = 4
    n_global_obs = 2
    n_hidden_local = 2
    n_hidden_global = 1
    n_actions = 3

    step_values = torch.arange(num_steps, dtype=torch.float32)
    local_obs = (
        base_value
        + step_values.view(num_steps, 1, 1)
        + torch.arange(n_agents, dtype=torch.float32).view(1, n_agents, 1) * 10.0
        + torch.arange(n_local_obs, dtype=torch.float32).view(1, 1, n_local_obs)
    )
    global_obs = (
        base_value
        + 100.0
        + step_values.view(num_steps, 1)
        + torch.arange(n_global_obs, dtype=torch.float32).view(1, n_global_obs)
    )
    hidden_local_vars = (
        base_value
        + 200.0
        + step_values.view(num_steps, 1, 1)
        + torch.arange(n_agents * n_hidden_local, dtype=torch.float32).view(1, n_agents, n_hidden_local)
    )
    hidden_global_vars = (
        base_value
        + 300.0
        + step_values.view(num_steps, 1)
        + torch.arange(n_hidden_global, dtype=torch.float32).view(1, n_hidden_global)
    )
    actions = (
        base_value
        + 400.0
        + step_values.view(num_steps, 1, 1)
        + torch.arange(n_agents, dtype=torch.float32).view(1, n_agents, 1) * 5.0
        + torch.arange(n_actions, dtype=torch.float32).view(1, 1, n_actions)
    )
    log_probs = (
        base_value
        + 500.0
        + step_values.view(num_steps, 1)
        + torch.arange(n_agents, dtype=torch.float32).view(1, n_agents)
    )
    values = base_value + 600.0 + step_values
    rewards = base_value + 700.0 + step_values
    returns = base_value + 800.0 + step_values
    advantages = base_value + 900.0 + step_values

    agent_mask = None
    final_agent_mask = None
    if with_agent_mask:
        agent_mask = torch.tensor(
            [
                [
                    True,
                    step_idx % 2 == 0,
                    (step_idx + int(base_value)) % 3 != 0,
                ]
                for step_idx in range(num_steps)
            ],
            dtype=torch.bool,
        )
        final_agent_mask = torch.tensor(
            [True, num_steps % 2 == 0, (num_steps + int(base_value)) % 3 != 0],
            dtype=torch.bool,
        )

    return PPOEpisodeSegment(
        local_obs=local_obs,
        global_obs=global_obs,
        hidden_local_vars=hidden_local_vars,
        hidden_global_vars=hidden_global_vars,
        agent_mask=agent_mask,
        actions=actions,
        rewards=rewards,
        log_probs=log_probs,
        values=values,
        final_local_obs=(
            base_value
            + 1000.0
            + torch.arange(n_agents * n_local_obs, dtype=torch.float32).view(n_agents, n_local_obs)
        ),
        final_global_obs=base_value + 1100.0 + torch.arange(n_global_obs, dtype=torch.float32),
        final_hidden_local_vars=(
            base_value
            + 1200.0
            + torch.arange(n_agents * n_hidden_local, dtype=torch.float32).view(n_agents, n_hidden_local)
        ),
        final_hidden_global_vars=base_value + 1300.0 + torch.arange(n_hidden_global, dtype=torch.float32),
        final_agent_mask=final_agent_mask,
        final_value=torch.tensor(base_value + 1400.0, dtype=torch.float32),
        initial_previous_actions=(
            base_value
            + 1500.0
            + torch.arange(n_agents * n_actions, dtype=torch.float32).view(n_agents, n_actions)
        ),
        is_true_episode_start=True,
        returns=returns,
        advantages=advantages,
    )


def _build_serial_windows(
        episodes: list[PPOEpisodeSegment],
        *,
        num_next_steps: int,
) -> WMEpisodeWindows:
    windows_per_episode = [
        build_wm_episode_windows(episode, num_next_steps=num_next_steps)
        for episode in episodes
    ]
    first = windows_per_episode[0]
    return WMEpisodeWindows(
        multi_step_actions=torch.cat([windows.multi_step_actions for windows in windows_per_episode], dim=0),
        next_local_obs=torch.cat([windows.next_local_obs for windows in windows_per_episode], dim=0),
        wm_target_time_mask=torch.cat([windows.wm_target_time_mask for windows in windows_per_episode], dim=0),
        next_global_obs=torch.cat([windows.next_global_obs for windows in windows_per_episode], dim=0),
        wm_agent_mask=(
            None
            if first.wm_agent_mask is None
            else torch.cat([windows.wm_agent_mask for windows in windows_per_episode], dim=0)
        ),
        wm_loss_agent_mask=(
            None
            if first.wm_loss_agent_mask is None
            else torch.cat([windows.wm_loss_agent_mask for windows in windows_per_episode], dim=0)
        ),
    )


def _build_batched_serial_windows(
        episodes: list[PPOEpisodeSegment],
        *,
        num_next_steps: int,
) -> WMEpisodeWindows:
    windows_per_episode = [
        build_wm_episode_windows(episode, num_next_steps=num_next_steps)
        for episode in episodes
    ]
    first = windows_per_episode[0]
    return WMEpisodeWindows(
        multi_step_actions=torch.stack([windows.multi_step_actions for windows in windows_per_episode], dim=0),
        next_local_obs=torch.stack([windows.next_local_obs for windows in windows_per_episode], dim=0),
        wm_target_time_mask=torch.stack([windows.wm_target_time_mask for windows in windows_per_episode], dim=0),
        next_global_obs=torch.stack([windows.next_global_obs for windows in windows_per_episode], dim=0),
        wm_agent_mask=(
            None
            if first.wm_agent_mask is None
            else torch.stack([windows.wm_agent_mask for windows in windows_per_episode], dim=0)
        ),
        wm_loss_agent_mask=(
            None
            if first.wm_loss_agent_mask is None
            else torch.stack([windows.wm_loss_agent_mask for windows in windows_per_episode], dim=0)
        ),
    )


def _assert_windows_equal(
        test_case: unittest.TestCase,
        actual: WMEpisodeWindows,
        expected: WMEpisodeWindows,
) -> None:
    for field in fields(WMEpisodeWindows):
        actual_value = getattr(actual, field.name)
        expected_value = getattr(expected, field.name)
        if expected_value is None:
            test_case.assertIsNone(actual_value, msg=field.name)
            continue
        test_case.assertIsNotNone(actual_value, msg=field.name)
        assert actual_value is not None
        test_case.assertTrue(
            torch.equal(actual_value, expected_value),
            msg=field.name,
        )


def _episode_to_device(episode: PPOEpisodeSegment, device: torch.device) -> PPOEpisodeSegment:
    return PPOEpisodeSegment(
        local_obs=episode.local_obs.to(device),
        global_obs=episode.global_obs.to(device),
        hidden_local_vars=episode.hidden_local_vars.to(device),
        hidden_global_vars=episode.hidden_global_vars.to(device),
        agent_mask=None if episode.agent_mask is None else episode.agent_mask.to(device),
        actions=episode.actions.to(device),
        rewards=None if episode.rewards is None else episode.rewards.to(device),
        log_probs=episode.log_probs.to(device),
        values=None if episode.values is None else episode.values.to(device),
        final_local_obs=None if episode.final_local_obs is None else episode.final_local_obs.to(device),
        final_global_obs=None if episode.final_global_obs is None else episode.final_global_obs.to(device),
        final_hidden_local_vars=(
            None if episode.final_hidden_local_vars is None else episode.final_hidden_local_vars.to(device)
        ),
        final_hidden_global_vars=(
            None if episode.final_hidden_global_vars is None else episode.final_hidden_global_vars.to(device)
        ),
        final_agent_mask=None if episode.final_agent_mask is None else episode.final_agent_mask.to(device),
        final_value=None if episode.final_value is None else episode.final_value.to(device),
        initial_previous_actions=(
            None if episode.initial_previous_actions is None else episode.initial_previous_actions.to(device)
        ),
        is_true_episode_start=episode.is_true_episode_start,
        returns=None if episode.returns is None else episode.returns.to(device),
        advantages=None if episode.advantages is None else episode.advantages.to(device),
    )


class PPOWMSamplerTests(unittest.TestCase):
    def test_batch_helper_matches_single_episode_helper(self) -> None:
        for with_agent_mask in (False, True):
            for num_next_steps in (1, 2, 4):
                with self.subTest(with_agent_mask=with_agent_mask, num_next_steps=num_next_steps):
                    episode = _make_episode(
                        base_value=10.0 + num_next_steps,
                        num_steps=3,
                        with_agent_mask=with_agent_mask,
                    )
                    expected = _build_batched_serial_windows([episode], num_next_steps=num_next_steps)
                    actual = build_wm_episode_windows_batch([episode], num_next_steps=num_next_steps)
                    _assert_windows_equal(self, actual, expected)

    def test_batch_helper_matches_serial_for_same_length_episodes(self) -> None:
        for with_agent_mask in (False, True):
            for num_next_steps in (1, 3, 5):
                with self.subTest(with_agent_mask=with_agent_mask, num_next_steps=num_next_steps):
                    episodes = [
                        _make_episode(base_value=10.0 * (idx + 1), num_steps=4, with_agent_mask=with_agent_mask)
                        for idx in range(5)
                    ]
                    expected = _build_batched_serial_windows(episodes, num_next_steps=num_next_steps)
                    actual = build_wm_episode_windows_batch(episodes, num_next_steps=num_next_steps)
                    _assert_windows_equal(self, actual, expected)

    def test_batch_helper_rejects_invalid_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "must not be empty"):
            build_wm_episode_windows_batch([], num_next_steps=2)

        with self.assertRaisesRegex(ValueError, "same number of steps"):
            build_wm_episode_windows_batch(
                [
                    _make_episode(base_value=10.0, num_steps=3, with_agent_mask=False),
                    _make_episode(base_value=20.0, num_steps=4, with_agent_mask=False),
                ],
                num_next_steps=2,
            )

        with self.assertRaisesRegex(ValueError, "agent_mask must be provided for all episodes or none"):
            build_wm_episode_windows_batch(
                [
                    _make_episode(base_value=10.0, num_steps=3, with_agent_mask=False),
                    _make_episode(base_value=20.0, num_steps=3, with_agent_mask=True),
                ],
                num_next_steps=2,
            )

    def test_sampler_matches_serial_window_construction_for_mixed_length_buckets(self) -> None:
        episode_lengths = [3, 2, 3, 1, 2, 5, 1, 5]
        for with_agent_mask in (False, True):
            for num_next_steps in (1, 2, 4):
                with self.subTest(with_agent_mask=with_agent_mask, num_next_steps=num_next_steps):
                    episodes = [
                        _make_episode(
                            base_value=10.0 * (idx + 1),
                            num_steps=length,
                            with_agent_mask=with_agent_mask,
                        )
                        for idx, length in enumerate(episode_lengths)
                    ]
                    expected_windows = _build_serial_windows(episodes, num_next_steps=num_next_steps)
                    sampler = PPOWMSampler(
                        episodes=episodes,
                        config=PPOWMSamplerConfig(batch_size=3, num_next_steps=num_next_steps),
                        requires_previous_actions=True,
                    )

                    self.assertTrue(torch.equal(sampler.multi_step_actions, expected_windows.multi_step_actions))
                    self.assertTrue(torch.equal(sampler.next_local_obs, expected_windows.next_local_obs))
                    self.assertTrue(torch.equal(sampler.wm_target_time_mask, expected_windows.wm_target_time_mask))
                    self.assertTrue(torch.equal(sampler.next_global_obs, expected_windows.next_global_obs))
                    if with_agent_mask:
                        self.assertIsNotNone(sampler.wm_agent_mask)
                        self.assertIsNotNone(sampler.wm_loss_agent_mask)
                        assert expected_windows.wm_agent_mask is not None
                        assert expected_windows.wm_loss_agent_mask is not None
                        self.assertTrue(torch.equal(sampler.wm_agent_mask, expected_windows.wm_agent_mask))
                        self.assertTrue(torch.equal(sampler.wm_loss_agent_mask, expected_windows.wm_loss_agent_mask))
                    else:
                        self.assertIsNone(sampler.wm_agent_mask)
                        self.assertIsNone(sampler.wm_loss_agent_mask)

                    expected_previous_actions = torch.cat(
                        tuple(
                            torch.cat((episode.initial_previous_actions.unsqueeze(0), episode.actions[:-1]), dim=0)
                            for episode in episodes
                        ),
                        dim=0,
                    )
                    self.assertTrue(torch.equal(sampler.previous_actions, expected_previous_actions))

    def test_sampler_matches_serial_construction_for_many_bucketed_episodes(self) -> None:
        generator = torch.Generator().manual_seed(1234)
        num_next_steps = 5
        for with_agent_mask in (False, True):
            with self.subTest(with_agent_mask=with_agent_mask):
                episode_lengths = torch.randint(
                    low=1,
                    high=7,
                    size=(32,),
                    generator=generator,
                ).tolist()
                episodes = [
                    _make_episode(
                        base_value=100.0 + idx * 7.0,
                        num_steps=length,
                        with_agent_mask=with_agent_mask,
                    )
                    for idx, length in enumerate(episode_lengths)
                ]
                expected_windows = _build_serial_windows(episodes, num_next_steps=num_next_steps)
                sampler = PPOWMSampler(
                    episodes=episodes,
                    config=PPOWMSamplerConfig(batch_size=8, num_next_steps=num_next_steps),
                )

                self.assertTrue(torch.equal(sampler.multi_step_actions, expected_windows.multi_step_actions))
                self.assertTrue(torch.equal(sampler.next_local_obs, expected_windows.next_local_obs))
                self.assertTrue(torch.equal(sampler.wm_target_time_mask, expected_windows.wm_target_time_mask))
                self.assertTrue(torch.equal(sampler.next_global_obs, expected_windows.next_global_obs))
                if with_agent_mask:
                    self.assertTrue(torch.equal(sampler.wm_agent_mask, expected_windows.wm_agent_mask))
                    self.assertTrue(torch.equal(sampler.wm_loss_agent_mask, expected_windows.wm_loss_agent_mask))
                else:
                    self.assertIsNone(sampler.wm_agent_mask)
                    self.assertIsNone(sampler.wm_loss_agent_mask)

    def test_sampler_compiled_helper_matches_eager_sampler(self) -> None:
        if not hasattr(torch, "compile"):
            self.skipTest("torch.compile unavailable")
        if sys.platform.startswith("win") and shutil.which("cl.exe") is None:
            self.skipTest("cl.exe unavailable")
        if sys.platform.startswith("win") and not torch.cuda.is_available():
            self.skipTest("Windows CPU Inductor compile is not reliable here without OpenMP headers")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        episodes = [
            _episode_to_device(
                _make_episode(base_value=10.0 * (idx + 1), num_steps=length, with_agent_mask=True),
                device,
            )
            for idx, length in enumerate([1, 2, 3, 4, 5, 6, 7, 8])
        ]

        eager_sampler = PPOWMSampler(
            episodes=episodes,
            config=PPOWMSamplerConfig(batch_size=4, num_next_steps=3),
            requires_previous_actions=True,
        )
        compiled_sampler = PPOWMSampler(
            episodes=episodes,
            config=PPOWMSamplerConfig(
                batch_size=4,
                num_next_steps=3,
                compile_wm_window_helper=True,
                wm_window_helper_compile_mode="default",
            ),
            requires_previous_actions=True,
        )

        self.assertTrue(torch.equal(compiled_sampler.multi_step_actions, eager_sampler.multi_step_actions))
        self.assertTrue(torch.equal(compiled_sampler.next_local_obs, eager_sampler.next_local_obs))
        self.assertTrue(torch.equal(compiled_sampler.wm_target_time_mask, eager_sampler.wm_target_time_mask))
        self.assertTrue(torch.equal(compiled_sampler.next_global_obs, eager_sampler.next_global_obs))
        self.assertTrue(torch.equal(compiled_sampler.wm_agent_mask, eager_sampler.wm_agent_mask))
        self.assertTrue(torch.equal(compiled_sampler.wm_loss_agent_mask, eager_sampler.wm_loss_agent_mask))


if __name__ == "__main__":
    unittest.main()
