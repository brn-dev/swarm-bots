import shutil
import sys
import unittest
from dataclasses import fields, replace
from unittest.mock import patch

import pytest
import torch

from swarmbots.learn.algos.ppo.ppo_rollout_buffer import PPOEpisodeAccumulator, PPOEpisodeSegment
from swarmbots.learn.algos.r_mat.r_mat_policy_mixin import RMATPolicyMixin
from swarmbots.learn.algos.r_mat.r_ppo_wm_sampler import RPPOSampler, RPPOWMSampler, RPPOWMSamplerConfig
from swarmbots.learn.algos.world_modeling.next_obs_pred_ppo_wrapper import NOPWorldModelConfig, NextObsPredWrapper
from swarmbots.learn.algos.world_modeling.wm_sampler_helper import (
    WMEpisodeWindows,
    build_wm_episode_windows_batch,
    reset_wm_window_helper_compile_cache,
)
from swarmbots.learn.algos.world_modeling.wm_recurrent_batch import (
    build_wm_target_time_mask,
    flatten_recurrent_wm_batch,
)


def _make_episode(
        *,
        num_steps: int,
        is_true_episode_start: bool = True,
        with_agent_mask: bool = True,
        step_offset: int = 0,
        rollout_env_idx: int | None = None,
        rollout_start_step: int = 0,
        initial_state_value: float = 0.0,
) -> PPOEpisodeSegment:
    n_agents = 2
    n_local_obs = 3
    n_global_obs = 2
    n_hidden_local = 1
    n_hidden_global = 1
    n_actions = 2

    step_values = torch.arange(step_offset, step_offset + num_steps, dtype=torch.float32)
    local_obs = (
        step_values.view(num_steps, 1, 1) * 100.0
        + torch.arange(n_agents, dtype=torch.float32).view(1, n_agents, 1) * 10.0
        + torch.arange(n_local_obs, dtype=torch.float32).view(1, 1, n_local_obs)
    )
    global_obs = step_values.view(num_steps, 1) * 100.0 + torch.arange(n_global_obs, dtype=torch.float32)
    hidden_local_vars = step_values.view(num_steps, 1, 1) + torch.arange(n_agents).view(1, n_agents, 1)
    hidden_global_vars = step_values.view(num_steps, 1)
    actions = (
        1000.0
        + step_values.view(num_steps, 1, 1) * 100.0
        + torch.arange(n_agents, dtype=torch.float32).view(1, n_agents, 1) * 10.0
        + torch.arange(n_actions, dtype=torch.float32).view(1, 1, n_actions)
    )
    log_probs = 2000.0 + step_values.view(num_steps, 1) + torch.arange(n_agents, dtype=torch.float32)
    values = 3000.0 + step_values
    rewards = 4000.0 + step_values
    returns = 5000.0 + step_values
    advantages = 6000.0 + step_values

    agent_mask = None
    final_agent_mask = None
    if with_agent_mask:
        agent_mask = torch.tensor(
            [[True, step_idx % 2 == 0] for step_idx in range(num_steps)],
            dtype=torch.bool,
        )
        final_agent_mask = torch.tensor([True, num_steps % 2 == 0], dtype=torch.bool)

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
        final_local_obs=9000.0 + torch.arange(n_agents * n_local_obs, dtype=torch.float32).view(n_agents, n_local_obs),
        final_global_obs=9100.0 + torch.arange(n_global_obs, dtype=torch.float32),
        final_hidden_local_vars=9200.0 + torch.arange(n_agents * n_hidden_local, dtype=torch.float32).view(
            n_agents,
            n_hidden_local,
        ),
        final_hidden_global_vars=9300.0 + torch.arange(n_hidden_global, dtype=torch.float32),
        final_agent_mask=final_agent_mask,
        final_value=torch.tensor(9400.0),
        initial_previous_actions=9500.0 + torch.arange(n_agents * n_actions, dtype=torch.float32).view(
            n_agents,
            n_actions,
        ),
        is_true_episode_start=is_true_episode_start,
        returns=returns,
        advantages=advantages,
        initial_temporal_state=torch.tensor([[initial_state_value]]),
        rollout_env_idx=rollout_env_idx,
        rollout_start_step=rollout_start_step,
    )


@pytest.mark.parametrize("config_subclass", [False, True])
@pytest.mark.parametrize("factory_kind", ["stock", "custom", "delegated"])
def test_recurrent_nop_preserves_custom_sampler_factory(config_subclass: bool, factory_kind: str) -> None:
    class CustomConfig(RPPOWMSamplerConfig):
        pass

    class CustomSampler(RPPOWMSampler):
        pass

    class RecurrentPolicy(RMATPolicyMixin, torch.nn.Module):
        def requires_previous_actions(self) -> bool:
            return True

    seen_configs: list[RPPOWMSamplerConfig] = []

    class CustomPolicy(RecurrentPolicy):
        def make_sampler(
                self, episodes: list[PPOEpisodeSegment], config: RPPOWMSamplerConfig,
        ) -> RPPOSampler:
            seen_configs.append(config)
            if factory_kind == "delegated":
                return super().make_sampler(episodes, config)
            return CustomSampler(episodes, config, requires_previous_actions=self.requires_previous_actions())

    config_type = CustomConfig if config_subclass else RPPOWMSamplerConfig
    config = config_type(batch_size=1, num_next_steps=2, sequence_length=2)
    policy = RecurrentPolicy() if factory_kind == "stock" else CustomPolicy()
    wrapper = NextObsPredWrapper(policy, world_model_config=NOPWorldModelConfig(
        n_agents=2, local_latent_dim=4, action_dim=2,
        d_model_transition_model=4, nhead_transition_model=1, dim_feedforward_transition_model=8,
    ))
    episodes = [_make_episode(num_steps=2)]
    sampler = wrapper.make_sampler(episodes, config)

    assert type(sampler) is (CustomSampler if factory_kind == "custom" else RPPOWMSampler)
    assert seen_configs == ([] if factory_kind == "stock" else [config])
    batch = next(sampler.sample())
    torch.testing.assert_close(batch.wm_actions[:, 0, 0], episodes[0].actions[None, 0])
    torch.testing.assert_close(batch.previous_actions[:, 0], episodes[0].initial_previous_actions[None])


def test_recurrent_wm_preserves_window_precision_across_episode_lengths() -> None:
    short = _make_episode(num_steps=1, rollout_env_idx=0)
    long = _make_episode(num_steps=2, rollout_env_idx=1, step_offset=2)
    episodes = [
        replace(
            short, actions=short.actions.half(), local_obs=short.local_obs.half(),
            final_local_obs=short.final_local_obs.half(), global_obs=short.global_obs.bfloat16(),
            final_global_obs=short.final_global_obs.bfloat16(),
        ),
        replace(long, final_local_obs=long.final_local_obs.double() + 0.123456789),
    ]
    config = RPPOWMSamplerConfig(batch_size=2, num_next_steps=2, sequence_length=2)
    sampler = RPPOWMSampler(episodes, config)
    assert sampler.wm_actions.dtype == torch.float32
    assert sampler.next_local_obs.dtype == torch.float64
    assert sampler.next_global_obs.dtype == torch.float32
    batch = next(sampler.sample())
    torch.testing.assert_close(batch.action_dtypes[torch.float16], torch.tensor([[True, False], [False, False]]))
    torch.testing.assert_close(batch.action_dtypes[torch.float32], torch.tensor([[False, False], [True, True]]))
    for lane, episode in enumerate(episodes):
        n_steps = episode.actions.shape[0]
        expected = build_wm_episode_windows_batch([episode], num_next_steps=config.num_next_steps)
        for actual, values in (
            (sampler.wm_actions, expected.multi_step_actions),
            (sampler.next_local_obs, expected.next_local_obs),
            (sampler.next_global_obs, expected.next_global_obs),
        ):
            torch.testing.assert_close(actual[lane:lane + 1, :n_steps], values.to(actual.dtype), rtol=0, atol=0)


def test_recurrent_rollout_keeps_mixed_action_precision_and_boundary_conditioning() -> None:
    accumulator = PPOEpisodeAccumulator(
        n_envs=2, max_episode_length=3, n_agents=1, agent_obs_shape=(1,), global_obs_shape=(0,),
        hidden_local_vars_shape=(0,), hidden_global_vars_shape=(0,), n_agent_actions=1, has_agent_mask=False,
        storage_device="cpu", storage_dtype=torch.float32,
    )
    obs = dict(
        local_obs=torch.zeros(2, 1, 1), global_obs=torch.zeros(2, 0),
        hidden_local_vars=torch.zeros(2, 1, 0), hidden_global_vars=torch.zeros(2, 0),
    )
    values = torch.zeros(2)
    precise = 0.5 + 2 ** -27
    previous_actions = torch.full((2, 1, 1), precise, dtype=torch.float64)
    episode_starts = torch.zeros(2, dtype=torch.bool)
    episodes = []
    for step, (dtype, value) in enumerate(((torch.float32, 1.), (torch.float64, precise), (torch.float16, 0.25))):
        actions = torch.full((2, 1, 1), value, dtype=dtype)
        dones = torch.tensor([step == 2, step != 1])
        episodes.extend(accumulator.add(
            **obs, agent_mask=None, actions=actions, rewards=values, log_probs=actions.squeeze(-1), values=values,
            previous_actions=previous_actions, episode_start_mask=episode_starts,
            temporal_state=torch.zeros(2, 1), rollout_step_idx=step, bootstrap_obs=obs,
            bootstrap_values=values, dones=dones,
        ))
        previous_actions = actions.masked_fill(dones[:, None, None], 0.)
        episode_starts = dones

    for episode in episodes:
        episode.compute_gae(gamma=0.99, gae_lambda=0.95)
    sampler = RPPOSampler(
        episodes, RPPOWMSamplerConfig(batch_size=2, num_next_steps=1, sequence_length=4),
        requires_previous_actions=True,
    )
    batch = next(sampler.sample())
    for dtype, step in ((torch.float32, 0), (torch.float64, 1), (torch.float16, 2)):
        expected = torch.zeros(2, 4, dtype=torch.bool)
        expected[:, step] = True
        torch.testing.assert_close(batch.action_dtypes[dtype], expected)
    torch.testing.assert_close(batch.actions[:, 1, 0, 0], torch.full((2,), precise, dtype=torch.float64), rtol=0, atol=0)
    torch.testing.assert_close(batch.log_probs[:, 1, 0], torch.full((2,), precise, dtype=torch.float64), rtol=0, atol=0)
    previous = torch.tensor([[precise, 1., precise, 0.], [precise, 0., precise, 0.]], dtype=torch.float64)
    torch.testing.assert_close(batch.previous_actions[..., 0, 0], previous, rtol=0, atol=0)


@pytest.mark.parametrize("with_agent_mask", [False, True])
def test_recurrent_wm_rows_own_reusable_compiled_window_outputs(with_agent_mask: bool) -> None:
    episodes = [
        _make_episode(
            num_steps=2, step_offset=start, rollout_env_idx=0,
            rollout_start_step=start, with_agent_mask=with_agent_mask,
        )
        for start in (0, 2)
    ]
    config = RPPOWMSamplerConfig(batch_size=1, num_next_steps=3, sequence_length=4)
    expected = RPPOWMSampler(episodes=episodes, config=config)
    storage: WMEpisodeWindows | None = None

    def reusable_windows(
        group: list[PPOEpisodeSegment], *, num_next_steps: int, compile_modules: bool, compile_mode: str,
    ) -> WMEpisodeWindows:
        nonlocal storage
        assert len(group) == 2
        result = build_wm_episode_windows_batch(group, num_next_steps=num_next_steps)
        if storage is None:
            storage = result
        else:
            for field in fields(WMEpisodeWindows):
                source = getattr(result, field.name)
                if source is not None:
                    getattr(storage, field.name).copy_(source)
        return storage

    with (
        patch("swarmbots.learn.algos.r_mat.r_ppo_wm_sampler.ensure_wm_window_helper_compile_available"),
        patch("swarmbots.learn.algos.r_mat.r_ppo_wm_sampler.build_wm_episode_windows_batch", side_effect=reusable_windows) as build_windows,
    ):
        actual = RPPOWMSampler(
            episodes=episodes,
            config=replace(config, compile_wm_window_helper=True, wm_window_helper_compile_mode="reduce-overhead"),
        )
        build_windows.assert_called_once()
        # Reusing the captured output storage must not overwrite the packed row.
        assert storage is not None
        for field in fields(WMEpisodeWindows):
            value = getattr(storage, field.name)
            if value is not None:
                value.zero_()
    for name in ("wm_actions", "next_local_obs", "next_global_obs", "wm_target_time_mask",
                 "wm_agent_mask", "wm_loss_agent_mask"):
        torch.testing.assert_close(getattr(actual, name), getattr(expected, name), rtol=0, atol=0)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("with_agent_mask", [False, True])
def test_compiled_recurrent_wm_windows_survive_episode_and_sampler_reuse(with_agent_mask: bool) -> None:
    if sys.platform.startswith("win") and shutil.which("cl.exe") is None:
        pytest.skip("cl.exe unavailable")
    reset_wm_window_helper_compile_cache()
    config = RPPOWMSamplerConfig(batch_size=2, num_next_steps=3, sequence_length=4)
    retained: list[tuple[RPPOWMSampler, dict[str, torch.Tensor | None]]] = []
    try:
        for offset in range(4):
            torch.compiler.cudagraph_mark_step_begin()
            episodes = []
            for env_idx in range(2):
                for start in (0, 2):
                    episode = _make_episode(
                        num_steps=2, step_offset=offset * 10 + env_idx * 4 + start,
                        rollout_env_idx=env_idx, rollout_start_step=start, with_agent_mask=with_agent_mask,
                    )
                    episodes.append(replace(episode, **{
                        field.name: value.cuda()
                        for field in fields(PPOEpisodeSegment)
                        if isinstance(value := getattr(episode, field.name), torch.Tensor)
                    }))
            expected = RPPOWMSampler(episodes=episodes, config=config)
            actual = RPPOWMSampler(
                episodes=episodes,
                config=replace(
                    config, compile_wm_window_helper=True, wm_window_helper_compile_mode="reduce-overhead",
                ),
            )
            snapshot = {}
            for name in ("wm_actions", "next_local_obs", "next_global_obs", "wm_target_time_mask",
                         "wm_agent_mask", "wm_loss_agent_mask"):
                value = getattr(actual, name)
                torch.testing.assert_close(value, getattr(expected, name), rtol=0, atol=0)
                snapshot[name] = None if value is None else value.clone()
            retained.append((actual, snapshot))
        for sampler, snapshot in retained:
            for name, expected_value in snapshot.items():
                torch.testing.assert_close(getattr(sampler, name), expected_value, rtol=0, atol=0)
    finally:
        reset_wm_window_helper_compile_cache()


class RPPOWMSamplerTests(unittest.TestCase):
    def test_env_fragments_form_one_tbptt_row_with_mid_sequence_reset(self) -> None:
        first_episode = _make_episode(
            num_steps=2,
            rollout_env_idx=0,
            rollout_start_step=0,
            initial_state_value=10.0,
        )
        second_episode = _make_episode(
            num_steps=2,
            step_offset=2,
            rollout_env_idx=0,
            rollout_start_step=2,
            initial_state_value=99.0,
        )

        sampler = RPPOWMSampler(
            episodes=[second_episode, first_episode],
            config=RPPOWMSamplerConfig(batch_size=1, num_next_steps=3, sequence_length=4),
            requires_previous_actions=True,
        )

        self.assertEqual(sampler.local_obs.shape[:2], (1, 4))
        self.assertTrue(torch.equal(
            sampler.local_obs[0, :, 0, 0],
            torch.tensor([0.0, 100.0, 200.0, 300.0]),
        ))
        self.assertTrue(torch.equal(sampler.time_mask, torch.ones((1, 4), dtype=torch.bool)))
        self.assertTrue(torch.equal(
            sampler.episode_start_mask,
            torch.tensor([[True, False, True, False]]),
        ))
        self.assertTrue(torch.equal(sampler.initial_temporal_state, torch.tensor([[10.0]])))

        assert sampler.previous_actions is not None
        self.assertTrue(torch.equal(sampler.previous_actions[0, 0], first_episode.initial_previous_actions))
        self.assertTrue(torch.equal(sampler.previous_actions[0, 1], first_episode.actions[0]))
        self.assertTrue(torch.equal(sampler.previous_actions[0, 2], second_episode.initial_previous_actions))

        self.assertTrue(torch.equal(sampler.next_local_obs[0, 1, 0], first_episode.final_local_obs))
        self.assertFalse(bool(sampler.wm_target_time_mask[0, 1, 1]))

    def test_mid_episode_row_uses_exact_initial_state_without_reset(self) -> None:
        episode = _make_episode(
            num_steps=3,
            is_true_episode_start=False,
            initial_state_value=7.0,
        )

        sampler = RPPOWMSampler(
            episodes=[episode],
            config=RPPOWMSamplerConfig(batch_size=1, num_next_steps=1, sequence_length=4),
        )

        self.assertTrue(torch.equal(sampler.time_mask, torch.tensor([[True, True, True, False]])))
        self.assertTrue(torch.equal(sampler.episode_start_mask, torch.zeros((1, 4), dtype=torch.bool)))
        self.assertTrue(torch.equal(sampler.initial_temporal_state, torch.tensor([[7.0]])))

    def test_rejects_batch_size_larger_than_tbptt_row_count(self) -> None:
        episodes = [
            _make_episode(num_steps=2, rollout_env_idx=env_idx)
            for env_idx in range(2)
        ]

        with self.assertRaisesRegex(ValueError, "batch_size counts TBPTT rows"):
            RPPOWMSampler(
                episodes=episodes,
                config=RPPOWMSamplerConfig(
                    batch_size=4,
                    num_next_steps=1,
                    sequence_length=2,
                ),
            )

    def test_rejects_empty_segments_and_mixed_agent_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "at least one non-empty"):
            RPPOWMSampler(
                episodes=[_make_episode(num_steps=0)],
                config=RPPOWMSamplerConfig(batch_size=1, num_next_steps=1, sequence_length=2),
            )

        with self.assertRaisesRegex(ValueError, "agent_mask must be provided for all episodes or none"):
            RPPOWMSampler(
                episodes=[
                    _make_episode(num_steps=1, with_agent_mask=True),
                    _make_episode(num_steps=1, with_agent_mask=False),
                ],
                config=RPPOWMSamplerConfig(batch_size=1, num_next_steps=1, sequence_length=2),
            )

    def test_rejects_invalid_recurrent_sampler_config(self) -> None:
        valid_episode = _make_episode(num_steps=1)
        invalid_configs = [
            (RPPOWMSamplerConfig(batch_size=1, num_next_steps=1, sequence_length=0), "sequence_length"),
            (RPPOWMSamplerConfig(batch_size=1, num_next_steps=0, sequence_length=2), "num_next_steps"),
        ]
        for config, message in invalid_configs:
            with self.subTest(config=config):
                with self.assertRaisesRegex(ValueError, message):
                    RPPOWMSampler(episodes=[valid_episode], config=config)


class RecurrentWMBatchHelperTests(unittest.TestCase):
    def test_flatten_recurrent_wm_batch_preserves_row_major_order_and_optional_masks(self) -> None:
        local_latents = torch.arange(2 * 3 * 2 * 4, dtype=torch.float32).reshape(2, 3, 2, 4)
        next_local_obs = torch.arange(2 * 3 * 2 * 2 * 5, dtype=torch.float32).reshape(2, 3, 2, 2, 5)
        actions = torch.arange(2 * 3 * 2 * 2 * 1, dtype=torch.float32).reshape(2, 3, 2, 2, 1)
        local_obs = torch.arange(2 * 3 * 2 * 6, dtype=torch.float32).reshape(2, 3, 2, 6)
        next_global_obs = torch.arange(2 * 3 * 2 * 7, dtype=torch.float32).reshape(2, 3, 2, 7)
        agent_mask = torch.tensor([
            [[True, True], [True, False], [False, False]],
            [[True, False], [True, True], [False, True]],
        ])
        loss_agent_mask = ~agent_mask
        time_mask = torch.tensor([
            [[True, True], [True, False], [False, False]],
            [[True, False], [True, True], [False, True]],
        ])

        flattened = flatten_recurrent_wm_batch(
            local_latents=local_latents,
            next_local_obs=next_local_obs,
            actions=actions,
            local_obs=local_obs,
            next_global_obs=next_global_obs,
            agent_mask=agent_mask,
            loss_agent_mask=loss_agent_mask,
            time_mask=time_mask,
        )

        self.assertIsNotNone(flattened)
        assert flattened is not None
        self.assertTrue(torch.equal(flattened.local_latents[0], local_latents[0, 0]))
        self.assertTrue(torch.equal(flattened.local_latents[3], local_latents[1, 0]))
        self.assertTrue(torch.equal(flattened.next_local_obs[5], next_local_obs[1, 2]))
        self.assertTrue(torch.equal(flattened.actions[4], actions[1, 1]))
        self.assertTrue(torch.equal(flattened.local_obs[2], local_obs[0, 2]))
        self.assertTrue(torch.equal(flattened.next_global_obs[1], next_global_obs[0, 1]))
        self.assertTrue(torch.equal(flattened.agent_mask, agent_mask.reshape(6, 2)))
        self.assertTrue(torch.equal(flattened.loss_agent_mask, loss_agent_mask.reshape(6, 2)))
        self.assertTrue(torch.equal(flattened.time_mask, time_mask.reshape(6, 2)))

    def test_flatten_recurrent_wm_batch_returns_none_for_flat_batches(self) -> None:
        result = flatten_recurrent_wm_batch(
            local_latents=torch.zeros(2, 3, 4),
            next_local_obs=torch.zeros(2, 1, 3, 5),
            actions=torch.zeros(2, 1, 3, 2),
        )
        self.assertIsNone(result)

    def test_build_wm_target_time_mask_combines_window_and_loss_masks(self) -> None:
        wm_target_time_mask = torch.tensor([
            [[True, True, False], [True, False, False], [True, True, True]],
            [[True, True, True], [False, False, False], [True, False, True]],
        ])
        sequence_time_mask = torch.tensor([
            [True, False, True],
            [False, True, True],
        ])

        actual = build_wm_target_time_mask(
            wm_target_time_mask=wm_target_time_mask,
            sequence_time_mask=sequence_time_mask,
        )

        self.assertTrue(torch.equal(actual, torch.tensor([
            [[True, True, False], [False, False, False], [True, True, True]],
            [[False, False, False], [False, False, False], [True, False, True]],
        ])))

    def test_recurrent_wm_helpers_reject_bad_masks(self) -> None:
        with self.assertRaisesRegex(ValueError, "time_mask dtype"):
            flatten_recurrent_wm_batch(
                local_latents=torch.zeros(1, 2, 3, 4),
                next_local_obs=torch.zeros(1, 2, 1, 3, 5),
                actions=torch.zeros(1, 2, 1, 3, 2),
                time_mask=torch.ones(1, 2, 1),
            )

        with self.assertRaisesRegex(ValueError, "agent_mask ndim"):
            flatten_recurrent_wm_batch(
                local_latents=torch.zeros(1, 2, 3, 4),
                next_local_obs=torch.zeros(1, 2, 1, 3, 5),
                actions=torch.zeros(1, 2, 1, 3, 2),
                agent_mask=torch.ones(1, 2, 1, 3, 1, dtype=torch.bool),
            )

        with self.assertRaisesRegex(ValueError, "prefix shape"):
            build_wm_target_time_mask(
                wm_target_time_mask=torch.ones(2, 3, 1, dtype=torch.bool),
                sequence_time_mask=torch.ones(2, 4, dtype=torch.bool),
            )


if __name__ == "__main__":
    unittest.main()
