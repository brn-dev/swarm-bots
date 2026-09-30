from __future__ import annotations

import pytest
import torch

from swarmbots import ALL_BENCHMARK_IDS, get_benchmark_spec, make_env
from swarmbots.benchmark.metadata import serialize_settings


@pytest.mark.integration
@pytest.mark.parametrize("benchmark_id", ALL_BENCHMARK_IDS)
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.cuda)])
def test_environment_reset_step_and_same_step_autoreset(benchmark_id: str, device: str) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    if device == "cuda":
        # Task shapes otherwise exhaust Dynamo's process-wide specialization cache.
        torch.compiler.reset()
    env = make_env(
        benchmark_id,
        num_envs=2,
        device=device,
        seed=7,
        episode_length=1,
        scenario_kwargs={"compile_reward_kernel": device == "cuda"},
        compile_tensor_operations=device == "cuda",
    )
    try:
        observations, _ = env.reset(seed=7)
        original = {key: value.clone() for key, value in observations.items()}
        assert (env._time >= env.scenario.reset_settle_time).all()
        repeated, _ = env.reset(seed=7)
        for key in original:
            torch.testing.assert_close(repeated[key], original[key])
        actions = {
            "actuators": torch.zeros(env.action_space["actuators"].shape, device=env.device),
            "connectors": torch.zeros(env.action_space["connectors"].shape, device=env.device),
        }

        next_observations, rewards, terminations, truncations, infos = env.step(actions)

        assert observations["local_obs"].device.type == device
        assert next_observations["local_obs"].shape == observations["local_obs"].shape
        assert rewards.shape == (2,)
        assert (terminations | truncations).all()
        assert torch.isfinite(rewards).all()
        assert all(torch.isfinite(value).all() for value in next_observations.values())
        assert infos["_final_obs"].all()
        assert (env._time >= env.scenario.reset_settle_time).all()
        assert set(infos["final_obs"]) == set(observations)
        if get_benchmark_spec(benchmark_id).supports_success_metric:
            assert infos["success"].shape == (2,)
        repeated, _ = env.reset(seed=7)
        for key in original:
            torch.testing.assert_close(repeated[key], original[key])
        settings = serialize_settings(env.get_settings())
        assert settings["scenario"]["swarm"]["unit_start_locations"]["pool_seeds"] == list(range(42_000, 42_050))
    finally:
        env.close()
        if device == "cuda":
            torch.compiler.reset()


@pytest.mark.integration
def test_multi_payload_partial_reset_updates_only_selected_world_observations() -> None:
    env = make_env(
        "SwarmBots-MultiPayloadGoal-v0", num_envs=2, device="cpu", seed=7,
        scenario_kwargs={"compile_reward_kernel": False},
        compile_tensor_operations=False,
    )
    try:
        observations, _ = env.reset(seed=7)
        original = {key: value.clone() for key, value in observations.items()}
        original_qpos = env._qpos.clone()
        original_qvel = env._qvel.clone()
        payload_records = original["global_obs"].reshape(2, -1, 12)
        assert (payload_records[:, :, 0].sum(dim=1) > 0).all()

        reset_observations, _ = env.reset(seed=8, options={"reset_mask": torch.tensor([True, False])})
        assert not torch.equal(reset_observations["global_obs"][0], original["global_obs"][0])
        for key, value in original.items():
            torch.testing.assert_close(reset_observations[key][1], value[1])
        torch.testing.assert_close(env._qpos[1], original_qpos[1])
        torch.testing.assert_close(env._qvel[1], original_qvel[1])
        assert (env._time >= env.scenario.reset_settle_time).all()
    finally:
        env.close()


@pytest.mark.integration
@pytest.mark.parametrize(
    ("device", "episode_length"),
    [("cpu", 7), pytest.param("cuda", 500, marks=pytest.mark.cuda)],
)
def test_staggered_episodes_with_settled_buffer_and_seeded_restart(device: str, episode_length: int) -> None:
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA device unavailable")
    if device == "cuda":
        torch.compiler.reset()
    first_lengths = [episode_length // 3, 2 * episode_length // 3]
    env = make_env(
        "SwarmBots-MoveTo-v0", num_envs=2, device=device, seed=7,
        episode_length=episode_length, first_episode_lengths=first_lengths,
        settled_reset_buffer_size=2, settled_reset_batch_size=1,
        scenario_kwargs={"compile_reward_kernel": device == "cuda"},
        compile_tensor_operations=device == "cuda",
    )
    try:
        observations, _ = env.reset(seed=7)
        original = {key: value.clone() for key, value in observations.items()}
        actions = {key: torch.zeros(space.shape, device=env.device) for key, space in env.action_space.items()}
        next_end = torch.tensor(first_lengths, device=env.device)
        completed = torch.zeros(2, device=env.device, dtype=torch.int64)
        previous_terminal: dict[str, torch.Tensor] = {}
        saved_terminal: dict[str, torch.Tensor] = {}

        for step in range(1, 3 * episode_length + 1):
            observations, rewards, terminated, truncated, info = env.step(actions)
            for key, terminal in previous_terminal.items():
                torch.testing.assert_close(terminal, saved_terminal[key])
            previous_terminal = {}
            expected_done = next_end == step
            assert not terminated.any(), f"Unexpected simulation failure at step {step}"
            torch.testing.assert_close(truncated, expected_done)
            assert torch.isfinite(rewards).all()
            assert all(torch.isfinite(value).all() for value in observations.values())
            if expected_done.any():
                torch.testing.assert_close(info["_final_obs"], expected_done)
                assert (env._time[expected_done] >= env.scenario.reset_settle_time).all()
                assert (env.current_step[expected_done] == 0).all()
                for key, terminal in info["final_obs"].items():
                    assert torch.isfinite(terminal[expected_done]).all()
                previous_terminal = info["final_obs"]
                saved_terminal = {key: value.clone() for key, value in previous_terminal.items()}
                completed += expected_done
                next_end += expected_done * episode_length

        assert (completed == 3).all()
        repeated, _ = env.reset(seed=7)
        for key in original:
            torch.testing.assert_close(repeated[key], original[key])
    finally:
        env.close()
        if device == "cuda":
            torch.compiler.reset()
