from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any
from unittest.mock import patch
import sys

import numpy as np
import pytest
import torch

import swarmbots.benchmark.cli as cli
import swarmbots.benchmark.recording as recording
from swarmbots import record_policy
from swarmbots.learn import as_benchmark_policy, make_training, record_checkpoint
from swarmbots.learn.env_wrappers.torch_feature_wise_obs_norm_wrapper import TorchFeatureWiseObsNormWrapper
from swarmbots.mjw_env.swarm.mjw_homogeneous_swarm import MJWPreConnectedUnitLocationsConfig


class _RecordingEnv:
    num_envs = 2
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.steps = 0
        self.closed = False
        self.reset_seed: int | None = None
        self.recording_options: dict[str, Any] = {}

    def _observations(self) -> dict[str, torch.Tensor]:
        return {
            "local_obs": torch.full((2, 1, 1), float(self.steps)),
            "global_obs": torch.zeros(2, 1),
            "agent_mask": torch.ones(2, 1, dtype=torch.bool),
            "hidden_local_vars": torch.ones(2, 1, 1),
            "hidden_global_vars": torch.ones(2, 1),
        }

    def reset(self, *, seed: int) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        self.reset_seed = seed
        self.steps = 0
        return self._observations(), {}

    def start_video_recording(self, **kwargs: Any) -> None:
        assert self.reset_seed is not None
        self.recording_options = kwargs

    def get_video_recording_status(self) -> dict[str, bool]:
        return {"active": self.steps < 4}

    def step(self, actions: dict[str, torch.Tensor]) -> tuple:
        assert actions == {}
        self.steps += 1
        return (
            self._observations(),
            torch.zeros(2),
            torch.tensor([self.steps % 2 == 0, False]),
            torch.tensor([False, self.steps % 3 == 0]),
            {},
        )

    def close(self) -> None:
        self.closed = True


def test_recording_uses_reset_observations_and_resets_policy_state_for_each_done_lane(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    env = _RecordingEnv()
    environment_options: dict[str, Any] = {}

    def make_env(benchmark_id: str, **kwargs: Any) -> _RecordingEnv:
        assert benchmark_id == "SwarmBots-WallEasy-v0"
        environment_options.update(kwargs)
        return env

    monkeypatch.setattr(recording, "make_env", make_env)
    starts: list[list[bool]] = []

    def policy(observations: Mapping[str, torch.Tensor], episode_starts: torch.Tensor) -> dict[str, torch.Tensor]:
        assert set(observations) == {"local_obs", "global_obs", "agent_mask"}
        assert torch.is_inference_mode_enabled()
        assert observations["local_obs"][0, 0, 0] == len(starts)
        starts.append(episode_starts.tolist())
        return {}

    output = record_policy(
        policy,
        "SwarmBots-WallEasy-v0",
        video_folder=tmp_path,
        num_episodes=5,
        max_parallel_episodes=2,
        seed=123,
        width=320,
        height=240,
        camera="overview",
        frame_stride=2,
    )
    assert output == tmp_path
    assert environment_options["num_envs"] == 2
    assert env.reset_seed == 123
    assert env.recording_options["num_episodes"] == 5
    assert env.recording_options["max_parallel_episodes"] == 2
    assert env.recording_options["camera"] == "overview"
    assert env.recording_options["width"] == 320
    assert env.recording_options["height"] == 240
    assert env.recording_options["frame_stride"] == 2
    assert starts == [[True, True], [False, False], [True, False], [False, True]]
    assert env.closed


def test_recording_closes_environment_when_policy_fails(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    env = _RecordingEnv()
    monkeypatch.setattr(recording, "make_env", lambda *args, **kwargs: env)

    def policy(observations: Any, episode_starts: torch.Tensor) -> dict[str, torch.Tensor]:
        raise RuntimeError("actor failed")

    with pytest.raises(RuntimeError, match="actor failed"):
        record_policy(policy, "SwarmBots-WallEasy-v0", video_folder=tmp_path)
    assert env.closed


def test_record_command_requires_checkpoint_variant() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main(["record", "SwarmBots-WallEasy-v0", "--checkpoint", "policy.pt"])


@pytest.mark.parametrize("exploration_noise", ["0.0", "0.4"])
def test_record_command_forwards_checkpoint_exploration_noise(
    exploration_noise: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    options: dict[str, Any] = {}

    def record(*args: Any, **kwargs: Any) -> Path:
        options.update(kwargs)
        return tmp_path

    monkeypatch.setattr("swarmbots.learn.record_checkpoint", record)
    cli.main([
        "record", "SwarmBots-WallEasy-v0", "--checkpoint", "policy.pt", "--variant", "matd3_mlp",
        "--stochastic", "--exploration-noise", exploration_noise, "--device", "cpu", "--output", str(tmp_path),
    ])
    assert options["exploration_noise"] == float(exploration_noise)
    assert options["deterministic"] is False


def test_record_command_rejects_exploration_noise_for_custom_factories() -> None:
    with pytest.raises(SystemExit, match="2"):
        cli.main([
            "record", "SwarmBots-WallEasy-v0", "--policy", "unused:factory", "--exploration-noise", "0.4",
        ])


def test_record_command_loads_custom_factory(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = ModuleType("recording_test_policy")
    factory_options: dict[str, Any] = {}
    calls: list[dict[str, Any]] = []

    def factory(**kwargs: Any) -> object:
        factory_options.update(kwargs)
        return "custom-policy"

    def record(policy: object, benchmark_id: str, **kwargs: Any) -> Path:
        assert policy == "custom-policy"
        assert benchmark_id == "SwarmBots-Bridge-v0"
        calls.append(kwargs)
        return tmp_path

    module.factory = factory
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(recording, "record_policy", record)
    cli.main(
        [
            "record",
            "SwarmBots-Bridge-v0",
            "--policy",
            "recording_test_policy:factory",
            "--output",
            str(tmp_path),
            "--device",
            "cpu",
            "--episodes",
            "3",
            "--parallel",
            "2",
            "--stochastic",
            "--camera",
            "side",
            "--scenario-kwargs",
            '{"reset_settle_time": 0}',
        ]
    )
    assert factory_options == {
        "benchmark_id": "SwarmBots-Bridge-v0",
        "device": torch.device("cpu"),
        "seed": 1_000,
        "deterministic": False,
    }
    assert calls[0]["num_episodes"] == 3
    assert calls[0]["max_parallel_episodes"] == 2
    assert calls[0]["camera"] == "side"
    assert calls[0]["scenario_kwargs"] == {"reset_settle_time": 0}


def test_checkpoint_mismatch_closes_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import swarmbots.learn.checkpoint_recording as checkpoint_recording

    env = _RecordingEnv()
    monkeypatch.setattr(checkpoint_recording, "_make_policy_env", lambda *args, **kwargs: (env, torch.nn.Linear(2, 1)))
    checkpoint_path = tmp_path / "wrong-architecture.pt"
    torch.save({"policy_state_dict": {"other.weight": torch.zeros(1)}}, checkpoint_path)
    with pytest.raises(RuntimeError, match="Missing key"):
        record_checkpoint(checkpoint_path, "SwarmBots-WallEasy-v0", "mat_qcx", video_folder=tmp_path)
    assert env.closed


@pytest.fixture
def single_torch_thread() -> Iterator[None]:
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def checkpoint_construction_options() -> dict[str, Any]:
    return {
        "device": "cpu",
        "episode_length": 4,
        "scenario_kwargs": {
            "unit_start_locations": MJWPreConnectedUnitLocationsConfig(
                num_units=2,
                pool_seeds=(42000,),
                max_radius=1.5,
                z_pos=0.5,
            ),
            "compile_reward_kernel": False,
            "reset_settle_time": 0,
        },
        "env_kwargs": {"compile_tensor_operations": False},
        "policy_kwargs": {
            "enc_d_model": 32, "dec_d_model": 16,
            "baseline_critic_hidden_dims": (32, 32),
            "baseline_critic_element_hidden_dims": (32, 32),
        },
    }


@pytest.mark.integration
@pytest.mark.parametrize("source", ["override", "checkpoint"])
@pytest.mark.parametrize("exploration_noise", [-0.1, float("nan"), float("inf")])
def test_invalid_recording_exploration_noise_closes_environment(
    source: str,
    exploration_noise: float,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_torch_thread: None,
    checkpoint_construction_options: dict[str, Any],
) -> None:
    import swarmbots.learn.checkpoint_recording as checkpoint_recording

    checkpoint_path = tmp_path / "model.pt"
    torch.save({"policy_hyper_parameters": {"exploration_noise": exploration_noise}}, checkpoint_path)
    original_make_policy_env = checkpoint_recording._make_policy_env
    closed: list[Any] = []

    def make_policy_env(*args: Any, **kwargs: Any) -> tuple[Any, Any]:
        env, policy = original_make_policy_env(*args, **kwargs)
        original_close = env.close

        def close() -> None:
            closed.append(env)
            original_close()

        monkeypatch.setattr(env, "close", close)
        return env, policy

    monkeypatch.setattr(checkpoint_recording, "_make_policy_env", make_policy_env)
    with pytest.raises(ValueError, match="exploration_noise must be finite and nonnegative"):
        record_checkpoint(
            checkpoint_path, "SwarmBots-WallEasy-v0", "matd3_mlp", video_folder=tmp_path,
            exploration_noise=exploration_noise if source == "override" else None,
            **checkpoint_construction_options,
        )
    assert len(closed) == 1


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["mat_ind", "tmasac_shared_encoder", "tmasac_slstm", "tmatd3"])
def test_checkpoint_recording_reconstructs_joint_observation_embedding(
    variant: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    single_torch_thread: None, checkpoint_construction_options: dict[str, Any],
) -> None:
    import swarmbots.learn.checkpoint_recording as checkpoint_recording

    options = {**checkpoint_construction_options, "policy_kwargs": {
        **checkpoint_construction_options["policy_kwargs"], "mat_joint_obs_embedding": True,
        "rmat_actor_d_model": 32, "transition_model_d_model": 16, "transition_model_nhead": 2,
    }}
    trainer = make_training("SwarmBots-PayloadPlane-v0", variant, num_envs=2, **options)
    checkpoint_path = tmp_path / "joint.pt"
    try:
        trainer.save(checkpoint_path)
        expected_policy = as_benchmark_policy(trainer)
        recorded = False

        def record(env: Any, policy: Any, **kwargs: Any) -> None:
            nonlocal recorded
            assert any(getattr(module, "joint_obs_embedding", False) for module in policy.policy.modules())
            torch.testing.assert_close(policy.policy.state_dict(), trainer.policy.state_dict(), rtol=0, atol=0)
            observations, _ = env.reset(seed=17)
            starts = torch.ones(2, dtype=torch.bool)
            expected_actions = expected_policy(observations, starts)
            actual_actions = policy(observations, starts)
            torch.testing.assert_close(actual_actions, expected_actions, rtol=0, atol=0)
            recorded = True

        monkeypatch.setattr(checkpoint_recording, "_record_episodes", record)
        record_checkpoint(checkpoint_path, "SwarmBots-PayloadPlane-v0", variant,
                          video_folder=tmp_path / "videos", num_episodes=2, **options)
        assert recorded
    finally:
        trainer.env.close()


def test_recording_rejects_exploration_noise_for_stochastic_policies(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    import swarmbots.learn.checkpoint_recording as checkpoint_recording

    env = _RecordingEnv()
    monkeypatch.setattr(checkpoint_recording, "_make_policy_env", lambda *args, **kwargs: (env, torch.nn.Linear(2, 1)))
    checkpoint_path = tmp_path / "model.pt"
    torch.save({}, checkpoint_path)
    with pytest.raises(ValueError, match="only supported for DDPG/TD3"):
        record_checkpoint(
            checkpoint_path, "SwarmBots-WallEasy-v0", "masac_mlp", video_folder=tmp_path, exploration_noise=0.4,
        )
    assert env.closed


@pytest.mark.integration
@pytest.mark.parametrize("variant,recurrent", [
    ("maddpg_mlp", False), ("matd3_deepset", False), ("tmatd3", False),
    ("matd3_deepset", True), ("tmatd3", True),
])
@pytest.mark.parametrize("exploration_noise", [0.0, 0.4])
@pytest.mark.parametrize("deterministic", [False, True])
def test_checkpoint_recording_preserves_custom_exploration_noise(
    variant: str,
    recurrent: bool,
    exploration_noise: float,
    deterministic: bool,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_torch_thread: None,
    checkpoint_construction_options: dict[str, Any],
) -> None:
    import swarmbots.learn.checkpoint_recording as checkpoint_recording

    construction_options = dict(checkpoint_construction_options)
    construction_options["policy_kwargs"] = dict(construction_options["policy_kwargs"])
    construction_options["policy_kwargs"]["td3_recurrent_actor"] = recurrent
    algorithm_options = {
        "exploration_noise": exploration_noise, "learning_starts": 0,
        "batch_size": 2, "buffer_capacity_per_env": 16,
    }
    if recurrent:
        algorithm_options.update(burn_in_steps=1, learning_steps=3, temporal_state_store_interval=1)
    trainer = make_training(
        "SwarmBots-WallEasy-v0", variant, num_envs=2, use_nop=False,
        algorithm_kwargs=algorithm_options,
        **construction_options,
    )
    checkpoint_path = tmp_path / "model.pt"
    try:
        trainer.save(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        # Recording must also restore inference settings when aligning compiled keys.
        checkpoint["policy_state_dict"] = {
            f"_orig_mod.{key}": value for key, value in checkpoint["policy_state_dict"].items()
        }
        torch.save(checkpoint, checkpoint_path)
        expected_policy = as_benchmark_policy(trainer, deterministic=deterministic)
        recorded = False

        def record(env: Any, policy: Any, **kwargs: Any) -> None:
            nonlocal recorded
            assert policy.policy.exploration_noise == exploration_noise
            assert policy.deterministic is deterministic
            observations, _ = env.reset(seed=17)
            starts = torch.ones(2, dtype=torch.bool)
            with patch("torch.randn_like", side_effect=torch.ones_like) as noise:
                expected_actions = expected_policy(observations, starts)
                actual_actions = policy(observations, starts)
            assert noise.call_count == (2 if exploration_noise > 0 and not deterministic else 0)
            for key in expected_actions:
                torch.testing.assert_close(actual_actions[key], expected_actions[key], rtol=0, atol=0)
            recorded = True

        monkeypatch.setattr(checkpoint_recording, "_record_episodes", record)
        record_checkpoint(
            checkpoint_path, "SwarmBots-WallEasy-v0", variant, video_folder=tmp_path / "videos",
            num_episodes=2, use_nop=False, deterministic=deterministic,
            **construction_options,
        )
        assert recorded
        assert trainer.policy.exploration_noise == exploration_noise
    finally:
        trainer.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("checkpoint_format", ["full", "legacy", "state_dict"])
@pytest.mark.parametrize("exploration_noise", [None, 0.0, 0.4])
def test_checkpoint_recording_noise_overrides_and_legacy_formats(
    checkpoint_format: str,
    exploration_noise: float | None,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_torch_thread: None,
    checkpoint_construction_options: dict[str, Any],
) -> None:
    import swarmbots.learn.checkpoint_recording as checkpoint_recording

    trainer = make_training(
        "SwarmBots-WallEasy-v0", "matd3_mlp", num_envs=2,
        algorithm_kwargs={
            "exploration_noise": 0.7, "learning_starts": 0, "batch_size": 2, "buffer_capacity_per_env": 16,
        },
        **checkpoint_construction_options,
    )
    checkpoint_path = tmp_path / "model.pt"
    try:
        trainer.save(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        if checkpoint_format == "legacy":
            del checkpoint["policy_hyper_parameters"]
        elif checkpoint_format == "state_dict":
            checkpoint = checkpoint["policy_state_dict"]
        torch.save(checkpoint, checkpoint_path)
        expected_noise = (0.7 if checkpoint_format == "full" else 0.1) if exploration_noise is None else exploration_noise
        trainer.policy.exploration_noise = expected_noise
        expected_policy = as_benchmark_policy(trainer, deterministic=False)
        recorded = False

        def record(env: Any, policy: Any, **kwargs: Any) -> None:
            nonlocal recorded
            assert policy.policy.exploration_noise == expected_noise
            observations, _ = env.reset(seed=17)
            starts = torch.ones(2, dtype=torch.bool)
            with patch("torch.randn_like", side_effect=torch.ones_like) as noise:
                expected_actions = expected_policy(observations, starts)
                actual_actions = policy(observations, starts)
            assert noise.call_count == (2 if expected_noise > 0 else 0)
            for key in expected_actions:
                torch.testing.assert_close(actual_actions[key], expected_actions[key], rtol=0, atol=0)
            recorded = True

        monkeypatch.setattr(checkpoint_recording, "_record_episodes", record)
        record_checkpoint(
            checkpoint_path, "SwarmBots-WallEasy-v0", "matd3_mlp", video_folder=tmp_path / "videos",
            num_episodes=2, deterministic=False, exploration_noise=exploration_noise,
            **checkpoint_construction_options,
        )
        assert recorded
    finally:
        trainer.env.close()


@pytest.mark.integration
@pytest.mark.parametrize("variant", [
    "mat_qcx", "mat_ind_lstm", "tmasac_slstm",
    "mappo_mlp",
    "maddpg_mlp", "matd3_deepset", "masac_mlp", "tmatd3",
])
def test_checkpoint_recording_restores_actor_and_normalization_and_writes_complete_episodes(
    variant: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_torch_thread: None,
    checkpoint_construction_options: dict[str, Any],
) -> None:
    import swarmbots.learn.checkpoint_recording as checkpoint_recording
    import swarmbots.mjw_env.mjw_live_recording as live_recording

    construction_options = checkpoint_construction_options
    trainer = make_training("SwarmBots-WallEasy-v0", variant, num_envs=2, **construction_options)
    checkpoint_path = tmp_path / "model.pt"
    try:
        env = trainer.env
        while hasattr(env, "env"):
            if isinstance(env, TorchFeatureWiseObsNormWrapper) and env.obs_rms is not None:
                env.obs_rms.mean.fill_(0.5)
                env.obs_rms.var.fill_(2.0)
                env.obs_rms.count.fill_(100)
            env = env.env
        trainer.save(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        checkpoint["policy_state_dict"] = {
            f"_orig_mod.{key}": value for key, value in checkpoint["policy_state_dict"].items()
        }
        torch.save(checkpoint, checkpoint_path)
        expected_policy = as_benchmark_policy(trainer)
        policy_calls = 0
        original_record = checkpoint_recording._record_episodes

        def record(env: Any, policy: Any, **kwargs: Any) -> None:
            def checked_policy(
                observations: Mapping[str, torch.Tensor], starts: torch.Tensor
            ) -> Mapping[str, torch.Tensor]:
                nonlocal policy_calls
                expected_actions = expected_policy(observations, starts)
                actions = policy(observations, starts)
                for key in expected_actions:
                    torch.testing.assert_close(actions[key], expected_actions[key])
                policy_calls += 1
                return actions

            original_record(env, checked_policy, **kwargs)

        written_frames: list[list[np.ndarray]] = []

        def write_video(*, frames: list[np.ndarray], output_path: Path, fps: int) -> None:
            assert fps == 15
            written_frames.append(frames)
            output_path.write_bytes(b"encoded video")

        monkeypatch.setattr(checkpoint_recording, "_record_episodes", record)
        monkeypatch.setattr(live_recording, "_write_video_file", write_video)
        output = record_checkpoint(
            checkpoint_path,
            "SwarmBots-WallEasy-v0",
            variant,
            video_folder=tmp_path / "videos",
            num_episodes=3,
            max_parallel_episodes=2,
            width=160,
            height=120,
            frame_stride=2,
            **construction_options,
        )
        assert len(list(output.glob("*.mp4"))) == 3
        assert policy_calls == 8
        assert len(written_frames) == 3
        for frames in written_frames:
            assert len(frames) == 3
            assert all(frame.shape == (120, 160, 3) for frame in frames)
    finally:
        trainer.env.close()
