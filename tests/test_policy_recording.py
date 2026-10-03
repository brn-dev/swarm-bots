from __future__ import annotations

from collections.abc import Iterator, Mapping
from pathlib import Path
from types import ModuleType
from typing import Any
import sys

import numpy as np
import pytest
import torch

import swarmbots.benchmark.cli as cli
import swarmbots.benchmark.recording as recording
from swarmbots import record_policy
from swarmbots.learn import as_benchmark_policy, make_training, record_checkpoint
from swarmbots.learn.checkpointing import capture_env_state
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


@pytest.mark.integration
@pytest.mark.parametrize("variant", ["mat_qcx", "mat_ind_lstm", "tmasac_slstm"])
def test_checkpoint_recording_restores_actor_and_normalization_and_writes_complete_episodes(
    variant: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    single_torch_thread: None,
) -> None:
    import swarmbots.learn.checkpoint_recording as checkpoint_recording
    import swarmbots.mjw_env.mjw_live_recording as live_recording

    scenario_kwargs = {
        "unit_start_locations": MJWPreConnectedUnitLocationsConfig(
            num_units=2,
            pool_seeds=(42000,),
            max_radius=1.5,
            z_pos=0.5,
        ),
        "compile_reward_kernel": False,
        "reset_settle_time": 0,
    }
    construction_options = {
        "device": "cpu",
        "episode_length": 4,
        "scenario_kwargs": scenario_kwargs,
        "env_kwargs": {"compile_tensor_operations": False},
        "policy_kwargs": {"enc_d_model": 32, "dec_d_model": 16},
    }
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
        torch.save(
            {
                "policy_state_dict": {f"_orig_mod.{key}": value for key, value in trainer.policy.state_dict().items()},
                "env_state": capture_env_state(trainer.env),
            },
            checkpoint_path,
        )
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
