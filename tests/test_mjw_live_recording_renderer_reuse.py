from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from swarmbots.mjw_env.mjw_live_recording import MJWLiveEpisodeRecorder, MJWRecordingConfig, MJWWorldSnapshot


class _DummyScenario:
    def build_model(self) -> object:
        return _FakeModel()

    def get_default_recording_camera_config(self) -> None:
        return None


class _FakeModel:
    def __init__(self) -> None:
        self.vis = _FakeVis()
        self.stat = _FakeStat()


class _FakeVis:
    def __init__(self) -> None:
        self.global_ = _FakeGlobalVisual()


class _FakeGlobalVisual:
    def __init__(self) -> None:
        self.offwidth = 640
        self.offheight = 480


class _FakeStat:
    def __init__(self) -> None:
        self.center = np.zeros(3, dtype=np.float64)
        self.extent = 1.0


class _FakeMjData:
    init_count = 0

    def __init__(self, model: object) -> None:
        _ = model
        type(self).init_count += 1
        self.qpos = np.zeros(1, dtype=np.float64)
        self.qvel = np.zeros(1, dtype=np.float64)
        self.eq_active = np.zeros(1, dtype=np.float64)
        self.mocap_pos = np.zeros(1, dtype=np.float64)
        self.mocap_quat = np.zeros(1, dtype=np.float64)
        self.time = 0.0


class _FakeRenderer:
    init_count = 0
    close_count = 0

    def __init__(self, model: object, *, height: int, width: int) -> None:
        _ = (model, height, width)
        type(self).init_count += 1
        self.scene = object()

    def update_scene(self, data: _FakeMjData, camera: object) -> None:
        _ = (data, camera)

    def render(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)

    def close(self) -> None:
        type(self).close_count += 1


class _FailingRenderer:
    def __init__(self, model: object, *, height: int, width: int) -> None:
        _ = (model, height, width)
        raise ValueError("renderer failed")


class MJWLiveRecordingRendererReuseTests(unittest.TestCase):
    def test_start_uses_a_single_shared_renderer(self) -> None:
        _FakeMjData.init_count = 0
        _FakeRenderer.init_count = 0
        _FakeRenderer.close_count = 0

        scenario = _DummyScenario()
        recorder = MJWLiveEpisodeRecorder(scenario=scenario)
        snapshots_by_world = {
            0: MJWWorldSnapshot(
                qpos=np.zeros(1, dtype=np.float64),
                qvel=np.zeros(1, dtype=np.float64),
                eq_active=np.zeros(1, dtype=np.float64),
                mocap_pos=np.zeros(1, dtype=np.float64),
                mocap_quat=np.zeros(1, dtype=np.float64),
                time=0.0,
            ),
            1: MJWWorldSnapshot(
                qpos=np.ones(1, dtype=np.float64),
                qvel=np.ones(1, dtype=np.float64),
                eq_active=np.ones(1, dtype=np.float64),
                mocap_pos=np.ones(1, dtype=np.float64),
                mocap_quat=np.ones(1, dtype=np.float64),
                time=1.0,
            ),
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MJWRecordingConfig(
                video_folder=Path(tmpdir) / "videos",
                video_name_prefix="test",
                num_episodes=2,
                max_parallel_episodes=2,
                fps=30,
                fps_mode="fixed",
                frame_stride=1,
                width=16,
                height=16,
                camera=0,
            )

            with (
                patch("swarmbots.mjw_env.mjw_live_recording.mujoco.MjData", _FakeMjData),
                patch("swarmbots.mjw_env.mjw_live_recording.mujoco.Renderer", _FakeRenderer),
                patch("swarmbots.mjw_env.mjw_live_recording.mujoco.mj_forward", lambda model, data: None),
                patch("swarmbots.mjw_env.mjw_live_recording.draw_accumulated_reward", lambda frame, *args, **kwargs: frame),
            ):
                recorder.start(
                    config=config,
                    episode_start_world_idx=np.array([0, 1], dtype=np.int64),
                    snapshots_by_world=snapshots_by_world,
                )
                status = recorder.get_status()
                recorder.close()

        self.assertEqual(_FakeMjData.init_count, 1)
        self.assertEqual(_FakeRenderer.init_count, 1)
        self.assertEqual(_FakeRenderer.close_count, 1)
        self.assertTrue(status["active"])
        self.assertEqual(status["active_worlds"], [0, 1])

    def test_start_failure_does_not_leave_recorder_active(self) -> None:
        recorder = MJWLiveEpisodeRecorder(scenario=_DummyScenario())

        with tempfile.TemporaryDirectory() as tmpdir:
            config = MJWRecordingConfig(
                video_folder=Path(tmpdir) / "videos",
                video_name_prefix="test",
                num_episodes=1,
                max_parallel_episodes=1,
                fps=30,
                fps_mode="fixed",
                frame_stride=1,
                width=1280,
                height=720,
                camera=0,
            )

            with (
                patch("swarmbots.mjw_env.mjw_live_recording.mujoco.MjData", _FakeMjData),
                patch("swarmbots.mjw_env.mjw_live_recording.mujoco.Renderer", _FailingRenderer),
            ):
                with self.assertRaisesRegex(ValueError, "renderer failed"):
                    recorder.start(
                        config=config,
                        episode_start_world_idx=np.array([0], dtype=np.int64),
                        snapshots_by_world={},
                    )

        self.assertFalse(recorder.is_active())
        self.assertFalse(recorder.get_status()["active"])


if __name__ == "__main__":
    unittest.main()
