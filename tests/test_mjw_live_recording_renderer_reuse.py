from __future__ import annotations

from contextlib import contextmanager
from concurrent.futures import Future
import shutil
import unittest
from pathlib import Path
from unittest.mock import patch
import uuid

import numpy as np

from swarmbots.mjw_env.mjw_live_recording import MJWLiveEpisodeRecorder, MJWRecordingConfig, MJWWorldSnapshot


@contextmanager
def _writable_temp_directory() -> object:
    root = Path(__file__).resolve().parents[1] / ".tmp" / "recording-tests"
    root.mkdir(parents=True, exist_ok=True)
    path = root / uuid.uuid4().hex
    path.mkdir()
    try:
        yield str(path)
    finally:
        shutil.rmtree(path, ignore_errors=True)


class _DummyScenario:
    def __init__(self) -> None:
        self.model_build_count = 0

    def build_model(self) -> object:
        self.model_build_count += 1
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


class _FailingSceneRenderer(_FakeRenderer):
    def update_scene(self, data: _FakeMjData, camera: object) -> None:
        if camera == "missing-camera" and data.time > 0:
            raise ValueError("camera does not exist")
        super().update_scene(data, camera)


class MJWLiveRecordingRendererReuseTests(unittest.TestCase):
    def test_unused_recorder_does_not_build_a_recording_model(self) -> None:
        scenario = _DummyScenario()
        recorder = MJWLiveEpisodeRecorder(scenario=scenario)
        try:
            self.assertFalse(recorder.get_status()["active"])
            self.assertEqual(scenario.model_build_count, 0)
        finally:
            recorder.close()
        self.assertEqual(scenario.model_build_count, 0)

    def test_writer_failure_still_shuts_down_and_can_be_closed_again(self) -> None:
        recorder = MJWLiveEpisodeRecorder(scenario=_DummyScenario())
        failed = Future()
        failed.set_exception(OSError("disk full"))
        recorder._writer_futures.append(failed)
        with self.assertRaisesRegex(OSError, "disk full"):
            recorder.close()
        self.assertTrue(recorder._writer_executor._shutdown)
        self.assertEqual(recorder._writer_futures, [])
        recorder.close()

    def test_start_uses_a_single_shared_renderer(self) -> None:
        _FakeMjData.init_count = 0
        _FakeRenderer.init_count = 0
        _FakeRenderer.close_count = 0

        scenario = _DummyScenario()
        recorder = MJWLiveEpisodeRecorder(scenario=scenario)
        self.assertEqual(scenario.model_build_count, 0)
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

        with _writable_temp_directory() as tmpdir:
            config = MJWRecordingConfig(
                video_folder=Path(tmpdir) / "videos",
                video_name_prefix="test",
                num_episodes=2,
                max_parallel_episodes=2,
                fps=30,
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
                old_slot = recorder._active_slots_by_world[0]
                other_slot = recorder._active_slots_by_world[1]
                old_slot.accumulated_reward = 7.0
                old_slot.step_count = 1
                old_slot.frames.append(np.zeros((2, 2, 3), dtype=np.uint8))
                recorder.on_explicit_resets(world_idx=np.array([0]))
                recorder.on_episode_starts(world_idx=np.array([0]), snapshots_by_world=snapshots_by_world)
                fresh_slot = recorder._active_slots_by_world[0]
                self.assertIsNot(fresh_slot, old_slot)
                self.assertIs(recorder._active_slots_by_world[1], other_slot)
                self.assertEqual(fresh_slot.accumulated_reward, 0.0)
                self.assertEqual(fresh_slot.step_count, 0)
                self.assertEqual(len(fresh_slot.frames), 1)
                self.assertGreater(fresh_slot.episode_idx, other_slot.episode_idx)
                self.assertEqual(recorder._episodes_started, 2)
                status = recorder.get_status()
                recorder.close()

        self.assertEqual(_FakeMjData.init_count, 1)
        self.assertEqual(_FakeRenderer.init_count, 1)
        self.assertEqual(_FakeRenderer.close_count, 1)
        self.assertEqual(scenario.model_build_count, 1)
        self.assertTrue(status["active"])
        self.assertCountEqual(status["active_worlds"], [0, 1])

    def test_start_failure_does_not_leave_recorder_active(self) -> None:
        recorder = MJWLiveEpisodeRecorder(scenario=_DummyScenario())

        with _writable_temp_directory() as tmpdir:
            config = MJWRecordingConfig(
                video_folder=Path(tmpdir) / "videos",
                video_name_prefix="test",
                num_episodes=1,
                max_parallel_episodes=1,
                fps=30,
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
        recorder.close()

    def test_render_failure_during_start_discards_partial_slots_and_allows_retry(self) -> None:
        _FakeRenderer.init_count = 0
        _FakeRenderer.close_count = 0
        _FailingSceneRenderer.init_count = 0
        _FailingSceneRenderer.close_count = 0
        scenario = _DummyScenario()
        recorder = MJWLiveEpisodeRecorder(scenario=scenario)
        snapshots = {
            world: MJWWorldSnapshot(
                qpos=np.zeros(1), qvel=np.zeros(1), eq_active=np.zeros(1),
                mocap_pos=np.zeros(1), mocap_quat=np.zeros(1), time=float(world),
            )
            for world in range(2)
        }
        with _writable_temp_directory() as tmpdir:
            config = MJWRecordingConfig(Path(tmpdir), "test", 2, 2, 30, 16, 16, "missing-camera")
            with (
                patch("swarmbots.mjw_env.mjw_live_recording.mujoco.MjData", _FakeMjData),
                patch("swarmbots.mjw_env.mjw_live_recording.mujoco.Renderer", _FailingSceneRenderer),
                patch("swarmbots.mjw_env.mjw_live_recording.mujoco.mj_forward", lambda model, data: None),
                patch("swarmbots.mjw_env.mjw_live_recording.draw_accumulated_reward", lambda frame, *args, **kwargs: frame),
                patch("swarmbots.mjw_env.mjw_live_recording._write_video_file") as write_video,
            ):
                try:
                    with self.assertRaisesRegex(ValueError, "camera does not exist"):
                        recorder.start(config=config, episode_start_world_idx=np.array([0, 1]), snapshots_by_world=snapshots)
                    status = recorder.get_status()
                    self.assertFalse(status["active"])
                    self.assertEqual(status["active_worlds"], [])
                    self.assertEqual(status["episodes_started"], 0)
                    self.assertEqual(_FailingSceneRenderer.close_count, 1)
                    config.camera = 0
                    recorder.start(config=config, episode_start_world_idx=np.array([0, 1]), snapshots_by_world=snapshots)
                    recorder.record_step(
                        rewards=np.zeros(2), reward_terms=None, dones=np.ones(2, dtype=bool),
                        unstable_mask=np.zeros(2, dtype=bool), snapshots_by_world=snapshots,
                    )
                finally:
                    recorder.close()
                self.assertEqual(write_video.call_count, 2)
                self.assertEqual(recorder.get_status()["episodes_completed"], 2)
                self.assertEqual(_FailingSceneRenderer.close_count, 2)
                self.assertEqual(scenario.model_build_count, 1)


if __name__ == "__main__":
    unittest.main()
