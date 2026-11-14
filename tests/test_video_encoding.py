from pathlib import Path

import imageio_ffmpeg
import numpy as np
import pytest

from swarmbots.mjw_env.mjw_live_recording import _write_video_file
from swarmbots.utils.recording_overlay import draw_accumulated_reward


@pytest.mark.parametrize("width, height", [(160, 120), (162, 122), (161, 121)])
@pytest.mark.parametrize("frame_count", [1, 4])
def test_recording_preserves_frame_count_rate_resolution_and_rgb_order(
    tmp_path: Path, width: int, height: int, frame_count: int,
) -> None:
    colors = [(210, 30, 50), (40, 210, 30), (30, 40, 210), (110, 120, 50)]
    frames = [np.full((height, width, 3), color, dtype=np.uint8) for color in colors[:frame_count]]
    output_path = tmp_path / "recording.mp4"
    _write_video_file(frames=frames, output_path=output_path, fps=13)
    reader = imageio_ffmpeg.read_frames(str(output_path))
    try:
        metadata = next(reader)
        decoded_frames = list(reader)
    finally:
        reader.close()
    assert metadata["size"] == (width, height)
    assert metadata["fps"] == pytest.approx(13.)
    assert len(decoded_frames) == frame_count
    for frame, color in zip(decoded_frames, colors[:frame_count], strict=True):
        pixels = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
        np.testing.assert_allclose(pixels.mean(axis=(0, 1)), color, atol=5.)


def test_reward_overlay_draws_statistics_without_changing_the_source_frame() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    annotated = draw_accumulated_reward(
        frame, 12.25, accumulated_reward_terms={"forward": 11., "towards_payload": 1.25},
    )
    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert annotated.any()
    assert not frame.any()
