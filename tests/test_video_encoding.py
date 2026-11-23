from pathlib import Path
import subprocess
import sys
from typing import Any

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
    assert metadata["codec"] == "h264"
    pixel_format = "yuv420p" if width % 2 == 0 and height % 2 == 0 else "yuv444p"
    assert metadata["pix_fmt"].split("(")[0] == pixel_format
    assert len(decoded_frames) == frame_count
    for frame, color in zip(decoded_frames, colors[:frame_count], strict=True):
        pixels = np.frombuffer(frame, dtype=np.uint8).reshape(height, width, 3)
        np.testing.assert_allclose(pixels.mean(axis=(0, 1)), color, atol=5.)


def test_failed_encoder_finalization_preserves_previous_video(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    output_path = tmp_path / "recording.mp4"
    output_path.write_bytes(b"existing video")
    frames = list(np.arange(48, dtype=np.uint8).reshape(2, 2, 4, 3))
    launch = subprocess.Popen
    failing_encoder = """
import pathlib, sys
frames = sys.stdin.buffer.read()
pathlib.Path(sys.argv[1]).write_bytes(b'incomplete video')
sys.exit(23 if frames == bytes(range(48)) else 24)
"""

    def launch_encoder(command: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
        return launch([command[0], "-c", failing_encoder, command[-1]], **kwargs)

    monkeypatch.setenv("IMAGEIO_FFMPEG_EXE", sys.executable)
    monkeypatch.setattr(subprocess, "Popen", launch_encoder)
    with pytest.raises(subprocess.CalledProcessError) as error:
        _write_video_file(frames=frames, output_path=output_path, fps=13)

    assert error.value.returncode == 23
    assert output_path.read_bytes() == b"existing video"
    assert set(tmp_path.iterdir()) == {output_path}


def test_reward_overlay_draws_statistics_without_changing_the_source_frame() -> None:
    frame = np.zeros((240, 320, 3), dtype=np.uint8)
    annotated = draw_accumulated_reward(
        frame, 12.25, accumulated_reward_terms={"forward": 11., "towards_payload": 1.25},
    )
    assert annotated.shape == frame.shape
    assert annotated.dtype == frame.dtype
    assert annotated.any()
    assert not frame.any()
