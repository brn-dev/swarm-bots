from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def draw_accumulated_reward(frame: np.ndarray, accumulated_reward: float) -> np.ndarray:
    if frame.ndim != 3 or frame.shape[2] < 3:
        return frame

    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image, "RGBA")
    font = ImageFont.load_default()
    label = f"{accumulated_reward:.3f}"

    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width = right - left
    text_height = bottom - top
    margin = 8

    x = image.width - text_width - margin
    y = image.height - text_height - margin

    draw.rectangle(
        [(x - 6, y - 4), (x + text_width + 6, y + text_height + 4)],
        fill=(0, 0, 0, 160),
    )
    draw.text((x, y), label, font=font, fill=(255, 255, 255, 255))
    return np.asarray(image)
