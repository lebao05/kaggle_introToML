"""Render text → ảnh và convert ảnh → pixel coordinates."""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def _get_font(font_size: int = 20) -> ImageFont.FreeTypeFont:
    paths = (
        ["C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/arial.ttf",
         "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf"]
        if os.name == "nt" else
        ["/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
         "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
         "C:/Windows/Fonts/times.ttf", "C:/Windows/Fonts/arial.ttf"]
    )
    for p in paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, font_size)
    return ImageFont.load_default()


def render_text_to_image(text: str, max_width: int,
                         font_size: int = 18, padding: int = 12) -> Image.Image:
    """Render text thành ảnh grayscale (mode L), chữ trắng nền đen, word-wrap."""
    font = _get_font(font_size)
    cw   = max_width - 2 * padding

    lines, cur = [], ""
    for word in text.split(" "):
        test = f"{cur} {word}".strip()
        if font.getbbox(test)[2] - font.getbbox(test)[0] <= cw:
            cur = test
        else:
            if cur:
                lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    if not lines:
        lines = ["."]

    lh  = font_size + 4
    img = Image.new("L", (max_width, len(lines) * lh + 2 * padding), 0)
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=255, font=font)
        y += lh
    return img


def image_to_pixels(img: Image.Image, sample_id: int,
                    threshold: int = 127) -> np.ndarray:
    """Chuyển PIL image → pixel coordinates (sample_id, row, col)."""
    arr = np.array(img)
    rows, cols = np.where(arr > threshold)
    if len(rows) == 0:
        print(f"[WARN] blank prediction sample_id={sample_id}", file=sys.stderr)
        return np.array([[sample_id, 0, 0]], dtype=np.int64)
    return np.stack([
        np.full(len(rows), sample_id, dtype=np.int64),
        rows.astype(np.int64),
        cols.astype(np.int64),
    ], axis=1)
