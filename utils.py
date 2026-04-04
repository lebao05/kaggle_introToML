"""Utilities: text rendering and pixel conversion."""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ============================================================
# Text Rendering
# ============================================================

def _get_font(font_size=20):
    """Try to load a serif font, fallback to default."""
    if os.name == "nt":
        font_paths = [
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/arial.ttf",
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        ]
    else:
        font_paths = [
            "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
            "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
            "C:/Windows/Fonts/times.ttf",
            "C:/Windows/Fonts/arial.ttf",
        ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, font_size)
    return ImageFont.load_default()


def render_text_to_image(text, max_width, font_size=20, padding=10):
    """Render text into a grayscale image with word wrapping.

    Returns a PIL Image (mode "L") with white text on black background.
    """
    font = _get_font(font_size)

    words = text.split(" ")
    lines = []
    current_line = ""
    content_width = max_width - 2 * padding

    for word in words:
        test = f"{current_line} {word}".strip()
        bbox = font.getbbox(test)
        w = bbox[2] - bbox[0]
        if w <= content_width:
            current_line = test
        else:
            if current_line:
                lines.append(current_line)
            current_line = word
    if current_line:
        lines.append(current_line)

    if not lines:
        lines = ["."]

    line_height = font_size + 4
    img_height = len(lines) * line_height + 2 * padding

    img = Image.new("L", (max_width, img_height), 0)
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=255, font=font)
        y += line_height

    return img


def render_target_to_image(text, max_width, font_size=20, padding=10):
    """Render target text to image."""
    return render_text_to_image(text, max_width, font_size, padding)


# ============================================================
# Pixel Conversion
# ============================================================

def image_to_pixels(img, sample_id, threshold=127):
    """Convert a PIL image to pixel coordinates (sample_id, row, col)."""
    arr = np.array(img)
    rows, cols = np.where(arr > threshold)

    if len(rows) == 0:
        print(
            f"[WARN] Blank rendered prediction for sample_id={sample_id}; "
            "fallback pixel (0,0) inserted.",
            file=sys.stderr,
        )
        return np.array([[sample_id, 0, 0]], dtype=np.int64)

    s_ids = np.full(len(rows), sample_id, dtype=np.int64)
    return np.stack([s_ids, rows.astype(np.int64), cols.astype(np.int64)], axis=1)
