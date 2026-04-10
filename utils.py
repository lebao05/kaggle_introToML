import os
import random
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont


LANG_PROMPTS = {
    "English": "English: ",
    "French": "French: ",
    "German": "German: ",
    "Italian": "Italian: ",
    "Spanish": "Spanish: ",
}


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | os.PathLike) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_font(font_size: int = 18):
    font_paths = [
        "C:/Windows/Fonts/times.ttf",
        "C:/Windows/Fonts/arial.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            return ImageFont.truetype(path, font_size)
    return ImageFont.load_default()


def render_text_to_image(text: str, max_width: int, font_size: int = 18, padding: int = 12) -> Image.Image:
    text = str(text or "")
    font = get_font(font_size)
    content_width = max(16, int(max_width) - 2 * padding)

    lines = []
    current = ""
    for word in text.split(" "):
        probe = f"{current} {word}".strip()
        bbox = font.getbbox(probe)
        if bbox[2] - bbox[0] <= content_width:
            current = probe
        else:
            if current:
                lines.append(current)
            current = word
    if current:
        lines.append(current)
    if not lines:
        lines = ["."]

    line_h = font_size + 4
    img = Image.new("L", (int(max_width), len(lines) * line_h + 2 * padding), 0)
    draw = ImageDraw.Draw(img)
    y = padding
    for line in lines:
        draw.text((padding, y), line, fill=255, font=font)
        y += line_h
    return img


def fit_image_to_canvas(img: Image.Image, image_size: int, anchor: str = "bottom") -> Image.Image:
    if img.mode != "RGB":
        img = img.convert("RGB")
    canvas = Image.new("RGB", (image_size, image_size), (0, 0, 0))
    src = img

    if src.size[0] > image_size:
        src = src.crop((max(0, src.size[0] - image_size), 0, src.size[0], src.size[1]))
    if src.size[1] > image_size:
        if anchor == "bottom":
            src = src.crop((0, src.size[1] - image_size, src.size[0], src.size[1]))
        else:
            src = src.crop((0, 0, src.size[0], image_size))

    x = 0
    if anchor == "bottom":
        y = image_size - src.size[1]
    else:
        y = 0
    canvas.paste(src, (x, y))
    return canvas


def white_pixel_density(img: Image.Image, threshold: int = 127) -> float:
    arr = np.array(img.convert("L"))
    return float((arr > threshold).mean())


def image_to_pixels(img: Image.Image, sample_id: int, threshold: int = 127) -> np.ndarray:
    arr = np.array(img)
    rows, cols = np.where(arr > threshold)
    if len(rows) == 0:
        return np.array([[sample_id, 0, 0]], dtype=np.int64)
    return np.stack(
        [
            np.full(len(rows), sample_id, dtype=np.int64),
            rows.astype(np.int64),
            cols.astype(np.int64),
        ],
        axis=1,
    )


def clean_text(text: str) -> str:
    return " ".join((text or "").split()).strip()


def get_language_prompt(language: str) -> str:
    return LANG_PROMPTS.get(str(language), f"{language}: ")


def tail_context(text: str, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def clean_prediction(text: str, language: str) -> str:
    text = clean_text(text)
    lower = text.lower()
    for prompt in LANG_PROMPTS.values():
        prefix = prompt.lower()
        if lower.startswith(prefix):
            text = clean_text(text[len(prompt) :])
            lower = text.lower()
    for prefix in ["answer:", "response:", "output:", "assistant:"]:
        if lower.startswith(prefix):
            text = clean_text(text[len(prefix) :])
            lower = text.lower()
    lang_prompt = get_language_prompt(language)
    if text.startswith(lang_prompt):
        text = clean_text(text[len(lang_prompt) :])
    return text
