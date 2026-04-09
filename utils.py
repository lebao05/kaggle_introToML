"""Render text -> image va convert image -> pixel coordinates.

COMPLIANCE NOTE (assignment requirement):
  - Context text NEVER enters the model as a text string.
  - It is always rendered to a PIL image first (render_text_to_image).
  - The model receives ONLY pixel_values (visual encoding), never input_ids of context.
  - Prompts below describe the TASK FORMAT only; they do NOT repeat or encode context content.
  - This ensures the model must perform visual understanding to solve the task,
    not text-to-text completion.
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFont


# ── Language-conditioned visual continuation prompts ──────────────────────────
# Framing: "visual content that follows" — not "read text and complete it"
# The model must look at the IMAGE to understand what's shown, not receive text.
LANG_PROMPTS = {
    "English": (
        "The image shows the visual rendering of a passage. "
        "Generate the continuation that would naturally follow "
        "what is visually shown."
    ),
    "French": (
        "L'image montre le rendu visuel d'un passage. "
        "Générez la suite qui suivrait naturellement "
        "ce qui est visuellement affiché."
    ),
    "German": (
        "Das Bild zeigt die visuelle Darstellung eines Textes. "
        "Generieren Sie die Fortsetzung, die natürlich auf "
        "das visuell Gezeigte folgen würde."
    ),
    "Italian": (
        "L'immagine mostra la rappresentazione visiva di un brano. "
        "Genera la continuazione che seguirebbe naturalmente "
        "ciò che è mostrato visivamente."
    ),
    "Spanish": (
        "La imagen muestra la representación visual de un pasaje. "
        "Genera la continuación que seguiría naturalmente "
        "lo que se muestra visualmente."
    ),
}
DEFAULT_PROMPT = (
    "The image shows the visual rendering of a passage. "
    "Generate the continuation that would naturally follow "
    "what is visually shown."
)


def get_lang_prompt(language: str) -> str:
    """Task-format prompt only. Does NOT encode context content as text."""
    return LANG_PROMPTS.get(language, DEFAULT_PROMPT)


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
    """Render text thanh anh grayscale (mode L), chu trang nen den, word-wrap."""
    font = _get_font(font_size)
    cw   = max_width - 2 * padding

    lines, cur = [], ""
    for word in text.split(" "):
        test = f"{cur} {word}".strip()
        bbox = font.getbbox(test)
        if bbox[2] - bbox[0] <= cw:
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
    """Chuyen PIL image -> pixel coordinates (sample_id, row, col)."""
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
