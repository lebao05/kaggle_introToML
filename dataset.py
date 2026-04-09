"""Dataset for direct pixel-space continuation prediction."""

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from config import LANG2ID
from utils import render_text_to_image


def _clean_split(context: str, target: str, window_frac: float = 0.05):
    """Move split boundary to nearby whitespace to avoid mid-word cuts."""
    full = context + target
    split = len(context)
    n = len(full)
    window = max(15, int(n * window_frac))
    lo = max(1, split - window)
    hi = min(n, split + window)
    spaces = [i for i in range(lo, hi) if full[i] == " "]
    if not spaces:
        return context, target
    best = min(spaces, key=lambda i: abs(i - split))
    new_ctx = full[:best].rstrip()
    new_tgt = full[best + 1 :].lstrip()
    if not new_ctx or not new_tgt:
        return context, target
    return new_ctx, new_tgt


def _pad_to_square(img: Image.Image) -> Image.Image:
    w, h = img.size
    s = max(w, h)
    out = Image.new(img.mode, (s, s), 0)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


def _render_rgb(text: str, max_width: int, font_size: int, padding: int) -> Image.Image:
    img = render_text_to_image(text, max_width, font_size, padding)
    return _pad_to_square(img).convert("RGB")


def _render_target_mask(text: str, max_width: int, config: dict) -> torch.Tensor:
    """Render target text and resize to fixed training mask size."""
    img = render_text_to_image(text, max_width, config["font_size"], config["padding"])
    img = img.resize((config["output_w"], config["output_h"]), Image.Resampling.NEAREST)
    arr = (np.array(img, dtype=np.float32) > 127).astype(np.float32)
    return torch.from_numpy(arr).unsqueeze(0)


def _normalize_pixel_values(x: torch.Tensor) -> torch.Tensor:
    """Collapse singleton batch/image dims from processor output to [3, H, W]."""
    while x.ndim > 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim == 4 and x.shape[0] == 1:
        x = x.squeeze(0)
    if x.ndim != 3:
        raise ValueError(f"Expected pixel_values with 3 dims after squeeze, got {tuple(x.shape)}")
    return x


class PixelTextDataset(Dataset):
    def __init__(self, df, processor, config: dict, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.cfg = config
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        context = str(row["context"])
        language = str(row["language"])
        max_width = int(row["max_width"])

        target = None
        if not self.is_test:
            target = str(row["target"])
            context, target = _clean_split(context, target)

        ctx_img = _render_rgb(context, max_width, self.cfg["font_size"], self.cfg["padding"])
        pixel_values = self.processor(
            images=ctx_img,
            return_tensors="pt",
            do_image_splitting=False,
        )["pixel_values"]
        pixel_values = _normalize_pixel_values(pixel_values)

        sample = {
            "pixel_values": pixel_values,
            "lang_ids": torch.tensor(LANG2ID[language], dtype=torch.long),
            "language": language,
            "context_text": context,
            "max_width": torch.tensor(max_width, dtype=torch.long),
        }

        if not self.is_test:
            sample["target_mask"] = _render_target_mask(target, max_width, self.cfg)
            sample["target_text"] = target

        if "id" in row.index:
            sample["id"] = torch.tensor(int(row["id"]), dtype=torch.long)
        return sample


def collate_fn(batch: list[dict]) -> dict:
    out = {}
    for key in batch[0]:
        if isinstance(batch[0][key], torch.Tensor):
            out[key] = torch.stack([b[key] for b in batch])
        else:
            out[key] = [b[key] for b in batch]
    return out
