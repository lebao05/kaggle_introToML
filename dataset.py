"""Dataset: context string → rendered image → TrOCR pixel_values tensor.

Pipeline mỗi sample:
  1. (Train) Fix mid-word split: 86% samples bị cắt giữa từ theo EDA
  2. Render context text → ảnh grayscale (PIL)
  3. Pad về hình vuông (giữ mode) → convert RGB
  4. TrOCR processor → pixel_values (3, 384, 384)
  5. (Train) mT5 tokenizer → labels
"""

import torch
from PIL import Image
from torch.utils.data import Dataset
from utils import render_text_to_image


def _clean_split(context: str, target: str, window_frac: float = 0.05):
    """Dời boundary về whitespace gần nhất để tránh cắt giữa từ."""
    full  = context + target
    split = len(context)
    n     = len(full)
    window = max(15, int(n * window_frac))
    lo = max(1, split - window)
    hi = min(n, split + window)
    spaces = [i for i in range(lo, hi) if full[i] == " "]
    if not spaces:
        return context, target
    best    = min(spaces, key=lambda i: abs(i - split))
    new_ctx = full[:best].rstrip()
    new_tgt = full[best + 1:].lstrip()
    if not new_ctx or not new_tgt:
        return context, target
    return new_ctx, new_tgt


def _pad_to_square(img: Image.Image) -> Image.Image:
    """Pad ảnh về hình vuông, giữ nguyên mode (L / RGB)."""
    w, h = img.size
    s    = max(w, h)
    out  = Image.new(img.mode, (s, s), 0)
    out.paste(img, ((s - w) // 2, (s - h) // 2))
    return out


class PixelTextDataset(Dataset):
    def __init__(self, df, trocr_processor, mt5_tokenizer, config, is_test=False):
        self.df            = df.reset_index(drop=True)
        self.processor     = trocr_processor   # xử lý ảnh → pixel_values
        self.tokenizer     = mt5_tokenizer      # tokenize target text
        self.cfg           = config
        self.is_test       = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row       = self.df.iloc[idx]
        context   = str(row["context"])
        max_width = int(row["max_width"])

        # ── 1. Fix word boundary ──────────────────────────────────────────────
        if not self.is_test:
            target          = str(row["target"])
            context, target = _clean_split(context, target)

        # ── 2. Render context → ảnh grayscale ────────────────────────────────
        img = render_text_to_image(
            context, max_width,
            font_size=self.cfg["font_size"],
            padding=self.cfg["padding"],
        )                                      # PIL mode "L"

        # ── 3. Pad vuông → RGB ────────────────────────────────────────────────
        img = _pad_to_square(img)              # vẫn mode "L", hình vuông
        img = img.convert("RGB")               # → mode "RGB" (TrOCR cần 3 channel)

        # ── 4. TrOCR processor → tensor ───────────────────────────────────────
        # Resize về 384×384, normalize → (3, 384, 384)
        pixel_values = self.processor(
            images=img, return_tensors="pt"
        )["pixel_values"].squeeze(0)

        sample = {"pixel_values": pixel_values}

        # ── 5. Tokenize target ────────────────────────────────────────────────
        if not self.is_test:
            enc    = self.tokenizer(
                target,
                max_length=self.cfg["max_target_len"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            labels = enc["input_ids"].squeeze(0).clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            sample["labels"] = labels

        return sample
