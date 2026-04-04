"""Dataset for stage-2 continuation LM in an OCR -> LM pipeline."""

import random
import torch
from torch.utils.data import Dataset


def _clean_split(context: str, target: str, window_frac: float = 0.05):
    full = context + target
    original_split = len(context)
    n = len(full)
    window = max(15, int(n * window_frac))
    lo = max(1, original_split - window)
    hi = min(n, original_split + window)
    spaces = [i for i in range(lo, hi) if full[i] == " "]
    if not spaces:
        return context, target
    best = min(spaces, key=lambda i: abs(i - original_split))
    new_context = full[:best]
    new_target = full[best + 1:]
    if not new_context.strip() or not new_target.strip():
        return context, target
    return new_context.rstrip(), new_target.lstrip()


def _inject_ocr_noise(text: str, cfg: dict) -> str:
    if not text or random.random() > cfg.get("train_context_noise_prob", 0.0):
        return text

    words = text.split()
    kept = []
    for w in words:
        if len(words) > 6 and random.random() < cfg.get("train_word_drop_prob", 0.0):
            continue
        chars = []
        for ch in w:
            if ch != " " and random.random() < cfg.get("train_char_drop_prob", 0.0):
                continue
            chars.append(ch)
        kept.append("".join(chars) or w)
    noisy = " ".join(kept).strip() or text
    if random.random() < cfg.get("train_space_noise_prob", 0.0):
        noisy = noisy.replace(" ", "", max(1, noisy.count(" ") // 6))
    return noisy


def build_lm_input(language: str, context_text: str, max_width: int) -> str:
    width_bucket = int(round(max_width / 64.0) * 64)
    return (
        f"continue text\n"
        f"language: {language}\n"
        f"max_width: {width_bucket}\n"
        f"context: {context_text}"
    )


class PixelTextDataset(Dataset):
    def __init__(self, df, tokenizer, config, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.tokenizer = tokenizer
        self.config = config
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        language = str(row["language"])
        context = str(row["context"])
        max_width = int(row["max_width"])

        if not self.is_test:
            target = str(row["target"])
            context, target = _clean_split(context, target)
            context = _inject_ocr_noise(context, self.config)
        else:
            target = None

        source_text = build_lm_input(language, context, max_width)
        src = self.tokenizer(
            source_text,
            max_length=self.config["max_source_len"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        sample = {
            "input_ids": src["input_ids"].squeeze(0),
            "attention_mask": src["attention_mask"].squeeze(0),
        }

        if not self.is_test:
            tgt = self.tokenizer(
                text_target=target,
                max_length=self.config["max_target_len"],
                truncation=True,
                padding="max_length",
                return_tensors="pt",
            )
            labels = tgt["input_ids"].squeeze(0).clone()
            labels[labels == self.tokenizer.pad_token_id] = -100
            sample["labels"] = labels

        return sample
