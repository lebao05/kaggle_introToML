import torch
from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import render_text_to_image


def _clean_split(context: str, target: str, window_frac: float = 0.05):
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


def _resize_for_model(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) <= max_side:
        return img
    out = img.copy()
    out.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
    canvas = Image.new(out.mode, (max_side, max_side), 0)
    canvas.paste(out, ((max_side - out.size[0]) // 2, (max_side - out.size[1]) // 2))
    return canvas


def build_prompt_text(processor, language: str) -> str:
    image_token = getattr(processor, "image_token", "<image>")
    return (
        f"{image_token}\n"
        f"Continue the visible passage in {language}. "
        "Output only the next text span. "
        "Do not add labels, explanations, or quotation marks."
    )


def build_full_text(processor, language: str, answer: str) -> str:
    prompt = build_prompt_text(processor, language)
    return f"{prompt}\n{answer}"


def _normalize_pixel_values(x: torch.Tensor) -> torch.Tensor:
    while x.ndim > 3 and x.shape[0] == 1:
        x = x.squeeze(0)
    return x


def _normalize_text_tensor(x: torch.Tensor) -> torch.Tensor:
    while x.ndim > 1 and x.shape[0] == 1:
        x = x.squeeze(0)
    return x


class VisualContinuationDataset(Dataset):
    def __init__(self, df, processor, config: dict, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.cfg = config
        self.is_test = is_test
        self.tokenizer = processor.tokenizer

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
        ctx_img = _resize_for_model(ctx_img, int(self.cfg["image_max_side"]))
        prompt_text = build_prompt_text(self.processor, language)

        if self.is_test:
            proc = self.processor(
                text=prompt_text,
                images=ctx_img,
                do_resize=False,
                truncation=True,
                max_length=self.cfg["max_prompt_length"],
                return_tensors="pt",
            )
            sample = {
                "context_text": context,
                "language": language,
                "max_width": max_width,
                "prompt_text": prompt_text,
                "prompt_len": int(proc["input_ids"].shape[-1]),
            }
        else:
            full_text = build_full_text(self.processor, language, target)
            proc = self.processor(
                text=full_text,
                images=ctx_img,
                do_resize=False,
                truncation=True,
                max_length=self.cfg["max_input_length"],
                return_tensors="pt",
            )
            prompt_ids = self.tokenizer(
                prompt_text,
                add_special_tokens=False,
                truncation=True,
                max_length=self.cfg["max_prompt_length"],
            )["input_ids"]
            labels = proc["input_ids"].clone()
            prompt_len = min(len(prompt_ids), labels.shape[-1] - 1)
            labels[:, :prompt_len] = -100
            labels[proc["attention_mask"] == 0] = -100
            sample = {
                "context_text": context,
                "language": language,
                "max_width": max_width,
                "prompt_text": prompt_text,
                "prompt_len": prompt_len,
                "target_text": target,
                "labels": labels.squeeze(0),
            }

        for key, value in proc.items():
            if isinstance(value, torch.Tensor):
                if key == "pixel_values":
                    sample[key] = _normalize_pixel_values(value)
                else:
                    sample[key] = _normalize_text_tensor(value)

        if "id" in row.index:
            sample["id"] = int(row["id"])
        return sample


def collate_fn(batch: list[dict]) -> dict:
    out = {}
    tensor_keys = {k for k, v in batch[0].items() if isinstance(v, torch.Tensor)}
    meta_keys = set(batch[0].keys()) - tensor_keys

    for key in tensor_keys:
        values = [b[key] for b in batch]
        if key in {"input_ids", "attention_mask"}:
            pad_value = 0
            out[key] = pad_sequence(values, batch_first=True, padding_value=pad_value)
        elif key == "labels":
            out[key] = pad_sequence(values, batch_first=True, padding_value=-100)
        else:
            out[key] = torch.stack(values)

    for key in meta_keys:
        out[key] = [b[key] for b in batch]
    return out
