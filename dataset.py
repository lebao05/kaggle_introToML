import pandas as pd
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset

from utils import fit_image_to_canvas, get_language_prompt, render_text_to_image, tail_context, white_pixel_density


def stratified_split(df: pd.DataFrame, col: str, frac: float, seed: int = 42):
    val_parts = []
    for _, g in df.groupby(col, group_keys=False):
        n = max(1, int(round(len(g) * frac))) if len(g) > 1 else 1
        n = min(n, len(g) - 1) if len(g) > 1 else 1
        val_parts.append(g.sample(n=n, random_state=seed))
    raw = pd.concat(val_parts)
    val = raw.sample(frac=1, random_state=seed).reset_index(drop=True)
    train = df.drop(raw.index).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, val


class OCRDataset(Dataset):
    def __init__(self, df: pd.DataFrame, processor, config: dict, is_test: bool = False):
        self.df = df.reset_index(drop=True)
        self.processor = processor
        self.cfg = config
        self.is_test = is_test

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        full_context = str(row["context"])
        context = tail_context(full_context, int(self.cfg["context_tail_chars"]))
        language = str(row["language"])
        max_width = int(row["max_width"])

        render_w = int(self.cfg["image_size"])
        raw_img = render_text_to_image(context, render_w, self.cfg["font_size"], self.cfg["padding"])
        img = fit_image_to_canvas(raw_img, self.cfg["image_size"], anchor="bottom")
        pixel_values = self.processor(images=img, return_tensors="pt").pixel_values.squeeze(0)

        sample = {
            "pixel_values": pixel_values,
            "context_text": context,
            "full_context_text": full_context,
            "context_char_len": len(context),
            "full_context_char_len": len(full_context),
            "language": language,
            "max_width": max_width,
            "orig_render_height": int(raw_img.size[1]),
            "orig_render_width": int(raw_img.size[0]),
            "render_height": int(img.size[1]),
            "render_width": int(img.size[0]),
            "resize_ratio_w": float(img.size[0] / max(raw_img.size[0], 1)),
            "resize_ratio_h": float(min(1.0, img.size[1] / max(raw_img.size[1], 1))),
            "context_white_density": white_pixel_density(raw_img),
            "pad_token_id": int(self.processor.tokenizer.pad_token_id),
        }

        if not self.is_test:
            target = str(row["target"])
            tokenizer = self.processor.tokenizer
            prompt_text = get_language_prompt(language)
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            target_ids = tokenizer(
                target,
                add_special_tokens=False,
                truncation=True,
                max_length=max(1, self.cfg["max_target_length"] - len(prompt_ids) - 1),
                return_tensors="pt",
            ).input_ids.squeeze(0)
            start_id = int(tokenizer.cls_token_id)
            eos_id = int(tokenizer.sep_token_id)
            sample["decoder_input_ids"] = torch.cat(
                [
                    torch.tensor([start_id], dtype=torch.long),
                    prompt_ids.to(torch.long),
                    target_ids.to(torch.long),
                ]
            )
            sample["labels"] = torch.cat(
                [
                    torch.full((len(prompt_ids),), -100, dtype=torch.long),
                    target_ids.to(torch.long),
                    torch.tensor([eos_id], dtype=torch.long),
                ]
            )
            sample["prompt_text"] = prompt_text
            sample["prompt_ids"] = prompt_ids.to(torch.long)
            sample["target_text"] = target
            sample["target_char_len"] = len(target)

        else:
            tokenizer = self.processor.tokenizer
            prompt_text = get_language_prompt(language)
            prompt_ids = tokenizer(
                prompt_text,
                add_special_tokens=False,
                return_tensors="pt",
            ).input_ids.squeeze(0)
            sample["prompt_text"] = prompt_text
            sample["prompt_ids"] = prompt_ids.to(torch.long)

        if "id" in row.index:
            sample["id"] = int(row["id"])
        return sample


def collate_fn(batch: list[dict]) -> dict:
    out = {}
    tensor_keys = {k for k, v in batch[0].items() if isinstance(v, torch.Tensor)}
    meta_keys = set(batch[0].keys()) - tensor_keys

    for key in tensor_keys:
        values = [b[key] for b in batch]
        if key == "labels":
            out[key] = pad_sequence(values, batch_first=True, padding_value=-100)
        elif key in {"decoder_input_ids", "prompt_ids"}:
            out[key] = pad_sequence(values, batch_first=True, padding_value=int(batch[0]["pad_token_id"]))
        else:
            out[key] = torch.stack(values)

    for key in meta_keys:
        out[key] = [b[key] for b in batch]
    return out
