import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor

from config_lora import get_config, resolve_config
from dataset_lora import _render_rgb, build_prompt_text
from model_lora import load_for_inference
from utils import image_to_pixels, render_text_to_image


def clean_generated_text(text: str) -> str:
    text = " ".join((text or "").split())
    bad_prefixes = [
        "Assistant:",
        "Assistant :",
        "Answer:",
        "Answer :",
        "Response:",
        "Response :",
        "The image shows",
        "The text is",
        "Theanswer",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in bad_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip(" -:\n\t")
                changed = True
    return text.strip()


def generate_submission():
    config = resolve_config(get_config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("INFERENCE: context image -> SmolVLM + LoRA -> continuation text")
    print("=" * 60)

    best_dir = os.path.join(config["output_dir"], "best_model")
    if not os.path.exists(best_dir):
        raise FileNotFoundError(f"Khong tim thay best_model tai {best_dir}. Chay train.py truoc.")

    print(f"\n[1/4] Loading model tu {best_dir} ...")
    t0 = time.time()
    cfg_path = os.path.join(best_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            saved_cfg = json.load(f)
        config.update(saved_cfg)
        config = resolve_config(config)

    processor = AutoProcessor.from_pretrained(best_dir)
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token or processor.tokenizer.unk_token
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False

    model = load_for_inference(config, best_dir, device)
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    print(f"  Trained: epochs={config['epochs']} lr={config['lr']} max_new_tokens={config['max_new_tokens']}")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\n[2/4] Loading test data...")
    df = pd.read_csv(config["test_file"])
    print(f"  {len(df)} samples | {df['language'].value_counts().to_dict()}")

    print(f"\n[3/4] Predicting {len(df)} sequences...")
    debug_dir = Path("debug_outputs/inference")
    debug_dir.mkdir(parents=True, exist_ok=True)
    pixel_list = []
    total_px = 0
    t0 = time.time()

    for i, row in df.iterrows():
        sid = int(row["id"])
        context = str(row["context"])
        language = str(row["language"])
        max_width = int(row["max_width"])

        prompt_text = build_prompt_text(processor, language)
        ctx_img = _render_rgb(context, max_width, config["font_size"], config["padding"])
        proc = processor(
            text=prompt_text,
            images=ctx_img,
            do_resize=False,
            truncation=True,
            max_length=config["max_prompt_length"],
            return_tensors="pt",
        )
        model_inputs = {}
        for k, v in proc.items():
            if isinstance(v, torch.Tensor):
                tensor = v.to(device)
                if k == "pixel_values" and tensor.ndim == 4:
                    tensor = tensor.unsqueeze(1)
                if k == "pixel_attention_mask" and tensor.ndim == 3:
                    tensor = tensor.unsqueeze(1)
                model_inputs[k] = tensor
        prompt_len = int(model_inputs["input_ids"].shape[1])

        with torch.no_grad():
            output_ids = model.generate(
                **model_inputs,
                max_new_tokens=config["max_new_tokens"],
                num_beams=config["num_beams"],
                do_sample=False,
                use_cache=True,
                repetition_penalty=float(config.get("repetition_penalty", 1.15)),
                no_repeat_ngram_size=int(config.get("no_repeat_ngram_size", 3)),
                pad_token_id=processor.tokenizer.pad_token_id,
                eos_token_id=processor.tokenizer.eos_token_id,
            )

        pred_text = processor.batch_decode(output_ids[:, prompt_len:], skip_special_tokens=True)[0].strip()
        pred_text = clean_generated_text(pred_text)
        pred_img = render_text_to_image(pred_text, max_width, config["font_size"], config["padding"])
        pixels = image_to_pixels(pred_img, sid, threshold=int(config["debug_threshold"]))
        pixel_list.append(pixels)
        total_px += len(pixels)

        if i % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(i + 1, 1) * (len(df) - i - 1)
            print(f"  [{i:>4d}/{len(df)}] ETA={eta:.0f}s chars={len(pred_text)} px={len(pixels)}")

        if i < 20:
            pred_img.save(debug_dir / f"pred_{sid}.png")
            render_text_to_image(context, max_width, config["font_size"], config["padding"]).save(
                debug_dir / f"context_{sid}.png"
            )
            (debug_dir / f"pred_{sid}.txt").write_text(pred_text, encoding="utf-8")

    print(f"\n[4/4] Exporting pixels...")
    all_px = np.concatenate(pixel_list, axis=0)
    pa = np.zeros(
        len(all_px),
        dtype=[("sample_id", np.int64), ("row_id", np.int64), ("col_id", np.int64)],
    )
    pa["sample_id"] = all_px[:, 0]
    pa["row_id"] = all_px[:, 1]
    pa["col_id"] = all_px[:, 2]

    Path("data").mkdir(parents=True, exist_ok=True)
    np.savez("data/pixels.npz", pixels=pa)
    pd.DataFrame({"id": df["id"].values, "sample_id": df["id"].values}).to_csv("submission.csv", index=False)

    print(f"\n{'=' * 60}")
    print("INFERENCE COMPLETE")
    print(f"  Predictions:   {len(df)}")
    print(f"  Total pixels:  {total_px:,}")
    print(f"  Avg px/sample: {total_px / max(len(df), 1):.0f}")
    print("  -> data/pixels.npz | submission.csv")
    print(f"  -> {debug_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    generate_submission()
