"""Compatibility wrapper for the active pixel-decoder inference pipeline."""

from inference_pixel import generate_submission as _pixel_infer

if __name__ == "__main__":
    _pixel_infer()
    raise SystemExit

import json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoProcessor

from config import get_config, resolve_config
from dataset import build_chat, _clean_split, _render_rgb
from model import load_for_inference
from utils import image_to_pixels, render_text_to_image


def generate_submission():
    config = resolve_config(get_config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("INFERENCE: SmolVLM-256M-Instruct + LoRA")
    print("=" * 60)

    # 1. Load best model (base + merged LoRA)
    best_dir = os.path.join(config["output_dir"], "best_adapter")
    if not os.path.exists(best_dir):
        raise FileNotFoundError(
            f"Khong tim thay best_adapter tai {best_dir}. "
            "Chay train.py truoc."
        )

    print(f"\n[1/4] Loading model tu {best_dir} ...")
    t0 = time.time()

    processor = AutoProcessor.from_pretrained(best_dir)
    processor.image_processor.do_image_splitting = False
    model = load_for_inference(config, best_dir, device)

    cfg_path = os.path.join(best_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            rc = json.load(f)
        print(f"  Trained: epochs={rc.get('epochs')} lr={rc.get('lr')} "
              f"lora_r={rc.get('lora_r')}")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # 2. Test data
    print("\n[2/4] Loading test data...")
    df = pd.read_csv(config["test_file"])
    print(f"  {len(df)} samples | {df['language'].value_counts().to_dict()}")

    # 3. Generate
    print(f"\n[3/4] Generating {len(df)} predictions...")
    tok = processor.tokenizer
    all_preds = []
    t0 = time.time()

    for i, row in df.iterrows():
        context   = str(row["context"])
        language  = str(row["language"])
        max_width = int(row["max_width"])

        img = _render_rgb(context, max_width,
                          config["font_size"], config["padding"])

        messages, images = build_chat(img, language, target=None)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = processor(
            text=text, images=images,
            return_tensors="pt",
            do_image_splitting=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens = config["max_new_tokens"],
                min_new_tokens = config["min_new_tokens"],
                do_sample      = False,
                pad_token_id   = tok.pad_token_id,
                eos_token_id   = tok.eos_token_id,
            )

        new_ids = out[0][prompt_len:]
        pred = tok.decode(new_ids, skip_special_tokens=True).strip()
        all_preds.append(pred)

        if i % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(i + 1, 1) * (len(df) - i - 1)
            print(f"  [{i:>4d}/{len(df)}] ETA={eta:.0f}s | \"{pred[:60]}\"")

    pred_lens = [len(p) for p in all_preds]
    print(f"\n  {len(all_preds)} predictions in {time.time()-t0:.1f}s")
    print(f"  Lengths: min={min(pred_lens)} "
          f"mean={np.mean(pred_lens):.0f} max={max(pred_lens)}")

    # 4. Render -> pixels
    print("\n[4/4] Rendering to pixels...")
    debug_dir = Path("debug_outputs/inference")
    debug_dir.mkdir(parents=True, exist_ok=True)

    pixel_list = []
    total_px   = 0

    for i, row in df.iterrows():
        sid       = int(row["id"])
        max_width = int(row["max_width"])
        pred_text = all_preds[i]

        pred_img = render_text_to_image(
            pred_text, max_width,
            config["font_size"], config["padding"],
        )
        pixels = image_to_pixels(pred_img, sid)
        pixel_list.append(pixels)
        total_px += len(pixels)

        if i < 20:
            pred_img.save(debug_dir / f"pred_{sid}.png")
            ctx_img = render_text_to_image(
                str(row["context"]), max_width,
                config["font_size"], config["padding"],
            )
            ctx_img.save(debug_dir / f"ctx_{sid}.png")
            print(f"  [{i:>2d}] {row['language']} w={max_width}")
            print(f"        ctx:  ...{str(row['context'])[-60:]}")
            print(f"        pred: {pred_text[:80]}")

    # 5. Export
    all_px = np.concatenate(pixel_list, axis=0)
    pa = np.zeros(len(all_px), dtype=[
        ("sample_id", np.int64), ("row_id", np.int64), ("col_id", np.int64),
    ])
    pa["sample_id"] = all_px[:, 0]
    pa["row_id"]    = all_px[:, 1]
    pa["col_id"]    = all_px[:, 2]

    Path("data").mkdir(parents=True, exist_ok=True)
    np.savez("data/pixels.npz", pixels=pa)
    pd.DataFrame({
        "id": df["id"].values,
        "sample_id": df["id"].values,
    }).to_csv("submission.csv", index=False)

    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"  Predictions:   {len(all_preds)}")
    print(f"  Total pixels:  {total_px:,}")
    print(f"  Avg px/sample: {total_px/max(len(all_preds),1):.0f}")
    print(f"  -> data/pixels.npz | submission.csv")
    print(f"  -> {debug_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    generate_submission()
