"""Inference for the row-structure decoder pipeline."""

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoProcessor

from config import LANG2ID, get_config, resolve_config
from dataset import _normalize_pixel_values, _render_rgb
from model_structure import load_for_inference
from utils import image_to_pixels, render_text_to_image


def _mask_to_image(logits: torch.Tensor, max_width: int, config: dict, threshold: float = 0.5) -> Image.Image:
    if logits.ndim == 3:
        logits = logits.squeeze(0)
    probs = torch.sigmoid(logits.detach()).float().cpu().numpy()
    arr = (probs >= threshold).astype(np.uint8) * 255
    img = Image.fromarray(arr, mode="L")
    return img.resize((max_width, config["output_h"]), Image.Resampling.NEAREST)


def generate_submission():
    config = resolve_config(get_config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("INFERENCE: SigLIP + row-structure decoder")
    print("=" * 60)

    best_dir = os.path.join(config["output_dir"], "best_model")
    if not os.path.exists(best_dir):
        raise FileNotFoundError(
            f"Khong tim thay best_model tai {best_dir}. "
            "Chay train.py truoc."
        )

    print(f"\n[1/4] Loading model tu {best_dir} ...")
    t0 = time.time()
    processor = AutoProcessor.from_pretrained(best_dir)
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False

    cfg_path = os.path.join(best_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            rc = json.load(f)
        config.update(rc)
        config = resolve_config(config)
        print(f"  Trained: epochs={rc.get('epochs')} lr={rc.get('lr')} output={rc.get('output_h')}x{rc.get('output_w')}")
    model = load_for_inference(config, best_dir, device)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\n[2/4] Loading test data...")
    df = pd.read_csv(config["test_file"])
    print(f"  {len(df)} samples | {df['language'].value_counts().to_dict()}")

    print(f"\n[3/4] Predicting {len(df)} masks...")
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

        ctx_img = _render_rgb(context, max_width, config["font_size"], config["padding"])
        pixel_values = processor(images=ctx_img, return_tensors="pt")["pixel_values"]
        pixel_values = _normalize_pixel_values(pixel_values).unsqueeze(0).to(device)
        lang_ids = torch.tensor([LANG2ID[language]], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(pixel_values, lang_ids)

        pred_img = _mask_to_image(logits[0], max_width, config, threshold=float(config.get("debug_threshold", 0.5)))
        pixels = image_to_pixels(pred_img, sid)
        pixel_list.append(pixels)
        total_px += len(pixels)

        if i % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(i + 1, 1) * (len(df) - i - 1)
            density = float((np.array(pred_img) > 127).mean())
            print(f"  [{i:>4d}/{len(df)}] ETA={eta:.0f}s density={density:.3f}")

        if i < 20:
            pred_img.save(debug_dir / f"pred_{sid}.png")
            render_text_to_image(context, max_width, config["font_size"], config["padding"]).save(
                debug_dir / f"ctx_{sid}.png"
            )

    print(f"\n[4/4] Exporting pixels...")
    all_px = np.concatenate(pixel_list, axis=0)
    pa = np.zeros(len(all_px), dtype=[
        ("sample_id", np.int64),
        ("row_id", np.int64),
        ("col_id", np.int64),
    ])
    pa["sample_id"] = all_px[:, 0]
    pa["row_id"] = all_px[:, 1]
    pa["col_id"] = all_px[:, 2]

    Path("data").mkdir(parents=True, exist_ok=True)
    np.savez("data/pixels.npz", pixels=pa)
    pd.DataFrame({"id": df["id"].values, "sample_id": df["id"].values}).to_csv("submission.csv", index=False)

    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"  Predictions:   {len(df)}")
    print(f"  Total pixels:  {total_px:,}")
    print(f"  Avg px/sample: {total_px/max(len(df),1):.0f}")
    print(f"  -> data/pixels.npz | submission.csv")
    print(f"  -> {debug_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    generate_submission()
