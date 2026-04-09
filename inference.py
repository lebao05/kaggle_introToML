"""Inference: load best model → generate → render → pixels → submit."""

import json, os, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import TrOCRProcessor, AutoTokenizer

from config import get_config, resolve_config
from dataset import PixelTextDataset
from model import TrOCRmT5Model
from utils import image_to_pixels, render_text_to_image, render_text_to_image as render_target_to_image


def generate_submission():
    config = resolve_config(get_config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("INFERENCE: TrOCR encoder + Projection + mT5 decoder")
    print("=" * 60)

    # ── 1. Load best model ────────────────────────────────────────────────────
    best_dir = os.path.join(config["output_dir"], "best_model")
    if not os.path.exists(best_dir):
        raise FileNotFoundError(
            f"Không tìm thấy best_model tại {best_dir}. "
            "Chạy train.py trước."
        )

    print(f"\n[1/4] Loading model từ {best_dir} ...")
    t0 = time.time()
    trocr_processor = TrOCRProcessor.from_pretrained(best_dir)
    mt5_tokenizer   = AutoTokenizer.from_pretrained(best_dir)
    model           = TrOCRmT5Model(config).to(device)
    model.load_state_dict(
        torch.load(os.path.join(best_dir, "model_weights.pt"),
                   map_location=device, weights_only=True)
    )
    model.eval()

    cfg_path = os.path.join(best_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            rc = json.load(f)
        print(f"  Trained: epochs={rc.get('epochs')} enc_lr={rc.get('enc_lr')} dec_lr={rc.get('dec_lr')}")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # ── 2. Test data ──────────────────────────────────────────────────────────
    print("\n[2/4] Loading test data...")
    df = pd.read_csv(config["test_file"])
    print(f"  {len(df)} samples | {df['language'].value_counts().to_dict()}")

    test_ds     = PixelTextDataset(df, trocr_processor, mt5_tokenizer, config, is_test=True)
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=config["batch_size"],
        shuffle=False, num_workers=0, pin_memory=True,
    )

    # ── 3. Generate ───────────────────────────────────────────────────────────
    print(f"\n[3/4] Generating ({len(test_loader)} batches)...")
    all_preds = []
    t0 = time.time()

    with torch.no_grad():
        for bi, batch in enumerate(test_loader):
            pv    = batch["pixel_values"].to(device)
            preds = model.generate(pv, mt5_tokenizer, config)
            all_preds.extend([p.strip() for p in preds])
            if bi % 5 == 0:
                eta = (time.time()-t0) / max(bi+1,1) * (len(test_loader)-bi-1)
                print(f"  B{bi:>3d}/{len(test_loader)} ETA={eta:.0f}s | \"{preds[0][:60]}\"")

    pred_lens = [len(p) for p in all_preds]
    print(f"  {len(all_preds)} predictions in {time.time()-t0:.1f}s")
    print(f"  Lengths: min={min(pred_lens)} mean={np.mean(pred_lens):.0f} max={max(pred_lens)}")

    # ── 4. Render → pixels ────────────────────────────────────────────────────
    print("\n[4/4] Rendering to pixels...")
    debug_dir = Path("debug_outputs/inference")
    debug_dir.mkdir(parents=True, exist_ok=True)

    pixel_list = []
    total_px   = 0

    for i, row in df.iterrows():
        sid       = int(row["id"])
        max_width = int(row["max_width"])
        pred_text = all_preds[i]

        pred_img = render_target_to_image(
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

    # ── 5. Export ─────────────────────────────────────────────────────────────
    all_px = np.concatenate(pixel_list, axis=0)
    pa = np.zeros(len(all_px), dtype=[
        ("sample_id", np.int64), ("row_id", np.int64), ("col_id", np.int64),
    ])
    pa["sample_id"] = all_px[:, 0]
    pa["row_id"]    = all_px[:, 1]
    pa["col_id"]    = all_px[:, 2]

    Path("data").mkdir(parents=True, exist_ok=True)
    np.savez("data/pixels.npz", pixels=pa)
    pd.DataFrame({"id": df["id"].values, "sample_id": df["id"].values}).to_csv(
        "submission.csv", index=False
    )

    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"  Predictions:   {len(all_preds)}")
    print(f"  Total pixels:  {total_px:,}")
    print(f"  Avg px/sample: {total_px/max(len(all_preds),1):.0f}")
    print(f"  → data/pixels.npz | submission.csv")
    print(f"  → {debug_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    generate_submission()
