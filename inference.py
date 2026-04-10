import json
import os
import time

import numpy as np
import pandas as pd
import torch

from config import get_config, resolve_config
from dataset import OCRDataset
from model import load_bundle
from utils import clean_prediction, ensure_dir, image_to_pixels, render_text_to_image


def generate_submission():
    config = resolve_config(get_config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    best_dir = os.path.join(config["output_dir"], "best_model")
    if not os.path.exists(best_dir):
        raise FileNotFoundError(f"Khong tim thay best_model tai {best_dir}. Chay train.py truoc.")

    print("=" * 60)
    print("INFERENCE: context image -> TrOCR -> continuation text")
    print("=" * 60)
    print(f"  Device:  {device}")

    cfg_path = os.path.join(best_dir, "run_config.json")
    if os.path.exists(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            saved_cfg = json.load(f)
        config.update(saved_cfg)
        config = resolve_config(config)

    print(f"\n[1/4] Loading model tu {best_dir} ...")
    t0 = time.time()
    model, processor = load_bundle(best_dir, device)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    print("\n[2/4] Loading test data...")
    test_df = pd.read_csv(config["test_file"])
    print(f"  {len(test_df)} samples | {test_df['language'].value_counts().to_dict()}")

    dataset = OCRDataset(test_df, processor, config, is_test=True)
    debug_dir = ensure_dir(os.path.join(config["debug_dir"], "inference"))

    print(f"\n[3/4] Predicting {len(dataset)} sequences...")
    pixel_list = []
    total_px = 0
    char_lengths = []
    px_lengths = []
    t0 = time.time()
    for i in range(len(dataset)):
        sample = dataset[i]
        start_id = int(processor.tokenizer.cls_token_id)
        decoder_prompt_ids = torch.cat(
            [torch.tensor([start_id], device=device, dtype=torch.long), sample["prompt_ids"].to(device)]
        ).unsqueeze(0)
        generated_ids = model.generate(
            pixel_values=sample["pixel_values"].unsqueeze(0).to(device),
            decoder_input_ids=decoder_prompt_ids,
            max_new_tokens=config["max_new_tokens"],
            min_new_tokens=config["min_new_tokens"],
            num_beams=config["num_beams"],
            early_stopping=True,
            repetition_penalty=float(config["repetition_penalty"]),
            no_repeat_ngram_size=int(config["no_repeat_ngram_size"]),
            length_penalty=float(config["length_penalty"]),
        )
        pred_text = clean_prediction(processor.batch_decode(generated_ids, skip_special_tokens=True)[0], sample["language"])
        pred_img = render_text_to_image(pred_text, sample["max_width"], config["font_size"], config["padding"])
        pixels = image_to_pixels(pred_img, sample["id"])
        pixel_list.append(pixels)
        total_px += len(pixels)
        char_lengths.append(len(pred_text))
        px_lengths.append(len(pixels))

        if i % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / max(i + 1, 1) * (len(dataset) - i - 1)
            print(
                f"  [{i:>4d}/{len(dataset)}] ETA={eta:.0f}s chars={len(pred_text)} px={len(pixels)} "
                f"ctxChars={len(sample['context_text'])} render={sample['render_width']}x{sample['render_height']}"
            )

        if i < 20:
            pred_img.save(debug_dir / f"pred_{sample['id']}.png")
            render_text_to_image(sample["context_text"], sample["max_width"], config["font_size"], config["padding"]).save(
                debug_dir / f"context_{sample['id']}.png"
            )
            (debug_dir / f"pred_{sample['id']}.txt").write_text(pred_text, encoding="utf-8")

    print("\n[4/4] Exporting pixels...")
    all_px = np.concatenate(pixel_list, axis=0)
    arr = np.zeros(len(all_px), dtype=[("sample_id", np.int64), ("row_id", np.int64), ("col_id", np.int64)])
    arr["sample_id"] = all_px[:, 0]
    arr["row_id"] = all_px[:, 1]
    arr["col_id"] = all_px[:, 2]

    ensure_dir("data")
    np.savez("data/pixels.npz", pixels=arr)
    pd.DataFrame({"id": test_df["id"].values, "sample_id": test_df["id"].values}).to_csv("submission.csv", index=False)

    print("\n" + "=" * 60)
    print("INFERENCE COMPLETE")
    print(f"  Predictions:   {len(dataset)}")
    print(f"  Total pixels:  {total_px:,}")
    print(f"  Avg px/sample: {total_px / max(len(dataset), 1):.0f}")
    if char_lengths:
        print(f"  Avg chars:     {float(np.mean(char_lengths)):.1f}")
        print(f"  Char range:    {int(np.min(char_lengths))}..{int(np.max(char_lengths))}")
        print(f"  Px range:      {int(np.min(px_lengths))}..{int(np.max(px_lengths))}")
    print("  -> data/pixels.npz | submission.csv")
    print(f"  -> {debug_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    generate_submission()
