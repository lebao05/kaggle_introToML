"""Inference for OCR -> continuation LM pipeline."""

import difflib
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from transformers import AutoTokenizer, TrOCRProcessor, VisionEncoderDecoderModel

from config import get_config, resolve_config
from dataset import build_lm_input
from model import TrOCRmT5Model
from utils import image_to_pixels, render_target_to_image, render_text_to_image


def _render_context_rgb(context: str, max_width: int, config: dict) -> Image.Image:
    return render_text_to_image(
        context,
        max_width,
        font_size=config["render_font_size"],
        padding=config["render_padding"],
    ).convert("RGB")


@torch.no_grad()
def _ocr_context_text(ocr_model, ocr_processor, context: str, max_width: int, config: dict, device) -> str:
    img = _render_context_rgb(context, max_width, config)
    pixel_values = ocr_processor(images=img, return_tensors="pt")["pixel_values"].to(device)
    out_ids = ocr_model.generate(pixel_values=pixel_values, num_beams=2, max_new_tokens=config["max_source_len"])
    text = ocr_processor.batch_decode(out_ids, skip_special_tokens=True)[0].strip()
    if config.get("ocr_fallback_to_raw_context", True):
        raw_norm = " ".join(context.split())
        ocr_norm = " ".join(text.split())
        short_ocr = len(ocr_norm) < max(12, int(len(raw_norm) * 0.25))
        star_ratio = ocr_norm.count("*") / max(1, len(ocr_norm))
        similarity = difflib.SequenceMatcher(None, raw_norm[:2000], ocr_norm[:2000]).ratio()
        receipt_like = any(token in ocr_norm.upper() for token in ("RECEIPT", "OFFER", "BACK OF"))
        if (
            not ocr_norm
            or short_ocr
            or star_ratio > 0.08
            or similarity < config.get("ocr_similarity_fallback_threshold", 0.55)
            or (receipt_like and similarity < 0.75)
        ):
            return context
    return text or context


def generate_submission():
    config = resolve_config(get_config())
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("INFERENCE: OCR -> continuation LM")
    print("=" * 60)
    print(f"  Device: {device}")

    print("\n[1/4] Loading OCR + LM...")
    ckpt_name = config.get("inference_checkpoint", "best_model")
    lm_dir = os.path.join(config["output_dir"], ckpt_name)
    if not os.path.isdir(lm_dir):
        fallback_dir = os.path.join(config["output_dir"], "final_model")
        print(f"  Checkpoint '{ckpt_name}' not found, fallback -> {fallback_dir}/")
        lm_dir = fallback_dir

    t0 = time.time()
    ocr_processor = TrOCRProcessor.from_pretrained(config["ocr_model_name"])
    ocr_model = VisionEncoderDecoderModel.from_pretrained(config["ocr_model_name"]).to(device)
    ocr_model.eval()
    tokenizer = AutoTokenizer.from_pretrained(lm_dir)
    lm_model = TrOCRmT5Model(config, model_dir=lm_dir).to(device)
    lm_model.eval()
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  OCR: {config['ocr_model_name']}")
    print(f"  LM checkpoint: {lm_dir}")

    print("\n[2/4] Loading test data...")
    df = pd.read_csv(config["test_file"])
    print(f"  {len(df)} samples | {df['language'].value_counts().to_dict()}")

    print("\n[3/4] OCR + continuation generation...")
    all_preds = []
    ocr_texts = []
    t0 = time.time()
    for i, row in df.iterrows():
        bt0 = time.time()
        context = str(row["context"])
        language = str(row["language"])
        max_width = int(row["max_width"])
        ocr_text = _ocr_context_text(ocr_model, ocr_processor, context, max_width, config, device)
        ocr_texts.append(ocr_text)
        source_text = build_lm_input(language, ocr_text, max_width)
        enc = tokenizer(
            source_text,
            max_length=config["max_source_len"],
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        pred = lm_model.generate(input_ids, attention_mask, tokenizer, config)[0].strip()
        all_preds.append(pred)
        bt = time.time() - bt0
        eta = bt * (len(df) - i - 1)
        if i % 10 == 0:
            print(f"  [{i:>4d}/{len(df)}] {bt:.2f}s ETA={eta:.0f}s | OCR=\"{ocr_text[:45]}...\" | PRED=\"{pred[:45]}...\"")

    pred_lens = [len(p) for p in all_preds]
    print(f"\n  {len(all_preds)} predictions in {time.time()-t0:.1f}s")
    print(f"  Lengths: min={min(pred_lens)} max={max(pred_lens)} mean={np.mean(pred_lens):.0f}")

    print("\n[4/4] Rendering to pixels...")
    debug_path = Path("debug_outputs/inference")
    debug_path.mkdir(parents=True, exist_ok=True)
    pixel_list = []
    total_px = 0
    for i, row in df.iterrows():
        sid = int(row["id"])
        max_width = int(row["max_width"])
        context = str(row["context"])
        lang = str(row["language"])
        pred_text = all_preds[i]
        pred_img = render_target_to_image(pred_text, max_width, config["render_font_size"], config["render_padding"])
        pixels = image_to_pixels(pred_img, sid)
        pixel_list.append(pixels)
        total_px += len(pixels)
        if i < 20:
            pred_img.save(debug_path / f"pred_{sid}.png")
            ctx_img = render_text_to_image(context, max_width, config["render_font_size"], config["render_padding"])
            ctx_img.save(debug_path / f"ctx_{sid}.png")
            print(f"  [{i:>2d}] {lang} w={max_width} px={len(pixels)}")
            print(f"        raw:  ...{context[-60:]}")
            print(f"        ocr:  ...{ocr_texts[i][-60:]}")
            print(f"        pred: {pred_text[:80]}")

    all_px = np.concatenate(pixel_list, axis=0)
    pa = np.zeros(len(all_px), dtype=[("sample_id", np.int64), ("row_id", np.int64), ("col_id", np.int64)])
    pa["sample_id"] = all_px[:, 0]
    pa["row_id"] = all_px[:, 1]
    pa["col_id"] = all_px[:, 2]
    Path("data").mkdir(parents=True, exist_ok=True)
    np.savez("data/pixels.npz", pixels=pa)
    pd.DataFrame({"id": df["id"].values, "sample_id": df["id"].values}).to_csv("submission.csv", index=False)

    print(f"\n{'='*60}")
    print("INFERENCE COMPLETE")
    print(f"{'='*60}")
    print(f"  Predictions:   {len(all_preds)}")
    print(f"  Total pixels:  {total_px:,}")
    print(f"  Avg px/sample: {total_px/len(all_preds):.0f}")
    print("  -> data/pixels.npz | submission.csv")
    print(f"  -> {debug_path}/")


if __name__ == "__main__":
    generate_submission()
