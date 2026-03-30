import re
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from config import get_config

# Hugging Face
from transformers import T5ForConditionalGeneration, T5Tokenizer


# =========================
# 1. Load Model & Tokenizer
# =========================
def load_model(config, device):
    # ✅ tokenizer from original model (SAFE)
    tokenizer = T5Tokenizer.from_pretrained(config["model_name"])

    # ✅ find latest checkpoint
    checkpoints = [
        os.path.join(config["output_dir"], d)
        for d in os.listdir(config["output_dir"])
        if d.startswith("checkpoint-")
    ]

    if len(checkpoints) == 0:
        raise ValueError("❌ No checkpoints found in output_dir")

    latest_checkpoint = sorted(
        checkpoints, key=lambda x: int(x.split("-")[-1])
    )[-1]

    print(f"🔥 Loading checkpoint: {latest_checkpoint}")

    model = T5ForConditionalGeneration.from_pretrained(latest_checkpoint).to(device)
    model.eval()

    return model, tokenizer


# =========================
# 2. Inference + Ratio
# =========================
def predict_text(model, tokenizer, context, device, max_len):
    inputs = tokenizer(
        context,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=max_len
    ).to(device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_len,
            num_beams=4
        )

    output_ids = outputs[0]

    raw_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    # clean punctuation spacing
    clean_text = re.sub(r'\s+([,.:])', r'\1', raw_text)

    # ✅ accurate lengths (ignore padding)
    input_len = inputs["attention_mask"].sum().item()
    output_len = (output_ids != tokenizer.pad_token_id).sum().item()

    return clean_text, input_len, output_len


# =========================
# 3. Text Rendering
# =========================
def text_to_image(text, max_width):
    img = Image.new("L", (max_width, 32), 0)
    draw = ImageDraw.Draw(img)

    if not text.strip():
        text = "."

    font_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "times.ttf"
    ]

    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, 14)
            break
        except:
            continue

    if font is None:
        font = ImageFont.load_default()

    draw.text((0, 0), text, fill=255, font=font)
    return img


# =========================
# 4. Generate Submission
# =========================
def generate_submission():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running inference on: {device}")

    model, tokenizer = load_model(config, device)
    df = pd.read_csv(config["test_file"])

    debug_path = Path("debug_outputs")
    debug_path.mkdir(parents=True, exist_ok=True)

    pixel_data_list = []

    # ✅ store ratios
    ratios = []

    for i, row in df.iterrows():
        sample_id = int(row["id"])
        max_width = int(row["max_width"])

        # =========================
        # 1. Prediction
        # =========================
        pred_text, input_len, output_len = predict_text(
            model,
            tokenizer,
            str(row["context"]),
            device,
            config["max_len"]
        )

        # ✅ compute ratio
        ratio = output_len / (input_len + output_len + 1e-8)
        ratios.append(ratio)

        # =========================
        # 2. Rendering
        # =========================
        img = text_to_image(pred_text, max_width)

        # =========================
        # 3. Pixels
        # =========================
        arr = np.array(img)
        rows, cols = np.where(arr > 127)

        if len(rows) > 0:
            s_ids = np.full(len(rows), sample_id, dtype=np.int64)
            pixel_block = np.stack(
                [s_ids, rows.astype(np.int64), cols.astype(np.int64)], axis=1
            )
            pixel_data_list.append(pixel_block)
        else:
            pixel_data_list.append(
                np.array([[sample_id, 0, 0]], dtype=np.int64)
            )

        # =========================
        # Debug print
        # =========================
        if i % 20 == 0:
            print(
                f"[{i}/{len(df)}] ID {sample_id} | "
                f"ratio={ratio:.4f} | "
                f"in={input_len} out={output_len} | "
                f"text={pred_text}"
            )
            img.save(debug_path / f"sample_{sample_id}_idx_{i}.png")

    # =========================
    # Combine pixels
    # =========================
    all_pixels = np.concatenate(pixel_data_list, axis=0)

    pixels_array = np.zeros(len(all_pixels), dtype=[
        ('sample_id', np.int64),
        ('row_id', np.int64),
        ('col_id', np.int64)
    ])

    pixels_array['sample_id'] = all_pixels[:, 0]
    pixels_array['row_id'] = all_pixels[:, 1]
    pixels_array['col_id'] = all_pixels[:, 2]

    Path("data").mkdir(parents=True, exist_ok=True)
    np.savez("data/pixels.npz", pixels=pixels_array)

    print("✅ Exported: data/pixels.npz")

    submission_df = pd.DataFrame({
        "id": df["id"].values,
        "sample_id": df["id"].values
    })

    submission_df.to_csv("submission.csv", index=False)

    print("✅ Exported: submission.csv")

    # =========================
    # Final stats
    # =========================
    avg_ratio = sum(ratios) / len(ratios)

    print(f"\n🔥 Average ratio (output / (input + output)): {avg_ratio:.4f}")
    print(f"Total white pixels: {len(pixels_array)}")


# =========================
# Run
# =========================
if __name__ == "__main__":
    generate_submission()