import re
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
    tokenizer = T5Tokenizer.from_pretrained(config["output_dir"])
    model = T5ForConditionalGeneration.from_pretrained(config["output_dir"]).to(device)
    model.eval()
    return model, tokenizer

# =========================
# 2. Optimized Inference Loop
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

    raw_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    # STANDARDIZE: remove spaces before commas and periods
    clean_text = re.sub(r'\s+([,.:])', r'\1', raw_text)
    
    return clean_text

# =========================
# 3. Text Rendering & Pixels
# =========================
def text_to_image(text, max_width):
    img = Image.new("L", (max_width, 32), 0)
    draw = ImageDraw.Draw(img)

    if not text.strip():
        text = "."  # fallback for empty text

    font_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "times.ttf"
    ]

    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, 16)
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

    for i, row in df.iterrows():
        sample_id = int(row["id"])
        max_width = int(row["max_width"])
        
        # 1. Prediction
        pred_text = predict_text(model, tokenizer, str(row["context"]), device, config["max_len"])
        
        # 2. Rendering
        img = text_to_image(pred_text, max_width)
        
        # 3. Pixel coordinates
        arr = np.array(img)
        rows, cols = np.where(arr > 127)
        
        if len(rows) > 0:
            s_ids = np.full(len(rows), sample_id, dtype=np.int64)
            pixel_block = np.stack([s_ids, rows.astype(np.int64), cols.astype(np.int64)], axis=1)
            pixel_data_list.append(pixel_block)
        else:
            pixel_data_list.append(np.array([[sample_id, 0, 0]], dtype=np.int64))

        if i % 20 == 0:
            print(f"[{i}/{len(df)}] ID {sample_id} Text: {pred_text}")
            img.save(debug_path / f"sample_{sample_id}_idx_{i}.png")

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
    print(f"\nTotal white pixels: {len(pixels_array)}")

if __name__ == "__main__":
    generate_submission()