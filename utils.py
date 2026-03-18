import re

import torch
import pandas as pd
import numpy as np
import zipfile
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

# Importing your specific project modules
from config import get_config, latest_weights_file_path
from model import build_transformer
from tokenizers import Tokenizer
from dataset import causal_mask

# =========================
# 1. Load Model & Tokenizer
# =========================
def load_model(config, device):
    tokenizer = Tokenizer.from_file(config["tokenizer_file"])

    model = build_transformer(
        tokenizer.get_vocab_size(),
        tokenizer.get_vocab_size(),
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"]
    ).to(device)

    model_path = latest_weights_file_path(config)
    if model_path is None:
        raise ValueError("No trained model found! Check your weights directory.")

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    return model, tokenizer

# =========================
# 2. Optimized Inference Loop
# =========================
def predict_text(model, tokenizer, context, device, max_len):
    # Encode context and add special tokens
    source_tokens = tokenizer.encode(context).ids
    source = torch.tensor([[tokenizer.token_to_id("[SOS]")] + source_tokens + [tokenizer.token_to_id("[EOS]")]], dtype=torch.long).to(device)
    source_mask = (source != tokenizer.token_to_id("[PAD]")).unsqueeze(1).unsqueeze(1).to(device)

    with torch.no_grad():
        encoder_output = model.encode(source, source_mask)

    # Start decoding with [SOS]
    decoder_input = torch.tensor([[tokenizer.token_to_id("[SOS]")]], dtype=torch.long).to(device)
    
    generated_tokens = []
    for _ in range(max_len):
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        
        with torch.no_grad():
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            
        token_id = next_word.item()
        if token_id == tokenizer.token_to_id("[EOS]"):
            break
            
        generated_tokens.append(token_id)
        decoder_input = torch.cat([decoder_input, next_word.unsqueeze(0)], dim=1)

    raw_text = tokenizer.decode(generated_tokens)
    
    # STANDARDIZE: Remove spaces before commas and colons
    # Examples: "word , word" -> "word, word" | "time : 10" -> "time: 10"
    clean_text = re.sub(r'\s+([, .])', r'\1', raw_text)
    
    return clean_text
# =========================
# 3. Text Rendering & Pixels
# =========================
def text_to_image(text, max_width):
    # Image height MUST be 32
    img = Image.new("L", (max_width, 32), 0)
    draw = ImageDraw.Draw(img)

    if not text or text.strip() == "":
        text = "." # OCR fails on empty images

    # Try common font paths for Kaggle/Linux environments
    font_paths = [
        "/usr/share/fonts/truetype/liberation/LiberationSerif-Regular.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
        "times.ttf"
    ]
    
    font = None
    for path in font_paths:
        try:
            font = ImageFont.truetype(path, 20)
            break
        except:
            continue
    
    if font is None:
        font = ImageFont.load_default()

    draw.text((0, 0), text, fill=255, font=font)
    return img

# =========================
# 4. Generate Submission
def generate_submission():
    config = get_config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Running inference on: {device}")

    model, tokenizer = load_model(config, device)
    df = pd.read_csv(config["test_file"])
    
    # Create debug folder
    debug_path = Path("debug_outputs")
    debug_path.mkdir(parents=True, exist_ok=True)
    
    pixel_data_list = []

    for i, row in df.iterrows():
        sample_id = int(row["id"])
        max_width = int(row["max_width"])
        
        # 1. Prediction
        pred_text = predict_text(model, tokenizer, str(row["context"]), device, config["seq_len"])
        
        # 2. Rendering
        img = text_to_image(pred_text, max_width)
        
        # 3. Coordinate Extraction
        arr = np.array(img)
        rows, cols = np.where(arr > 127) 
        
        if len(rows) > 0:
            s_ids = np.full(len(rows), sample_id, dtype=np.int64)
            pixel_block = np.stack([s_ids, rows.astype(np.int64), cols.astype(np.int64)], axis=1)
            pixel_data_list.append(pixel_block)
        else:
            # Fallback coordinate
            pixel_data_list.append(np.array([[sample_id, 0, 0]], dtype=np.int64))

        if i % 20 == 0:
            print(f"[{i}/{len(df)}] ID {sample_id} Text: {pred_text}")
            img.save(debug_path / f"sample_{sample_id}_idx_{i}.png")

    # Combine pixels
    all_pixels = np.concatenate(pixel_data_list, axis=0)

    # Create Structured Array
    pixels_array = np.zeros(len(all_pixels), dtype=[
        ('sample_id', np.int64), 
        ('row_id', np.int64), 
        ('col_id', np.int64)
    ])
    pixels_array['sample_id'] = all_pixels[:, 0]
    pixels_array['row_id'] = all_pixels[:, 1]
    pixels_array['col_id'] = all_pixels[:, 2]

    # Save CSV and NPZ (No ZIP)
    # Ensure 'data' folder exists for the npz
    Path("data").mkdir(parents=True, exist_ok=True)
    
    # Export NPZ
    np.savez("data/pixels.npz", pixels=pixels_array)
    print("✅ Exported: data/pixels.npz")

    # Export CSV
    submission_df = pd.DataFrame({
        "id": df["id"].values,
        "sample_id": df["id"].values
    })
    submission_df.to_csv("submission.csv", index=False)
    print("✅ Exported: submission.csv")

    print(f"\nTotal white pixels: {len(pixels_array)}")

if __name__ == "__main__":
    generate_submission()