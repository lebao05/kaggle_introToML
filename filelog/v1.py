from pathlib import Path

# Stable language mapping used across dataset/train/inference
LANG2ID = {
    "English": 0,
    "French": 1,
    "German": 2,
    "Italian": 3,
    "Spanish": 4,
}
ID2LANG = {v: k for k, v in LANG2ID.items()}


def get_config():
    root = Path(".")
    return {
        # Data
        "train_file": str(root / "train-3.csv"),
        "test_file": str(root / "test.csv"),
        "output_dir": str(root / "outputs_trocr_mt5"),
        "checkpoint_dir": str(root / "outputs_trocr_mt5" / "checkpoints"),
        "val_split": 0.10,

        # Model
        "encoder_name": "microsoft/trocr-base-printed",
        "encoder_hidden_dim": 768,
        "mt5_model": "google/mt5-small",
        "mt5_d_model": 512,
        "num_languages": len(LANG2ID),
        "proj_dropout": 0.10,

        # Rendering / dataset
        "render_font_size": 18,
        "render_padding": 12,
        "max_target_len": 96,
        "batch_size": 8,

        # Training schedule
        "phase1_epochs": 3,
        "phase2_epochs": 8,
        "phase1_lr": 5e-4,
        "phase2_encoder_lr": 1e-5,
        "phase2_decoder_lr": 3e-4,
        "phase2_proj_lr": 3e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "label_smoothing": 0.0,
        "enc_patch_drop": 0.08,
        "fp16": True,

        # Generation
        "num_beams": 4,
        "length_penalty": 0.9,
        "no_repeat_ngram_size": 3,
        "repetition_penalty": 1.2,

        # Diagnostics / sampling
        # High entropy means attention is diffuse and decoder may ignore encoder.
        "attn_entropy_warn": 4.5,
        "debug_every": 2,
        "tf_start": 1.0,
        "tf_end": 1.0,
    }
