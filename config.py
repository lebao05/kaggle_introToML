from pathlib import Path

LANG2ID = {"English": 0, "French": 1, "German": 2, "Italian": 3, "Spanish": 4}
ID2LANG = {v: k for k, v in LANG2ID.items()}


def get_config():
    root = Path(".")
    return {
        # ── Model ─────────────────────────────────────────────────────────────
        # SigLIP vision encoder (extracted from SmolVLM, pretrained, frozen).
        # ONLY the pixel decoder is trained — no text tokens ever.
        #
        # COMPLIANCE:
        #   Input  : context text rendered to PIL image → pixel_values (visual)
        #   Process: SigLIP patch features → pooled visual embedding
        #   Output : pixel decoder generates continuation image DIRECTLY
        #            in pixel space — never passes through text token space.
        "model_name": "HuggingFaceTB/SmolVLM-256M-Instruct",

        # ── Data ──────────────────────────────────────────────────────────────
        "train_file": str(root / "train-3.csv"),
        "test_file":  str(root / "test.csv"),
        "output_dir": str(root / "outputs"),
        "val_split":  0.10,

        # ── Render (context image + target image use same params) ──────────────
        "font_size": 18,
        "padding":   12,

        # ── Pixel decoder output size ─────────────────────────────────────────
        # Model generates a fixed (output_h × output_w) image.
        # At inference the width is scaled to match each sample's max_width.
        "output_h": 96,    # rows in predicted continuation image
        "output_w": 384,   # columns (base width; scaled at inference)

        # ── Pixel decoder architecture ────────────────────────────────────────
        "lang_emb_dim":    32,    # language conditioning embedding size
        "decoder_hidden":  128,   # channels at start of ConvTranspose stack
        "decoder_dropout": 0.15,  # dropout inside decoder

        # ── Training ──────────────────────────────────────────────────────────
        "epochs":       25,       # more epochs ok: decoder is small, low overfit risk
        "batch_size":   8,        # larger batch fine without LM forward
        "lr":           3e-4,
        "weight_decay": 1e-4,
        "warmup_ratio": 0.10,
        "grad_clip":    1.0,
        "bf16":         True,

        # BCE loss weight for positive (white) pixels.
        # Text images are sparse: ~5-15% white pixels → upweight positives.
        "pos_weight":   6.0,

        # ── Eval & save ───────────────────────────────────────────────────────
        "eval_samples":             30,   # samples used for pixel-F1 eval
        "best_min_f1":              0.0,  # save only if f1 > this
        "early_stop_patience":      5,
        "early_stop_warmup_epochs": 3,
        "debug_image_count":        4,
        "debug_threshold":          0.5,
    }


def resolve_config(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["batch_size"] = int(max(1, cfg.get("batch_size", 8)))
    cfg["epochs"]     = int(max(1, cfg.get("epochs", 25)))
    cfg["output_h"]   = int(max(16, cfg.get("output_h", 96)))
    cfg["output_w"]   = int(max(16, cfg.get("output_w", 384)))
    return cfg
