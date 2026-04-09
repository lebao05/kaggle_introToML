from pathlib import Path

LANG2ID = {"English": 0, "French": 1, "German": 2, "Italian": 3, "Spanish": 4}
ID2LANG = {v: k for k, v in LANG2ID.items()}


def get_config():
    root = Path(".")
    return {
        # ── Tên model ──────────────────────────────────────────────────────────
        # Encoder: TrOCR ViT — đọc ảnh rendered text → patch embeddings
        # Decoder: mT5-small — sinh text continuation (multilingual)
        "trocr_name": "microsoft/trocr-base-printed",
        "mt5_name":   "google/mt5-small",

        # ── Data ───────────────────────────────────────────────────────────────
        "train_file":     str(root / "train-3.csv"),
        "test_file":      str(root / "test.csv"),
        "output_dir":     str(root / "outputs"),
        "val_split":      0.10,

        # ── Render ─────────────────────────────────────────────────────────────
        # Phải nhất quán giữa train, val, inference
        "font_size": 18,
        "padding":   12,

        # ── Tokenization ───────────────────────────────────────────────────────
        # EDA: target mean=57 chars, 95th pct ~25 tokens → 64 đủ
        "max_target_len": 64,

        # ── Generation ─────────────────────────────────────────────────────────
        "max_gen_len":           64,
        "min_gen_len":           5,
        "num_beams":             4,
        "length_penalty":        0.9,
        "repetition_penalty":    1.1,
        "no_repeat_ngram_size":  2,

        # ── Training ───────────────────────────────────────────────────────────
        "epochs":        12,
        "batch_size":    8,
        "enc_lr":        1e-5,   # encoder fine-tune nhẹ
        "proj_lr":       1e-4,   # projection học nhanh hơn
        "dec_lr":        3e-4,   # decoder học task mới
        "weight_decay":  0.01,
        "warmup_ratio":  0.06,
        "grad_clip":     1.0,
        "fp16":          False,
        "enc_patch_drop": 0.10,  # dropout patch encoder — tránh decoder ignore encoder

        # ── Projection ─────────────────────────────────────────────────────────
        # TrOCR-base encoder: 768 dim
        # mT5-small decoder:  512 dim
        # Cần projection bridge để nối 2 model
        "encoder_dim":  768,
        "decoder_dim":  512,
        "proj_dropout": 0.05,

        # ── Save & eval ────────────────────────────────────────────────────────
        "eval_gen_samples":        60,
        "sample_pred_count":       4,
        "best_max_short_pct":      35.0,
        "early_stop_patience":     4,
        "early_stop_min_delta":    0.002,
        "early_stop_warmup_epochs": 3,
    }


def resolve_config(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["max_target_len"] = int(max(8,  cfg.get("max_target_len", 64)))
    cfg["max_gen_len"]    = int(max(1,  min(cfg.get("max_gen_len", 64), cfg["max_target_len"])))
    cfg["min_gen_len"]    = int(max(1,  min(cfg.get("min_gen_len", 5),  cfg["max_gen_len"])))
    cfg["batch_size"]     = int(max(1,  cfg.get("batch_size", 8)))
    cfg["epochs"]         = int(max(1,  cfg.get("epochs", 12)))
    if not 0 <= cfg["best_max_short_pct"] <= 100:
        raise ValueError("best_max_short_pct phải trong [0, 100]")
    return cfg
