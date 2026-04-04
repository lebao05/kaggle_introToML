from pathlib import Path

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
        "profile": "vision_ocr_then_lm_v1",
        "task_mode": "vision_ocr_then_lm",
        "ocr_model_name": "microsoft/trocr-small-printed",
        "lm_model_name": "google/mt5-small",
        "train_file": str(root / "train-3.csv"),
        "test_file": str(root / "test.csv"),
        "output_dir": str(root / "outputs_trocr_mt5"),
        "checkpoint_dir": str(root / "outputs_trocr_mt5" / "checkpoints"),
        "val_split": 0.10,
        "render_font_size": 18,
        "render_padding": 12,
        "max_source_len": 192,
        "max_target_len": 72,
        "max_gen_len": 72,
        "min_gen_len": 8,
        "batch_size": 8,
        "epochs": 12,
        "lr": 2e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "warmup_ratio": 0.08,
        "fp16": False,
        "num_beams": 4,
        "length_penalty": 0.9,
        "repetition_penalty": 1.08,
        "no_repeat_ngram_size": 2,
        "inference_checkpoint": "best_model",
        "best_max_short_pct": 35.0,
        "early_stop_on_cer": True,
        "early_stop_patience": 3,
        "early_stop_min_delta": 0.002,
        "early_stop_warmup_epochs": 3,
        "eval_gen_samples": 80,
        "sample_pred_count": 5,
        "debug_every": 1,
        "ocr_fallback_to_raw_context": True,
        "ocr_similarity_fallback_threshold": 0.55,
        "train_context_noise_prob": 0.20,
        "train_word_drop_prob": 0.08,
        "train_char_drop_prob": 0.03,
        "train_space_noise_prob": 0.03,
    }


def resolve_config(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["batch_size"] = int(max(1, cfg.get("batch_size", 8)))
    cfg["epochs"] = int(max(1, cfg.get("epochs", 12)))
    cfg["max_source_len"] = int(max(32, cfg.get("max_source_len", 192)))
    cfg["max_target_len"] = int(max(8, cfg.get("max_target_len", 72)))
    cfg["max_gen_len"] = int(max(1, min(cfg.get("max_gen_len", cfg["max_target_len"]), cfg["max_target_len"])))
    cfg["min_gen_len"] = int(max(1, min(cfg.get("min_gen_len", 8), cfg["max_gen_len"])))
    if not 0 <= cfg["best_max_short_pct"] <= 100:
        raise ValueError("best_max_short_pct must be in [0, 100].")
    return cfg
