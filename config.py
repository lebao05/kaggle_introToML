from pathlib import Path


LANGUAGES = ["English", "French", "German", "Italian", "Spanish"]


def get_config() -> dict:
    root = Path(".")
    return {
        "seed": 42,
        "train_file": str(root / "train-3.csv"),
        "test_file": str(root / "test.csv"),
        "output_dir": str(root / "outputs"),
        "debug_dir": str(root / "debug_outputs"),
        "model_name": "microsoft/trocr-base-printed",
        "font_size": 18,
        "padding": 12,
        "image_size": 384,
        "context_tail_chars": 160,
        "freeze_encoder": True,
        "decoder_tune_mode": "top_layers",
        "decoder_train_layers": 4,
        "val_split": 0.10,
        "epochs": 10,
        "batch_size": 4,
        "grad_accum_steps": 2,
        "lr": 1.0e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,
        "grad_clip": 1.0,
        "bf16": True,
        "max_target_length": 128,
        "max_new_tokens": 72,
        "min_new_tokens": 8,
        "num_beams": 2,
        "repetition_penalty": 1.20,
        "no_repeat_ngram_size": 3,
        "length_penalty": 0.8,
        "prefix_token_count": 12,
        "prefix_loss_weight": 3.0,
        "debug_topk": 5,
        "eval_samples": 24,
        "debug_image_count": 4,
        "log_interval": 20,
        "early_stop_patience": 4,
        "early_stop_warmup_epochs": 2,
    }


def resolve_config(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["epochs"] = int(max(1, cfg.get("epochs", 10)))
    cfg["batch_size"] = int(max(1, cfg.get("batch_size", 4)))
    cfg["grad_accum_steps"] = int(max(1, cfg.get("grad_accum_steps", 2)))
    cfg["image_size"] = int(max(128, cfg.get("image_size", 384)))
    cfg["context_tail_chars"] = int(max(32, cfg.get("context_tail_chars", 160)))
    cfg["decoder_train_layers"] = int(max(1, cfg.get("decoder_train_layers", 4)))
    cfg["max_target_length"] = int(max(16, cfg.get("max_target_length", 128)))
    cfg["max_new_tokens"] = int(max(8, cfg.get("max_new_tokens", 72)))
    cfg["min_new_tokens"] = int(max(0, min(cfg["max_new_tokens"] - 1, cfg.get("min_new_tokens", 8))))
    cfg["num_beams"] = int(max(1, cfg.get("num_beams", 4)))
    cfg["no_repeat_ngram_size"] = int(max(0, cfg.get("no_repeat_ngram_size", 3)))
    cfg["prefix_token_count"] = int(max(1, cfg.get("prefix_token_count", 12)))
    cfg["debug_topk"] = int(max(1, cfg.get("debug_topk", 5)))
    cfg["eval_samples"] = int(max(1, cfg.get("eval_samples", 24)))
    cfg["debug_image_count"] = int(max(1, cfg.get("debug_image_count", 4)))
    cfg["log_interval"] = int(max(1, cfg.get("log_interval", 20)))
    cfg["early_stop_patience"] = int(max(1, cfg.get("early_stop_patience", 4)))
    cfg["early_stop_warmup_epochs"] = int(max(1, cfg.get("early_stop_warmup_epochs", 2)))
    return cfg
