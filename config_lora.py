from pathlib import Path


LANG2ID = {"English": 0, "French": 1, "German": 2, "Italian": 3, "Spanish": 4}
ID2LANG = {v: k for k, v in LANG2ID.items()}


def get_config():
    root = Path(".")
    return {
        "model_name": "HuggingFaceTB/SmolVLM-256M-Instruct",
        "train_file": str(root / "train-3.csv"),
        "test_file": str(root / "test.csv"),
        "output_dir": str(root / "outputs"),
        "val_split": 0.10,
        "font_size": 18,
        "padding": 12,
        "image_max_side": 384,
        "epochs": 12,
        "batch_size": 1,
        "grad_accum_steps": 8,
        "lr": 2e-4,
        "weight_decay": 0.01,
        "warmup_ratio": 0.10,
        "grad_clip": 1.0,
        "bf16": True,
        "max_input_length": 768,
        "max_prompt_length": 128,
        "max_new_tokens": 80,
        "num_beams": 1,
        "repetition_penalty": 1.15,
        "no_repeat_ngram_size": 3,
        "eval_samples": 24,
        "debug_image_count": 4,
        "early_stop_patience": 4,
        "early_stop_warmup_epochs": 2,
        "best_max_cer": 2.0,
        "debug_threshold": 127,
        "lora_r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
    }


def resolve_config(cfg: dict) -> dict:
    cfg = dict(cfg)
    cfg["batch_size"] = int(max(1, cfg.get("batch_size", 1)))
    cfg["grad_accum_steps"] = int(max(1, cfg.get("grad_accum_steps", 8)))
    cfg["epochs"] = int(max(1, cfg.get("epochs", 12)))
    cfg["max_input_length"] = int(max(128, cfg.get("max_input_length", 768)))
    cfg["max_prompt_length"] = int(max(64, cfg.get("max_prompt_length", 640)))
    cfg["max_new_tokens"] = int(max(8, cfg.get("max_new_tokens", 80)))
    cfg["eval_samples"] = int(max(1, cfg.get("eval_samples", 24)))
    cfg["debug_image_count"] = int(max(1, cfg.get("debug_image_count", 4)))
    cfg["early_stop_patience"] = int(max(1, cfg.get("early_stop_patience", 4)))
    cfg["early_stop_warmup_epochs"] = int(max(1, cfg.get("early_stop_warmup_epochs", 2)))
    cfg["num_beams"] = int(max(1, cfg.get("num_beams", 1)))
    cfg["lora_r"] = int(max(1, cfg.get("lora_r", 8)))
    cfg["lora_alpha"] = int(max(1, cfg.get("lora_alpha", 16)))
    cfg["image_max_side"] = int(max(128, cfg.get("image_max_side", 384)))
    if cfg["max_prompt_length"] >= cfg["max_input_length"]:
        cfg["max_prompt_length"] = max(64, cfg["max_input_length"] - 64)
    return cfg
