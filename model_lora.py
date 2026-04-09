import torch
from peft import LoraConfig, PeftModel, TaskType, get_peft_model
from transformers import AutoModelForVision2Seq


def _freeze_base(model) -> None:
    for param in model.parameters():
        param.requires_grad = False


def build_model(config: dict, device: torch.device):
    dtype = torch.bfloat16 if config.get("bf16") and device.type == "cuda" else torch.float32
    print(f"  Loading {config['model_name']} ...")
    model = AutoModelForVision2Seq.from_pretrained(
        config["model_name"],
        torch_dtype=dtype,
        _attn_implementation="eager",
    )
    _freeze_base(model)
    model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=config["lora_r"],
        lora_alpha=config["lora_alpha"],
        target_modules=list(config["lora_target_modules"]),
        lora_dropout=float(config["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_cfg)
    model = model.to(device)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    return model


def load_for_inference(config: dict, save_dir: str, device: torch.device):
    dtype = torch.bfloat16 if config.get("bf16") and device.type == "cuda" else torch.float32
    base = AutoModelForVision2Seq.from_pretrained(
        config["model_name"],
        torch_dtype=dtype,
        _attn_implementation="eager",
    )
    model = PeftModel.from_pretrained(base, save_dir)
    model = model.to(device)
    return model.eval()
