import torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


def build_processor(model_name: str):
    return TrOCRProcessor.from_pretrained(model_name)


def freeze_decoder_top_layers(model, num_layers: int) -> dict:
    decoder = model.decoder.model.decoder
    layers = decoder.layers
    keep_from = max(0, len(layers) - int(num_layers))

    total_decoder = 0
    trainable_decoder = 0
    unfrozen_names = []

    for name, param in model.decoder.named_parameters():
        total_decoder += param.numel()
        param.requires_grad = False

    for idx, layer in enumerate(layers):
        if idx < keep_from:
            continue
        for name, param in layer.named_parameters():
            param.requires_grad = True
            trainable_decoder += param.numel()
            unfrozen_names.append(f"layers.{idx}.{name}")

    extra_modules = []
    if hasattr(model.decoder.model, "decoder") and hasattr(model.decoder.model.decoder, "layernorm_embedding"):
        extra_modules.append(("decoder.layernorm_embedding", model.decoder.model.decoder.layernorm_embedding))
    if hasattr(model.decoder.model, "decoder") and hasattr(model.decoder.model.decoder, "embed_positions"):
        extra_modules.append(("decoder.embed_positions", model.decoder.model.decoder.embed_positions))
    if hasattr(model.decoder.model, "decoder") and hasattr(model.decoder.model.decoder, "embed_tokens"):
        extra_modules.append(("decoder.embed_tokens", model.decoder.model.decoder.embed_tokens))
    if hasattr(model.decoder, "output_projection"):
        extra_modules.append(("output_projection", model.decoder.output_projection))
    if hasattr(model.decoder, "lm_head"):
        extra_modules.append(("lm_head", model.decoder.lm_head))

    for prefix, module in extra_modules:
        for name, param in module.named_parameters():
            if not param.requires_grad:
                param.requires_grad = True
                trainable_decoder += param.numel()
                unfrozen_names.append(f"{prefix}.{name}")

    return {
        "mode": "top_layers",
        "decoder_total": total_decoder,
        "decoder_trainable": trainable_decoder,
        "decoder_kept_layers": list(range(keep_from, len(layers))),
        "decoder_unfrozen_count": len(unfrozen_names),
        "decoder_unfrozen_preview": unfrozen_names[:16],
    }


def build_model(config: dict, processor, device: torch.device):
    dtype = torch.bfloat16 if config.get("bf16") and device.type == "cuda" else torch.float32
    model = VisionEncoderDecoderModel.from_pretrained(
        config["model_name"],
        torch_dtype=dtype,
    )
    model.config.decoder_start_token_id = processor.tokenizer.cls_token_id
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    model.config.eos_token_id = processor.tokenizer.sep_token_id
    model.config.vocab_size = model.config.decoder.vocab_size
    model.generation_config.decoder_start_token_id = model.config.decoder_start_token_id
    model.generation_config.pad_token_id = model.config.pad_token_id
    model.generation_config.eos_token_id = model.config.eos_token_id
    if config.get("freeze_encoder", True):
        for param in model.encoder.parameters():
            param.requires_grad = False
    freeze_info = None
    if config.get("decoder_tune_mode") == "top_layers":
        freeze_info = freeze_decoder_top_layers(model, int(config.get("decoder_train_layers", 4)))
    model._freeze_info = freeze_info
    model = model.to(device)
    return model


def save_bundle(model, processor, config: dict, save_dir: str) -> None:
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    import json, os
    with open(os.path.join(save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def load_bundle(save_dir: str, device: torch.device):
    processor = TrOCRProcessor.from_pretrained(save_dir)
    model = VisionEncoderDecoderModel.from_pretrained(save_dir)
    model = model.to(device).eval()
    return model, processor
