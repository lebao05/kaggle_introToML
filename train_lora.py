import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from config_lora import get_config, resolve_config
from dataset_lora import VisualContinuationDataset, collate_fn
from model_lora import build_model
from utils import render_text_to_image


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split(df, col: str, frac: float, seed: int = 42):
    val_parts = []
    for _, g in df.groupby(col, group_keys=False):
        n = max(1, int(round(len(g) * frac))) if len(g) > 1 else 1
        n = min(n, len(g) - 1) if len(g) > 1 else 1
        val_parts.append(g.sample(n=n, random_state=seed))
    raw = pd.concat(val_parts)
    val = raw.sample(frac=1, random_state=seed).reset_index(drop=True)
    train = df.drop(raw.index).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, val


def build_optimizer(model, config: dict, total_steps: int):
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = AdamW(trainable, lr=config["lr"], weight_decay=config["weight_decay"])
    warmup = max(1, int(total_steps * config["warmup_ratio"]))
    decay = max(1, total_steps - warmup)
    scheduler = SequentialLR(
        optimizer,
        schedulers=[
            LinearLR(optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(optimizer, T_max=decay, eta_min=1e-7),
        ],
        milestones=[warmup],
    )
    return optimizer, scheduler


def char_error_rate(pred: str, target: str) -> float:
    a = pred or ""
    b = target or ""
    if not b:
        return 0.0 if not a else 1.0
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        prev = dp[0]
        dp[0] = i
        for j, cb in enumerate(b, start=1):
            old = dp[j]
            cost = 0 if ca == cb else 1
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + cost)
            prev = old
    return dp[-1] / max(len(b), 1)


def _tensor_inputs(batch: dict, device: torch.device) -> dict:
    out = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor) and key != "labels":
            tensor = value.to(device)
            if key == "pixel_values" and tensor.ndim == 4:
                tensor = tensor.unsqueeze(1)
            if key == "pixel_attention_mask" and tensor.ndim == 3:
                tensor = tensor.unsqueeze(1)
            out[key] = tensor
    return out


def _single_inputs(sample: dict, device: torch.device) -> dict:
    out = {}
    for key, value in sample.items():
        if isinstance(value, torch.Tensor) and key != "labels":
            tensor = value.unsqueeze(0).to(device)
            if key == "pixel_values" and tensor.ndim == 4:
                tensor = tensor.unsqueeze(1)
            if key == "pixel_attention_mask" and tensor.ndim == 3:
                tensor = tensor.unsqueeze(1)
            out[key] = tensor
    return out


def clean_generated_text(text: str) -> str:
    text = " ".join((text or "").split())
    bad_prefixes = [
        "Assistant:",
        "Assistant :",
        "Answer:",
        "Answer :",
        "Response:",
        "Response :",
        "The image shows",
        "The text is",
        "Theanswer",
    ]
    changed = True
    while changed:
        changed = False
        for prefix in bad_prefixes:
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip(" -:\n\t")
                changed = True
    return text.strip()


def generate_prediction(model, processor, sample: dict, config: dict, device: torch.device) -> str:
    model_inputs = _single_inputs(sample, device)
    prompt_len = int(model_inputs["input_ids"].shape[1])
    with torch.no_grad():
        output_ids = model.generate(
            **model_inputs,
            max_new_tokens=config["max_new_tokens"],
            num_beams=config["num_beams"],
            do_sample=False,
            use_cache=True,
            repetition_penalty=float(config.get("repetition_penalty", 1.15)),
            no_repeat_ngram_size=int(config.get("no_repeat_ngram_size", 3)),
            pad_token_id=processor.tokenizer.pad_token_id,
            eos_token_id=processor.tokenizer.eos_token_id,
        )
    new_tokens = output_ids[:, prompt_len:]
    text = processor.batch_decode(new_tokens, skip_special_tokens=True)[0].strip()
    return clean_generated_text(text)


@torch.no_grad()
def evaluate_generation(model, processor, dataset, config: dict, device, epoch: int) -> dict:
    model.eval()
    limit = min(int(config["eval_samples"]), len(dataset))
    out_dir = Path("debug_outputs/train") / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cers = []
    short_flags = []
    pred_lens = []
    true_lens = []
    lang_rows = []
    examples = []

    for i in range(limit):
        sample = dataset[i]
        pred = generate_prediction(model, processor, sample, config, device)
        true = sample["target_text"]
        cer = char_error_rate(pred, true)
        cers.append(cer)
        pred_lens.append(len(pred))
        true_lens.append(len(true))
        short_flags.append(float(len(pred) < max(4, int(0.5 * max(len(true), 1)))))
        lang_rows.append(sample["language"])
        examples.append((sample, pred, true, cer))

    per_lang = {}
    for lang in sorted(set(lang_rows)):
        idxs = [i for i, row_lang in enumerate(lang_rows) if row_lang == lang]
        per_lang[lang] = float(np.mean([cers[i] for i in idxs]))

    for i, (sample, pred, true, cer) in enumerate(examples[: int(config["debug_image_count"])]):
        max_width = int(sample["max_width"])
        render_text_to_image(sample["context_text"], max_width, config["font_size"], config["padding"]).save(
            out_dir / f"context_{i}.png"
        )
        render_text_to_image(true, max_width, config["font_size"], config["padding"]).save(
            out_dir / f"target_{i}.png"
        )
        render_text_to_image(pred, max_width, config["font_size"], config["padding"]).save(
            out_dir / f"pred_{i}.png"
        )
        txt = [
            f"language: {sample['language']}",
            f"cer: {cer:.4f}",
            "",
            f"TRUE: {true}",
            "",
            f"PRED: {pred}",
        ]
        (out_dir / f"sample_{i}.txt").write_text("\n".join(txt), encoding="utf-8")

    return {
        "cer_mean": float(np.mean(cers)) if cers else float("inf"),
        "pred_len": float(np.mean(pred_lens)) if pred_lens else 0.0,
        "true_len": float(np.mean(true_lens)) if true_lens else 0.0,
        "short_pct": 100.0 * float(np.mean(short_flags)) if short_flags else 0.0,
        "cer_per_lang": per_lang,
        "examples": examples[:4],
        "debug_dir": str(out_dir),
    }


def run_epoch(model, loader, optimizer, scheduler, config: dict, device, tag: str, train_mode: bool) -> float:
    model.train() if train_mode else model.eval()
    use_bf16 = bool(config.get("bf16") and device.type == "cuda")
    losses = []
    grad_accum = int(config["grad_accum_steps"])

    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    for bi, batch in enumerate(loader):
        labels = batch["labels"].to(device)
        model_inputs = _tensor_inputs(batch, device)

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                outputs = model(**model_inputs, labels=labels)
                loss = outputs.loss

            if not torch.isfinite(loss):
                print(f"  [{tag}] non-finite loss batch {bi}, skip")
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                continue

            step_loss = loss / grad_accum if train_mode else loss
            if train_mode:
                step_loss.backward()

        losses.append(float(loss.item()))

        if train_mode:
            do_step = (bi + 1) % grad_accum == 0 or (bi + 1) == len(loader)
            if do_step:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config["grad_clip"],
                )
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()

            if bi % 20 == 0:
                lr = scheduler.get_last_lr()[-1] if scheduler else config["lr"]
                print(f"  [{tag}] B{bi:>3d}/{len(loader)} loss={loss.item():.4f} lr={lr:.2e}")

    return float(np.mean(losses)) if losses else float("nan")


def save(model, processor, config: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def train():
    config = resolve_config(get_config())
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TRAINING: context image -> SmolVLM + LoRA -> continuation text")
    print("=" * 60)
    print(f"  Device:  {device}")
    print(f"  Model:   {config['model_name']}")
    print(f"  Epochs:  {config['epochs']}  LR={config['lr']}")
    print(f"  LoRA:    r={config['lora_r']} alpha={config['lora_alpha']} dropout={config['lora_dropout']}")

    os.makedirs(config["output_dir"], exist_ok=True)

    print("\n[1/4] Loading processor...")
    processor = AutoProcessor.from_pretrained(config["model_name"])
    processor.tokenizer.padding_side = "right"
    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token or processor.tokenizer.unk_token
    if hasattr(processor, "image_processor"):
        processor.image_processor.do_image_splitting = False

    print("\n[2/4] Loading data...")
    df = pd.read_csv(config["train_file"])
    print(f"  {len(df)} samples | {df['language'].value_counts().to_dict()}")
    train_df, val_df = stratified_split(df, "language", config["val_split"])
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = VisualContinuationDataset(train_df, processor, config, is_test=False)
    val_ds = VisualContinuationDataset(val_df, processor, config, is_test=False)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    sample = train_ds[0]
    print(f"\n  [SANITY] pixel_values shape = {tuple(sample['pixel_values'].shape)}")
    if "pixel_attention_mask" in sample:
        print(f"  [SANITY] pixel_attn shape   = {tuple(sample['pixel_attention_mask'].shape)}")
    print(f"  [SANITY] image_max_side    = {config['image_max_side']}")
    print(f"  [SANITY] prompt            = '{sample['prompt_text']}'")
    print(f"  [SANITY] prompt_len        = {sample['prompt_len']}")
    print(f"  [SANITY] language          = {sample['language']}")
    print(f"  [SANITY] target text       = '{sample['target_text'][:80]}'")

    print("\n[3/4] Building model...")
    model = build_model(config, device)
    model.config.use_cache = False
    model.config.pad_token_id = processor.tokenizer.pad_token_id
    total_steps = max(1, (len(train_loader) * config["epochs"]) // config["grad_accum_steps"])
    optimizer, scheduler = build_optimizer(model, config, total_steps)

    print(f"\n[4/4] Training ({total_steps} optimizer steps)...")
    best_cer = float("inf")
    no_improve = 0
    history = []

    for ep in range(1, config["epochs"] + 1):
        print(f"\n{'=' * 50} Epoch {ep}/{config['epochs']} {'=' * 50}")
        t0 = time.time()
        tr_loss = run_epoch(model, train_loader, optimizer, scheduler, config, device, "tr", train_mode=True)
        val_loss = run_epoch(model, val_loader, None, None, config, device, "val", train_mode=False)
        gen = evaluate_generation(model, processor, val_ds, config, device, ep)
        elapsed = time.time() - t0

        print(f"  train={tr_loss:.4f}  val={val_loss:.4f}  gap={val_loss-tr_loss:+.4f}  {elapsed:.0f}s")
        print(
            f"  [Gen] CER={gen['cer_mean']:.3f}  pred={gen['pred_len']:.0f}/true={gen['true_len']:.0f}  "
            f"short={gen['short_pct']:.1f}%"
        )
        for lang, cer in gen["cer_per_lang"].items():
            flag = "  !!!" if cer > 0.95 else ""
            print(f"    {lang:>10s}: {cer:.3f}{flag}")
        for sample_row, pred, true, _ in gen["examples"]:
            print(f"    [{sample_row['language']}]")
            print(f"      TRUE: {true}")
            print(f"      PRED: {pred}")

        history.append(
            {
                "ep": ep,
                "train": tr_loss,
                "val": val_loss,
                "cer": gen["cer_mean"],
                "short_pct": gen["short_pct"],
            }
        )

        can_save = gen["cer_mean"] <= float(config["best_max_cer"])
        improved = (not np.isfinite(best_cer)) or (gen["cer_mean"] < best_cer - 1e-6)
        if can_save and improved:
            prev = best_cer
            best_cer = gen["cer_mean"]
            best_dir = os.path.join(config["output_dir"], "best_model")
            save(model, processor, config, best_dir)
            no_improve = 0
            print(f"  >>> Saved best -> {best_dir}  (CER {prev:.4f} -> {best_cer:.4f})")
            print(f"  >>> Debug -> {gen['debug_dir']}")
        else:
            no_improve += 1

        if ep >= config["early_stop_warmup_epochs"] and no_improve >= config["early_stop_patience"]:
            print(f"  EARLY STOP: {no_improve} epochs without improvement.")
            break

    final_dir = os.path.join(config["output_dir"], "final_model")
    save(model, processor, config, final_dir)
    print(f"\n  Saved final -> {final_dir}/")

    print(f"\n{'=' * 60}")
    print("TRAINING COMPLETE")
    print(f"{'=' * 60}")
    if np.isfinite(best_cer):
        print(f"  Best CER: {best_cer:.4f}")
    print(f"\n  {'Ep':>3}  {'Train':>8}  {'Val':>8}  {'CER':>7}  {'Short%':>7}")
    for row in history:
        mark = " *" if np.isfinite(best_cer) and abs(row["cer"] - best_cer) < 1e-6 else ""
        print(
            f"  {row['ep']:>3}  {row['train']:>8.4f}  {row['val']:>8.4f}  "
            f"{row['cer']:>7.3f}  {row['short_pct']:>7.1f}{mark}"
        )
    print("=" * 60)


if __name__ == "__main__":
    train()
