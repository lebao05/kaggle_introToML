"""Compatibility wrapper for the active pixel-decoder training pipeline."""

from train_pixel import train as _pixel_train

if __name__ == "__main__":
    _pixel_train()
    raise SystemExit

import json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from config import get_config, resolve_config
from dataset import PixelTextDataset, collate_fn, _clean_split, _render_rgb, build_chat
from model import build_model
from utils import get_lang_prompt


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ── Data split ────────────────────────────────────────────────────────────────

def stratified_split(df, col: str, frac: float, seed: int = 42):
    val_parts = []
    for _, g in df.groupby(col, group_keys=False):
        n = max(1, int(round(len(g) * frac))) if len(g) > 1 else 1
        n = min(n, len(g) - 1) if len(g) > 1 else 1
        val_parts.append(g.sample(n=n, random_state=seed))
    raw   = pd.concat(val_parts)
    val   = raw.sample(frac=1, random_state=seed).reset_index(drop=True)
    train = df.drop(raw.index).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, val


# ── Metrics ───────────────────────────────────────────────────────────────────

def _cer(a: str, b: str) -> float:
    d = max(len(a), len(b), 1)
    return (sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))) / d


# ── Eval generation ───────────────────────────────────────────────────────────

@torch.no_grad()
def eval_generate(model, processor, val_df, config, device, n: int = 30) -> list[dict]:
    """Generate predictions cho n val samples, tra ve records."""
    model.eval()
    records = []
    tok = processor.tokenizer

    for i in range(min(n, len(val_df))):
        row       = val_df.iloc[i]
        context   = str(row["context"])
        target    = str(row["target"])
        language  = str(row["language"])
        max_width = int(row["max_width"])

        context, target = _clean_split(context, target)
        img = _render_rgb(context, max_width, config["font_size"], config["padding"])

        messages, images = build_chat(img, language, target=None)
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        enc = processor(
            text=text, images=images,
            return_tensors="pt",
            do_image_splitting=False,
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        prompt_len = enc["input_ids"].shape[1]

        out = model.generate(
            **enc,
            max_new_tokens = config["max_new_tokens"],
            min_new_tokens = config["min_new_tokens"],
            do_sample      = False,
            pad_token_id   = tok.pad_token_id,
            eos_token_id   = tok.eos_token_id,
        )
        new_ids = out[0][prompt_len:]
        pred = tok.decode(new_ids, skip_special_tokens=True).strip()

        records.append({
            "lang":     language,
            "pred":     pred,
            "true":     target,
            "cer":      _cer(pred, target),
            "pred_len": len(pred),
            "true_len": len(target),
            "short":    len(pred) < 0.2 * max(len(target), 1),
        })

    return records


def summarise_metrics(records: list[dict]) -> dict:
    if not records:
        return {}
    per_lang = {
        lang: float(np.mean([r["cer"] for r in records if r["lang"] == lang]))
        for lang in sorted({r["lang"] for r in records})
    }
    return {
        "cer_mean":      float(np.mean([r["cer"] for r in records])),
        "cer_per_lang":  per_lang,
        "mean_pred_len": float(np.mean([r["pred_len"] for r in records])),
        "mean_true_len": float(np.mean([r["true_len"] for r in records])),
        "pct_too_short": float(np.mean([r["short"] for r in records])) * 100,
    }


def print_metrics(m: dict):
    if not m:
        return
    print(f"  [Gen] CER={m['cer_mean']:.3f}  "
          f"pred={m['mean_pred_len']:.0f}/true={m['mean_true_len']:.0f}  "
          f"short={m['pct_too_short']:.1f}%")
    for lang, cer in m["cer_per_lang"].items():
        flag = "  !!!" if cer > 0.7 else ""
        print(f"    {lang:>10s}: {cer:.3f}{flag}")


def should_save_best(m: dict, best_cer: float, config: dict) -> bool:
    if not m:
        return False
    if m["pct_too_short"] > config["best_max_short_pct"]:
        return False
    return m["cer_mean"] < best_cer


# ── Optimizer ─────────────────────────────────────────────────────────────────

def build_optimizer(model, config: dict, total_steps: int):
    """Chi optimize LoRA params (requires_grad=True)."""
    lora_params = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(lora_params, lr=config["lr"], weight_decay=config["weight_decay"])

    warmup = max(1, int(total_steps * config["warmup_ratio"]))
    decay  = max(1, total_steps - warmup)
    sch = SequentialLR(opt, schedulers=[
        LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
        CosineAnnealingLR(opt, T_max=decay, eta_min=1e-7),
    ], milestones=[warmup])
    return opt, sch


# ── Epoch ─────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scheduler,
              config: dict, device, tag: str, train_mode: bool) -> float:
    model.train() if train_mode else model.eval()
    # bfloat16 khong can GradScaler (khac voi fp16)
    use_bf16 = bool(config.get("bf16") and device.type == "cuda")
    losses   = []
    ctx      = torch.enable_grad if train_mode else torch.no_grad

    with ctx():
        for bi, batch in enumerate(loader):
            input_ids      = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            pixel_values   = batch["pixel_values"].to(device)
            labels         = batch["labels"].to(device)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            # SmolVLM forward: tinh loss tu causal LM objective
            # labels da mask prompt tokens = -100, chi tinh loss tren response
            with torch.autocast(device_type="cuda",
                                dtype=torch.bfloat16 if use_bf16 else torch.float32,
                                enabled=use_bf16):
                out = model(
                    input_ids      = input_ids,
                    attention_mask = attention_mask,
                    pixel_values   = pixel_values,
                    labels         = labels,
                )
                loss = out.loss

            if not torch.isfinite(loss):
                print(f"  [{tag}] non-finite loss batch {bi}, skip")
                continue

            if train_mode:
                loss.backward()
                nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config["grad_clip"],
                )
                optimizer.step()
                if scheduler:
                    scheduler.step()

            losses.append(loss.item())
            if train_mode and bi % 20 == 0:
                lr = scheduler.get_last_lr()[-1] if scheduler else config["lr"]
                print(f"  [{tag}] B{bi:>3d}/{len(loader)} "
                      f"loss={loss.item():.4f} lr={lr:.2e}")

    return float(np.mean(losses)) if losses else float("nan")


# ── Save ──────────────────────────────────────────────────────────────────────

def save(model, processor, config: dict, save_dir: str):
    """Luu LoRA adapter + processor + config."""
    os.makedirs(save_dir, exist_ok=True)
    # Luu chi LoRA adapter (khong luu base weights)
    model.save_pretrained(save_dir)
    processor.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    config = resolve_config(get_config())
    set_seed(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TRAINING: SmolVLM-256M-Instruct + LoRA")
    print("=" * 60)
    print(f"  Device:  {device}")
    print(f"  Model:   {config['model_name']}")
    print(f"  LoRA:    r={config['lora_r']} alpha={config['lora_alpha']}")
    print(f"  Epochs:  {config['epochs']}  LR={config['lr']}")

    os.makedirs(config["output_dir"], exist_ok=True)

    # 1. Processor
    print("\n[1/4] Loading processor...")
    processor = AutoProcessor.from_pretrained(config["model_name"])
    # Tat image splitting de co consistent pixel_values shape cho batching
    processor.image_processor.do_image_splitting = False
    print(f"  vocab_size={processor.tokenizer.vocab_size}")
    print(f"  pad_token_id={processor.tokenizer.pad_token_id}  "
          f"eos_token_id={processor.tokenizer.eos_token_id}")

    # 2. Data
    print("\n[2/4] Loading data...")
    df = pd.read_csv(config["train_file"])
    print(f"  {len(df)} samples | {df['language'].value_counts().to_dict()}")

    train_df, val_df = stratified_split(df, "language", config["val_split"])
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = PixelTextDataset(train_df, processor, config, is_test=False)
    val_ds   = PixelTextDataset(val_df,   processor, config, is_test=False)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, num_workers=0, pin_memory=True,
                              collate_fn=collate_fn)
    val_loader   = DataLoader(val_ds, batch_size=config["batch_size"],
                              shuffle=False, num_workers=0, pin_memory=True,
                              collate_fn=collate_fn)

    # Sanity check
    s = train_ds[0]
    resp_ids = s["labels"][s["labels"] != -100]
    print(f"\n  [SANITY] input_ids shape   = {s['input_ids'].shape}")
    print(f"  [SANITY] pixel_values shape = {s['pixel_values'].shape}")
    print(f"  [SANITY] response tokens   = {len(resp_ids)}")
    print(f"  [SANITY] response text     = "
          f"'{processor.tokenizer.decode(resp_ids, skip_special_tokens=True)[:80]}'")

    # 3. Model
    print("\n[3/4] Building model...")
    model = build_model(config, device)

    total_steps = len(train_loader) * config["epochs"]
    optimizer, scheduler = build_optimizer(model, config, total_steps)

    # 4. Training loop
    print(f"\n[4/4] Training ({total_steps} steps)...")
    best_cer   = float("inf")
    no_improve = 0
    history    = []

    for ep in range(1, config["epochs"] + 1):
        print(f"\n{'='*50} Epoch {ep}/{config['epochs']} {'='*50}")
        t0 = time.time()

        tr_loss  = run_epoch(model, train_loader, optimizer, scheduler,
                             config, device, "tr", train_mode=True)
        val_loss = run_epoch(model, val_loader, None, None,
                             config, device, "val", train_mode=False)

        elapsed = time.time() - t0
        print(f"  train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"gap={val_loss-tr_loss:+.4f}  {elapsed:.0f}s")

        # Generation metrics (on raw val_df, not val_ds)
        records = eval_generate(model, processor, val_df, config, device,
                                n=config["eval_gen_samples"])
        metrics = summarise_metrics(records)
        print_metrics(metrics)

        # Sample predictions
        for r in records[:config["sample_pred_count"]]:
            print(f"    [{r['lang']}]")
            print(f"      TRUE: {r['true'][:80]}")
            print(f"      PRED: {r['pred'][:80]}")

        history.append({
            "ep": ep, "tr": tr_loss, "val": val_loss,
            "cer": metrics.get("cer_mean", float("nan")),
            "short": metrics.get("pct_too_short", float("nan")),
        })

        # Save best
        if should_save_best(metrics, best_cer, config):
            prev     = best_cer
            best_cer = metrics["cer_mean"]
            best_dir = os.path.join(config["output_dir"], "best_adapter")
            save(model, processor, config, best_dir)
            no_improve = 0
            print(f"  >>> Saved best -> {best_dir}  "
                  f"(CER {prev:.4f} -> {best_cer:.4f})")
        else:
            no_improve += 1

        # Early stopping
        if (ep >= config["early_stop_warmup_epochs"]
                and np.isfinite(best_cer)
                and no_improve >= config["early_stop_patience"]):
            print(f"  EARLY STOP: {no_improve} epochs no improvement.")
            break

    # Save final
    final_dir = os.path.join(config["output_dir"], "final_adapter")
    save(model, processor, config, final_dir)
    print(f"\n  Saved final -> {final_dir}/")

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    if np.isfinite(best_cer):
        print(f"  Best CER: {best_cer:.4f}")
    print(f"\n  {'Ep':>3}  {'Train':>8}  {'Val':>8}  {'CER':>7}  {'Short%':>7}")
    for h in history:
        mark = " *" if np.isfinite(best_cer) and abs(h["cer"] - best_cer) < 1e-6 else ""
        print(f"  {h['ep']:>3}  {h['tr']:>8.4f}  {h['val']:>8.4f}  "
              f"{h['cer']:>7.3f}  {h['short']:>6.1f}%{mark}")
    print("=" * 60)


if __name__ == "__main__":
    train()
