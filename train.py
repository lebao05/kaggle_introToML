"""Training: TrOCR encoder + Projection + mT5 decoder."""

import json, os, random, time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from pathlib import Path
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import TrOCRProcessor, AutoTokenizer

from config import get_config, resolve_config, LANG2ID
from dataset import PixelTextDataset
from model import TrOCRmT5Model
from utils import render_text_to_image


# ── Seed ──────────────────────────────────────────────────────────────────────

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)


# ── Data split ────────────────────────────────────────────────────────────────

def stratified_split(df, col, frac, seed=42):
    val_parts = []
    for _, g in df.groupby(col, group_keys=False):
        n = max(1, int(round(len(g) * frac))) if len(g) > 1 else 1
        n = min(n, len(g) - 1) if len(g) > 1 else 1
        val_parts.append(g.sample(n=n, random_state=seed))
    raw = pd.concat(val_parts)
    val   = raw.sample(frac=1, random_state=seed).reset_index(drop=True)
    train = df.drop(raw.index).sample(frac=1, random_state=seed).reset_index(drop=True)
    return train, val


# ── Metrics ───────────────────────────────────────────────────────────────────

def _cer(a: str, b: str) -> float:
    d = max(len(a), len(b), 1)
    return (sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))) / d


@torch.no_grad()
def generation_metrics(model, val_df, val_ds, tokenizer, config, device, n=60):
    model.eval()
    records = []
    for i in range(min(n, len(val_ds))):
        s    = val_ds[i]
        pv   = s["pixel_values"].unsqueeze(0).to(device)
        pred = model.generate(pv, tokenizer, config)[0]
        true_ids = s["labels"][s["labels"] != -100]
        true = tokenizer.decode(true_ids, skip_special_tokens=True).strip()
        records.append({
            "lang": str(val_df.iloc[i]["language"]),
            "cer":  _cer(pred, true),
            "pred_len": len(pred),
            "true_len": len(true),
            "short": len(pred) < 0.2 * max(len(true), 1),
        })
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


def print_metrics(m):
    if not m: return
    print(f"  [Gen] CER={m['cer_mean']:.3f}  "
          f"pred={m['mean_pred_len']:.0f}/true={m['mean_true_len']:.0f}  "
          f"short={m['pct_too_short']:.1f}%")
    for lang, cer in m["cer_per_lang"].items():
        print(f"    {lang:>10s}: {cer:.3f}{'  ⚠' if cer > 0.7 else ''}")


@torch.no_grad()
def sample_preds(model, val_df, val_ds, tokenizer, config, device, n=4):
    model.eval()
    rows = []
    for i in range(min(n, len(val_ds))):
        s    = val_ds[i]
        pv   = s["pixel_values"].unsqueeze(0).to(device)
        pred = model.generate(pv, tokenizer, config)[0]
        true_ids = s["labels"][s["labels"] != -100]
        true = tokenizer.decode(true_ids, skip_special_tokens=True).strip()
        rows.append((str(val_df.iloc[i]["language"]), true, pred))
    return rows


def should_save_best(m, best_cer, config):
    if not m: return False
    if m["pct_too_short"] > config["best_max_short_pct"]: return False
    return m["cer_mean"] < best_cer


# ── Optimizer ─────────────────────────────────────────────────────────────────

def build_optimizer(model, config, total_steps):
    """3 param groups với LR riêng: encoder (thấp) / proj / decoder (cao)."""
    enc_p  = [p for n, p in model.named_parameters() if "encoder" in n and p.requires_grad]
    dec_p  = [p for n, p in model.named_parameters() if ("decoder" in n or "lm_head" in n) and p.requires_grad]
    proj_p = [p for n, p in model.named_parameters()
              if not any(k in n for k in ("encoder", "decoder", "lm_head")) and p.requires_grad]

    opt = AdamW([
        {"params": enc_p,  "lr": config["enc_lr"]},
        {"params": dec_p,  "lr": config["dec_lr"]},
        {"params": proj_p, "lr": config["proj_lr"]},
    ], weight_decay=config["weight_decay"])

    warmup = max(1, int(total_steps * config["warmup_ratio"]))
    decay  = max(1, total_steps - warmup)
    sch = SequentialLR(opt, schedulers=[
        LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
        CosineAnnealingLR(opt, T_max=decay, eta_min=1e-7),
    ], milestones=[warmup])
    return opt, sch


# ── Epoch ─────────────────────────────────────────────────────────────────────

def run_epoch(model, loader, optimizer, scheduler, scaler,
              config, device, tag, train_mode):
    model.train() if train_mode else model.eval()
    patch_drop = config["enc_patch_drop"] if train_mode else 0.0
    use_amp    = bool(config.get("fp16") and device.type == "cuda")
    losses     = []
    ctx        = torch.enable_grad if train_mode else torch.no_grad

    with ctx():
        for bi, batch in enumerate(loader):
            pv = batch["pixel_values"].to(device)
            lb = batch["labels"].to(device)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with autocast(device_type="cuda", enabled=use_amp):
                loss = model(pv, lb, patch_drop=patch_drop)

            if not torch.isfinite(loss):
                print(f"  [{tag}] non-finite loss batch {bi}, skip")
                continue

            if train_mode:
                if use_amp and scaler:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    scaler.step(optimizer); scaler.update()
                else:
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                    optimizer.step()
                if scheduler: scheduler.step()

            losses.append(loss.item())
            if train_mode and bi % 20 == 0:
                lr = scheduler.get_last_lr()[-1] if scheduler else config["dec_lr"]
                print(f"  [{tag}] B{bi:>3d}/{len(loader)} loss={loss.item():.4f} lr={lr:.2e}")

    return float(np.mean(losses)) if losses else float("nan")


# ── Save ──────────────────────────────────────────────────────────────────────

def save(model, trocr_processor, mt5_tokenizer, config, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model_weights.pt"))
    trocr_processor.save_pretrained(save_dir)
    mt5_tokenizer.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


# ── Main ──────────────────────────────────────────────────────────────────────

def train():
    config = resolve_config(get_config())
    set_seed(42)

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("fp16") and device.type == "cuda")
    scaler  = GradScaler("cuda") if use_amp else None

    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(os.path.join(config["output_dir"], "checkpoints"), exist_ok=True)

    print("=" * 60)
    print("TRAINING: TrOCR encoder + Projection + mT5 decoder")
    print("=" * 60)
    print(f"  Device:  {device}")
    print(f"  Encoder: {config['trocr_name']}")
    print(f"  Decoder: {config['mt5_name']}")
    print(f"  Epochs:  {config['epochs']}")

    # ── 1. Processors ─────────────────────────────────────────────────────────
    print("\n[1/4] Loading processors...")
    trocr_processor = TrOCRProcessor.from_pretrained(config["trocr_name"])
    mt5_tokenizer   = AutoTokenizer.from_pretrained(config["mt5_name"])
    print(f"  mT5 vocab_size={mt5_tokenizer.vocab_size}")

    # ── 2. Data ───────────────────────────────────────────────────────────────
    print("\n[2/4] Loading data...")
    df = pd.read_csv(config["train_file"])
    print(f"  {len(df)} samples | {df['language'].value_counts().to_dict()}")

    train_df, val_df = stratified_split(df, "language", config["val_split"])
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = PixelTextDataset(train_df, trocr_processor, mt5_tokenizer, config)
    val_ds   = PixelTextDataset(val_df,   trocr_processor, mt5_tokenizer, config)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True,  num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    # Sanity check
    s = train_ds[0]
    true_ids = s["labels"][s["labels"] != -100]
    print(f"\n  [SANITY] pixel_values={s['pixel_values'].shape}")
    print(f"  [SANITY] target: '{mt5_tokenizer.decode(true_ids, skip_special_tokens=True)}'")

    # ── 3. Model ──────────────────────────────────────────────────────────────
    print("\n[3/4] Building model...")
    model = TrOCRmT5Model(config).to(device)
    total     = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    enc_p     = sum(p.numel() for n, p in model.named_parameters() if "encoder" in n)
    dec_p     = sum(p.numel() for n, p in model.named_parameters() if "decoder" in n or "lm_head" in n)
    proj_p    = sum(p.numel() for n, p in model.named_parameters()
                   if not any(k in n for k in ("encoder","decoder","lm_head")))
    print(f"  Encoder:    {enc_p:>12,}")
    print(f"  Decoder:    {dec_p:>12,}")
    print(f"  Projection: {proj_p:>12,}")
    print(f"  Total:      {total:>12,}  trainable={trainable:,}")

    total_steps = len(train_loader) * config["epochs"]
    optimizer, scheduler = build_optimizer(model, config, total_steps)

    # ── 4. Training loop ──────────────────────────────────────────────────────
    print("\n[4/4] Training...")
    best_cer   = float("inf")
    no_improve = 0
    history    = []

    for ep in range(1, config["epochs"] + 1):
        print(f"\n── Epoch {ep}/{config['epochs']} ──")
        t0 = time.time()

        tr_loss  = run_epoch(model, train_loader, optimizer, scheduler,
                             scaler, config, device, "tr", train_mode=True)
        val_loss = run_epoch(model, val_loader, None, None, None,
                             config, device, "val", train_mode=False)
        metrics  = generation_metrics(model, val_df, val_ds, mt5_tokenizer,
                                      config, device, n=config["eval_gen_samples"])

        print(f"  train={tr_loss:.4f}  val={val_loss:.4f}  "
              f"gap={val_loss-tr_loss:+.4f}  {time.time()-t0:.0f}s")
        print(f"  memory_scale={model.memory_scale.item():.3f}")
        print_metrics(metrics)

        rows = sample_preds(model, val_df, val_ds, mt5_tokenizer, config, device)
        for lang, true, pred in rows:
            print(f"    {lang}: TRUE={true[:80]}")
            print(f"           PRED={pred[:80]}")

        history.append({
            "ep": ep, "tr": tr_loss, "val": val_loss,
            "cer": metrics.get("cer_mean", float("nan")),
            "short": metrics.get("pct_too_short", float("nan")),
        })

        # Save best
        saved = should_save_best(metrics, best_cer, config)
        if saved:
            prev     = best_cer
            best_cer = metrics["cer_mean"]
            best_dir = os.path.join(config["output_dir"], "best_model")
            save(model, trocr_processor, mt5_tokenizer, config, best_dir)
            improved   = (prev - best_cer) >= config["early_stop_min_delta"]
            no_improve = 0
            print(f"  ★ Saved best → {best_dir}  (CER {prev:.4f} → {best_cer:.4f})")
        else:
            improved    = False
            no_improve += 1

        # Early stopping
        if (ep >= config["early_stop_warmup_epochs"]
                and np.isfinite(best_cer)
                and no_improve >= config["early_stop_patience"]):
            print(f"  ⏹ Early stop: {no_improve} epochs không cải thiện.")
            break

    # Save final
    final_dir = os.path.join(config["output_dir"], "final_model")
    save(model, trocr_processor, mt5_tokenizer, config, final_dir)
    print(f"\n  Saved final → {final_dir}/")

    # Summary
    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    if np.isfinite(best_cer):
        print(f"  Best CER: {best_cer:.4f}")
    print(f"\n  {'Ep':>3}  {'Train':>8}  {'Val':>8}  {'CER':>7}  {'Short%':>7}")
    for h in history:
        mk = " ★" if np.isfinite(best_cer) and abs(h["cer"] - best_cer) < 1e-6 else ""
        print(f"  {h['ep']:>3}  {h['tr']:>8.4f}  {h['val']:>8.4f}  "
              f"{h['cer']:>7.3f}  {h['short']:>6.1f}%{mk}")
    print("=" * 60)


if __name__ == "__main__":
    train()
