"""Training script for OCR -> continuation LM pipeline."""

import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
from torch.amp import autocast, GradScaler
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup

from config import get_config, resolve_config
from dataset import PixelTextDataset
from model import TrOCRmT5Model


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def stratified_split_df(df, label_col: str, val_fraction: float, random_state: int = 42):
    val_parts = []
    for _, grp in df.groupby(label_col, group_keys=False):
        n_val = max(1, int(round(len(grp) * val_fraction))) if len(grp) > 1 else 1
        n_val = min(n_val, len(grp) - 1) if len(grp) > 1 else 1
        val_parts.append(grp.sample(n=n_val, random_state=random_state))
    raw_val = pd.concat(val_parts)
    val_df = raw_val.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    train_df = df.drop(raw_val.index).sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, val_df


def _char_cer(a: str, b: str) -> float:
    if max(len(a), len(b)) == 0:
        return 0.0
    mismatch = sum(x != y for x, y in zip(a, b)) + abs(len(a) - len(b))
    return mismatch / max(len(a), len(b), 1)


@torch.no_grad()
def generation_metrics(model, val_df, val_ds, tokenizer, config, device, n=80):
    records = []
    model.eval()
    for i in range(min(n, len(val_ds))):
        sample = val_ds[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        pred = model.generate(input_ids, attention_mask, tokenizer, config)[0].strip()
        true_ids = sample["labels"].clone()
        true_ids = true_ids[true_ids != -100]
        true = tokenizer.decode(true_ids, skip_special_tokens=True).strip()
        lang = str(val_df.iloc[i]["language"])
        records.append({
            "lang": lang,
            "cer": _char_cer(pred, true),
            "pred_len": len(pred),
            "true_len": len(true),
            "too_short": len(pred) < 0.2 * max(len(true), 1),
        })
    if not records:
        return {}
    cer_per_lang = {}
    for lang in sorted({r["lang"] for r in records}):
        cer_per_lang[lang] = float(np.mean([r["cer"] for r in records if r["lang"] == lang]))
    return {
        "cer_mean": float(np.mean([r["cer"] for r in records])),
        "cer_per_lang": cer_per_lang,
        "mean_pred_len": float(np.mean([r["pred_len"] for r in records])),
        "mean_true_len": float(np.mean([r["true_len"] for r in records])),
        "pct_too_short": float(np.mean([r["too_short"] for r in records])) * 100.0,
    }


@torch.no_grad()
def sample_preds(model, val_df, val_ds, tokenizer, config, device, n=5):
    rows = []
    model.eval()
    for i in range(min(n, len(val_ds))):
        sample = val_ds[i]
        input_ids = sample["input_ids"].unsqueeze(0).to(device)
        attention_mask = sample["attention_mask"].unsqueeze(0).to(device)
        pred = model.generate(input_ids, attention_mask, tokenizer, config)[0].strip()
        true_ids = sample["labels"].clone()
        true_ids = true_ids[true_ids != -100]
        true = tokenizer.decode(true_ids, skip_special_tokens=True).strip()
        rows.append((str(val_df.iloc[i]["language"]), true, pred))
    return rows


def print_generation_metrics(metrics: dict):
    if not metrics:
        return
    print(
        f"  [Gen metrics] CER={metrics['cer_mean']:.3f}  "
        f"pred_len={metrics['mean_pred_len']:.0f}/{metrics['mean_true_len']:.0f}(true)  "
        f"too_short={metrics['pct_too_short']:.1f}%"
    )
    for lang, cer in metrics["cer_per_lang"].items():
        flag = "  ?" if cer > 0.7 else ""
        print(f"    {lang:>10s}: CER={cer:.3f}{flag}")


@torch.no_grad()
def evaluate_loss(model, loader, device, use_amp):
    model.eval()
    losses = []
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        with autocast(device_type="cuda", enabled=use_amp):
            loss = model(input_ids, attention_mask, labels)
        losses.append(loss.item())
    return float(np.mean(losses)) if losses else float("nan")


def train_one_epoch(model, loader, optimizer, scheduler, scaler, device, use_amp, grad_clip):
    model.train()
    losses = []
    t0 = time.time()
    for bi, batch in enumerate(loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optimizer.zero_grad(set_to_none=True)
        with autocast(device_type="cuda", enabled=use_amp):
            loss = model(input_ids, attention_mask, labels)
        if scaler is not None and use_amp:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
        scheduler.step()
        losses.append(loss.item())
        if bi % 20 == 0:
            dt = time.time() - t0
            eta = dt / max(1, bi + 1) * (len(loader) - bi - 1)
            lr = scheduler.get_last_lr()[0]
            print(f"  [tr] B {bi:>3d}/{len(loader)} loss={loss.item():.4f} lr={lr:.2e} {dt/max(1, bi+1):.2f}s ETA={eta:.0f}s")
    return float(np.mean(losses)) if losses else float("nan")


def should_save_best(metrics, best_cer, config):
    if not metrics:
        return False
    if metrics["pct_too_short"] > config["best_max_short_pct"]:
        return False
    return metrics["cer_mean"] < best_cer


def train():
    config = resolve_config(get_config())
    set_seed(42)
    os.makedirs(config["output_dir"], exist_ok=True)
    os.makedirs(config["checkpoint_dir"], exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    use_amp = bool(config.get("fp16", False) and device.type == "cuda")
    scaler = GradScaler("cuda") if use_amp else None

    print("=" * 60)
    print("TRAINING: OCR -> continuation LM")
    print("=" * 60)
    print(f"  Device:   {device}")
    print(f"  OCR:      {config['ocr_model_name']}")
    print(f"  LM:       {config['lm_model_name']}")
    print(f"  Epochs:   {config['epochs']}")

    print("\n[1/4] Loading tokenizer...")
    t0 = time.time()
    tokenizer = AutoTokenizer.from_pretrained(config["lm_model_name"])
    print(f"  Done {time.time()-t0:.1f}s")
    print(
        "  Token ids: "
        f"pad={tokenizer.pad_token_id} "
        f"eos={getattr(tokenizer, 'eos_token_id', None)} "
        f"bos={getattr(tokenizer, 'bos_token_id', None)}"
    )

    print("\n[2/4] Loading data...")
    df = pd.read_csv(config["train_file"])
    print(f"  {len(df)} samples | {df['language'].value_counts().to_dict()}")
    target_lens = df["target"].astype(str).str.len()
    print(f"  target: {target_lens.min()}-{target_lens.max()} chars (mean={target_lens.mean():.0f})")
    train_df, val_df = stratified_split_df(df, "language", config["val_split"], random_state=42)
    print(f"  Train: {len(train_df)}, Val: {len(val_df)}")
    print(f"  Train lang dist: {train_df['language'].value_counts().to_dict()}")
    print(f"  Val   lang dist: {val_df['language'].value_counts().to_dict()}")

    train_ds = PixelTextDataset(train_df, tokenizer, config, is_test=False)
    val_ds = PixelTextDataset(val_df, tokenizer, config, is_test=False)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True)

    sample = train_ds[0]
    sample_true_ids = sample["labels"][sample["labels"] != -100]
    print(f"\n  [SANITY] input_ids={sample['input_ids'].shape} labels={sample['labels'].shape}")
    print(f"  [SANITY] target: {tokenizer.decode(sample_true_ids, skip_special_tokens=True)}")

    print("\n[3/4] Building LM...")
    t0 = time.time()
    model = TrOCRmT5Model(config).to(device)
    built_s = time.time() - t0
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Built {built_s:.1f}s")
    print(f"  Total params:     {total_params:,}")
    print(f"  Trainable params: {trainable_params:,}")

    total_steps = len(train_loader) * config["epochs"]
    warmup_steps = int(total_steps * config["warmup_ratio"])
    optimizer = AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)

    best_cer = float("inf")
    no_improve = 0
    history = []

    print("\n[4/4] Training...")
    for epoch in range(1, config["epochs"] + 1):
        print(f"\n-- Epoch {epoch}/{config['epochs']} --")
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, scheduler, scaler, device, use_amp, config["grad_clip"])
        val_loss = evaluate_loss(model, val_loader, device, use_amp)
        metrics = generation_metrics(model, val_df, val_ds, tokenizer, config, device, n=config["eval_gen_samples"])
        epoch_s = time.time() - t0

        print(f"  train={train_loss:.4f}  val={val_loss:.4f}  {epoch_s:.0f}s")
        print_generation_metrics(metrics)
        rows = sample_preds(model, val_df, val_ds, tokenizer, config, device, n=config["sample_pred_count"])
        for lang, true, pred in rows:
            print(f"    {lang}: TRUE={true[:80]}")
            print(f"           PRED={pred[:80]}")

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "cer": metrics.get("cer_mean", float("nan")),
            "short_pct": metrics.get("pct_too_short", float("nan")),
        })

        saved_this_epoch = should_save_best(metrics, best_cer, config)
        if saved_this_epoch:
            prev_best = best_cer
            best_cer = metrics["cer_mean"]
            best_dir = os.path.join(config["output_dir"], "best_model")
            os.makedirs(best_dir, exist_ok=True)
            model.save_pretrained(best_dir)
            tokenizer.save_pretrained(best_dir)
            with open(os.path.join(best_dir, "run_config.json"), "w", encoding="utf-8") as f:
                json.dump(config, f, ensure_ascii=False, indent=2)
            print(f"  ? Saved best (CER={best_cer:.4f}, short={metrics['pct_too_short']:.1f}%)")
            improved = (prev_best - best_cer) >= config["early_stop_min_delta"]
            no_improve = 0
        else:
            improved = False

        if config.get("early_stop_on_cer", False) and epoch >= config["early_stop_warmup_epochs"] and np.isfinite(best_cer):
            if not saved_this_epoch:
                no_improve += 1
            if no_improve >= config["early_stop_patience"]:
                print(f"  ? Early stop: no CER improvement for {no_improve} epoch(s).")
                break

    final_dir = os.path.join(config["output_dir"], "final_model")
    os.makedirs(final_dir, exist_ok=True)
    model.save_pretrained(final_dir)
    tokenizer.save_pretrained(final_dir)
    with open(os.path.join(final_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    if np.isfinite(best_cer):
        print(f"  Best CER (short<={config['best_max_short_pct']:.1f}%): {best_cer:.4f}")
    else:
        print("  Best CER: n/a")
    for row in history:
        print(f"   ep={row['epoch']:>2} train={row['train_loss']:.4f} val={row['val_loss']:.4f} cer={row['cer']:.3f} short={row['short_pct']:.1f}%")


if __name__ == "__main__":
    train()
