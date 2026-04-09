"""Training: SmolVLM vision encoder + direct pixel decoder."""

import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader
from transformers import AutoProcessor

from config import get_config, resolve_config
from dataset import PixelTextDataset, collate_fn
from model import build_model
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


def weighted_bce(logits: torch.Tensor, target: torch.Tensor, pos_weight: float) -> torch.Tensor:
    pos_weight_t = torch.tensor(float(pos_weight), device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight_t)


def batch_f1(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> float:
    pred_bin = torch.sigmoid(logits) >= threshold
    target_bin = target >= threshold
    tp = (pred_bin & target_bin).sum(dim=(1, 2, 3)).float()
    fp = (pred_bin & ~target_bin).sum(dim=(1, 2, 3)).float()
    fn = (~pred_bin & target_bin).sum(dim=(1, 2, 3)).float()
    f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
    return float(f1.mean().item())


def logits_to_uint8(logits: torch.Tensor, threshold: float = 0.5) -> np.ndarray:
    probs = torch.sigmoid(logits.detach()).float().cpu().numpy()
    if probs.ndim == 3:
        probs = probs[0]
    return ((probs >= threshold).astype(np.uint8) * 255)


@torch.no_grad()
def save_epoch_debug(model, loader, config: dict, device, epoch: int):
    model.eval()
    out_dir = Path("debug_outputs/train") / f"epoch_{epoch:03d}"
    out_dir.mkdir(parents=True, exist_ok=True)
    batch = next(iter(loader), None)
    if batch is None:
        return

    pixel_values = batch["pixel_values"].to(device)
    lang_ids = batch["lang_ids"].to(device)
    logits = model(pixel_values, lang_ids)
    n = min(int(config.get("debug_image_count", 4)), logits.shape[0])
    thr = float(config.get("debug_threshold", 0.5))

    for i in range(n):
        pred_arr = logits_to_uint8(logits[i], threshold=thr)
        tgt_arr = (batch["target_mask"][i].squeeze(0).cpu().numpy() >= 0.5).astype(np.uint8) * 255
        max_width = int(batch["max_width"][i].item())
        context_text = batch["context_text"][i]

        pred_img = Image.fromarray(pred_arr, mode="L").resize((max_width, config["output_h"]), Image.Resampling.NEAREST)
        target_img = Image.fromarray(tgt_arr, mode="L").resize((max_width, config["output_h"]), Image.Resampling.NEAREST)
        ctx_img = render_text_to_image(context_text, max_width, config["font_size"], config["padding"])

        pred_img.save(out_dir / f"pred_{i}.png")
        target_img.save(out_dir / f"target_{i}.png")
        ctx_img.save(out_dir / f"context_{i}.png")

    dens = float((torch.sigmoid(logits).detach() >= thr).float().mean().item())
    print(f"  [Dbg] saved {n} samples -> {out_dir}  density={dens:.3f}")


@torch.no_grad()
def eval_pixel_metrics(model, loader, config: dict, device) -> dict:
    model.eval()
    losses = []
    f1s = []
    lang_rows = []

    for batch in loader:
        pixel_values = batch["pixel_values"].to(device)
        lang_ids = batch["lang_ids"].to(device)
        target_mask = batch["target_mask"].to(device)

        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16 if config.get("bf16") else torch.float32,
            enabled=bool(config.get("bf16") and device.type == "cuda"),
        ):
            logits = model(pixel_values, lang_ids)
            loss = weighted_bce(logits, target_mask, config["pos_weight"])

        losses.append(float(loss.item()))
        pred_bin = torch.sigmoid(logits) >= 0.5
        tgt_bin = target_mask >= 0.5
        tp = (pred_bin & tgt_bin).sum(dim=(1, 2, 3)).float()
        fp = (pred_bin & ~tgt_bin).sum(dim=(1, 2, 3)).float()
        fn = (~pred_bin & tgt_bin).sum(dim=(1, 2, 3)).float()
        f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        f1s.extend([float(x.item()) for x in f1])
        lang_rows.extend(list(batch["language"]))

    if not losses:
        return {}

    per_lang = {}
    for lang in sorted(set(lang_rows)):
        idxs = [i for i, l in enumerate(lang_rows) if l == lang]
        per_lang[lang] = float(np.mean([f1s[i] for i in idxs]))

    return {
        "loss": float(np.mean(losses)),
        "f1_mean": float(np.mean(f1s)),
        "f1_per_lang": per_lang,
    }


def print_metrics(m: dict):
    if not m:
        return
    print(f"  [Pix] loss={m['loss']:.4f}  f1={m['f1_mean']:.3f}")
    for lang, f1 in m["f1_per_lang"].items():
        print(f"    {lang:>10s}: {f1:.3f}")


def should_save_best(m: dict, best_f1: float, config: dict) -> bool:
    if not m:
        return False
    return m["f1_mean"] > max(best_f1, config["best_min_f1"])


def build_optimizer(model, config: dict, total_steps: int):
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = AdamW(trainable, lr=config["lr"], weight_decay=config["weight_decay"])
    warmup = max(1, int(total_steps * config["warmup_ratio"]))
    decay = max(1, total_steps - warmup)
    sch = SequentialLR(
        opt,
        schedulers=[
            LinearLR(opt, start_factor=0.1, end_factor=1.0, total_iters=warmup),
            CosineAnnealingLR(opt, T_max=decay, eta_min=1e-7),
        ],
        milestones=[warmup],
    )
    return opt, sch


def run_epoch(model, loader, optimizer, scheduler, config: dict, device, tag: str, train_mode: bool) -> float:
    model.train() if train_mode else model.eval()
    use_bf16 = bool(config.get("bf16") and device.type == "cuda")
    losses = []
    ctx = torch.enable_grad if train_mode else torch.no_grad

    with ctx():
        for bi, batch in enumerate(loader):
            pixel_values = batch["pixel_values"].to(device)
            lang_ids = batch["lang_ids"].to(device)
            target_mask = batch["target_mask"].to(device)

            if train_mode:
                optimizer.zero_grad(set_to_none=True)

            with torch.autocast(
                device_type="cuda",
                dtype=torch.bfloat16 if use_bf16 else torch.float32,
                enabled=use_bf16,
            ):
                logits = model(pixel_values, lang_ids)
                loss = weighted_bce(logits, target_mask, config["pos_weight"])

            if not torch.isfinite(loss):
                print(f"  [{tag}] non-finite loss batch {bi}, skip")
                continue

            if train_mode:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad],
                    config["grad_clip"],
                )
                optimizer.step()
                if scheduler:
                    scheduler.step()

            losses.append(float(loss.item()))
            if train_mode and bi % 20 == 0:
                lr = scheduler.get_last_lr()[-1] if scheduler else config["lr"]
                f1 = batch_f1(logits.detach(), target_mask.detach())
                print(f"  [{tag}] B{bi:>3d}/{len(loader)} loss={loss.item():.4f} f1={f1:.3f} lr={lr:.2e}")

    return float(np.mean(losses)) if losses else float("nan")


def save(model, processor, config: dict, save_dir: str):
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(save_dir, "model.pt"))
    processor.save_pretrained(save_dir)
    with open(os.path.join(save_dir, "run_config.json"), "w", encoding="utf-8") as f:
        json.dump(config, f, ensure_ascii=False, indent=2)


def train():
    config = resolve_config(get_config())
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TRAINING: SmolVLM vision encoder + pixel decoder")
    print("=" * 60)
    print(f"  Device:  {device}")
    print(f"  Model:   {config['model_name']}")
    print(f"  Epochs:  {config['epochs']}  LR={config['lr']}")
    print(f"  Output:  {config['output_h']}x{config['output_w']}  pos_weight={config['pos_weight']}")

    os.makedirs(config["output_dir"], exist_ok=True)

    print("\n[1/4] Loading processor...")
    processor = AutoProcessor.from_pretrained(config["model_name"])
    processor.image_processor.do_image_splitting = False

    print("\n[2/4] Loading data...")
    df = pd.read_csv(config["train_file"])
    print(f"  {len(df)} samples | {df['language'].value_counts().to_dict()}")
    train_df, val_df = stratified_split(df, "language", config["val_split"])
    print(f"  Train: {len(train_df)} | Val: {len(val_df)}")

    train_ds = PixelTextDataset(train_df, processor, config, is_test=False)
    val_ds = PixelTextDataset(val_df, processor, config, is_test=False)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    s = train_ds[0]
    print(f"\n  [SANITY] pixel_values shape = {tuple(s['pixel_values'].shape)}")
    print(f"  [SANITY] target_mask shape  = {tuple(s['target_mask'].shape)}")
    print(f"  [SANITY] language           = {s['language']}")
    print(f"  [SANITY] target text        = '{s['target_text'][:80]}'")

    print("\n[3/4] Building model...")
    model = build_model(config, device)

    total_steps = len(train_loader) * config["epochs"]
    optimizer, scheduler = build_optimizer(model, config, total_steps)

    print(f"\n[4/4] Training ({total_steps} steps)...")
    best_f1 = float("-inf")
    no_improve = 0
    history = []

    for ep in range(1, config["epochs"] + 1):
        print(f"\n{'='*50} Epoch {ep}/{config['epochs']} {'='*50}")
        t0 = time.time()
        tr_loss = run_epoch(model, train_loader, optimizer, scheduler, config, device, "tr", train_mode=True)
        val_loss = run_epoch(model, val_loader, None, None, config, device, "val", train_mode=False)
        metrics = eval_pixel_metrics(model, val_loader, config, device)
        elapsed = time.time() - t0

        print(f"  train={tr_loss:.4f}  val={val_loss:.4f}  gap={val_loss-tr_loss:+.4f}  {elapsed:.0f}s")
        print_metrics(metrics)
        save_epoch_debug(model, val_loader, config, device, ep)

        history.append({
            "ep": ep,
            "tr": tr_loss,
            "val": val_loss,
            "f1": metrics.get("f1_mean", float('nan')),
        })

        if should_save_best(metrics, best_f1, config):
            prev = best_f1
            best_f1 = metrics["f1_mean"]
            best_dir = os.path.join(config["output_dir"], "best_model")
            save(model, processor, config, best_dir)
            no_improve = 0
            print(f"  >>> Saved best -> {best_dir}  (F1 {prev:.4f} -> {best_f1:.4f})")
        else:
            no_improve += 1

        if ep >= config["early_stop_warmup_epochs"] and no_improve >= config["early_stop_patience"]:
            print(f"  EARLY STOP: {no_improve} epochs without improvement.")
            break

    final_dir = os.path.join(config["output_dir"], "final_model")
    save(model, processor, config, final_dir)
    print(f"\n  Saved final -> {final_dir}/")

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    if np.isfinite(best_f1):
        print(f"  Best F1: {best_f1:.4f}")
    print(f"\n  {'Ep':>3}  {'Train':>8}  {'Val':>8}  {'F1':>7}")
    for h in history:
        mark = " *" if np.isfinite(best_f1) and abs(h["f1"] - best_f1) < 1e-6 else ""
        print(f"  {h['ep']:>3}  {h['tr']:>8.4f}  {h['val']:>8.4f}  {h['f1']:>7.3f}{mark}")
    print("=" * 60)


if __name__ == "__main__":
    train()
