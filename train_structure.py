"""Training: vision encoder -> row-structure decoder -> fixed mask."""

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
from model_structure import build_model
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


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    inter = (probs * target).sum(dim=(1, 2, 3))
    denom = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
    dice = (2 * inter + 1e-6) / (denom + 1e-6)
    return 1.0 - dice.mean()


def density_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_density = probs.mean(dim=(1, 2, 3))
    true_density = target.mean(dim=(1, 2, 3))
    return (pred_density - true_density).abs().mean()


def projection_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    probs = torch.sigmoid(logits)
    pred_rows = probs.mean(dim=3)
    true_rows = target.mean(dim=3)
    pred_cols = probs.mean(dim=2)
    true_cols = target.mean(dim=2)
    row_err = (pred_rows - true_rows).abs().mean()
    col_err = (pred_cols - true_cols).abs().mean()
    return 0.5 * (row_err + col_err)


def structure_loss(logits: torch.Tensor, target: torch.Tensor, config: dict) -> tuple[torch.Tensor, dict]:
    bce = weighted_bce(logits, target, config["pos_weight"])
    dice = soft_dice_loss(logits, target)
    dens = density_loss(logits, target)
    proj = projection_loss(logits, target)
    total = (
        float(config.get("loss_bce_weight", 1.0)) * bce
        + float(config.get("loss_dice_weight", 0.7)) * dice
        + float(config.get("loss_density_weight", 0.3)) * dens
        + float(config.get("loss_projection_weight", 0.4)) * proj
    )
    parts = {
        "bce": float(bce.detach().item()),
        "dice": float(dice.detach().item()),
        "dens": float(dens.detach().item()),
        "proj": float(proj.detach().item()),
    }
    return total, parts


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


def density_pair(logits: torch.Tensor, target: torch.Tensor, threshold: float = 0.5) -> tuple[float, float]:
    pred_bin = (torch.sigmoid(logits).detach() >= threshold).float()
    target_bin = (target.detach() >= threshold).float()
    return float(pred_bin.mean().item()), float(target_bin.mean().item())


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
    pred_density, target_density = density_pair(logits, batch["target_mask"].to(device), threshold=thr)

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

    print(
        f"  [Dbg] saved {n} samples -> {out_dir}  "
        f"pred_density={pred_density:.3f} target_density={target_density:.3f}"
    )


@torch.no_grad()
def eval_pixel_metrics(model, loader, config: dict, device) -> dict:
    model.eval()
    losses = []
    f1s = []
    lang_rows = []
    pred_densities = []
    target_densities = []
    row_errors = []
    col_errors = []

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
            loss, _ = structure_loss(logits, target_mask, config)

        losses.append(float(loss.item()))
        pred_bin = torch.sigmoid(logits) >= 0.5
        tgt_bin = target_mask >= 0.5
        tp = (pred_bin & tgt_bin).sum(dim=(1, 2, 3)).float()
        fp = (pred_bin & ~tgt_bin).sum(dim=(1, 2, 3)).float()
        fn = (~pred_bin & tgt_bin).sum(dim=(1, 2, 3)).float()
        f1 = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        f1s.extend([float(x.item()) for x in f1])
        lang_rows.extend(list(batch["language"]))
        pred_densities.extend((pred_bin.float().mean(dim=(1, 2, 3))).cpu().tolist())
        target_densities.extend((tgt_bin.float().mean(dim=(1, 2, 3))).cpu().tolist())
        row_errors.extend((pred_bin.float().mean(dim=3) - tgt_bin.float().mean(dim=3)).abs().mean(dim=(1, 2)).cpu().tolist())
        col_errors.extend((pred_bin.float().mean(dim=2) - tgt_bin.float().mean(dim=2)).abs().mean(dim=(1, 2)).cpu().tolist())

    if not losses:
        return {}

    per_lang = {}
    for lang in sorted(set(lang_rows)):
        idxs = [i for i, l in enumerate(lang_rows) if l == lang]
        per_lang[lang] = float(np.mean([f1s[i] for i in idxs]))

    pred_density = float(np.mean(pred_densities))
    target_density = float(np.mean(target_densities))
    row_error = float(np.mean(row_errors))
    col_error = float(np.mean(col_errors))
    score = (
        float(np.mean(f1s))
        - float(config.get("best_density_penalty", 0.75)) * abs(pred_density - target_density)
        - float(config.get("best_projection_penalty", 0.50)) * 0.5 * (row_error + col_error)
    )

    return {
        "loss": float(np.mean(losses)),
        "f1_mean": float(np.mean(f1s)),
        "pred_density": pred_density,
        "target_density": target_density,
        "row_error": row_error,
        "col_error": col_error,
        "score": score,
        "f1_per_lang": per_lang,
    }


def print_metrics(m: dict):
    if not m:
        return
    print(f"  [Pix] loss={m['loss']:.4f}  f1={m['f1_mean']:.3f}")
    print(
        f"        density pred={m['pred_density']:.3f} true={m['target_density']:.3f}  "
        f"row_err={m['row_error']:.3f} col_err={m['col_error']:.3f} score={m['score']:.3f}"
    )
    for lang, f1 in m["f1_per_lang"].items():
        print(f"    {lang:>10s}: {f1:.3f}")


def should_save_best(m: dict, best_score: float, config: dict) -> bool:
    if not m:
        return False
    return m["score"] > max(best_score, float(config.get("best_min_score", 0.0)))


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
                loss, parts = structure_loss(logits, target_mask, config)

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
                pred_density, target_density = density_pair(logits.detach(), target_mask.detach())
                print(
                    f"  [{tag}] B{bi:>3d}/{len(loader)} loss={loss.item():.4f} "
                    f"f1={f1:.3f} dens={pred_density:.3f}/{target_density:.3f} "
                    f"bce={parts['bce']:.3f} dice={parts['dice']:.3f} proj={parts['proj']:.3f} "
                    f"lr={lr:.2e}"
                )

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
    print("TRAINING: SigLIP + row-structure decoder")
    print("=" * 60)
    print(f"  Device:  {device}")
    print(f"  Model:   {config['model_name']}")
    print(f"  Epochs:  {config['epochs']}  LR={config['lr']}")
    print(f"  Output:  {config['output_h']}x{config['output_w']}  grid_w={config['structure_grid_w']}")

    os.makedirs(config["output_dir"], exist_ok=True)

    print("\n[1/4] Loading processor...")
    processor = AutoProcessor.from_pretrained(config["model_name"])
    if hasattr(processor, "image_processor"):
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
    best_score = float("-inf")
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
            "f1": metrics.get("f1_mean", float("nan")),
            "score": metrics.get("score", float("nan")),
        })

        if should_save_best(metrics, best_score, config):
            prev = best_score
            best_score = metrics["score"]
            best_dir = os.path.join(config["output_dir"], "best_model")
            save(model, processor, config, best_dir)
            no_improve = 0
            print(f"  >>> Saved best -> {best_dir}  (score {prev:.4f} -> {best_score:.4f})")
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
    if np.isfinite(best_score):
        print(f"  Best score: {best_score:.4f}")
    print(f"\n  {'Ep':>3}  {'Train':>8}  {'Val':>8}  {'F1':>7}  {'Score':>7}")
    for h in history:
        mark = " *" if np.isfinite(best_score) and abs(h["score"] - best_score) < 1e-6 else ""
        print(f"  {h['ep']:>3}  {h['tr']:>8.4f}  {h['val']:>8.4f}  {h['f1']:>7.3f}  {h['score']:>7.3f}{mark}")
    print("=" * 60)


if __name__ == "__main__":
    train()
