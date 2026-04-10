import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
from torch.utils.data import DataLoader

from config import get_config, resolve_config
from dataset import OCRDataset, collate_fn, stratified_split
from metrics import summarize_prediction
from model import build_model, build_processor, save_bundle
from utils import clean_prediction, ensure_dir, render_text_to_image, set_seed


def build_optimizer(model, config: dict, total_steps: int):
    optimizer = AdamW([p for p in model.parameters() if p.requires_grad], lr=config["lr"], weight_decay=config["weight_decay"])
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


def describe_batch(batch: dict) -> dict:
    pixel_values = batch["pixel_values"]
    labels = batch["labels"]
    valid_label_counts = (labels != -100).sum(dim=1).float()
    stats = {
        "pixel_shape": tuple(pixel_values.shape),
        "pixel_min": float(pixel_values.min().item()),
        "pixel_max": float(pixel_values.max().item()),
        "pixel_mean": float(pixel_values.mean().item()),
        "pixel_std": float(pixel_values.std().item()),
        "label_tokens_mean": float(valid_label_counts.mean().item()),
        "label_tokens_min": int(valid_label_counts.min().item()),
        "label_tokens_max": int(valid_label_counts.max().item()),
    }
    if "target_char_len" in batch:
        tgt = np.array(batch["target_char_len"], dtype=np.float32)
        ctx = np.array(batch["context_char_len"], dtype=np.float32)
        stats.update(
            {
                "target_chars_mean": float(tgt.mean()),
                "context_chars_mean": float(ctx.mean()),
                "orig_render_h_mean": float(np.mean(batch["orig_render_height"])),
                "orig_render_w_mean": float(np.mean(batch["orig_render_width"])),
                "render_h_mean": float(np.mean(batch["render_height"])),
                "max_width_mean": float(np.mean(batch["max_width"])),
                "resize_ratio_h_mean": float(np.mean(batch["resize_ratio_h"])),
                "resize_ratio_w_mean": float(np.mean(batch["resize_ratio_w"])),
                "white_density_mean": float(np.mean(batch["context_white_density"])),
            }
        )
    return stats


def compute_weighted_loss(logits, labels, prefix_token_count: int, prefix_loss_weight: float):
    vocab = logits.size(-1)
    flat_loss = F.cross_entropy(
        logits.view(-1, vocab),
        labels.view(-1),
        ignore_index=-100,
        reduction="none",
    ).view_as(labels)

    valid = labels != -100
    weights = torch.ones_like(flat_loss)

    if prefix_loss_weight > 1.0 and prefix_token_count > 0:
        prefix_mask = torch.zeros_like(valid)
        for i in range(labels.size(0)):
            valid_idx = torch.nonzero(valid[i], as_tuple=False).squeeze(-1)
            if valid_idx.numel() == 0:
                continue
            take = valid_idx[:prefix_token_count]
            prefix_mask[i, take] = True
        weights = torch.where(prefix_mask, torch.full_like(weights, float(prefix_loss_weight)), weights)
    else:
        prefix_mask = torch.zeros_like(valid)

    weighted = flat_loss * weights * valid
    denom = (weights * valid).sum().clamp_min(1.0)
    loss = weighted.sum() / denom

    prefix_count = prefix_mask.sum().item()
    nonprefix_mask = valid & (~prefix_mask)
    prefix_loss = (flat_loss * prefix_mask).sum() / prefix_mask.sum().clamp_min(1)
    nonprefix_loss = (flat_loss * nonprefix_mask).sum() / nonprefix_mask.sum().clamp_min(1)
    return loss, float(prefix_loss.item()), float(nonprefix_loss.item()), int(prefix_count)


@torch.no_grad()
def debug_first_step(model, processor, sample: dict, device, topk: int = 5):
    pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
    start_id = int(processor.tokenizer.cls_token_id)
    decoder_prompt_ids = torch.cat(
        [torch.tensor([start_id], device=device, dtype=torch.long), sample["prompt_ids"].to(device)]
    ).unsqueeze(0)
    outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_prompt_ids)
    logits = outputs.logits[:, -1, :]
    probs = torch.softmax(logits, dim=-1)
    vals, ids = torch.topk(probs, k=min(int(topk), probs.size(-1)), dim=-1)
    parts = []
    for prob, tok_id in zip(vals[0].tolist(), ids[0].tolist()):
        tok = processor.tokenizer.decode([int(tok_id)]).replace("\n", "\\n").strip()
        if not tok:
            tok = "<blank>"
        parts.append(f"{tok}:{prob:.2f}")
    return " | ".join(parts)


def run_epoch(model, loader, optimizer, scheduler, config: dict, device, tag: str, train_mode: bool) -> float:
    model.train() if train_mode else model.eval()
    use_bf16 = bool(config.get("bf16") and device.type == "cuda")
    losses = []
    prefix_losses = []
    nonprefix_losses = []
    grad_accum = int(config["grad_accum_steps"])

    if train_mode:
        optimizer.zero_grad(set_to_none=True)

    for bi, batch in enumerate(loader):
        if bi == 0:
            stats = describe_batch(batch)
            print(
                f"  [{tag}] batch_stats shape={stats['pixel_shape']} pix=({stats['pixel_min']:.3f},{stats['pixel_max']:.3f}) "
                f"mean={stats['pixel_mean']:.3f} std={stats['pixel_std']:.3f} "
                f"labelTok={stats['label_tokens_mean']:.1f}[{stats['label_tokens_min']},{stats['label_tokens_max']}]"
            )
            if "target_chars_mean" in stats:
                print(
                    f"  [{tag}] text_stats ctxChars={stats['context_chars_mean']:.1f} tgtChars={stats['target_chars_mean']:.1f} "
                    f"origRender={stats['orig_render_w_mean']:.1f}x{stats['orig_render_h_mean']:.1f} "
                    f"canvasH={stats['render_h_mean']:.1f} maxW={stats['max_width_mean']:.1f} "
                    f"rScale=({stats['resize_ratio_w_mean']:.2f},{stats['resize_ratio_h_mean']:.2f}) density={stats['white_density_mean']:.4f}"
                )
        pixel_values = batch["pixel_values"].to(device)
        decoder_input_ids = batch["decoder_input_ids"].to(device)
        labels = batch["labels"].to(device)

        with torch.set_grad_enabled(train_mode):
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=use_bf16):
                outputs = model(pixel_values=pixel_values, decoder_input_ids=decoder_input_ids, labels=labels)
                loss, prefix_loss, nonprefix_loss, prefix_count = compute_weighted_loss(
                    outputs.logits,
                    labels,
                    int(config["prefix_token_count"]),
                    float(config["prefix_loss_weight"]),
                )

            if not torch.isfinite(loss):
                print(f"  [{tag}] non-finite loss batch {bi}, skip")
                if train_mode:
                    optimizer.zero_grad(set_to_none=True)
                continue

            if train_mode:
                (loss / grad_accum).backward()

        losses.append(float(loss.item()))
        prefix_losses.append(prefix_loss)
        nonprefix_losses.append(nonprefix_loss)

        if train_mode:
            do_step = (bi + 1) % grad_accum == 0 or (bi + 1) == len(loader)
            if do_step:
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config["grad_clip"])
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler:
                    scheduler.step()
            else:
                grad_norm = None

            if bi % int(config["log_interval"]) == 0:
                lr = scheduler.get_last_lr()[-1] if scheduler else config["lr"]
                if grad_norm is None:
                    print(
                        f"  [{tag}] B{bi:>3d}/{len(loader)} loss={loss.item():.4f} "
                        f"prefix={prefix_loss:.4f} tail={nonprefix_loss:.4f} pTok={prefix_count} lr={lr:.2e}"
                    )
                else:
                    print(
                        f"  [{tag}] B{bi:>3d}/{len(loader)} loss={loss.item():.4f} "
                        f"prefix={prefix_loss:.4f} tail={nonprefix_loss:.4f} pTok={prefix_count} "
                        f"lr={lr:.2e} gnorm={float(grad_norm):.3f}"
                    )

    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def evaluate_generation(model, processor, dataset, config: dict, device, epoch: int) -> dict:
    model.eval()
    limit = min(int(config["eval_samples"]), len(dataset))
    debug_dir = ensure_dir(os.path.join(config["debug_dir"], "train", f"epoch_{epoch:03d}"))

    rows = []
    for i in range(limit):
        sample = dataset[i]
        pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
        start_id = int(processor.tokenizer.cls_token_id)
        decoder_prompt_ids = torch.cat(
            [torch.tensor([start_id], device=device, dtype=torch.long), sample["prompt_ids"].to(device)]
        ).unsqueeze(0)
        generated_ids = model.generate(
            pixel_values=pixel_values,
            decoder_input_ids=decoder_prompt_ids,
            max_new_tokens=config["max_new_tokens"],
            min_new_tokens=config["min_new_tokens"],
            num_beams=config["num_beams"],
            early_stopping=True,
            repetition_penalty=float(config["repetition_penalty"]),
            no_repeat_ngram_size=int(config["no_repeat_ngram_size"]),
            length_penalty=float(config["length_penalty"]),
        )
        pred = clean_prediction(processor.batch_decode(generated_ids, skip_special_tokens=True)[0], sample["language"])
        true = sample["target_text"]
        summary = summarize_prediction(
            pred,
            true,
            int(sample["max_width"]),
            config["font_size"],
            config["padding"],
            sample["language"],
        )
        rows.append(
            {
                "language": sample["language"],
                "pred": pred,
                "true": true,
                "max_width": int(sample["max_width"]),
                **summary,
            }
        )

    df = pd.DataFrame(rows)
    per_lang = df.groupby("language")["cer"].mean().to_dict() if not df.empty else {}
    per_lang_pix = df.groupby("language")["pixel_f1"].mean().to_dict() if not df.empty else {}
    unique_pred_ratio = (df["pred"].nunique() / max(len(df), 1)) if not df.empty else 0.0
    alpha_start_match = float(df["alpha_start_match"].mean()) if not df.empty else 0.0
    lang_consistency = float(df["lang_consistency"].mean()) if not df.empty else 0.0
    first_token_mode = df["first_token"].mode().iat[0] if not df.empty and not df["first_token"].mode().empty else ""
    first_token_top = (
        df["first_token"].value_counts().head(5).to_dict()
        if not df.empty
        else {}
    )
    first_token_by_lang = {}
    english_opener_rate = {}
    if not df.empty:
        english_function_openers = {"and", "the", "to", "for", "of", "in", "is"}
        for language, group in df.groupby("language"):
            mode = group["first_token"].mode()
            first_token_by_lang[language] = mode.iat[0] if not mode.empty else ""
            english_opener_rate[language] = float(group["first_token"].isin(english_function_openers).mean())
    dbg_first = debug_first_step(model, processor, dataset[0], device, topk=int(config["debug_topk"]))

    for i, row in enumerate(rows[: int(config["debug_image_count"])]):
        render_text_to_image(dataset[i]["context_text"], row["max_width"], config["font_size"], config["padding"]).save(
            debug_dir / f"context_{i}.png"
        )
        render_text_to_image(row["true"], row["max_width"], config["font_size"], config["padding"]).save(
            debug_dir / f"target_{i}.png"
        )
        render_text_to_image(row["pred"], row["max_width"], config["font_size"], config["padding"]).save(
            debug_dir / f"pred_{i}.png"
        )
        (debug_dir / f"sample_{i}.txt").write_text(
            "\n".join(
                [
                    f"language: {row['language']}",
                    f"cer: {row['cer']:.4f}",
                    f"prefix_cer: {row['prefix_cer']:.4f}",
                    f"pixel_f1: {row['pixel_f1']:.4f}",
                    f"repeat: {row['repeat']:.4f}",
                    f"len_ratio: {row['len_ratio']:.4f}",
                    f"pred_chars: {row['pred_char_len']}",
                    f"target_chars: {row['target_char_len']}",
                    f"first_token: {row['first_token']}",
                    f"best_marker_language: {row['best_marker_language']}",
                    f"lang_consistency: {row['lang_consistency']:.4f}",
                    f"marker_counts: en={row['marker_english']} fr={row['marker_french']} de={row['marker_german']} it={row['marker_italian']} es={row['marker_spanish']}",
                    "",
                    f"TRUE: {row['true']}",
                    "",
                    f"PRED: {row['pred']}",
                ]
            ),
            encoding="utf-8",
        )

    return {
        "cer_mean": float(df["cer"].mean()) if not df.empty else float("inf"),
        "pixel_f1": float(df["pixel_f1"].mean()) if not df.empty else 0.0,
        "prefix_cer": float(df["prefix_cer"].mean()) if not df.empty else float("inf"),
        "repeat": float(df["repeat"].mean()) if not df.empty else 0.0,
        "len_ratio": float(df["len_ratio"].mean()) if not df.empty else 0.0,
        "pred_len": float(df["pred"].str.len().mean()) if not df.empty else 0.0,
        "true_len": float(df["true"].str.len().mean()) if not df.empty else 0.0,
        "short_pct": 100.0 * float((df["len_ratio"] < 0.5).mean()) if not df.empty else 0.0,
        "long_pct": 100.0 * float((df["len_ratio"] > 1.5).mean()) if not df.empty else 0.0,
        "unique_pred_ratio": float(unique_pred_ratio),
        "alpha_start_match": alpha_start_match,
        "lang_consistency": lang_consistency,
        "first_token_mode": first_token_mode,
        "first_token_top": first_token_top,
        "first_token_by_lang": first_token_by_lang,
        "english_opener_rate": english_opener_rate,
        "cer_per_lang": per_lang,
        "pix_per_lang": per_lang_pix,
        "examples": rows[:4],
        "debug_dir": str(debug_dir),
        "debug_first_step": dbg_first,
    }


def train():
    config = resolve_config(get_config())
    set_seed(config["seed"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("TRAINING: context image -> TrOCR -> continuation text")
    print("=" * 60)
    print(f"  Device:  {device}")
    print(f"  Model:   {config['model_name']}")
    print(f"  Epochs:  {config['epochs']}  LR={config['lr']}")
    print(f"  Render:  image_size={config['image_size']} tail_chars={config['context_tail_chars']} font={config['font_size']} pad={config['padding']}")
    print(f"  Decode:  max_new={config['max_new_tokens']} min_new={config['min_new_tokens']} beams={config['num_beams']} rep_pen={config['repetition_penalty']} no_repeat_ngram={config['no_repeat_ngram_size']}")
    print(f"  Tune:    encoder_frozen={config['freeze_encoder']} decoder_mode={config.get('decoder_tune_mode')} top_layers={config.get('decoder_train_layers')}")

    ensure_dir(config["output_dir"])
    ensure_dir(config["debug_dir"])

    print("\n[1/4] Loading processor...")
    processor = build_processor(config["model_name"])

    print("\n[2/4] Loading data...")
    train_df = pd.read_csv(config["train_file"])
    print(f"  {len(train_df)} samples | {train_df['language'].value_counts().to_dict()}")
    tr_df, va_df = stratified_split(train_df, "language", config["val_split"], seed=config["seed"])
    print(f"  Train: {len(tr_df)} | Val: {len(va_df)}")
    train_lang_counts = tr_df["language"].value_counts().to_dict()
    print(f"  Train language counts: {train_lang_counts}")

    train_ds = OCRDataset(tr_df, processor, config, is_test=False)
    val_ds = OCRDataset(va_df, processor, config, is_test=False)
    print("  Balanced sampling: OFF")
    train_loader = DataLoader(
        train_ds,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], shuffle=False, num_workers=0, pin_memory=True, collate_fn=collate_fn)

    sample = train_ds[0]
    print(f"\n  [SANITY] pixel_values shape = {tuple(sample['pixel_values'].shape)}")
    print(f"  [SANITY] image_size        = {config['image_size']}")
    print(f"  [SANITY] context_tail_chars= {config['context_tail_chars']}")
    print(f"  [SANITY] prompt text       = '{sample['prompt_text']}'")
    print(f"  [SANITY] language          = {sample['language']}")
    print(f"  [SANITY] context chars     = {len(sample['context_text'])}/{len(sample['full_context_text'])}")
    print(f"  [SANITY] orig render size  = {sample['orig_render_width']}x{sample['orig_render_height']}")
    print(f"  [SANITY] render size       = {sample['render_width']}x{sample['render_height']}")
    print(f"  [SANITY] resize ratio      = ({sample['resize_ratio_w']:.2f}, {sample['resize_ratio_h']:.2f})")
    print(f"  [SANITY] white density     = {sample['context_white_density']:.4f}")
    print(f"  [SANITY] prompt ids len    = {len(sample['prompt_ids'])}")
    print(f"  [SANITY] decoder ids len   = {len(sample['decoder_input_ids'])}")
    print(f"  [SANITY] label tokens      = {int((sample['labels'] != -100).sum().item())}")
    print(f"  [SANITY] target text       = '{sample['target_text'][:80]}'")

    print("\n[3/4] Building model...")
    model = build_model(config, processor, device)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100.0 * trainable / max(total, 1):.2f}%)")
    if getattr(model, "_freeze_info", None):
        info = model._freeze_info
        print(
            f"  Decoder tune mode: {info['mode']} trainable={info['decoder_trainable']:,}/{info['decoder_total']:,} "
            f"kept_layers={info['decoder_kept_layers']}"
        )
        print(f"  Decoder unfrozen preview: {info['decoder_unfrozen_preview']}")
    total_steps = max(1, (len(train_loader) * config["epochs"]) // config["grad_accum_steps"])
    optimizer, scheduler = build_optimizer(model, config, total_steps)

    print(f"\n[4/4] Training ({total_steps} optimizer steps)...")
    best_cer = float("inf")
    no_improve = 0
    history = []

    for epoch in range(1, config["epochs"] + 1):
        print(f"\n{'=' * 50} Epoch {epoch}/{config['epochs']} {'=' * 50}")
        t0 = time.time()
        tr_loss = run_epoch(model, train_loader, optimizer, scheduler, config, device, "tr", True)
        va_loss = run_epoch(model, val_loader, None, None, config, device, "val", False)
        gen = evaluate_generation(model, processor, val_ds, config, device, epoch)
        elapsed = time.time() - t0

        print(f"  train={tr_loss:.4f}  val={va_loss:.4f}  gap={va_loss-tr_loss:+.4f}  {elapsed:.0f}s")
        print(
            f"  [Gen] CER={gen['cer_mean']:.3f}  pred={gen['pred_len']:.0f}/true={gen['true_len']:.0f}  "
            f"short={gen['short_pct']:.1f}%  pixF1={gen['pixel_f1']:.3f}  prefixCER={gen['prefix_cer']:.3f}  "
            f"repeat={gen['repeat']:.3f}  lenRatio={gen['len_ratio']:.2f}  long={gen['long_pct']:.1f}%  "
            f"unique={gen['unique_pred_ratio']:.2f}  alphaStart={gen['alpha_start_match']:.2f}  "
            f"langCons={gen['lang_consistency']:.2f}"
        )
        print(f"  [Dbg] next-token: {gen['debug_first_step']}")
        print(f"  [Dbg] first-token mode: {gen['first_token_mode'] or '<blank>'}  top={gen['first_token_top']}")
        for lang, cer in gen["cer_per_lang"].items():
            flag = "  !!!" if cer > 0.95 else ""
            pix = gen["pix_per_lang"].get(lang, 0.0)
            first_tok = gen["first_token_by_lang"].get(lang, "")
            opener_rate = gen["english_opener_rate"].get(lang, 0.0)
            print(
                f"    {lang:>10s}: CER={cer:.3f} pixF1={pix:.3f} "
                f"firstTok={first_tok or '<blank>'} engOpen={opener_rate:.2f}{flag}"
            )
        for ex in gen["examples"]:
            print(f"    [{ex['language']}]")
            print(f"      TRUE: {ex['true']}")
            print(f"      PRED: {ex['pred']}")

        history.append({"epoch": epoch, "train": tr_loss, "val": va_loss, "cer": gen["cer_mean"], "pixf1": gen["pixel_f1"]})

        if gen["cer_mean"] < best_cer - 1e-6:
            prev = best_cer
            best_cer = gen["cer_mean"]
            save_bundle(model, processor, config, os.path.join(config["output_dir"], "best_model"))
            no_improve = 0
            print(f"  >>> Saved best -> {os.path.join(config['output_dir'], 'best_model')}  (CER {prev:.4f} -> {best_cer:.4f})")
            print(f"  >>> Debug -> {gen['debug_dir']}")
        else:
            no_improve += 1

        if epoch >= config["early_stop_warmup_epochs"] and no_improve >= config["early_stop_patience"]:
            print(f"  EARLY STOP: {no_improve} epochs without improvement.")
            break

    save_bundle(model, processor, config, os.path.join(config["output_dir"], "final_model"))
    print(f"\n  Saved final -> {os.path.join(config['output_dir'], 'final_model')}/")
    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best CER: {best_cer:.4f}")


if __name__ == "__main__":
    train()
