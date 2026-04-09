"""Model: SigLIP vision encoder + row-structure decoder + raster head."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForVision2Seq

from config import LANG2ID


class RowStructureDecoder(nn.Module):
    """Decode continuation image row-by-row from spatial visual features."""

    def __init__(self, enc_dim: int, config: dict):
        super().__init__()
        self.out_h = int(config["output_h"])
        self.out_w = int(config["output_w"])
        self.grid_w = int(config.get("structure_grid_w", 96))
        self.hidden = int(config.get("structure_hidden", 192))
        self.row_hidden = int(config.get("row_hidden", 256))
        self.col_hidden = int(config.get("col_hidden", 192))
        self.lang_dim = int(config["lang_emb_dim"])

        self.spatial_proj = nn.Sequential(
            nn.Conv2d(enc_dim, self.hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(),
            nn.Conv2d(self.hidden, self.hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(),
        )
        self.lang_proj = nn.Linear(self.lang_dim, self.lang_dim)
        self.row_pos = nn.Embedding(self.out_h, self.hidden)
        self.row_decoder = nn.GRU(
            input_size=self.hidden + self.lang_dim,
            hidden_size=self.row_hidden,
            num_layers=int(config.get("row_layers", 2)),
            batch_first=True,
            dropout=float(config.get("decoder_dropout", 0.15)),
            bidirectional=False,
        )
        self.col_pos = nn.Embedding(self.grid_w, self.hidden)
        self.col_decoder = nn.GRU(
            input_size=self.hidden + self.lang_dim,
            hidden_size=self.col_hidden,
            num_layers=int(config.get("col_layers", 1)),
            batch_first=True,
            dropout=float(config.get("decoder_dropout", 0.15)) if int(config.get("col_layers", 1)) > 1 else 0.0,
            bidirectional=False,
        )
        self.fuse = nn.Sequential(
            nn.Conv2d(self.hidden + self.row_hidden + self.col_hidden, self.hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden),
            nn.ReLU(),
            nn.Dropout2d(float(config.get("decoder_dropout", 0.15))),
            nn.Conv2d(self.hidden, self.hidden // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(self.hidden // 2),
            nn.ReLU(),
            nn.Conv2d(self.hidden // 2, 1, kernel_size=1),
        )

    def forward(self, feat_map: torch.Tensor, lang_vec: torch.Tensor) -> torch.Tensor:
        x = self.spatial_proj(feat_map)
        x = F.interpolate(x, size=(self.out_h, self.grid_w), mode="bilinear", align_corners=False)

        row_feat = x.mean(dim=3).transpose(1, 2)  # [B, H, C]
        pos_ids = torch.arange(self.out_h, device=x.device)
        row_feat = row_feat + self.row_pos(pos_ids).unsqueeze(0)

        lang = self.lang_proj(lang_vec).unsqueeze(1).expand(-1, self.out_h, -1)
        row_in = torch.cat([row_feat, lang], dim=-1)
        row_out, _ = self.row_decoder(row_in)  # [B, H, row_hidden]

        row_map = row_out.transpose(1, 2).unsqueeze(-1).expand(-1, -1, self.out_h, self.grid_w)

        col_feat = x.mean(dim=2).transpose(1, 2)  # [B, W, C]
        col_ids = torch.arange(self.grid_w, device=x.device)
        col_feat = col_feat + self.col_pos(col_ids).unsqueeze(0)
        col_lang = self.lang_proj(lang_vec).unsqueeze(1).expand(-1, self.grid_w, -1)
        col_in = torch.cat([col_feat, col_lang], dim=-1)
        col_out, _ = self.col_decoder(col_in)  # [B, W, col_hidden]

        col_map = col_out.transpose(1, 2).unsqueeze(2).expand(-1, -1, self.out_h, self.grid_w)
        fused = torch.cat([x, row_map, col_map], dim=1)
        logits = self.fuse(fused)
        logits = F.interpolate(logits, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        return logits


class VisualStructureModel(nn.Module):
    """Context image -> continuation image mask via row-structure decoding."""

    def __init__(self, vision_encoder: nn.Module, enc_dim: int, config: dict):
        super().__init__()
        self.vision_encoder = vision_encoder
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        self.lang_emb = nn.Embedding(len(LANG2ID) + 1, int(config["lang_emb_dim"]))
        self.decoder = RowStructureDecoder(enc_dim, config)

    @staticmethod
    def _patches_to_map(patches: torch.Tensor) -> torch.Tensor:
        b, n, d = patches.shape
        side = int(round(n ** 0.5))
        if side * side == n:
            usable = patches
        elif int(round((n - 1) ** 0.5)) ** 2 == n - 1:
            usable = patches[:, 1:, :]
            side = int(round((n - 1) ** 0.5))
        else:
            raise ValueError(f"Cannot reshape {n} patches into a square map.")
        return usable.transpose(1, 2).reshape(b, d, side, side)

    def forward(self, pixel_values: torch.Tensor, lang_ids: torch.Tensor) -> torch.Tensor:
        if pixel_values.ndim == 5 and pixel_values.shape[1] == 1:
            pixel_values = pixel_values.squeeze(1)
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values shape [B, 3, H, W], got {tuple(pixel_values.shape)}")

        enc_param = next(self.vision_encoder.parameters())
        pixel_values = pixel_values.to(device=enc_param.device, dtype=enc_param.dtype)
        vis_out = self.vision_encoder(pixel_values)
        feat_map = self._patches_to_map(vis_out.last_hidden_state)
        lang_vec = self.lang_emb(lang_ids)
        dec_param = next(self.decoder.parameters())
        feat_map = feat_map.to(dtype=dec_param.dtype)
        lang_vec = lang_vec.to(dtype=dec_param.dtype)
        return self.decoder(feat_map, lang_vec)


def build_model(config: dict, device: torch.device) -> VisualStructureModel:
    dtype = torch.bfloat16 if config.get("bf16") else torch.float32
    print(f"  Loading {config['model_name']} to extract SigLIP encoder ...")
    smolvlm = AutoModelForVision2Seq.from_pretrained(
        config["model_name"],
        torch_dtype=dtype,
        _attn_implementation="eager",
    )
    vision_encoder = smolvlm.model.vision_model
    enc_dim = vision_encoder.config.hidden_size
    del smolvlm
    torch.cuda.empty_cache()
    print(f"  SigLIP extracted: enc_dim={enc_dim}")

    model = VisualStructureModel(vision_encoder, enc_dim, config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
    return model.to(device)


def load_for_inference(config: dict, save_dir: str, device: torch.device) -> VisualStructureModel:
    model = build_model(config, device)
    ckpt = torch.load(f"{save_dir}/model.pt", map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    return model.eval()
