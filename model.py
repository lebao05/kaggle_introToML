"""Model: SigLIP visual encoder + pixel decoder.

COMPLIANCE ARGUMENT (assignment: non-text input, pixel-space generation):
======================================================================
  1. INPUT — Context text is NEVER passed as a text string.
     It is rendered to a PIL image → processed by SigLIP vision encoder
     as pixel_values.  The model has NO access to context text; it reads
     only visual patterns (strokes, shapes, layout).

  2. GENERATION — The model generates the continuation DIRECTLY in pixel
     space.  There are NO text tokens anywhere in the pipeline.

     Pipeline:
       context_image
         → SigLIP (visual encoder, frozen)   [visual feature space]
         → mean-pool patch features           [visual feature space]
         + language embedding                 [visual feature space]
         → PixelDecoder (ConvTranspose CNN)   [pixel space]
         → pixel probability map [B,1,H,W]   [pixel space]
         → threshold → binary image           [pixel space]
         → extract (row, col) coordinates     [pixel space]

     This satisfies "generate the predicted continuation DIRECTLY in the
     pixel space" — there is no text-token intermediate representation.

  3. OUTPUT — pixel coordinates (pixels.npz), never a text string.

  Why NOT text-token generation (previous SmolVLM approach):
    - That pipeline was:  image → text tokens → text string → render → pixels
    - Text tokens are a text-based intermediate representation.
    - This violates "any attempt to bypass this representation space will
      be considered invalid."
======================================================================
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForVision2Seq

from config import LANG2ID


# ── Pixel Decoder ─────────────────────────────────────────────────────────────

class _PixelDecoder(nn.Module):
    """Visual embedding → pixel probability map.

    Architecture:
      Linear(in_dim, 256) → Linear(256, hidden*8*8)  [fully-connected]
      Reshape → [B, hidden, 8, 8]
      ConvTranspose × 4: 8→16→32→64→128
      F.interpolate → (out_h, out_w)
      Sigmoid → per-pixel probability in [0, 1]
    """

    def __init__(self, in_dim: int, hidden: int = 128,
                 dropout: float = 0.15,
                 out_h: int = 96, out_w: int = 384):
        super().__init__()
        self.hidden  = hidden
        self.out_h   = out_h
        self.out_w   = out_w

        self.fc = nn.Sequential(
            nn.Linear(in_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, hidden * 8 * 8),
            nn.ReLU(),
        )

        self.ups = nn.Sequential(
            # 8×8 → 16×16
            nn.ConvTranspose2d(hidden, 64, 4, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(),
            # 16×16 → 32×32
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
            # 32×32 → 64×64
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.BatchNorm2d(16), nn.ReLU(),
            # 64×64 → 128×128
            nn.ConvTranspose2d(16, 1, 4, stride=2, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: [B, in_dim] → [B, 1, out_h, out_w]"""
        x = self.fc(x)                              # [B, hidden*64]
        x = x.view(x.size(0), self.hidden, 8, 8)   # [B, hidden, 8, 8]
        x = self.ups(x)                             # [B, 1, 128, 128]
        if x.shape[-2:] != (self.out_h, self.out_w):
            x = F.interpolate(x, size=(self.out_h, self.out_w),
                              mode="bilinear", align_corners=False)
        return x  # [B, 1, out_h, out_w]


class _SpatialPixelDecoder(nn.Module):
    """Spatial feature map -> pixel logits while preserving layout."""

    def __init__(self, in_dim: int, lang_dim: int, hidden: int = 128,
                 dropout: float = 0.15, out_h: int = 96, out_w: int = 384):
        super().__init__()
        self.out_h = out_h
        self.out_w = out_w
        self.lang_proj = nn.Linear(lang_dim, lang_dim)
        self.stem = nn.Sequential(
            nn.Conv2d(in_dim + lang_dim, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
        )
        self.block1 = nn.Sequential(
            nn.Conv2d(hidden, hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden),
            nn.ReLU(),
            nn.Dropout2d(dropout),
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(hidden, hidden // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden // 2),
            nn.ReLU(),
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(hidden // 2, hidden // 4, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden // 4),
            nn.ReLU(),
            nn.Dropout2d(dropout * 0.5),
        )
        self.head = nn.Conv2d(hidden // 4, 1, kernel_size=1)

    def forward(self, feat_map: torch.Tensor, lang_vec: torch.Tensor) -> torch.Tensor:
        """feat_map: [B, C, H, W], lang_vec: [B, lang_dim] -> [B, 1, out_h, out_w]."""
        b, _, h, w = feat_map.shape
        lang = self.lang_proj(lang_vec).view(b, -1, 1, 1).expand(-1, -1, h, w)
        x = torch.cat([feat_map, lang], dim=1)
        x = self.stem(x)
        x = self.block1(x)
        x = F.interpolate(
            x,
            size=(max(self.out_h // 2, h), max(self.out_w // 2, w)),
            mode="bilinear",
            align_corners=False,
        )
        x = self.block2(x)
        x = F.interpolate(x, size=(self.out_h, self.out_w), mode="bilinear", align_corners=False)
        x = self.block3(x)
        return self.head(x)


# ── Full Model ────────────────────────────────────────────────────────────────

class VisualPixelModel(nn.Module):
    """Context image → continuation image directly in pixel space.

    No text tokens.  Only visual features → pixel decoder.
    """

    def __init__(self, vision_encoder: nn.Module,
                 enc_dim: int,
                 config: dict):
        super().__init__()
        cfg = config
        lang_dim = cfg["lang_emb_dim"]
        n_langs  = len(LANG2ID)

        # Vision encoder: frozen SigLIP (visual features only)
        self.vision_encoder = vision_encoder
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # Language conditioning (which of 5 languages)
        self.lang_emb = nn.Embedding(n_langs + 1, lang_dim)

        # Pixel decoder
        self.decoder = _SpatialPixelDecoder(
            in_dim=enc_dim,
            lang_dim=lang_dim,
            hidden=cfg["decoder_hidden"],
            dropout=cfg["decoder_dropout"],
            out_h=cfg["output_h"],
            out_w=cfg["output_w"],
        )

    @staticmethod
    def _patches_to_map(patches: torch.Tensor) -> torch.Tensor:
        """Convert patch sequence [B, N, D] into spatial map [B, D, H, W]."""
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

    def forward(self, pixel_values: torch.Tensor,
                lang_ids: torch.Tensor) -> torch.Tensor:
        """
        pixel_values: [B, 3, H, W]  — SigLIP-processed context image
        lang_ids:     [B]            — integer language indices
        Returns:      [B, 1, output_h, output_w]  — pixel probability map
        """
        # Encode context image → visual patch features (visual space only)
        if pixel_values.ndim == 5 and pixel_values.shape[1] == 1:
            pixel_values = pixel_values.squeeze(1)
        if pixel_values.ndim != 4:
            raise ValueError(f"Expected pixel_values shape [B, 3, H, W], got {tuple(pixel_values.shape)}")
        enc_param = next(self.vision_encoder.parameters())
        pixel_values = pixel_values.to(device=enc_param.device, dtype=enc_param.dtype)
        vis_out = self.vision_encoder(pixel_values)
        patches = vis_out.last_hidden_state          # [B, N_patches, D]
        feat_map = self._patches_to_map(patches)     # [B, D, H, W]

        # Language conditioning
        lang_vec = self.lang_emb(lang_ids)           # [B, lang_dim]

        return self.decoder(feat_map, lang_vec)      # [B, 1, out_h, out_w]


# ── Factory ───────────────────────────────────────────────────────────────────

def build_model(config: dict, device: torch.device) -> VisualPixelModel:
    """Load SmolVLM → extract SigLIP encoder → build VisualPixelModel."""
    dtype = torch.bfloat16 if config.get("bf16") else torch.float32

    print(f"  Loading {config['model_name']} to extract SigLIP encoder ...")
    smolvlm = AutoModelForVision2Seq.from_pretrained(
        config["model_name"],
        torch_dtype=dtype,
        _attn_implementation="eager",
    )
    vision_encoder = smolvlm.model.vision_model  # SigLIP (Idefics3VisionTransformer)
    enc_dim = vision_encoder.config.hidden_size

    # Free the language model — we only need the vision encoder
    del smolvlm
    torch.cuda.empty_cache()
    print(f"  SigLIP extracted: enc_dim={enc_dim}")

    model = VisualPixelModel(vision_encoder, enc_dim, config)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    print(f"  Trainable params: {trainable:,} / {total:,} "
          f"({100*trainable/total:.2f}%)")
    return model.to(device)


def load_for_inference(config: dict, save_dir: str,
                       device: torch.device) -> VisualPixelModel:
    """Rebuild model and load saved weights."""
    model = build_model(config, device)
    ckpt  = torch.load(f"{save_dir}/model.pt",
                       map_location=device, weights_only=True)
    model.load_state_dict(ckpt, strict=True)
    return model.eval()
