"""Model: TrOCR encoder + Projection bridge + mT5 decoder.

Tại sao kiến trúc này?
  - TrOCR encoder (ViT): đọc ảnh rendered text → patch embeddings
    Pretrain trên OCR → đã biết cách đọc chữ từ ảnh
    Không có transcription bias ở đây — chỉ là feature extractor

  - mT5 decoder: sinh text continuation
    Pretrain trên text generation task → đúng bản chất continuation
    Multilingual → hỗ trợ tốt cả 5 ngôn ngữ trong dataset
    KHÔNG có transcription bias như TrOCR decoder

  - Projection bridge (768 → 512):
    TrOCR encoder output: 768 dim
    mT5 decoder input:    512 dim
    Cần MLP để ánh xạ giữa 2 không gian này
    Residual skip: tránh MLP collapse tất cả token về 1 điểm
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoConfig,
    MT5ForConditionalGeneration,
    VisionEncoderDecoderModel,
)


class Projection(nn.Module):
    """MLP bridge: encoder_dim → decoder_dim với residual skip.

    Residual skip quan trọng: không có nó, MLP có thể collapse
    tất cả 577 patch embeddings về cùng 1 vector → decoder mất
    khả năng attend vào các patch khác nhau.
    """

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.05):
        super().__init__()
        self.fc1  = nn.Linear(in_dim, out_dim * 2)
        self.act  = nn.GELU()
        self.fc2  = nn.Linear(out_dim * 2, out_dim)
        self.norm = nn.LayerNorm(out_dim)
        self.drop = nn.Dropout(dropout)
        self.skip = nn.Linear(in_dim, out_dim, bias=False)  # residual path

    def forward(self, x):
        # x: (B, seq_len, in_dim)
        return self.norm(self.drop(self.fc2(self.act(self.fc1(x)))) + self.skip(x))


class TrOCRmT5Model(nn.Module):
    """TrOCR ViT encoder + Projection + mT5 decoder.

    Forward (training):
        pixel_values (B,3,384,384) + labels (B,T) → cross-entropy loss

    Generate (inference):
        pixel_values (B,3,384,384) → List[str]
    """

    def __init__(self, config):
        super().__init__()
        self.cfg = config

        # ── 1. TrOCR encoder (ViT) ────────────────────────────────────────────
        # Chỉ lấy encoder, bỏ TrOCR decoder (không dùng)
        trocr          = VisionEncoderDecoderModel.from_pretrained(config["trocr_name"])
        self.encoder   = trocr.encoder          # ViT: (B,3,384,384) → (B,577,768)
        del trocr

        # ── 2. Projection bridge ─────────────────────────────────────────────
        self.proj = Projection(
            in_dim  = config["encoder_dim"],    # 768
            out_dim = config["decoder_dim"],    # 512
            dropout = config["proj_dropout"],
        )
        # Scale để đưa memory magnitude gần với decoder embedding magnitude
        # Quan trọng: nếu memory quá nhỏ so với dec embedding, decoder ignore encoder
        self.memory_scale = nn.Parameter(torch.tensor(8.0))
        self.memory_norm  = nn.LayerNorm(config["decoder_dim"])

        # ── 3. mT5 decoder + lm_head ─────────────────────────────────────────
        mt5_cfg = AutoConfig.from_pretrained(config["mt5_name"])
        mt5_cfg.tie_word_embeddings = False     # tránh ambiguous tie behavior
        mt5             = MT5ForConditionalGeneration.from_pretrained(
                              config["mt5_name"], config=mt5_cfg)
        self.decoder    = mt5.decoder           # T5 stack decoder
        self.lm_head    = mt5.lm_head           # linear: 512 → vocab_size
        self.mt5_config = mt5.config
        del mt5

        self._init_projection()

    def _init_projection(self):
        """Xavier init cho projection, zero/one cho norm — không đụng encoder/decoder."""
        for name, p in self.named_parameters():
            if any(k in name for k in ("encoder", "decoder", "lm_head", "memory_scale")):
                continue
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            elif "norm" in name.lower():
                nn.init.ones_(p)
            else:
                nn.init.zeros_(p)

    # ── Encode ────────────────────────────────────────────────────────────────

    def encode(self, pixel_values: torch.Tensor,
               patch_drop: float = 0.0) -> torch.Tensor:
        """
        pixel_values: (B, 3, 384, 384)
        Returns memory: (B, 577, 512)
        """
        # TrOCR encoder: ảnh → 577 patch embeddings, mỗi cái 768-dim
        enc_out = self.encoder(pixel_values=pixel_values).last_hidden_state
        # (B, 577, 768)

        # Patch dropout (training only): buộc decoder phải dùng nhiều patch
        # thay vì chỉ attend vào 1-2 patch rồi ignore phần còn lại
        if patch_drop > 0.0 and self.training:
            B, S, _ = enc_out.shape
            mask    = torch.bernoulli(
                torch.full((B, S, 1), 1.0 - patch_drop, device=enc_out.device)
            )
            enc_out = enc_out * mask

        # Projection: 768 → 512
        memory = self.proj(enc_out)                          # (B, 577, 512)
        memory = self.memory_norm(memory) * self.memory_scale
        return memory

    # ── Helpers ───────────────────────────────────────────────────────────────

    @staticmethod
    def _ones_mask(x: torch.Tensor) -> torch.Tensor:
        return torch.ones(x.shape[:2], dtype=torch.long, device=x.device)

    def _to_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        if bool(getattr(self.mt5_config, "tie_word_embeddings", False)):
            hidden = hidden * (self.mt5_config.d_model ** -0.5)
        return self.lm_head(hidden).float()

    @staticmethod
    def _shift_right(labels: torch.Tensor,
                     start_id: int, pad_id: int) -> torch.Tensor:
        """Tạo decoder input: shift labels sang phải, đặt start_id ở đầu."""
        shifted        = labels.new_full(labels.shape, pad_id)
        shifted[:, 1:] = labels[:, :-1].clone()
        shifted[:, 0]  = start_id
        shifted[shifted == -100] = pad_id
        return shifted

    # ── Forward (training) ────────────────────────────────────────────────────

    def forward(self, pixel_values: torch.Tensor,
                labels: torch.Tensor,
                patch_drop: float = 0.0) -> torch.Tensor:
        """
        pixel_values: (B, 3, 384, 384)
        labels:       (B, T) — pad positions = -100
        Returns: scalar loss
        """
        memory      = self.encode(pixel_values, patch_drop)
        memory_mask = self._ones_mask(memory)

        pad_id   = self.mt5_config.pad_token_id
        start_id = self.mt5_config.decoder_start_token_id or pad_id
        dec_in   = self._shift_right(labels, start_id, pad_id)
        dec_mask = (dec_in != pad_id).long()
        dec_mask[:, 0] = 1

        dec_out = self.decoder(
            input_ids             = dec_in,
            attention_mask        = dec_mask,
            encoder_hidden_states = memory,
            encoder_attention_mask= memory_mask,
            return_dict           = True,
        )
        logits = self._to_logits(dec_out.last_hidden_state)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30, 30)

        loss = nn.CrossEntropyLoss(ignore_index=-100)(
            logits.reshape(-1, logits.size(-1)),
            labels.reshape(-1),
        )
        return loss

    # ── Generate (inference) ──────────────────────────────────────────────────

    @torch.no_grad()
    def generate(self, pixel_values: torch.Tensor,
                 tokenizer, config) -> list[str]:
        """Beam search generation. Returns list[str] length B."""
        memory = self.encode(pixel_values, patch_drop=0.0)
        return self._beam_search(memory, tokenizer, config)

    @torch.no_grad()
    def _beam_search(self, memory: torch.Tensor,
                     tokenizer, config) -> list[str]:
        B      = memory.size(0)
        device = memory.device

        pad_id  = self.mt5_config.pad_token_id
        eos_id  = tokenizer.eos_token_id or pad_id
        max_len = config["max_gen_len"]
        min_len = config["min_gen_len"]
        beam_k  = config["num_beams"]
        rep_pen = config["repetition_penalty"]
        len_pen = config["length_penalty"]
        ngram_n = config["no_repeat_ngram_size"]

        # Block mT5 sentinel tokens (<extra_id_0>...<extra_id_99>)
        sentinel_ids = set()
        for i in range(100):
            tid = tokenizer.convert_tokens_to_ids(f"<extra_id_{i}>")
            if tid != tokenizer.unk_token_id:
                sentinel_ids.add(tid)

        results = []
        for b in range(B):
            mem      = memory[b:b+1]
            mem_mask = self._ones_mask(mem)
            beams, done = [(0.0, [pad_id])], []

            for _ in range(max_len):
                cands = []
                for score, toks in beams:
                    if toks[-1] == eos_id:
                        done.append((score, toks))
                        continue
                    inp = torch.tensor([toks], dtype=torch.long, device=device)
                    msk = (inp != pad_id).long()
                    msk[:, 0] = 1
                    out = self.decoder(
                        input_ids=inp, attention_mask=msk,
                        encoder_hidden_states=mem,
                        encoder_attention_mask=mem_mask,
                        return_dict=True,
                    )
                    logits = self._to_logits(out.last_hidden_state[:, -1])[0]
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=30.0, neginf=-30.0).clamp(-30, 30)
                    logp   = F.log_softmax(logits, dim=-1)
                    V      = logp.numel()

                    # Min length: chặn EOS sớm
                    if len(toks) - 1 < min_len:
                        logp[eos_id] = -1e9

                    # Block sentinel tokens
                    for tid in sentinel_ids:
                        if tid != eos_id and 0 <= tid < V:
                            logp[tid] = -1e9

                    # Repetition penalty
                    for t in set(toks):
                        if t in {pad_id, eos_id} or not (0 <= t < V):
                            continue
                        logp[t] = logp[t] * rep_pen if logp[t] < 0 else logp[t] / rep_pen

                    # No-repeat ngram
                    if ngram_n > 0 and len(toks) >= ngram_n:
                        prefix = tuple(toks[-(ngram_n-1):])
                        for i in range(len(toks) - ngram_n + 1):
                            if tuple(toks[i:i+ngram_n-1]) == prefix:
                                banned = toks[i+ngram_n-1]
                                if 0 <= banned < V:
                                    logp[banned] = -1e9

                    top_vals, top_ids = logp.topk(beam_k)
                    for v, tid in zip(top_vals, top_ids):
                        cands.append((score + v.item(), toks + [tid.item()]))

                if not cands:
                    break
                cands.sort(key=lambda x: x[0] / max((len(x[1])-1)**len_pen, 1e-6), reverse=True)
                beams = cands[:beam_k]
                if all(t[-1] == eos_id for _, t in beams):
                    done.extend(beams)
                    break

            done.extend(beams)
            best = max(done, key=lambda x: x[0] / max((len(x[1])-1)**len_pen, 1e-6))
            ids  = [t for t in best[1] if t not in {pad_id, eos_id}]
            results.append(tokenizer.decode(ids, skip_special_tokens=True).strip())

        return results
