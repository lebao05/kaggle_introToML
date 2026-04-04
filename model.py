"""Seq2seq continuation model for OCR -> LM pipeline."""

import torch.nn as nn
from transformers import AutoConfig, AutoModelForSeq2SeqLM


class TrOCRmT5Model(nn.Module):
    def __init__(self, config, model_dir=None):
        super().__init__()
        source = model_dir or config["lm_model_name"]
        lm_cfg = AutoConfig.from_pretrained(source)
        if getattr(lm_cfg, "tie_word_embeddings", None) is not None:
            lm_cfg.tie_word_embeddings = False
        self.model = AutoModelForSeq2SeqLM.from_pretrained(source, config=lm_cfg)

    def forward(self, input_ids, attention_mask, labels):
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels).loss

    def generate(self, input_ids, attention_mask, tokenizer, config):
        gen_kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "max_new_tokens": config["max_gen_len"],
            "num_beams": config["num_beams"],
            "repetition_penalty": config["repetition_penalty"],
            "length_penalty": config["length_penalty"],
            "no_repeat_ngram_size": config["no_repeat_ngram_size"],
            "early_stopping": True,
        }
        try:
            gen_ids = self.model.generate(**gen_kwargs, min_new_tokens=config["min_gen_len"])
        except TypeError:
            gen_ids = self.model.generate(**gen_kwargs, min_length=config["min_gen_len"] + 1)
        return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

    def save_pretrained(self, output_dir):
        self.model.save_pretrained(output_dir)
