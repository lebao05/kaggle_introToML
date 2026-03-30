import pandas as pd
import torch
from torch.utils.data import Dataset

from transformers import (
    BartConfig,
    BartForConditionalGeneration,
    AutoTokenizer,
    Trainer,
    TrainingArguments
)


# =========================
# Dataset
# =========================
class Seq2SeqDataset(Dataset):
    def __init__(self, df, tokenizer, max_len):
        self.df = df
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # CHANGE THESE COLUMN NAMES to match your CSV
        src_text = str(row["context"])
        tgt_text = str(row["target"])

        # Tokenize input
        src = self.tokenizer(
            src_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        # Tokenize target
        tgt = self.tokenizer(
            tgt_text,
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = src["input_ids"].squeeze()
        attention_mask = src["attention_mask"].squeeze()

        labels = tgt["input_ids"].squeeze()

        # IMPORTANT: ignore padding in loss
        labels[labels == self.tokenizer.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels
        }


# =========================
# Train function
# =========================
def train():
    config = {
        "train_file": "train.csv",
        "output_dir": "./outputs",
        "max_len": 200,
        "batch_size": 8,
        "num_epochs": 5,
        "lr": 5e-4,
    }

    # =========================
    # Tokenizer
    # =========================
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    # BART needs these tokens
    tokenizer.pad_token = tokenizer.pad_token or "[PAD]"

    # =========================
    # Model (FROM SCRATCH)
    # =========================
    model_config = BartConfig(
        vocab_size=tokenizer.vocab_size,
        d_model=512,
        encoder_layers=6,
        decoder_layers=6,
        encoder_attention_heads=8,
        decoder_attention_heads=8,
        encoder_ffn_dim=2048,
        decoder_ffn_dim=2048,
        max_position_embeddings=512,
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.cls_token_id,
        eos_token_id=tokenizer.sep_token_id,
    )

    model = BartForConditionalGeneration(model_config)

    # =========================
    # Load data
    # =========================
    df = pd.read_csv(config["train_file"])

    train_size = int(0.95 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]

    train_ds = Seq2SeqDataset(train_df, tokenizer, config["max_len"])
    val_ds = Seq2SeqDataset(val_df, tokenizer, config["max_len"])

    # =========================
    # Training arguments
    # =========================
    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        num_train_epochs=config["num_epochs"],
        learning_rate=config["lr"],

        evaluation_strategy="steps",
        eval_steps=100,

        logging_strategy="steps",
        logging_steps=20,
        logging_dir="./logs",

        save_strategy="steps",
        save_steps=500,

        report_to="tensorboard",

        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",

        fp16=torch.cuda.is_available(),
    )

    # =========================
    # Trainer
    # =========================
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
    )

    # =========================
    # Train
    # =========================
    trainer.train()

    # =========================
    # Save
    # =========================
    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    train()