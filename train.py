import pandas as pd
from transformers import (
    T5ForConditionalGeneration,
    T5Tokenizer,
    Trainer,
    TrainingArguments
)
from dataset import KaggleDataset
from config import get_config


def train():
    config = get_config()

    # Load tokenizer + model
    tokenizer = T5Tokenizer.from_pretrained(config["model_name"])
    model = T5ForConditionalGeneration.from_pretrained(config["model_name"])

    # Load data
    df = pd.read_csv(config["train_file"])

    train_size = int(0.99 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]

    train_ds = KaggleDataset(train_df, tokenizer, config["max_len"])
    val_ds = KaggleDataset(val_df, tokenizer, config["max_len"])

    training_args = TrainingArguments(
        output_dir=config["output_dir"],
        per_device_train_batch_size=config["batch_size"],
        per_device_eval_batch_size=1,
        num_train_epochs=config["num_epochs"],
        learning_rate=config["lr"],
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_dir="./logs",
        fp16=True  # GPU speedup
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer
    )

    trainer.train()

    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])


if __name__ == "__main__":
    train()