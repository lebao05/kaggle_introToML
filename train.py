import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import warnings

from model import build_transformer
from dataset import KaggleDataset, causal_mask
from config import get_config, get_weights_file_path, latest_weights_file_path

# Tokenizer
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace


# =========================
# Tokenizer
# =========================
def get_or_build_tokenizer(config, df):
    path = Path(config["tokenizer_file"])

    if not path.exists():
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2
        )

        def iterator():
            for _, row in df.iterrows():
                yield row["context"]
                yield row["target"]

        tokenizer.train_from_iterator(iterator(), trainer)
        tokenizer.save(str(path))
    else:
        tokenizer = Tokenizer.from_file(str(path))

    return tokenizer


# =========================
# Dataset
# =========================
def get_ds(config):
    df = pd.read_csv(config["train_file"])

    tokenizer = get_or_build_tokenizer(config, df)

    train_size = int(0.9 * len(df))
    train_df = df[:train_size]
    val_df = df[train_size:]

    train_ds = KaggleDataset(train_df, tokenizer, config["seq_len"])
    val_ds = KaggleDataset(val_df, tokenizer, config["seq_len"])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=1)

    return train_loader, val_loader, tokenizer


# =========================
# Greedy Decode
# =========================
def greedy_decode(model, source, source_mask, tokenizer, max_len, device):
    sos_idx = tokenizer.token_to_id('[SOS]')
    eos_idx = tokenizer.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)

    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    while True:
        if decoder_input.size(1) == max_len:
            break

        decoder_mask = causal_mask(decoder_input.size(1)).to(device)

        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        prob = model.project(out[:, -1])

        _, next_word = torch.max(prob, dim=1)

        decoder_input = torch.cat(
            [decoder_input, next_word.unsqueeze(0)], dim=1
        )

        if next_word.item() == eos_idx:
            break

    return decoder_input.squeeze(0)


# =========================
# Validation
# =========================
def run_validation(model, val_loader, tokenizer, device, max_len=50):
    model.eval()

    with torch.no_grad():
        for i, batch in enumerate(val_loader):
            encoder_input = batch["encoder_input"].to(device)

            encoder_mask = (encoder_input != tokenizer.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(1)

            output = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer,
                max_len,
                device
            )

            pred_text = tokenizer.decode(output.cpu().numpy())

            print("=" * 50)
            print("PRED:", pred_text)

            if i == 2:
                break


# =========================
# Model
# =========================
def get_model(config, vocab_size):
    return build_transformer(
        vocab_size,
        vocab_size,
        config["seq_len"],
        config["seq_len"],
        d_model=config["d_model"]
    )


# =========================
# Training
# =========================
def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    model_folder = f"{config['datasource']}_{config['model_folder']}"
    Path(model_folder).mkdir(parents=True, exist_ok=True)
    train_loader, val_loader, tokenizer = get_ds(config)

    model = get_model(config, tokenizer.get_vocab_size()).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer.token_to_id('[PAD]'),
        label_smoothing=0.1
    ).to(device)

    # Load checkpoint if exists
    initial_epoch = 0
    global_step = 0

    model_filename = latest_weights_file_path(config)

    if model_filename:
        print("Loading checkpoint:", model_filename)
        state = torch.load(model_filename)

        model.load_state_dict(state["model_state_dict"])
        optimizer.load_state_dict(state["optimizer_state_dict"])
        initial_epoch = state["epoch"] + 1
        global_step = state["global_step"]

    # Training loop
    for epoch in range(initial_epoch, config["num_epochs"]):
        model.train()
        batch_iterator = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch in batch_iterator:
            encoder_input = batch["encoder_input"].to(device)
            decoder_input = batch["decoder_input"].to(device)
            label = batch["label"].to(device)

            encoder_mask = (encoder_input != tokenizer.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(1)
            decoder_mask = (decoder_input != tokenizer.token_to_id('[PAD]')).unsqueeze(1) & causal_mask(decoder_input.size(1)).to(device)

            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask, decoder_input, decoder_mask)

            proj_output = model.project(decoder_output)

            loss = loss_fn(
                proj_output.view(-1, tokenizer.get_vocab_size()),
                label.view(-1)
            )

            batch_iterator.set_postfix({"loss": f"{loss.item():.4f}"})

            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            global_step += 1

        # Validation
        run_validation(model, val_loader, tokenizer, device)

        # Save model
        model_path = get_weights_file_path(config, f"{epoch:02d}")

        torch.save({
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "global_step": global_step
        }, model_path)


# =========================
# Main
# =========================
if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)